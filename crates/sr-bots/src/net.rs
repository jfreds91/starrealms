//! Native forward pass of the Python `starrealms.model.Policy`, for serving
//! trained models without PyTorch (server, wasm).
//!
//! File format (`.srnn`, written by `python -m starrealms.export`):
//! `b"SRNN0001"`, u32 LE header length, JSON header, then little-endian f32
//! tensors at the byte offsets (relative to the data start) listed in the header.

use crate::Agent;
use serde::Deserialize;
use sr_core::obs::{LAYOUT, N_IDS};
use sr_core::{Game, Rng, N_ACTIONS, N_CARDS, OBS_DIM};

const MAGIC: &[u8; 8] = b"SRNN0001";
const COUNT_SCALE: f32 = 0.25;

#[derive(Deserialize)]
struct Header {
    obs_dim: usize,
    n_actions: usize,
    n_cards: usize,
    embed: usize,
    hidden: usize,
    layers: usize,
    tensors: Vec<TensorInfo>,
}

#[derive(Deserialize)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    offset: usize,
}

struct Linear {
    /// Stored transposed: `[in][out]`, so sparse inputs can skip columns.
    wt: Vec<f32>,
    b: Vec<f32>,
    n_in: usize,
    n_out: usize,
}

impl Linear {
    fn forward(&self, x: &[f32], y: &mut Vec<f32>) {
        y.clear();
        y.extend_from_slice(&self.b);
        for (i, &xi) in x.iter().enumerate() {
            if xi != 0.0 {
                let row = &self.wt[i * self.n_out..(i + 1) * self.n_out];
                for (yo, &w) in y.iter_mut().zip(row) {
                    *yo += xi * w;
                }
            }
        }
    }
}

fn seg(name: &str) -> (usize, usize) {
    let &(_, off, len) = LAYOUT.iter().find(|s| s.0 == name).expect("layout segment");
    (off, off + len)
}

pub struct Net {
    embed: Vec<f32>,
    d: usize,
    trunk: Vec<Linear>,
    pi: Linear,
    v: Linear,
}

impl Net {
    pub fn from_bytes(bytes: &[u8]) -> Result<Net, String> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err("not an SRNN0001 file".into());
        }
        let hlen = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let header: Header =
            serde_json::from_slice(bytes.get(12..12 + hlen).ok_or("truncated header")?).map_err(|e| e.to_string())?;
        if header.obs_dim != OBS_DIM || header.n_actions != N_ACTIONS || header.n_cards != N_CARDS {
            return Err(format!(
                "model built for obs_dim {} / {} actions / {} cards; engine has {OBS_DIM} / {N_ACTIONS} / {N_CARDS}",
                header.obs_dim, header.n_actions, header.n_cards
            ));
        }
        let data = &bytes[12 + hlen..];
        let tensor = |name: &str| -> Result<(Vec<f32>, Vec<usize>), String> {
            let t = header.tensors.iter().find(|t| t.name == name).ok_or(format!("missing tensor {name}"))?;
            let n: usize = t.shape.iter().product();
            let raw = data.get(t.offset..t.offset + 4 * n).ok_or(format!("tensor {name} out of bounds"))?;
            Ok((raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(), t.shape.clone()))
        };
        let linear = |prefix: &str| -> Result<Linear, String> {
            let (w, shape) = tensor(&format!("{prefix}.weight"))?;
            let (b, _) = tensor(&format!("{prefix}.bias"))?;
            let (n_out, n_in) = (shape[0], shape[1]);
            let mut wt = vec![0.0; n_in * n_out];
            for o in 0..n_out {
                for i in 0..n_in {
                    wt[i * n_out + o] = w[o * n_in + i];
                }
            }
            Ok(Linear { wt, b, n_in, n_out })
        };
        let (embed, eshape) = tensor("embed.weight")?;
        if eshape != [N_CARDS + 1, header.embed] {
            return Err("embedding shape mismatch".into());
        }
        let trunk = (0..header.layers).map(|i| linear(&format!("trunk.{}", 2 * i))).collect::<Result<Vec<_>, _>>()?;
        let net = Net { embed, d: header.embed, trunk, pi: linear("pi")?, v: linear("v")? };
        let expect = net.input_dim();
        if net.trunk[0].n_in != expect || net.trunk.last().unwrap().n_out != header.hidden {
            return Err(format!("trunk input {} != expected {expect}", net.trunk[0].n_in));
        }
        Ok(net)
    }

    fn input_dim(&self) -> usize {
        let (cs, ce) = seg("counts");
        let (is, _) = seg("ids");
        let zones = (ce - cs) / N_CARDS;
        (cs + (is - ce)) + (ce - cs) + N_IDS * self.d + zones * self.d
    }

    fn features(&self, obs: &[f32]) -> Vec<f32> {
        let (cs, ce) = seg("counts");
        let (is, ie) = seg("ids");
        let d = self.d;
        let zones = (ce - cs) / N_CARDS;
        let mut x = Vec::with_capacity(self.input_dim());
        x.extend_from_slice(&obs[..cs]);
        x.extend_from_slice(&obs[ce..is]);
        x.extend(obs[cs..ce].iter().map(|v| v * COUNT_SCALE));
        for &id in &obs[is..ie] {
            let k = id as usize;
            x.extend_from_slice(&self.embed[k * d..(k + 1) * d]);
        }
        for z in 0..zones {
            let mut acc = vec![0.0f32; d];
            for c in 0..N_CARDS {
                let n = obs[cs + z * N_CARDS + c];
                if n != 0.0 {
                    let e = &self.embed[(c + 1) * d..(c + 2) * d];
                    for (a, &w) in acc.iter_mut().zip(e) {
                        *a += n * w;
                    }
                }
            }
            x.extend(acc.into_iter().map(|v| v * COUNT_SCALE));
        }
        x
    }

    /// Unmasked logits and value for an observation.
    pub fn forward(&self, obs: &[f32]) -> (Vec<f32>, f32) {
        let mut h = self.features(obs);
        let mut y = Vec::new();
        for l in &self.trunk {
            l.forward(&h, &mut y);
            for v in y.iter_mut() {
                *v = v.max(0.0);
            }
            std::mem::swap(&mut h, &mut y);
        }
        let mut logits = Vec::new();
        self.pi.forward(&h, &mut logits);
        let mut v = Vec::new();
        self.v.forward(&h, &mut v);
        (logits, v[0])
    }
}

/// Plays a `Net`. `temperature == 0` picks the most likely legal action.
pub struct NetAgent {
    net: std::sync::Arc<Net>,
    pub temperature: f32,
    name: String,
    obs: Vec<f32>,
}

impl NetAgent {
    pub fn new(net: std::sync::Arc<Net>, name: impl Into<String>, temperature: f32) -> NetAgent {
        NetAgent { net, temperature, name: name.into(), obs: vec![0.0; OBS_DIM] }
    }

    /// Legal actions with their probabilities, plus the value estimate.
    pub fn policy(&mut self, g: &Game, legal: &[u16]) -> (Vec<(u16, f32)>, f32) {
        g.observe(&mut self.obs);
        let (logits, v) = self.net.forward(&self.obs);
        let t = if self.temperature > 0.0 { self.temperature } else { 1.0 };
        let m = legal.iter().map(|&a| logits[a as usize]).fold(f32::NEG_INFINITY, f32::max);
        let ws: Vec<f32> = legal.iter().map(|&a| ((logits[a as usize] - m) / t).exp()).collect();
        let z: f32 = ws.iter().sum();
        (legal.iter().zip(ws).map(|(&a, w)| (a, w / z)).collect(), v)
    }
}

impl Agent for NetAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&mut self, g: &Game, legal: &[u16], rng: &mut Rng) -> u16 {
        let greedy = self.temperature <= 0.0;
        let (probs, _) = self.policy(g, legal);
        if greedy {
            return probs.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|p| p.0).unwrap_or(legal[0]);
        }
        let mut r = rng.f32();
        for &(a, p) in &probs {
            r -= p;
            if r <= 0.0 {
                return a;
            }
        }
        probs.last().map(|p| p.0).unwrap_or(legal[0])
    }
}
