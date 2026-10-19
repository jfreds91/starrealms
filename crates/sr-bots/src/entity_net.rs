//! Native forward pass of the Python `starrealms.model.EntityPolicy` (entity
//! transformer with a pointer action head), built on candle.
//!
//! File format: safetensors written by `python -m starrealms.export`, with the
//! model config and the tokenizer constants in the `__metadata__` map. The
//! tokenizer mirrors `python/starrealms/tokens.py`; since one observation is
//! evaluated at a time, padding tokens are simply left out instead of masked.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear};
use serde::Deserialize;
use sr_core::action::{MAX_OPP_BASES, MAX_OWN_SLOTS, ROW};
use sr_core::obs::*;
use sr_core::{db, N_ACTIONS, N_CARDS, OBS_DIM};
use std::collections::HashMap;

const HAND: usize = 12;
const DISCARD: usize = 32;
const COUNT_SCALE: f32 = 0.25;
const LN_EPS: f64 = 1e-5;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Group {
    Own,
    Opp,
    Row,
    Explorer,
    Hand,
    Discard,
}

/// Token type ids, as in `tokens.token_types`: 0 global, 1..=9 zones, then one per group.
fn type_id(g: Group) -> usize {
    1 + COUNT_ZONES.len()
        + match g {
            Group::Own => 0,
            Group::Opp => 1,
            Group::Row => 2,
            Group::Explorer => 3,
            Group::Hand => 4,
            Group::Discard => 5,
        }
}

/// How each action family is scored, in action-id order (`sr_core::action`).
enum Family {
    /// `n` logits from the global head, starting at this offset of its output.
    Global(usize, usize),
    /// Pointer logits of verb `.0` over group `.1`.
    Pointer(usize, Group),
}

const VERBS: [&str; 14] = [
    "play",
    "primary",
    "ally",
    "ally2",
    "scrap",
    "buy",
    "buy_explorer",
    "attack_base",
    "pick_hand",
    "pick_discard",
    "pick_row",
    "pick_explorer",
    "pick_opp_base",
    "pick_own",
];

fn families() -> [(usize, Family); 18] {
    use Family::*;
    use Group::*;
    [
        (1, Global(0, 1)),           // end
        (1, Global(1, 1)),           // skip
        (sr_core::action::MAX_CHOICES, Global(2, sr_core::action::MAX_CHOICES)), // choose
        (N_CARDS, Pointer(0, Hand)), // play
        (MAX_OWN_SLOTS, Pointer(1, Own)),
        (MAX_OWN_SLOTS, Pointer(2, Own)),
        (MAX_OWN_SLOTS, Pointer(3, Own)),
        (MAX_OWN_SLOTS, Pointer(4, Own)),
        (ROW, Pointer(5, Row)),
        (1, Pointer(6, Explorer)),
        (1, Global(2 + sr_core::action::MAX_CHOICES, 1)), // attack player
        (MAX_OPP_BASES, Pointer(7, Opp)),
        (N_CARDS, Pointer(8, Hand)),
        (N_CARDS, Pointer(9, Discard)),
        (ROW, Pointer(10, Row)),
        (1, Pointer(11, Explorer)),
        (MAX_OPP_BASES, Pointer(12, Opp)),
        (MAX_OWN_SLOTS, Pointer(13, Own)),
    ]
}
const N_GLOBAL_LOGITS: usize = 3 + sr_core::action::MAX_CHOICES;

#[derive(Deserialize)]
struct Config {
    d: usize,
    layers: usize,
    heads: usize,
}

struct Block {
    ln1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

pub struct EntityNet {
    d: usize,
    heads: usize,
    /// `[N_CARDS + 1, d]` card vectors, row 0 zero.
    cards: Tensor,
    /// `[N_CARDS, d]` pointer keys of the bare cards (hand/discard overflow).
    card_keys: Tensor,
    type_emb: Tensor,
    global_in: Linear,
    zone_in: Linear,
    copy_in: Linear,
    own_in: Linear,
    opp_in: Linear,
    row_in: Linear,
    explorer_in: Linear,
    hand_in: Linear,
    discard_in: Linear,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    value1: Linear,
    value2: Linear,
    global_out: Linear,
    query: Linear,
    key: Linear,
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Distinct card ids (0-based) with a nonzero count, ascending, at most `k`, with their counts.
fn distinct(counts: &[f32], k: usize) -> Vec<(usize, f32)> {
    counts.iter().enumerate().filter(|(_, &n)| n > 0.0).take(k).map(|(c, &n)| (c, n)).collect()
}

impl EntityNet {
    pub fn from_bytes(bytes: &[u8]) -> Result<EntityNet, String> {
        let (_, meta) = safetensors::SafeTensors::read_metadata(bytes).map_err(err)?;
        let meta = meta.metadata().clone().unwrap_or_default();
        let get = |k: &str| meta.get(k).cloned().ok_or_else(|| format!("missing metadata {k}"));
        if get("arch")? != "entity" {
            return Err("not an entity model".into());
        }
        let expect = [
            ("obs_dim", OBS_DIM.to_string()),
            ("n_actions", N_ACTIONS.to_string()),
            ("n_cards", N_CARDS.to_string()),
            ("hand", HAND.to_string()),
            ("discard", DISCARD.to_string()),
            ("verbs", VERBS.join(",")),
        ];
        for (k, v) in expect {
            let got = get(k)?;
            if got != v {
                return Err(format!("model built for {k} = {got}, engine has {v}"));
            }
        }
        let cfg: Config = serde_json::from_str(&get("config")?).map_err(err)?;
        let dev = Device::Cpu;
        let ts: HashMap<String, Tensor> = candle_core::safetensors::load_buffer(bytes, &dev).map_err(err)?;
        let t = |name: &str| -> Result<Tensor, String> {
            ts.get(name).ok_or_else(|| format!("missing tensor {name}"))?.to_dtype(DType::F32).map_err(err)
        };
        let linear = |p: &str| -> Result<Linear, String> {
            Ok(Linear::new(t(&format!("{p}.weight"))?, ts.get(&format!("{p}.bias")).cloned()))
        };
        let ln = |p: &str| -> Result<LayerNorm, String> {
            Ok(LayerNorm::new(t(&format!("{p}.weight"))?, t(&format!("{p}.bias"))?, LN_EPS))
        };
        let blocks = (0..cfg.layers)
            .map(|i| {
                let p = format!("blocks.{i}");
                Ok(Block {
                    ln1: ln(&format!("{p}.ln1"))?,
                    qkv: linear(&format!("{p}.qkv"))?,
                    proj: linear(&format!("{p}.proj"))?,
                    ln2: ln(&format!("{p}.ln2"))?,
                    ff1: linear(&format!("{p}.ff1"))?,
                    ff2: linear(&format!("{p}.ff2"))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let card_feat = linear("card_feat")?;
        let table = (t("card_id.weight")? + card_feat.forward(&t("card_feats")?).map_err(err)?).map_err(err)?;
        let zero = Tensor::zeros((1, cfg.d), DType::F32, &dev).map_err(err)?;
        let cards = Tensor::cat(&[&zero, &table.narrow(0, 1, N_CARDS).map_err(err)?], 0).map_err(err)?;
        let ln_f = ln("ln_f")?;
        let key = linear("key")?;
        let card_keys = key
            .forward(&ln_f.forward(&cards.narrow(0, 1, N_CARDS).map_err(err)?).map_err(err)?)
            .map_err(err)?;
        if cards.dims() != [N_CARDS + 1, cfg.d] {
            return Err(format!("card table shape {:?}", cards.dims()));
        }
        Ok(EntityNet {
            d: cfg.d,
            heads: cfg.heads,
            cards,
            card_keys,
            type_emb: t("type_emb.weight")?,
            global_in: linear("global_in")?,
            zone_in: linear("zone_in")?,
            copy_in: linear("copy_in")?,
            own_in: linear("own_in")?,
            opp_in: linear("opp_in")?,
            row_in: linear("row_in")?,
            explorer_in: linear("explorer_in")?,
            hand_in: linear("hand_in")?,
            discard_in: linear("discard_in")?,
            blocks,
            ln_f,
            value1: linear("value1")?,
            value2: linear("value2")?,
            global_out: linear("global_out")?,
            query: linear("query")?,
            key,
        })
    }

    /// Unmasked logits and value for an observation.
    pub fn forward(&self, obs: &[f32]) -> (Vec<f32>, f32) {
        self.try_forward(obs).expect("entity forward pass")
    }

    fn try_forward(&self, obs: &[f32]) -> candle_core::Result<(Vec<f32>, f32)> {
        let dev = Device::Cpu;
        let d = self.d;
        let row = |x: &[f32]| Tensor::from_slice(x, (1, x.len()), &dev);
        let card = |id: usize| self.cards.narrow(0, id, 1);
        let ids = &obs[OFF_IDS..OFF_IDS + N_IDS];
        let id_at = |i: usize| ids[i] as usize;

        // Token inputs, global first; `slots` maps (group, index) to the token position.
        let mut xs: Vec<Tensor> = Vec::new();
        let mut types: Vec<u32> = Vec::new();
        let mut slots: Vec<(Group, usize, usize)> = Vec::new();
        let mut dense = obs[OFF_SCALARS..OFF_SCALARS + N_SCALARS].to_vec();
        dense.extend_from_slice(&obs[OFF_PENDING..OFF_PENDING + N_PENDING]);
        xs.push((self.global_in.forward(&row(&dense)?)? + card(id_at(N_IDS - 1))?)?);
        types.push(0);

        let counts = &obs[OFF_COUNTS..OFF_COUNTS + N_COUNTS];
        let zones: Vec<f32> = counts.iter().map(|c| c * COUNT_SCALE).collect();
        let zt = Tensor::from_slice(&zones, (COUNT_ZONES.len(), N_CARDS), &dev)?;
        let bag = zt.matmul(&self.cards.narrow(0, 1, N_CARDS)?)?;
        let total = zt.sum_keepdim(1)?;
        xs.push(self.zone_in.forward(&Tensor::cat(&[&bag, &total], 1)?)?);
        types.extend((1..=COUNT_ZONES.len()).map(|z| z as u32));

        let mut push = |xs: &mut Vec<Tensor>, x: Tensor, g: Group, i: usize| {
            slots.push((g, i, types.len()));
            types.push(type_id(g) as u32);
            xs.push(x);
        };
        for j in 0..MAX_OWN_SLOTS {
            let id = id_at(j);
            if id == 0 {
                continue;
            }
            let f = &obs[OFF_OWN + j * OWN_SLOT_FEATS..OFF_OWN + (j + 1) * OWN_SLOT_FEATS];
            let x = ((card(id)? + self.copy_in.forward(&card(id_at(MAX_OWN_SLOTS + j))?)?)? + self.own_in.forward(&row(f)?)?)?;
            push(&mut xs, x, Group::Own, j);
        }
        for b in 0..MAX_OPP_BASES {
            let id = id_at(2 * MAX_OWN_SLOTS + b);
            if id == 0 {
                continue;
            }
            let f = &obs[OFF_OPP + b * OPP_BASE_FEATS..OFF_OPP + (b + 1) * OPP_BASE_FEATS];
            push(&mut xs, (card(id)? + self.opp_in.forward(&row(f)?)?)?, Group::Opp, b);
        }
        for k in 0..ROW {
            let id = id_at(2 * MAX_OWN_SLOTS + MAX_OPP_BASES + k);
            if id == 0 {
                continue;
            }
            let f = &obs[OFF_ROW + k * ROW_FEATS..OFF_ROW + (k + 1) * ROW_FEATS];
            push(&mut xs, (card(id)? + self.row_in.forward(&row(f)?)?)?, Group::Row, k);
        }
        let explorers = obs[OFF_SCALARS + SCALAR_NAMES.iter().position(|s| *s == "explorers").unwrap()];
        let x = (card(db().explorer as usize + 1)? + self.explorer_in.forward(&row(&[explorers])?)?)?;
        push(&mut xs, x, Group::Explorer, 0);
        let zone = |name: &str| {
            let z = COUNT_ZONES.iter().position(|s| *s == name).unwrap();
            &counts[z * N_CARDS..(z + 1) * N_CARDS]
        };
        for (c, n) in distinct(zone("my_hand"), HAND) {
            let x = (card(c + 1)? + self.hand_in.forward(&row(&[n * COUNT_SCALE])?)?)?;
            push(&mut xs, x, Group::Hand, c);
        }
        for (c, n) in distinct(zone("my_discard"), DISCARD) {
            let x = (card(c + 1)? + self.discard_in.forward(&row(&[n * COUNT_SCALE])?)?)?;
            push(&mut xs, x, Group::Discard, c);
        }

        let n = types.len();
        let types = Tensor::from_slice(&types, n, &dev)?;
        let mut x = (Tensor::cat(&xs, 0)? + self.type_emb.index_select(&types, 0)?)?;
        let (h, dh) = (self.heads, d / self.heads);
        let scale = (dh as f64).powf(-0.5);
        for blk in &self.blocks {
            let qkv = blk.qkv.forward(&blk.ln1.forward(&x)?)?;
            let mut heads = Vec::with_capacity(h);
            for i in 0..h {
                let q = qkv.narrow(1, i * dh, dh)?;
                let k = qkv.narrow(1, d + i * dh, dh)?;
                let v = qkv.narrow(1, 2 * d + i * dh, dh)?;
                let att = candle_nn::ops::softmax_last_dim(&(q.matmul(&k.t()?)? * scale)?)?;
                heads.push(att.matmul(&v)?);
            }
            x = (&x + blk.proj.forward(&Tensor::cat(&heads, 1)?)?)?;
            let ff = blk.ff2.forward(&blk.ff1.forward(&blk.ln2.forward(&x)?)?.gelu_erf()?)?;
            x = (x + ff)?;
        }
        let hs = self.ln_f.forward(&x)?;
        let g = hs.narrow(0, 0, 1)?;
        let value = self.value2.forward(&self.value1.forward(&g)?.gelu_erf()?)?.flatten_all()?.to_vec1::<f32>()?[0];
        let glob = self.global_out.forward(&g)?.flatten_all()?.to_vec1::<f32>()?;
        let q = self.query.forward(&g)?.reshape((VERBS.len(), d))?;
        let inv = 1.0 / (d as f64).sqrt();
        let scores = (q.matmul(&self.key.forward(&hs)?.t()?)? * inv)?.to_vec2::<f32>()?;
        let card_scores = (q.matmul(&self.card_keys.t()?)? * inv)?.to_vec2::<f32>()?;
        debug_assert_eq!(glob.len(), N_GLOBAL_LOGITS);

        let mut logits = Vec::with_capacity(N_ACTIONS);
        for (count, fam) in families() {
            match fam {
                Family::Global(o, n) => logits.extend_from_slice(&glob[o..o + n]),
                Family::Pointer(v, grp) => {
                    for i in 0..count {
                        let pos = slots.iter().find(|s| s.0 == grp && s.1 == i).map(|s| s.2);
                        logits.push(match (pos, grp) {
                            (Some(p), _) => scores[v][p],
                            (None, Group::Hand | Group::Discard) => card_scores[v][i],
                            (None, _) => 0.0,
                        });
                    }
                }
            }
        }
        debug_assert_eq!(logits.len(), N_ACTIONS);
        Ok((logits, value))
    }
}
