//! Turn lookahead on top of a policy/value `Net`, without peeking at hidden information.
//!
//! At each decision with a real choice, the `width` most likely actions under the policy are
//! each tried on `samples` determinized copies of the game (`Game::determinize`: hidden cards
//! re-dealt from what the player knows). After a candidate, play continues greedily by the
//! policy (for both players, on the copy) through `turns` hand-overs, and the position is
//! scored by the value head (or by the result, if the game ended). Each candidate is judged on
//! the same copies, and the best average wins.
//!
//! The leaf must come after a hand-over: mid-turn, the value head assumes unspent trade and
//! combat will be used well, so it cannot tell "end the turn now" from playing on.
//!
//! The value head was trained with potential-based shaping on the authority difference, so it
//! estimates `return - shaping * phi`; `shaping * phi` is added back to compare positions with
//! different authority totals (e.g. after attacking).

use crate::net::Net;
use crate::Agent;
use sr_core::obs::{OFF_SCALARS, SCALAR_NAMES};
use sr_core::{Game, Outcome, Rng, OBS_DIM};
use std::sync::Arc;

pub struct SearchAgent {
    net: Arc<Net>,
    name: String,
    /// Candidate actions per decision.
    pub width: usize,
    /// Determinized copies per decision.
    pub samples: usize,
    /// Hand-overs to play through before scoring (1 = end of this turn, 2 = after the
    /// opponent's reply, ...).
    pub turns: usize,
    /// Skip the search when the policy already puts this much probability on one action.
    pub confident: f32,
    /// Candidates are ranked by mean value + `prior_weight * ln(policy probability)`.
    pub prior_weight: f32,
    /// Shaping coefficient the value head was trained with.
    pub shaping: f32,
    /// Safety cap on greedy steps per rollout.
    pub max_steps: usize,
    obs: Vec<f32>,
    legal: Vec<u16>,
}

pub(crate) fn phi_index() -> usize {
    OFF_SCALARS + SCALAR_NAMES.iter().position(|s| *s == "authority_diff").expect("authority_diff scalar")
}

impl SearchAgent {
    pub fn new(net: Arc<Net>, name: impl Into<String>, width: usize, samples: usize) -> SearchAgent {
        SearchAgent {
            net,
            name: name.into(),
            width,
            samples,
            turns: 1,
            confident: 0.97,
            prior_weight: 0.0,
            shaping: 0.3,
            max_steps: 200,
            obs: vec![0.0; OBS_DIM],
            legal: Vec::with_capacity(64),
        }
    }

    /// Legal actions with policy probabilities, and the shaping-corrected value, for the player
    /// to move in `g`.
    fn evaluate(&mut self, g: &Game, legal: &[u16]) -> (Vec<(u16, f32)>, f32) {
        g.observe(&mut self.obs);
        let (logits, v) = self.net.forward(&self.obs);
        let m = legal.iter().map(|&a| logits[a as usize]).fold(f32::NEG_INFINITY, f32::max);
        let ws: Vec<f32> = legal.iter().map(|&a| (logits[a as usize] - m).exp()).collect();
        let z: f32 = ws.iter().sum();
        let probs = legal.iter().zip(ws).map(|(&a, w)| (a, w / z)).collect();
        (probs, v + self.shaping * self.obs[phi_index()])
    }

    /// Value for `me` of `g` after playing on greedily through `self.turns` hand-overs.
    fn rollout(&mut self, mut g: Game, me: u8) -> f32 {
        let mut legal = std::mem::take(&mut self.legal);
        let (mut steps, mut handovers, mut mover) = (0, 0, me);
        let value = loop {
            if let Some(o) = g.outcome {
                break match o {
                    Outcome::Win(s) if s == me => 1.0,
                    Outcome::Win(_) => -1.0,
                    Outcome::Draw => 0.0,
                };
            }
            if g.active != mover {
                mover = g.active;
                handovers += 1;
            }
            g.legal_actions(&mut legal);
            let (probs, v) = self.evaluate(&g, &legal);
            if handovers >= self.turns || steps >= self.max_steps {
                break if g.active == me { v } else { -v };
            }
            let a = probs.iter().max_by(|x, y| x.1.total_cmp(&y.1)).unwrap().0;
            g.apply(a);
            steps += 1;
        };
        self.legal = legal;
        value
    }
}

impl Agent for SearchAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&mut self, g: &Game, legal: &[u16], rng: &mut Rng) -> u16 {
        if legal.len() == 1 {
            return legal[0];
        }
        let (mut probs, _) = self.evaluate(g, legal);
        probs.sort_by(|x, y| y.1.total_cmp(&x.1));
        if probs[0].1 >= self.confident || self.width <= 1 || self.samples == 0 {
            return probs[0].0;
        }
        let cands = &probs[..self.width.min(probs.len())];
        let me = g.active;
        let mut total = vec![0.0f32; cands.len()];
        for _ in 0..self.samples {
            let det = g.determinize(me, rng.next_u64());
            for (i, &(a, _)) in cands.iter().enumerate() {
                let mut sim = det.clone();
                sim.apply(a);
                total[i] += self.rollout(sim, me);
            }
        }
        let score: Vec<f32> = cands
            .iter()
            .zip(&total)
            .map(|(&(_, p), t)| t / self.samples as f32 + self.prior_weight * p.max(1e-6).ln())
            .collect();
        // ties (e.g. interchangeable plays) go to the policy's favourite, which comes first
        let best = (0..cands.len()).fold(0, |b, i| if score[i] > score[b] + 1e-6 { i } else { b });
        cands[best].0
    }
}
