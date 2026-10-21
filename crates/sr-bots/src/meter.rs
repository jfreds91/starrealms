//! Win chance from a value head, for a "who's winning" meter.
//!
//! The value head only ever saw positions from the side of the player to move, so it is asked
//! about the player to move and the other seat gets the complement. When that is not `seat`,
//! the mover's hidden cards are re-dealt from what `seat` knows (`Game::determinize`) and the
//! values averaged, so the meter never reflects the opponent's actual hand.

use crate::net::Net;
use crate::search::phi_index;
use sr_core::{Game, Outcome, OBS_DIM};

/// Shaping coefficient the value head was trained with (see `search`).
const SHAPING: f32 = 0.3;

/// Probability in [0, 1] that `seat` wins `g`, as judged by `net`. `samples` determinizations
/// are averaged when it is the other seat's move.
pub fn win_chance(net: &Net, g: &Game, seat: u8, samples: usize) -> f32 {
    if let Some(o) = g.outcome {
        return match o {
            Outcome::Win(s) if s == seat => 1.0,
            Outcome::Win(_) => 0.0,
            Outcome::Draw => 0.5,
        };
    }
    let mut obs = vec![0.0; OBS_DIM];
    let mut value = |g: &Game| {
        g.observe(&mut obs);
        let (_, v) = net.forward(&obs);
        (v + SHAPING * obs[phi_index()]).clamp(-1.0, 1.0)
    };
    let mover = if g.active == seat {
        value(g)
    } else {
        let n = samples.max(1);
        (0..n as u64).map(|i| value(&g.determinize(seat, i))).sum::<f32>() / n as f32
    };
    let v = if g.active == seat { mover } else { -mover };
    (v + 1.0) / 2.0
}
