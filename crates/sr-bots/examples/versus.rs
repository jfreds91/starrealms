//! Head-to-head between two agent specs (see `sr_bots::factory`), alternating seats, on all
//! cores. Prints A's score with a 95% Wilson interval.
//!
//!     cargo run -p sr-bots --release --example versus -- <a> <b> [games] [sets] [seed] [mirror]
//!     cargo run -p sr-bots --release --example versus -- search:models/phase_d.safetensors model:models/phase_c.srnn 1000 core
//!
//! `mirror` plays each seed twice with the seats swapped and reports how many pairs were
//! player-decided (the same agent won both) or deal-decided (the same seat won both).

use sr_bots::{factory, play_game};
use sr_core::{Config, Outcome};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// 1 if A won, -1 if B won, 0 on a draw. `swap` means A sat in seat 1.
fn agent_score(outcome: Option<Outcome>, swap: bool) -> i8 {
    match outcome.unwrap() {
        Outcome::Win(s) if (s == 0) != swap => 1,
        Outcome::Win(_) => -1,
        Outcome::Draw => 0,
    }
}

/// A mirrored pair: the same agent won both games, the same seat won both, or neither
/// (a draw, or the two games split without agreeing).
fn pair_kind(first: i8, second: i8) -> &'static str {
    if first != 0 && first == second {
        "player"
    } else if first != 0 && first == -second {
        "deal"
    } else {
        "other"
    }
}

fn wilson(points: f64, n: f64) -> (f64, f64, f64) {
    let p = points / n;
    let z = 1.96f64;
    let centre = (p + z * z / (2.0 * n)) / (1.0 + z * z / n);
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / (1.0 + z * z / n);
    (p, centre - half, centre + half)
}

fn main() {
    // One game per core already; candle's matmuls would otherwise each fan out to every core.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }
    let args: Vec<String> = std::env::args().collect();
    let mirror = args.iter().any(|s| s == "mirror");
    let positional: Vec<&String> = args.iter().skip(1).filter(|s| s.as_str() != "mirror").collect();
    let (a, b) = (positional[0], positional[1]);
    let games: u64 = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let sets: Vec<String> = positional.get(3).map_or(vec!["core".into()], |s| s.split(',').map(String::from).collect());
    let seed0: u64 = positional.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let cfg = Config { sets, ..Config::default() };
    let (fa, fb) = (factory(a).expect("agent a"), factory(b).expect("agent b"));
    let next = AtomicU64::new(0);
    let (wins, losses, draws) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let (player_a, player_b, deal, other) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= games {
                    break;
                }
                let seed = seed0.wrapping_mul(1_000_003).wrapping_add(i);
                if mirror {
                    let r0 = agent_score(play_game(&cfg, seed, &mut [fa(), fb()]).outcome, false);
                    let r1 = agent_score(play_game(&cfg, seed, &mut [fb(), fa()]).outcome, true);
                    match pair_kind(r0, r1) {
                        "player" if r0 == 1 => player_a.fetch_add(1, Ordering::Relaxed),
                        "player" => player_b.fetch_add(1, Ordering::Relaxed),
                        "deal" => deal.fetch_add(1, Ordering::Relaxed),
                        _ => other.fetch_add(1, Ordering::Relaxed),
                    };
                } else {
                    let swap = i % 2 == 1;
                    let mut agents = if swap { [fb(), fa()] } else { [fa(), fb()] };
                    match agent_score(play_game(&cfg, seed, &mut agents).outcome, swap) {
                        1 => wins.fetch_add(1, Ordering::Relaxed),
                        -1 => losses.fetch_add(1, Ordering::Relaxed),
                        _ => draws.fetch_add(1, Ordering::Relaxed),
                    };
                }
            });
        }
    });
    let secs = t0.elapsed().as_secs_f64();
    if mirror {
        let (pa, pb, de, ot) = (player_a.into_inner(), player_b.into_inner(), deal.into_inner(), other.into_inner());
        let n = (pa + pb + de + ot) as f64;
        // A pair is a point for A only when A won both seats; a split or a draw is half.
        let points = pa as f64 + 0.5 * (de + ot) as f64;
        let (p, lo, hi) = wilson(points, n);
        println!(
            "{a} vs {b} mirror: {p:.3} [{lo:.3}, {hi:.3}] over {n:.0} pairs ({pa} player-A, {pb} player-B, {de} deal-decided, {ot} other) in {secs:.0}s"
        );
        println!(
            "deal-decided {:.3}  player-decided {:.3}",
            de as f64 / n,
            (pa + pb) as f64 / n
        );
    } else {
        let (w, l, d) = (wins.into_inner(), losses.into_inner(), draws.into_inner());
        let n = (w + l + d) as f64;
        let (p, lo, hi) = wilson(w as f64 + 0.5 * d as f64, n);
        println!(
            "{a} vs {b}: {p:.3} [{lo:.3}, {hi:.3}] over {n} games ({w}W {d}D {l}L) in {secs:.0}s"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::pair_kind;

    #[test]
    fn mirrored_pairs_split_skill_from_the_deal() {
        assert_eq!(pair_kind(1, 1), "player");
        assert_eq!(pair_kind(-1, -1), "player");
        assert_eq!(pair_kind(1, -1), "deal");
        assert_eq!(pair_kind(-1, 1), "deal");
        assert_eq!(pair_kind(1, 0), "other");
        assert_eq!(pair_kind(0, 0), "other");
    }
}
