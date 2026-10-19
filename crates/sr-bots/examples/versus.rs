//! Head-to-head between two agent specs (see `sr_bots::factory`), alternating seats, on all
//! cores. Prints A's score with a 95% Wilson interval.
//!
//!     cargo run -p sr-bots --release --example versus -- <a> <b> [games] [sets] [seed]
//!     cargo run -p sr-bots --release --example versus -- search:models/phase_d.safetensors model:models/phase_c.srnn 1000 core

use sr_bots::{factory, play_game};
use sr_core::{Config, Outcome};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

fn main() {
    // One game per core already; candle's matmuls would otherwise each fan out to every core.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }
    let args: Vec<String> = std::env::args().collect();
    let (a, b) = (&args[1], &args[2]);
    let games: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let sets: Vec<String> = args.get(4).map_or(vec!["core".into()], |s| s.split(',').map(String::from).collect());
    let seed0: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
    let cfg = Config { sets, ..Config::default() };
    let (fa, fb) = (factory(a).expect("agent a"), factory(b).expect("agent b"));
    let next = AtomicU64::new(0);
    let (wins, losses, draws) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= games {
                    break;
                }
                let swap = i % 2 == 1;
                let mut agents = if swap { [fb(), fa()] } else { [fa(), fb()] };
                let g = play_game(&cfg, seed0.wrapping_mul(1_000_003).wrapping_add(i), &mut agents);
                match g.outcome.unwrap() {
                    Outcome::Win(s) if (s == 0) != swap => wins.fetch_add(1, Ordering::Relaxed),
                    Outcome::Win(_) => losses.fetch_add(1, Ordering::Relaxed),
                    Outcome::Draw => draws.fetch_add(1, Ordering::Relaxed),
                };
            });
        }
    });
    let (w, l, d) = (wins.into_inner(), losses.into_inner(), draws.into_inner());
    let n = (w + l + d) as f64;
    let p = (w as f64 + 0.5 * d as f64) / n;
    let z = 1.96f64;
    let centre = (p + z * z / (2.0 * n)) / (1.0 + z * z / n);
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / (1.0 + z * z / n);
    println!(
        "{a} vs {b}: {:.3} [{:.3}, {:.3}] over {n} games ({w}W {d}D {l}L) in {:.0}s",
        p,
        centre - half,
        centre + half,
        t0.elapsed().as_secs_f64()
    );
}
