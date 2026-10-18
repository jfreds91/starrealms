//! Random-vs-random throughput on one core.
//!
//!     cargo run -p sr-core --release --example bench -- [games] [sets,comma,separated]

use sr_core::{action, Config, Game, Outcome, Rng};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let sets: Vec<String> = args.get(2).map_or(vec!["core".into()], |s| s.split(',').map(String::from).collect());
    let cfg = Config { sets: sets.clone(), ..Config::default() };

    let mut rng = Rng::new(1);
    let mut legal = Vec::with_capacity(64);
    let (mut actions, mut turns, mut wins, mut draws) = (0u64, 0u64, [0u64; 2], 0u64);
    let start = Instant::now();
    for seed in 0..games {
        let mut g = Game::new(&cfg, seed).unwrap();
        while !g.is_over() {
            g.legal_actions(&mut legal);
            // Random but purposeful: rarely end the turn while other moves exist.
            let a = if legal.len() > 1 && rng.below(8) != 0 {
                legal[1 + rng.below(legal.len() - 1)]
            } else {
                legal[rng.below(legal.len())]
            };
            debug_assert!(a < action::N_ACTIONS as u16);
            g.apply(a);
            actions += 1;
        }
        turns += g.turn as u64;
        match g.outcome.unwrap() {
            Outcome::Win(s) => wins[s as usize] += 1,
            Outcome::Draw => draws += 1,
        }
    }
    let secs = start.elapsed().as_secs_f64();
    println!("sets {:?}: {games} games in {secs:.2}s", sets);
    println!("  {:.0} games/s, {:.0} actions/s", games as f64 / secs, actions as f64 / secs);
    println!(
        "  avg {:.1} player-turns, {:.1} actions per game; wins p0 {} p1 {} draws {}",
        turns as f64 / games as f64,
        actions as f64 / games as f64,
        wins[0],
        wins[1],
        draws
    );
}
