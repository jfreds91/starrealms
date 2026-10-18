//! Round-robin between scripted agents, alternating seats.
//!
//!     cargo run -p sr-bots --release --example arena -- [games] [sets]

use sr_bots::{by_name, play_game};
use sr_core::{Config, Outcome};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let sets: Vec<String> = args.get(2).map_or(vec!["core".into()], |s| s.split(',').map(String::from).collect());
    let cfg = Config { sets, ..Config::default() };
    let names = ["random", "greedy", "heuristic"];
    for (i, a) in names.iter().enumerate() {
        for b in &names[i + 1..] {
            let (mut wa, mut wb, mut draws, mut turns) = (0, 0, 0, 0u64);
            for seed in 0..games {
                let swap = seed % 2 == 1;
                let mut agents = if swap { [by_name(b).unwrap(), by_name(a).unwrap()] } else { [by_name(a).unwrap(), by_name(b).unwrap()] };
                let g = play_game(&cfg, seed, &mut agents);
                turns += g.turn as u64;
                match g.outcome.unwrap() {
                    Outcome::Win(s) => {
                        if (s == 0) != swap {
                            wa += 1
                        } else {
                            wb += 1
                        }
                    }
                    Outcome::Draw => draws += 1,
                }
            }
            println!(
                "{a:>9} vs {b:<9}  {wa:>5} - {wb:<5} draws {draws:<4} ({:.1}% for {b}), avg {:.1} turns",
                100.0 * wb as f64 / games as f64,
                turns as f64 / games as f64
            );
        }
    }
}
