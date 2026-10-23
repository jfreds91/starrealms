//! Label close decisions with paired determinized playouts.
//!
//! Each input line is a JSON object `{"replay", "actions": [a, b], "seat"}`. For every
//! sample the hidden cards are re-dealt once and both actions are played out on that copy
//! by `spec` (greedy). The output line is `{"wins": [a, b], "losses": [a, b], "draws": [a, b]}`
//! counted for `seat`.
//!
//!     cargo run -p sr-bots --release --example label_moves -- positions.jsonl labels.jsonl model:models/phase_f.safetensors 64 7

use serde::Deserialize;
use serde_json::json;
use sr_bots::{factory, Agent};
use sr_core::{Game, Outcome, Replay, Rng, Session};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
struct Position {
    replay: String,
    actions: [u16; 2],
    seat: u8,
}

fn mix(a: u64, b: u64) -> u64 {
    let mut r = Rng::new(a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    r.next_u64()
}

fn playout(base: &Game, action: u16, seat: u8, det_seed: u64, mut agent: Box<dyn Agent>) -> i8 {
    let mut g = base.determinize(seat, det_seed);
    let mut legal = Vec::new();
    g.legal_actions(&mut legal);
    if !legal.contains(&action) {
        return 0;
    }
    g.apply(action);
    let mut rng = Rng::new(det_seed ^ 0x5eed);
    for _ in 0..4_000 {
        if g.is_over() {
            break;
        }
        g.legal_actions(&mut legal);
        if legal.is_empty() {
            break;
        }
        let a = agent.act(&g, &legal, &mut rng);
        g.apply(a);
    }
    match g.outcome {
        Some(Outcome::Win(s)) if s == seat => 1,
        Some(Outcome::Win(_)) => -1,
        _ => 0,
    }
}

fn main() {
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }
    let args: Vec<String> = std::env::args().collect();
    let spec = &args[3];
    let samples: usize = args[4].parse().expect("samples");
    let seed0: u64 = args[5].parse().expect("seed");
    let make = factory(spec).expect("agent");
    let positions: Vec<Position> = BufReader::new(File::open(&args[1]).expect("positions"))
        .lines()
        .map(|l| serde_json::from_str(&l.expect("line")).expect("position"))
        .collect();
    let replays: Vec<Replay> = positions
        .iter()
        .map(|p| serde_json::from_str(&p.replay).expect("replay"))
        .collect();
    let out = Arc::new(Mutex::new(File::create(&args[2]).expect("output")));
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let n = positions.len();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let base = Session::from_replay(replays[i].clone()).expect("replay").game;
                let mut outcomes = [vec![0i8; samples], vec![0i8; samples]];
                for sample in 0..samples {
                    let det = mix(seed0, (i as u64) << 32 | sample as u64);
                    for (k, &action) in positions[i].actions.iter().enumerate() {
                        outcomes[k][sample] = playout(&base, action, positions[i].seat, det, make());
                    }
                }
                let line = json!({"i": i, "a": outcomes[0], "b": outcomes[1]});
                let mut f = out.lock().unwrap();
                writeln!(f, "{line}").unwrap();
                if (i + 1) % 25 == 0 {
                    eprintln!("labeled {}/{n}", i + 1);
                }
            });
        }
    });
    eprintln!("labeled {n} positions, {samples} samples");
}
