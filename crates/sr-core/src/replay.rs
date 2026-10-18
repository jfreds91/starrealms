use crate::game::{Config, Game};
use serde::{Deserialize, Serialize};

/// Everything needed to reproduce a game exactly.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Replay {
    pub config: Config,
    /// Written as a string: JSON numbers above 2^53 do not survive JavaScript.
    #[serde(with = "seed_string")]
    pub seed: u64,
    pub actions: Vec<u16>,
}

mod seed_string {
    use serde::{de, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(seed: &u64, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(seed)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
        struct V;
        impl de::Visitor<'_> for V {
            type Value = u64;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a u64 seed as a string or integer")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<u64, E> {
                Ok(v)
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<u64, E> {
                v.parse().map_err(E::custom)
            }
        }
        d.deserialize_any(V)
    }
}

impl Replay {
    pub fn new(config: Config, seed: u64) -> Replay {
        Replay { config, seed, actions: Vec::new() }
    }

    /// Re-simulate the whole game.
    pub fn run(&self) -> Result<Game, String> {
        self.run_prefix(self.actions.len())
    }

    /// Re-simulate the first `n` actions.
    pub fn run_prefix(&self, n: usize) -> Result<Game, String> {
        let mut g = Game::new(&self.config, self.seed)?;
        for (i, &a) in self.actions.iter().take(n).enumerate() {
            g.step(a).map_err(|e| format!("action #{i}: {e}"))?;
        }
        Ok(g)
    }
}

/// A game that records its own replay.
#[derive(Clone)]
pub struct Session {
    pub game: Game,
    pub replay: Replay,
}

impl Session {
    pub fn new(config: Config, seed: u64) -> Result<Session, String> {
        let game = Game::new(&config, seed)?;
        Ok(Session { game, replay: Replay::new(config, seed) })
    }

    pub fn from_replay(replay: Replay) -> Result<Session, String> {
        let game = replay.run()?;
        Ok(Session { game, replay })
    }

    pub fn step(&mut self, a: u16) -> Result<(), String> {
        self.game.step(a)?;
        self.replay.actions.push(a);
        Ok(())
    }
}
