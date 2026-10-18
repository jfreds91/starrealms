//! The engine in the browser: hot-seat and offline games against bots, and
//! the replay viewer. Built by `web/scripts/build-wasm.sh`.
//!
//! Everything crosses the boundary as JSON strings shaped like the server's
//! messages, so the client renders local and online games with the same code.

use sr_bots::{by_name, Agent, Net, NetAgent};
use sr_core::{Config, Replay, Rng, Session};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

fn js(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn catalog() -> String {
    serde_json::to_string(&sr_core::view::catalog()).unwrap()
}

/// A live local game. Seats are either human (driven by `step`) or a bot set
/// with `set_bot`, which `bot_step` plays one action at a time.
#[wasm_bindgen]
pub struct Engine {
    s: Session,
    bots: [Option<Box<dyn Agent>>; 2],
    rng: Rng,
    model: Option<Arc<Net>>,
}

#[wasm_bindgen]
impl Engine {
    /// `config_json`: a `sr_core::Config` (missing fields take defaults).
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, seed: u64) -> Result<Engine, JsError> {
        let mut cfg: Config = serde_json::from_str(config_json).map_err(js)?;
        cfg.log = true;
        Ok(Engine::wrap(Session::new(cfg, seed).map_err(js)?))
    }

    /// Resume from a saved replay (see `replay`).
    pub fn from_replay(json: &str) -> Result<Engine, JsError> {
        let r: Replay = serde_json::from_str(json).map_err(js)?;
        Ok(Engine::wrap(Session::from_replay(r).map_err(js)?))
    }

    fn wrap(s: Session) -> Engine {
        let rng = Rng::new(s.replay.seed ^ 0xb075_eed5);
        Engine { s, bots: [None, None], rng, model: None }
    }

    /// Load an exported `.srnn` network for the `"model"` bot spec.
    pub fn load_model(&mut self, bytes: &[u8]) -> Result<(), JsError> {
        self.model = Some(Arc::new(Net::from_bytes(bytes).map_err(js)?));
        Ok(())
    }

    /// `spec`: "random" | "greedy" | "heuristic" | "model" (after `load_model`), or "" for a human seat.
    pub fn set_bot(&mut self, seat: u8, spec: &str) -> Result<(), JsError> {
        let bot: Option<Box<dyn Agent>> = match spec {
            "" => None,
            "model" => {
                let net = self.model.clone().ok_or_else(|| js("no model loaded"))?;
                Some(Box::new(NetAgent::new(net, "model", 0.0)))
            }
            name => Some(by_name(name).ok_or_else(|| js(format!("unknown bot {name:?}")))?),
        };
        *self.bots.get_mut(seat as usize).ok_or_else(|| js("seat"))? = bot;
        Ok(())
    }

    pub fn active(&self) -> u8 {
        self.s.game.active
    }

    pub fn is_over(&self) -> bool {
        self.s.game.is_over()
    }

    /// True if the seat to move is a bot.
    pub fn bot_to_move(&self) -> bool {
        !self.s.game.is_over() && self.bots[self.s.game.active as usize].is_some()
    }

    /// JSON `View` for `seat`.
    pub fn view(&self, seat: u8, log_tail: usize) -> String {
        serde_json::to_string(&self.s.game.view(seat, log_tail)).unwrap()
    }

    pub fn step(&mut self, action: u16) -> Result<(), JsError> {
        self.s.step(action).map_err(js)
    }

    /// Next "play all" action, or -1.
    pub fn next_simple_play(&self) -> i32 {
        self.s.game.next_simple_play().map_or(-1, i32::from)
    }

    /// Play one action for the bot whose turn it is. Returns the action.
    pub fn bot_step(&mut self) -> Result<u16, JsError> {
        let g = &self.s.game;
        let bot = self.bots[g.active as usize].as_mut().ok_or_else(|| js("not a bot's turn"))?;
        let legal = g.legal_vec();
        if legal.is_empty() {
            return Err(js("game is over"));
        }
        let a = bot.act(g, &legal, &mut self.rng);
        self.s.step(a).map_err(js)?;
        Ok(a)
    }

    pub fn replay(&self) -> String {
        serde_json::to_string(&self.s.replay).unwrap()
    }
}

/// Step-through viewer for a finished game's replay.
#[wasm_bindgen]
pub struct ReplayViewer {
    replay: Replay,
}

#[wasm_bindgen]
impl ReplayViewer {
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> Result<ReplayViewer, JsError> {
        let mut replay: Replay = serde_json::from_str(json).map_err(js)?;
        replay.config.log = true;
        replay.run().map_err(js)?;
        Ok(ReplayViewer { replay })
    }

    pub fn len(&self) -> usize {
        self.replay.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.replay.actions.is_empty()
    }

    /// JSON `View` for `seat` after the first `n` actions.
    pub fn view_at(&self, n: usize, seat: u8, log_tail: usize) -> Result<String, JsError> {
        let g = self.replay.run_prefix(n).map_err(js)?;
        Ok(serde_json::to_string(&g.view(seat, log_tail)).unwrap())
    }

    /// Indices `n` at which a new turn starts (for "next turn" navigation).
    pub fn turn_starts(&self) -> Vec<u32> {
        let mut g = sr_core::Game::new(&self.replay.config, self.replay.seed).unwrap();
        let mut out = vec![0];
        for (i, &a) in self.replay.actions.iter().enumerate() {
            let t = g.turn;
            g.apply(a);
            if g.turn != t {
                out.push(i as u32 + 1);
            }
        }
        out
    }
}
