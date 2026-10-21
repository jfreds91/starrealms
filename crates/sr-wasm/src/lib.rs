//! The engine in the browser, for the replay viewer only: games are always
//! played on the server. Built by `web/scripts/build-wasm.sh`.
//!
//! Views cross the boundary as JSON strings shaped like the server's
//! messages, so the client renders replays and live games with the same code.

use sr_core::Replay;
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
