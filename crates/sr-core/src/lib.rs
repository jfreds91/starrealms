//! Star Realms rules engine (Core Set, Colony Wars, Frontiers; two players).

pub mod action;
pub mod cards;
pub mod game;
pub mod obs;
pub mod replay;
pub mod rng;
pub mod text;
pub mod view;

pub use action::{Action, N_ACTIONS};
pub use cards::{card, db, Card, CardId, Faction, Kind, N_CARDS};
pub use game::{Config, Game, Outcome, Pending, PendingKind};
pub use obs::OBS_DIM;
pub use replay::{Replay, Session};
pub use rng::Rng;

#[cfg(test)]
mod tests;
