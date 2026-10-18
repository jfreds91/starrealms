//! SQLite persistence. A game row is its replay (config, seed, actions) plus
//! seat owners; the live state is always re-derived by replaying it.

use crate::rooms::Seat;
use rusqlite::{params, Connection, OptionalExtension};
use sr_core::{Config, Replay};
use std::path::Path;
use std::sync::Mutex;

pub struct Db(Mutex<Connection>);

#[derive(Clone, Debug)]
pub struct Player {
    pub id: String,
    pub name: String,
}

pub struct GameRow {
    pub replay: Replay,
    pub seats: [Seat; 2],
    pub created: i64,
    pub updated: i64,
    pub resigned: Option<u8>,
}

/// One line of a player's game list.
#[derive(serde::Serialize)]
pub struct GameSummary {
    pub id: String,
    pub seat: u8,
    pub seats: [Seat; 2],
    pub status: String,
    pub active: u8,
    pub turn: u16,
    pub winner: Option<i8>,
    pub sets: Vec<String>,
    pub updated: i64,
}

/// What changes on every move.
pub struct Progress<'a> {
    pub actions: &'a [u16],
    pub seats: &'a [Seat; 2],
    pub status: &'a str,
    pub active: u8,
    pub turn: u16,
    pub winner: Option<i8>,
    pub resigned: Option<u8>,
    pub updated: i64,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS players (
    id      TEXT PRIMARY KEY,
    token   TEXT NOT NULL UNIQUE,
    name    TEXT NOT NULL,
    created INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    id       TEXT PRIMARY KEY,
    created  INTEGER NOT NULL,
    updated  INTEGER NOT NULL,
    config   TEXT NOT NULL,
    seed     TEXT NOT NULL,
    actions  TEXT NOT NULL,
    seat0    TEXT NOT NULL,
    seat1    TEXT NOT NULL,
    p0       TEXT,
    p1       TEXT,
    status   TEXT NOT NULL,
    active   INTEGER NOT NULL,
    turn     INTEGER NOT NULL,
    winner   INTEGER,
    resigned INTEGER
);
CREATE INDEX IF NOT EXISTS games_p0 ON games (p0, updated);
CREATE INDEX IF NOT EXISTS games_p1 ON games (p1, updated);
";

fn player_id(s: &Seat) -> Option<&str> {
    match s {
        Seat::Player { id, .. } => Some(id),
        _ => None,
    }
}

impl Db {
    pub fn open(path: &Path) -> rusqlite::Result<Db> {
        let c = Connection::open(path)?;
        c.pragma_update(None, "journal_mode", "WAL")?;
        c.pragma_update(None, "synchronous", "NORMAL")?;
        c.execute_batch(SCHEMA)?;
        Ok(Db(Mutex::new(c)))
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn create_player(&self, id: &str, token: &str, name: &str, now: i64) -> rusqlite::Result<()> {
        self.conn().execute(
            "INSERT INTO players (id, token, name, created) VALUES (?1, ?2, ?3, ?4)",
            params![id, token, name, now],
        )?;
        Ok(())
    }

    pub fn player_by_token(&self, token: &str) -> rusqlite::Result<Option<Player>> {
        self.conn()
            .query_row("SELECT id, name FROM players WHERE token = ?1", [token], |r| {
                Ok(Player { id: r.get(0)?, name: r.get(1)? })
            })
            .optional()
    }

    pub fn rename_player(&self, id: &str, name: &str) -> rusqlite::Result<()> {
        self.conn().execute("UPDATE players SET name = ?2 WHERE id = ?1", params![id, name])?;
        Ok(())
    }

    pub fn insert_game(&self, id: &str, replay: &Replay, created: i64, p: &Progress) -> rusqlite::Result<()> {
        self.conn().execute(
            "INSERT INTO games (id, created, updated, config, seed, actions, seat0, seat1, p0, p1,
                                status, active, turn, winner, resigned)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                id,
                created,
                p.updated,
                serde_json::to_string(&replay.config).unwrap(),
                replay.seed.to_string(),
                serde_json::to_string(p.actions).unwrap(),
                serde_json::to_string(&p.seats[0]).unwrap(),
                serde_json::to_string(&p.seats[1]).unwrap(),
                player_id(&p.seats[0]),
                player_id(&p.seats[1]),
                p.status,
                p.active,
                p.turn,
                p.winner,
                p.resigned,
            ],
        )?;
        Ok(())
    }

    pub fn save_progress(&self, id: &str, p: &Progress) -> rusqlite::Result<()> {
        self.conn().execute(
            "UPDATE games SET updated = ?2, actions = ?3, seat0 = ?4, seat1 = ?5, p0 = ?6, p1 = ?7,
                              status = ?8, active = ?9, turn = ?10, winner = ?11, resigned = ?12
             WHERE id = ?1",
            params![
                id,
                p.updated,
                serde_json::to_string(p.actions).unwrap(),
                serde_json::to_string(&p.seats[0]).unwrap(),
                serde_json::to_string(&p.seats[1]).unwrap(),
                player_id(&p.seats[0]),
                player_id(&p.seats[1]),
                p.status,
                p.active,
                p.turn,
                p.winner,
                p.resigned,
            ],
        )?;
        Ok(())
    }

    pub fn load_game(&self, id: &str) -> rusqlite::Result<Option<GameRow>> {
        self.conn()
            .query_row(
                "SELECT config, seed, actions, seat0, seat1, created, updated, resigned FROM games WHERE id = ?1",
                [id],
                |r| {
                    let json = |i: usize| -> rusqlite::Result<String> { r.get(i) };
                    let bad = |e: serde_json::Error| {
                        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
                    };
                    let config: Config = serde_json::from_str(&json(0)?).map_err(bad)?;
                    let seed: u64 = json(1)?.parse().map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(e))
                    })?;
                    Ok(GameRow {
                        replay: Replay { config, seed, actions: serde_json::from_str(&json(2)?).map_err(bad)? },
                        seats: [
                            serde_json::from_str(&json(3)?).map_err(bad)?,
                            serde_json::from_str(&json(4)?).map_err(bad)?,
                        ],
                        created: r.get(5)?,
                        updated: r.get(6)?,
                        resigned: r.get(7)?,
                    })
                },
            )
            .optional()
    }

    pub fn games_of(&self, player: &str, limit: usize) -> rusqlite::Result<Vec<GameSummary>> {
        let c = self.conn();
        let mut st = c.prepare(
            "SELECT id, CASE WHEN p0 = ?1 THEN 0 ELSE 1 END, seat0, seat1, status, active, turn, winner, config, updated
             FROM games WHERE p0 = ?1 OR p1 = ?1 ORDER BY updated DESC LIMIT ?2",
        )?;
        let rows = st.query_map(params![player, limit as i64], |r| {
            let seat = |i: usize| -> rusqlite::Result<Seat> {
                Ok(serde_json::from_str(&r.get::<_, String>(i)?).unwrap_or(Seat::Open))
            };
            let config: Config = serde_json::from_str(&r.get::<_, String>(8)?).unwrap_or_default();
            Ok(GameSummary {
                id: r.get(0)?,
                seat: r.get(1)?,
                seats: [seat(2)?, seat(3)?],
                status: r.get(4)?,
                active: r.get(5)?,
                turn: r.get(6)?,
                winner: r.get(7)?,
                sets: config.sets,
                updated: r.get(9)?,
            })
        })?;
        rows.collect()
    }
}
