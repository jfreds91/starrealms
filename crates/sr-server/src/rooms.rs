//! Live games. A `Room` owns one `Session` (game + replay). Every change bumps
//! the room's version; each connected socket re-renders its own seat's view
//! when the version moves, so no client ever receives another seat's view.

use crate::db::{Db, Player, Progress};
use serde::{Deserialize, Serialize};
use sr_bots::{by_name, Agent, Net, NetAgent};
use sr_core::{view::View, Config, Outcome, Replay, Rng, Session};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::watch;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Seat {
    Open,
    Player { id: String, name: String },
    Bot { spec: String, name: String },
}

pub struct Opponent {
    pub spec: String,
    pub name: String,
    model: Option<Arc<Net>>,
}

impl Opponent {
    fn agent(&self) -> Box<dyn Agent> {
        match &self.model {
            Some(net) => Box::new(NetAgent::new(net.clone(), self.spec.clone(), 0.0)),
            None => by_name(&self.spec).expect("built-in bot"),
        }
    }
}

pub struct App {
    pub db: Db,
    rooms: Mutex<HashMap<String, Arc<Room>>>,
    pub opponents: BTreeMap<String, Opponent>,
    pub catalog: String,
    pub bot_delay: Duration,
}

pub struct Room {
    pub id: String,
    state: Mutex<RoomState>,
    version: watch::Sender<u64>,
}

pub struct RoomState {
    pub session: Session,
    pub seats: [Seat; 2],
    pub created: i64,
    pub updated: i64,
    pub resigned: Option<u8>,
    pub online: [u32; 2],
    bot_running: bool,
}

#[derive(Serialize)]
pub struct SeatMeta<'a> {
    #[serde(flatten)]
    pub seat: &'a Seat,
    pub online: bool,
}

#[derive(Serialize)]
pub struct Meta<'a> {
    pub id: &'a str,
    pub seats: [SeatMeta<'a>; 2],
    pub status: &'static str,
    /// Winning seat, -1 for a draw, None while running.
    pub winner: Option<i8>,
    pub resigned: Option<u8>,
    pub sets: &'a [String],
    pub created: i64,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg<'a> {
    State {
        game: Meta<'a>,
        /// The viewer's seat, or None for a spectator.
        seat: Option<u8>,
        /// Number of actions taken so far; echo it back with each action.
        n: usize,
        view: View,
    },
    Error {
        message: String,
    },
}

pub fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

pub fn random_id(len: usize) -> String {
    const ALPHABET: &[u8] = b"abcdefghjkmnpqrstuvwxyz23456789";
    let mut buf = vec![0u8; len];
    getrandom::fill(&mut buf).expect("OS randomness");
    buf.iter().map(|b| ALPHABET[*b as usize % ALPHABET.len()] as char).collect()
}

fn random_seed() -> u64 {
    let mut b = [0u8; 8];
    getrandom::fill(&mut b).expect("OS randomness");
    u64::from_le_bytes(b)
}

impl RoomState {
    pub fn seat_of(&self, player: Option<&Player>) -> Option<u8> {
        let p = player?;
        self.seats.iter().position(|s| matches!(s, Seat::Player { id, .. } if *id == p.id)).map(|i| i as u8)
    }

    pub fn winner(&self) -> Option<i8> {
        if let Some(r) = self.resigned {
            return Some(1 - r as i8);
        }
        self.session.game.outcome.map(|o| match o {
            Outcome::Win(s) => s as i8,
            Outcome::Draw => -1,
        })
    }

    pub fn status(&self) -> &'static str {
        if self.winner().is_some() {
            "finished"
        } else if self.seats.contains(&Seat::Open) {
            "waiting"
        } else {
            "active"
        }
    }

    fn finished(&self) -> bool {
        self.winner().is_some()
    }

    fn bot_to_move(&self) -> bool {
        !self.finished()
            && !self.seats.contains(&Seat::Open)
            && matches!(self.seats[self.session.game.active as usize], Seat::Bot { .. })
    }

    fn progress(&self) -> Progress<'_> {
        Progress {
            actions: &self.session.replay.actions,
            seats: &self.seats,
            status: self.status(),
            active: self.session.game.active,
            turn: self.session.game.turn,
            winner: self.winner(),
            resigned: self.resigned,
            updated: self.updated,
        }
    }

    pub fn message(&self, id: &str, seat: Option<u8>) -> String {
        let g = &self.session.game;
        let mut view = match seat {
            Some(s) => g.view(s, 200),
            None => g.spectator_view(200),
        };
        if self.status() != "active" {
            view.actions.clear();
        }
        let seats = [0, 1].map(|i| SeatMeta {
            seat: &self.seats[i],
            online: matches!(self.seats[i], Seat::Bot { .. }) || self.online[i] > 0,
        });
        serde_json::to_string(&ServerMsg::State {
            game: Meta {
                id,
                seats,
                status: self.status(),
                winner: self.winner(),
                resigned: self.resigned,
                sets: &self.session.replay.config.sets,
                created: self.created,
            },
            seat,
            n: self.session.replay.actions.len(),
            view,
        })
        .unwrap()
    }
}

pub enum Move {
    Action(u16),
    PlayAll,
    Resign,
}

impl App {
    pub fn new(db: Db, opponents: BTreeMap<String, Opponent>, bot_delay: Duration) -> App {
        App {
            db,
            rooms: Mutex::new(HashMap::new()),
            opponents,
            catalog: serde_json::to_string(&sr_core::view::catalog()).unwrap(),
            bot_delay,
        }
    }

    /// Built-in bots plus every `<dir>/*.srnn` (MLP) and `<dir>/*.safetensors` (entity) model.
    pub fn load_opponents(models: &std::path::Path) -> BTreeMap<String, Opponent> {
        let mut out = BTreeMap::new();
        for (spec, name) in [("heuristic", "Heuristic bot"), ("greedy", "Greedy bot"), ("random", "Random bot")] {
            out.insert(spec.into(), Opponent { spec: spec.into(), name: name.into(), model: None });
        }
        let Ok(dir) = std::fs::read_dir(models) else { return out };
        let mut files: Vec<_> = dir.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "srnn" || x == "safetensors")).collect();
        files.sort();
        for path in files {
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            match std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| Net::from_bytes(&b)) {
                Ok(net) => {
                    let spec = format!("model:{stem}");
                    tracing::info!("loaded model opponent {spec} from {}", path.display());
                    out.insert(spec.clone(), Opponent { spec, name: format!("Model {stem}"), model: Some(Arc::new(net)) });
                }
                Err(e) => tracing::warn!("skipping {}: {e}", path.display()),
            }
        }
        out
    }

    pub fn room(&self, id: &str) -> Result<Option<Arc<Room>>, String> {
        if let Some(r) = self.rooms.lock().unwrap().get(id) {
            return Ok(Some(r.clone()));
        }
        let Some(row) = self.db.load_game(id).map_err(|e| e.to_string())? else { return Ok(None) };
        let session = Session::from_replay(row.replay)?;
        let room = Arc::new(Room {
            id: id.to_string(),
            state: Mutex::new(RoomState {
                session,
                seats: row.seats,
                created: row.created,
                updated: row.updated,
                resigned: row.resigned,
                online: [0, 0],
                bot_running: false,
            }),
            version: watch::Sender::new(0),
        });
        Ok(Some(self.rooms.lock().unwrap().entry(id.to_string()).or_insert(room).clone()))
    }

    pub fn create_game(&self, sets: Vec<String>, seats: [Seat; 2]) -> Result<Arc<Room>, String> {
        let cfg = Config { sets, log: true, ..Config::default() };
        let session = Session::new(cfg, random_seed())?;
        let id = random_id(10);
        let t = now();
        let state =
            RoomState { session, seats, created: t, updated: t, resigned: None, online: [0, 0], bot_running: false };
        self.db.insert_game(&id, &state.session.replay, t, &state.progress()).map_err(|e| e.to_string())?;
        let room = Arc::new(Room { id: id.clone(), state: Mutex::new(state), version: watch::Sender::new(0) });
        self.rooms.lock().unwrap().insert(id, room.clone());
        Ok(room)
    }

    fn opponent(&self, seat: &Seat) -> Option<&Opponent> {
        match seat {
            Seat::Bot { spec, .. } => self.opponents.get(spec),
            _ => None,
        }
    }
}

impl Room {
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.version.subscribe()
    }

    fn bump(&self) {
        self.version.send_modify(|v| *v += 1);
    }

    pub fn with<T>(&self, f: impl FnOnce(&RoomState) -> T) -> T {
        f(&self.state.lock().unwrap())
    }

    pub fn set_online(&self, seat: Option<u8>, delta: i32) {
        if let Some(s) = seat {
            let mut st = self.state.lock().unwrap();
            st.online[s as usize] = st.online[s as usize].saturating_add_signed(delta);
            drop(st);
            self.bump();
        }
    }

    fn persist(&self, app: &App, st: &mut RoomState) {
        st.updated = now();
        if let Err(e) = app.db.save_progress(&self.id, &st.progress()) {
            tracing::error!("saving game {}: {e}", self.id);
        }
    }

    /// Give an open seat to `player`. Returns their seat.
    pub fn join(self: &Arc<Self>, app: &Arc<App>, player: &Player) -> Result<u8, String> {
        let mut st = self.state.lock().unwrap();
        if let Some(s) = st.seat_of(Some(player)) {
            return Ok(s);
        }
        let s = st.seats.iter().position(|s| *s == Seat::Open).ok_or("this game is full")?;
        st.seats[s] = Seat::Player { id: player.id.clone(), name: player.name.clone() };
        self.persist(app, &mut st);
        drop(st);
        self.bump();
        self.kick_bot(app);
        Ok(s as u8)
    }

    /// Apply a move from `player`. `n` is the action count the client last saw,
    /// so a stale click (double-click, lag) is rejected instead of misapplied.
    pub fn play(self: &Arc<Self>, app: &Arc<App>, player: &Player, n: Option<usize>, mv: Move) -> Result<(), String> {
        let mut st = self.state.lock().unwrap();
        let seat = st.seat_of(Some(player)).ok_or("you are not playing in this game")?;
        if st.finished() {
            return Err("the game is over".into());
        }
        if let Move::Resign = mv {
            st.resigned = Some(seat);
        } else {
            if st.seats.contains(&Seat::Open) {
                return Err("waiting for an opponent to join".into());
            }
            if st.session.game.active != seat {
                return Err("it is not your turn".into());
            }
            if n.is_some_and(|n| n != st.session.replay.actions.len()) {
                return Err("the game moved on; try again".into());
            }
            match mv {
                Move::Action(a) => st.session.step(a)?,
                Move::PlayAll => {
                    while let Some(a) = st.session.game.next_simple_play() {
                        st.session.step(a)?;
                    }
                }
                Move::Resign => unreachable!(),
            }
        }
        self.persist(app, &mut st);
        drop(st);
        self.bump();
        self.kick_bot(app);
        Ok(())
    }

    /// Start the bot loop if a bot is to move. Bots act one action at a time
    /// with a pause, so the other player can follow what they do.
    pub fn kick_bot(self: &Arc<Self>, app: &Arc<App>) {
        {
            let mut st = self.state.lock().unwrap();
            if st.bot_running || !st.bot_to_move() {
                return;
            }
            st.bot_running = true;
        }
        let (room, app) = (self.clone(), app.clone());
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(app.bot_delay).await;
                let mut st = room.state.lock().unwrap();
                if !st.bot_to_move() {
                    st.bot_running = false;
                    return;
                }
                let g = &st.session.game;
                let Some(opp) = app.opponent(&st.seats[g.active as usize]) else {
                    tracing::error!("game {}: unknown bot {:?}", room.id, st.seats[g.active as usize]);
                    st.bot_running = false;
                    return;
                };
                let legal = g.legal_vec();
                let n = st.session.replay.actions.len() as u64;
                let mut rng = Rng::new(st.session.replay.seed ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let a = opp.agent().act(g, &legal, &mut rng);
                if let Err(e) = st.session.step(a) {
                    tracing::error!("game {}: bot chose an illegal action: {e}", room.id);
                    st.bot_running = false;
                    return;
                }
                room.persist(&app, &mut st);
                drop(st);
                room.bump();
            }
        });
    }

    /// Replay of a finished game (it contains the seed, so never while running).
    pub fn finished_replay(&self) -> Option<(Replay, [Seat; 2], Option<u8>)> {
        let st = self.state.lock().unwrap();
        st.finished().then(|| (st.session.replay.clone(), st.seats.clone(), st.resigned))
    }
}
