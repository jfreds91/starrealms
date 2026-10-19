//! `starrealms._engine`: the Rust engine exposed to Python.
//!
//! `VecEnv` steps many games in parallel (Rayon) and returns NumPy arrays.
//! Both seats are decided from Python by default (self-play); pass
//! `opponent="heuristic"` (or "greedy"/"random") to have one seat played by a
//! scripted bot inside Rust, so Python only sees the learner's decisions.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use sr_bots::{by_name, Agent, Net, NetAgent};
use sr_core::{action, obs, Config, Game, Outcome, Replay, Rng, Session, N_ACTIONS, OBS_DIM};

fn err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn mix(a: u64, b: u64) -> u64 {
    let mut r = Rng::new(a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    r.next_u64()
}

/// Agent constructor for a spec: a bot name, or `model:<path.srnn>[@temperature]`
/// (weights loaded once and shared).
fn agent_factory(spec: &str) -> PyResult<Box<dyn Fn() -> Box<dyn Agent>>> {
    if let Some(rest) = spec.strip_prefix("model:") {
        let (path, temp) = match rest.rsplit_once('@') {
            Some((p, t)) => (p, t.parse::<f32>().map_err(err)?),
            None => (rest, 0.0),
        };
        let net = std::sync::Arc::new(Net::from_bytes(&std::fs::read(path).map_err(err)?).map_err(err)?);
        let name = spec.to_string();
        return Ok(Box::new(move || Box::new(NetAgent::new(net.clone(), name.clone(), temp)) as Box<dyn Agent>));
    }
    by_name(spec).ok_or_else(|| err(format!("unknown opponent {spec:?}")))?;
    let name = spec.to_string();
    Ok(Box::new(move || by_name(&name).unwrap()))
}

struct Slot {
    game: Game,
    opp: Option<Box<dyn Agent>>,
    learner: u8,
    rng: Rng,
    episode: u64,
    index: u64,
}

#[derive(Default, Clone, Copy)]
struct StepOut {
    done: bool,
    rewards: [f32; 2],
    winner: i8,
    turns: u16,
    learner_before: u8,
}

impl Slot {
    fn new_game(&mut self, cfg: &Config, base_seed: u64) {
        let seed = mix(base_seed, self.index << 32 | self.episode);
        self.game = Game::new(cfg, seed).expect("validated config");
        self.learner = ((self.episode + self.index) % 2) as u8;
        self.episode += 1;
    }

    /// Apply forced moves and opponent moves until the learner (or, in
    /// self-play, anyone) must decide. Resets finished games.
    fn advance(&mut self, cfg: &Config, base_seed: u64, auto_forced: bool, out: &mut StepOut) {
        let mut legal = Vec::with_capacity(64);
        loop {
            if let Some(o) = self.game.outcome {
                out.done = true;
                out.turns = self.game.turn;
                out.learner_before = self.learner;
                match o {
                    Outcome::Win(s) => {
                        out.winner = s as i8;
                        out.rewards[s as usize] = 1.0;
                        out.rewards[1 - s as usize] = -1.0;
                    }
                    Outcome::Draw => out.winner = -1,
                }
                self.new_game(cfg, base_seed);
                continue;
            }
            self.game.legal_actions(&mut legal);
            if auto_forced && legal.len() == 1 {
                self.game.apply(legal[0]);
                continue;
            }
            if let Some(bot) = self.opp.as_mut() {
                if self.game.active != self.learner {
                    let a = bot.act(&self.game, &legal, &mut self.rng);
                    self.game.apply(a);
                    continue;
                }
            }
            return;
        }
    }

    fn write(&self, obs_out: &mut [f32], mask_out: &mut [u8]) {
        self.game.observe(obs_out);
        self.game.legal_mask(mask_out);
    }
}

#[pyclass]
struct VecEnv {
    slots: Vec<Slot>,
    cfg: Config,
    seed: u64,
    auto_forced: bool,
}

type Batch<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i8>>);

impl VecEnv {
    fn batch<'py>(&self, py: Python<'py>, obs: Vec<f32>, mask: Vec<u8>) -> PyResult<Batch<'py>> {
        let n = self.slots.len();
        let o = PyArray1::from_vec(py, obs).reshape([n, OBS_DIM])?.into_any();
        let m = PyArray1::from_vec(py, mask).reshape([n, N_ACTIONS])?.call_method1("astype", ("bool",))?;
        let seat = PyArray1::from_vec(py, self.slots.iter().map(|s| s.game.active as i8).collect());
        let learner = PyArray1::from_vec(py, self.slots.iter().map(|s| s.learner as i8).collect());
        Ok((o, m, seat, learner))
    }
}

#[pymethods]
impl VecEnv {
    /// `opponent`: None for self-play, or "random" | "greedy" | "heuristic".
    #[new]
    #[pyo3(signature = (n, sets=None, seed=0, opponent=None, max_turns=200, auto_forced=true))]
    fn new(
        n: usize,
        sets: Option<Vec<String>>,
        seed: u64,
        opponent: Option<String>,
        max_turns: u16,
        auto_forced: bool,
    ) -> PyResult<VecEnv> {
        let cfg = Config { sets: sets.unwrap_or_else(|| vec!["core".into()]), max_turns, ..Config::default() };
        Game::new(&cfg, 0).map_err(err)?;
        let factory = opponent.as_deref().map(agent_factory).transpose()?;
        let mut slots = Vec::with_capacity(n);
        for i in 0..n {
            let opp = factory.as_ref().map(|f| f());
            let mut s = Slot {
                game: Game::new(&cfg, 0).unwrap(),
                opp,
                learner: 0,
                rng: Rng::new(mix(seed, i as u64 + 1_000_003)),
                episode: 0,
                index: i as u64,
            };
            s.new_game(&cfg, seed);
            slots.push(s);
        }
        Ok(VecEnv { slots, cfg, seed, auto_forced })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.slots.len()
    }

    /// Returns (obs [n, OBS_DIM] f32, mask [n, N_ACTIONS] bool, seat [n] i8, learner_seat [n] i8).
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Batch<'py>> {
        let (cfg, seed, af) = (&self.cfg, self.seed, self.auto_forced);
        let n = self.slots.len();
        let mut obs = vec![0f32; n * OBS_DIM];
        let mut mask = vec![0u8; n * N_ACTIONS];
        py.detach(|| {
            self.slots.par_iter_mut().zip(obs.par_chunks_mut(OBS_DIM)).zip(mask.par_chunks_mut(N_ACTIONS)).for_each(
                |((s, o), m)| {
                    s.new_game(cfg, seed);
                    let mut out = StepOut::default();
                    s.advance(cfg, seed, af, &mut out);
                    s.write(o, m);
                },
            )
        });
        self.batch(py, obs, mask)
    }

    /// Apply one action per env. Finished games restart automatically.
    ///
    /// Returns (obs, mask, seat, learner_seat, rewards [n, 2] f32, done [n] bool,
    /// winner [n] i8 (-1 draw, -2 not done), turns [n] u16, done_learner [n] i8).
    /// `rewards[i, s]` is the terminal reward for seat `s` of the game that ended.
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let acts = actions.as_slice()?.to_vec();
        let n = self.slots.len();
        if acts.len() != n {
            return Err(err(format!("expected {n} actions, got {}", acts.len())));
        }
        let (cfg, seed, af) = (&self.cfg, self.seed, self.auto_forced);
        let mut obs = vec![0f32; n * OBS_DIM];
        let mut mask = vec![0u8; n * N_ACTIONS];
        let mut outs = vec![StepOut { winner: -2, ..Default::default() }; n];
        let bad: Option<String> = py.detach(|| {
            self.slots
                .par_iter_mut()
                .zip(obs.par_chunks_mut(OBS_DIM))
                .zip(mask.par_chunks_mut(N_ACTIONS))
                .zip(outs.par_iter_mut())
                .zip(acts.par_iter())
                .map(|((((s, o), m), out), &a)| {
                    let mut legal = Vec::with_capacity(64);
                    s.game.legal_actions(&mut legal);
                    if a < 0 || !legal.contains(&(a as u16)) {
                        return Some(format!("env {}: illegal action {a}", s.index));
                    }
                    s.game.apply(a as u16);
                    s.advance(cfg, seed, af, out);
                    s.write(o, m);
                    None
                })
                .reduce(|| None, |a, b| a.or(b))
        });
        if let Some(e) = bad {
            return Err(err(e));
        }
        let (o, m, seat, learner) = self.batch(py, obs, mask)?;
        let rewards = PyArray1::from_vec(py, outs.iter().flat_map(|x| x.rewards).collect()).reshape([n, 2])?;
        let done = PyArray1::from_vec(py, outs.iter().map(|x| x.done).collect());
        let winner = PyArray1::from_vec(py, outs.iter().map(|x| x.winner).collect());
        let turns = PyArray1::from_vec(py, outs.iter().map(|x| x.turns).collect());
        let done_learner = PyArray1::from_vec(py, outs.iter().map(|x| x.learner_before as i8).collect());
        pyo3::types::PyTuple::new(
            py,
            [
                o,
                m,
                seat.into_any(),
                learner.into_any(),
                rewards.into_any(),
                done.into_any(),
                winner.into_any(),
                turns.into_any(),
                done_learner.into_any(),
            ],
        )
    }

    /// Named per-env quantities for the seat to move, for shaped rewards and dashboards.
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let col = |f: &dyn Fn(&Game) -> f32| PyArray1::from_vec(py, self.slots.iter().map(|s| f(&s.game)).collect());
        d.set_item("authority", col(&|g| g.me().authority as f32))?;
        d.set_item("opp_authority", col(&|g| g.opp().authority as f32))?;
        d.set_item("trade", col(&|g| g.t.trade as f32))?;
        d.set_item("combat", col(&|g| g.t.combat as f32))?;
        d.set_item("turn", col(&|g| g.turn as f32))?;
        d.set_item("deck_size", col(&|g| g.me().total_cards() as f32))?;
        d.set_item("bases", col(&|g| g.me().num_bases() as f32))?;
        d.set_item("opp_bases", col(&|g| g.opp().num_bases() as f32))?;
        d.set_item("scrap_heap", col(&|g| g.scrap_heap.len() as f32))?;
        Ok(d)
    }

    /// Text rendering of env `i` from the mover's perspective.
    fn render(&self, i: usize) -> PyResult<String> {
        let s = self.slots.get(i).ok_or_else(|| err("env index"))?;
        Ok(s.game.render(s.game.active))
    }

    fn describe(&self, i: usize, a: u16) -> PyResult<String> {
        let s = self.slots.get(i).ok_or_else(|| err("env index"))?;
        Ok(s.game.describe(a))
    }
}

/// A single game for scripting, debugging, and tests.
#[pyclass(name = "Game")]
struct PyGame {
    s: Session,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (sets=None, seed=0, max_turns=200, log=true))]
    fn new(sets: Option<Vec<String>>, seed: u64, max_turns: u16, log: bool) -> PyResult<PyGame> {
        let cfg = Config { sets: sets.unwrap_or_else(|| vec!["core".into()]), max_turns, log, ..Config::default() };
        Ok(PyGame { s: Session::new(cfg, seed).map_err(err)? })
    }

    #[staticmethod]
    fn from_replay(json: &str) -> PyResult<PyGame> {
        let r: Replay = serde_json::from_str(json).map_err(err)?;
        Ok(PyGame { s: Session::from_replay(r).map_err(err)? })
    }

    fn replay(&self) -> String {
        serde_json::to_string(&self.s.replay).unwrap()
    }

    #[getter]
    fn active(&self) -> u8 {
        self.s.game.active
    }

    #[getter]
    fn turn(&self) -> u16 {
        self.s.game.turn
    }

    /// None while running, -1 for a draw, else the winning seat.
    #[getter]
    fn winner(&self) -> Option<i8> {
        self.s.game.outcome.map(|o| match o {
            Outcome::Win(s) => s as i8,
            Outcome::Draw => -1,
        })
    }

    fn legal(&self) -> Vec<u16> {
        self.s.game.legal_vec()
    }

    fn step(&mut self, a: u16) -> PyResult<()> {
        self.s.step(a).map_err(err)
    }

    fn describe(&self, a: u16) -> String {
        self.s.game.describe(a)
    }

    fn observe<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.s.game.observation())
    }

    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut m = vec![0u8; N_ACTIONS];
        self.s.game.legal_mask(&mut m);
        PyArray1::from_vec(py, m).call_method1("astype", ("bool",))
    }

    /// JSON view for `seat` (what that seat can see).
    #[pyo3(signature = (seat, log_tail=20))]
    fn view(&self, seat: u8, log_tail: usize) -> String {
        serde_json::to_string(&self.s.game.view(seat, log_tail)).unwrap()
    }

    #[pyo3(signature = (seat=None))]
    fn render(&self, seat: Option<u8>) -> String {
        self.s.game.render(seat.unwrap_or(self.s.game.active))
    }

    /// Action a scripted bot would take now.
    #[pyo3(signature = (name, seed=0))]
    fn bot_action(&self, name: &str, seed: u64) -> PyResult<u16> {
        let mut bot = by_name(name).ok_or_else(|| err(format!("unknown bot {name:?}")))?;
        let legal = self.s.game.legal_vec();
        if legal.is_empty() {
            return Err(err("game is over"));
        }
        Ok(bot.act(&self.s.game, &legal, &mut Rng::new(seed)))
    }
}

/// Play `games` games between two scripted bots (alternating seats).
/// Returns (wins_a, wins_b, draws).
#[pyfunction]
#[pyo3(signature = (a, b, games=1000, sets=None, seed=0))]
fn arena(py: Python<'_>, a: &str, b: &str, games: u64, sets: Option<Vec<String>>, seed: u64) -> PyResult<(u64, u64, u64)> {
    let cfg = Config { sets: sets.unwrap_or_else(|| vec!["core".into()]), ..Config::default() };
    by_name(a).ok_or_else(|| err(format!("unknown bot {a:?}")))?;
    by_name(b).ok_or_else(|| err(format!("unknown bot {b:?}")))?;
    Ok(py.detach(|| {
        (0..games)
            .into_par_iter()
            .map(|i| {
                let swap = i % 2 == 1;
                let (x, y) = if swap { (b, a) } else { (a, b) };
                let mut agents = [by_name(x).unwrap(), by_name(y).unwrap()];
                let g = sr_bots::play_game(&cfg, mix(seed, i), &mut agents);
                match g.outcome {
                    Some(Outcome::Win(s)) if (s == 0) != swap => (1, 0, 0),
                    Some(Outcome::Win(_)) => (0, 1, 0),
                    _ => (0, 0, 1),
                }
            })
            .reduce(|| (0, 0, 0), |p, q| (p.0 + q.0, p.1 + q.1, p.2 + q.2))
    }))
}

#[pyfunction]
fn card_catalog() -> String {
    serde_json::to_string(&sr_core::view::catalog()).unwrap()
}

#[pyfunction]
fn obs_layout() -> Vec<(&'static str, usize, usize)> {
    obs::LAYOUT.to_vec()
}

/// (name, first id, count) of each action family, in id order.
#[pyfunction]
fn action_layout() -> Vec<(&'static str, u16, u16)> {
    use action::*;
    let starts = [
        ("end", END),
        ("skip", SKIP),
        ("choose", CHOOSE),
        ("play", PLAY),
        ("primary", PRIMARY),
        ("ally", ALLY),
        ("ally2", ALLY2),
        ("scrap", SCRAP),
        ("buy", BUY),
        ("buy_explorer", BUY_EXPLORER),
        ("attack_player", ATTACK_PLAYER),
        ("attack_base", ATTACK_BASE),
        ("pick_hand", PICK_HAND),
        ("pick_discard", PICK_DISCARD),
        ("pick_row", PICK_ROW),
        ("pick_explorer", PICK_EXPLORER),
        ("pick_opp_base", PICK_OPP_BASE),
        ("pick_own", PICK_OWN),
    ];
    let mut out = Vec::new();
    for (i, &(name, s)) in starts.iter().enumerate() {
        let e = starts.get(i + 1).map_or(N_ACTIONS as u16, |x| x.1);
        out.push((name, s, e - s));
    }
    out
}

/// `[N_CARDS, N_CARD_FEATS]` static card features (see `sr_core::obs::card_features`).
#[pyfunction]
fn card_features<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rows: Vec<Vec<f32>> = obs::card_features().iter().map(|r| r.to_vec()).collect();
    PyArray2::from_vec2(py, &rows).map_err(err)
}

/// Rust forward pass of an exported model (`.srnn` or entity `.safetensors`): (unmasked logits, value).
#[pyfunction]
fn srnn_forward<'py>(
    py: Python<'py>,
    path: &str,
    obs: PyReadonlyArray1<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, f32)> {
    let net = Net::from_bytes(&std::fs::read(path).map_err(err)?).map_err(err)?;
    let o = obs.as_slice()?;
    if o.len() != OBS_DIM {
        return Err(err(format!("expected {OBS_DIM} features, got {}", o.len())));
    }
    let (logits, v) = net.forward(o);
    Ok((PyArray1::from_vec(py, logits), v))
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_class::<PyGame>()?;
    m.add_function(wrap_pyfunction!(arena, m)?)?;
    m.add_function(wrap_pyfunction!(card_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(obs_layout, m)?)?;
    m.add_function(wrap_pyfunction!(srnn_forward, m)?)?;
    m.add_function(wrap_pyfunction!(action_layout, m)?)?;
    m.add_function(wrap_pyfunction!(card_features, m)?)?;
    m.add("OBS_DIM", OBS_DIM)?;
    m.add("N_ACTIONS", N_ACTIONS)?;
    m.add("N_CARDS", sr_core::N_CARDS)?;
    m.add("N_IDS", obs::N_IDS)?;
    m.add("COUNT_ZONES", obs::COUNT_ZONES.to_vec())?;
    m.add("SCALAR_NAMES", obs::SCALAR_NAMES.to_vec())?;
    m.add("END", action::END)?;
    m.add("SKIP", action::SKIP)?;
    Ok(())
}
