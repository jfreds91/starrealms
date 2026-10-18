//! Scripted agents. Agents only read information visible to the player to
//! move: they never inspect the opponent's hand or any draw-pile order, and
//! never simulate ahead on a cloned game (which would reveal the shuffle).

mod heuristic;
pub mod net;

pub use heuristic::Heuristic;
pub use net::{Net, NetAgent};
use sr_core::{action, Game, Rng};

pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    /// Pick one of `legal` (non-empty) for the player to move in `g`.
    fn act(&mut self, g: &Game, legal: &[u16], rng: &mut Rng) -> u16;
}

/// Uniform over legal actions.
pub struct RandomAgent;

impl Agent for RandomAgent {
    fn name(&self) -> &str {
        "random"
    }
    fn act(&mut self, _g: &Game, legal: &[u16], rng: &mut Rng) -> u16 {
        legal[rng.below(legal.len())]
    }
}

/// Play everything, use every non-scrap ability, buy the most expensive card,
/// attack outposts then face. The sanity floor for anything smarter.
pub struct Greedy;

impl Agent for Greedy {
    fn name(&self) -> &str {
        "greedy"
    }
    fn act(&mut self, g: &Game, legal: &[u16], _rng: &mut Rng) -> u16 {
        use sr_core::Action as A;
        if g.pending.is_some() {
            // First non-skip option.
            return *legal.iter().find(|&&a| a != action::SKIP).unwrap_or(&legal[0]);
        }
        let decoded: Vec<(u16, A)> = legal.iter().map(|&a| (a, A::decode(a).unwrap())).collect();
        let pick = |pred: &dyn Fn(&A) -> bool| decoded.iter().find(|(_, x)| pred(x)).map(|(a, _)| *a);
        if let Some(a) = pick(&|x| matches!(x, A::Play(_))) {
            return a;
        }
        if let Some(a) = pick(&|x| matches!(x, A::Primary(_) | A::Ally(_) | A::Ally2(_))) {
            return a;
        }
        let best_buy = decoded
            .iter()
            .filter_map(|(a, x)| match x {
                A::Buy(k) => g.trade_row[*k as usize].map(|c| (sr_core::card(c).cost, *a)),
                A::BuyExplorer => Some((2, *a)),
                _ => None,
            })
            .max();
        if let Some((_, a)) = best_buy {
            return a;
        }
        if let Some(a) = pick(&|x| matches!(x, A::AttackBase(_))) {
            return a;
        }
        if let Some(a) = pick(&|x| matches!(x, A::AttackPlayer)) {
            return a;
        }
        action::END
    }
}

pub fn by_name(name: &str) -> Option<Box<dyn Agent>> {
    match name {
        "random" => Some(Box::new(RandomAgent)),
        "greedy" => Some(Box::new(Greedy)),
        "heuristic" => Some(Box::new(Heuristic::default())),
        _ => None,
    }
}

/// Like `by_name`, plus `model:<path.srnn>` or `model:<path.srnn>@<temperature>`.
pub fn from_spec(spec: &str) -> Result<Box<dyn Agent>, String> {
    if let Some(rest) = spec.strip_prefix("model:") {
        let (path, temp) = match rest.rsplit_once('@') {
            Some((p, t)) => (p, t.parse::<f32>().map_err(|e| format!("bad temperature: {e}"))?),
            None => (rest, 0.0),
        };
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let net = Net::from_bytes(&bytes)?;
        return Ok(Box::new(NetAgent::new(std::sync::Arc::new(net), spec, temp)));
    }
    by_name(spec).ok_or_else(|| format!("unknown agent {spec:?}"))
}

/// Play one game between two agents. Returns the finished game.
pub fn play_game(cfg: &sr_core::Config, seed: u64, agents: &mut [Box<dyn Agent>; 2]) -> Game {
    let mut g = Game::new(cfg, seed).expect("valid config");
    let mut rng = Rng::new(seed ^ 0x5eed);
    let mut legal = Vec::with_capacity(64);
    while !g.is_over() {
        g.legal_actions(&mut legal);
        let a = agents[g.active as usize].act(&g, &legal, &mut rng);
        debug_assert!(legal.contains(&a), "{} chose illegal {a}", agents[g.active as usize].name());
        g.apply(a);
    }
    g
}
