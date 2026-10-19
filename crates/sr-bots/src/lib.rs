//! Scripted agents. Agents only read information visible to the player to
//! move: they never inspect the opponent's hand or any draw-pile order, and
//! never simulate ahead on a cloned game (which would reveal the shuffle).

#[cfg(feature = "candle")]
pub mod entity_net;
mod heuristic;
pub mod net;
pub mod search;

#[cfg(feature = "candle")]
pub use entity_net::EntityNet;
pub use heuristic::Heuristic;
pub use net::{MlpNet, Net, NetAgent};
pub use search::SearchAgent;
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

pub type AgentFactory = Box<dyn Fn() -> Box<dyn Agent> + Send + Sync>;

fn load_net(path: &str) -> Result<std::sync::Arc<Net>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(std::sync::Arc::new(Net::from_bytes(&bytes)?))
}

/// Constructor for agents of a spec, loading any weights once:
/// - a bot name (`by_name`);
/// - `model:<path>[@temperature]`: a network (`.srnn` or `.safetensors`), greedy by default;
/// - `search:<path>[@width,samples[,prior_weight[,turns]]]`: a network with turn lookahead
///   (`SearchAgent`, default 4,4,0,1).
pub fn factory(spec: &str) -> Result<AgentFactory, String> {
    let name = spec.to_string();
    if let Some(rest) = spec.strip_prefix("model:") {
        let (path, temp) = match rest.rsplit_once('@') {
            Some((p, t)) => (p, t.parse::<f32>().map_err(|e| format!("bad temperature: {e}"))?),
            None => (rest, 0.0),
        };
        let net = load_net(path)?;
        return Ok(Box::new(move || Box::new(NetAgent::new(net.clone(), name.clone(), temp))));
    }
    if let Some(rest) = spec.strip_prefix("search:") {
        let (path, opts) = rest.rsplit_once('@').unwrap_or((rest, "4,4"));
        let opts: Vec<f32> = opts.split(',').map(|s| s.parse::<f32>().map_err(|e| format!("bad search option {s:?}: {e}"))).collect::<Result<_, _>>()?;
        if !(2..=4).contains(&opts.len()) {
            return Err("search options are @width,samples[,prior_weight[,turns]]".into());
        }
        let net = load_net(path)?;
        return Ok(Box::new(move || {
            let mut agent = SearchAgent::new(net.clone(), name.clone(), opts[0] as usize, opts[1] as usize);
            agent.prior_weight = opts.get(2).copied().unwrap_or(0.0);
            agent.turns = opts.get(3).map_or(1, |&t| t as usize);
            Box::new(agent)
        }));
    }
    by_name(spec).ok_or_else(|| format!("unknown agent {spec:?}"))?;
    Ok(Box::new(move || by_name(&name).unwrap()))
}

/// One agent for a spec (see `factory`).
pub fn from_spec(spec: &str) -> Result<Box<dyn Agent>, String> {
    Ok(factory(spec)?())
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
