//! Game state and rules.
//!
//! `Game` is a plain value: `Game::new(config, seed)` plus a sequence of legal
//! actions reproduces any game. Only the active player ever makes decisions;
//! "opponent discards" effects accumulate on the opponent and are resolved as
//! the first decision of their next turn.

use crate::action::{self, Action, MAX_CHOICES, MAX_OPP_BASES, MAX_OWN_SLOTS, ROW};
use crate::cards::*;
use crate::rng::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    /// Trade-deck sets to shuffle together: any of "core", "colony_wars", "frontiers".
    pub sets: Vec<String>,
    /// Total player-turns before the game is declared a draw.
    pub max_turns: u16,
    /// Automatically use abilities that only gain trade/combat/authority or
    /// make the opponent discard. Never changes the set of reachable outcomes.
    pub auto_pure: bool,
    pub explorers: u8,
    pub starting_authority: i16,
    /// Record a human-readable log of actions (for the UI and debugging).
    pub log: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config { sets: vec!["core".into()], max_turns: 200, auto_pure: true, explorers: 10, starting_authority: 50, log: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Win(u8),
    Draw,
}

pub const F_PRIMARY: u8 = 1;
pub const F_ALLY: u8 = 2;
pub const F_ALLY2: u8 = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct InPlay {
    pub card: CardId,
    /// Card being copied (Stealth Needle / Stealth Tower).
    pub copy: Option<CardId>,
    pub uid: u16,
    pub used: u8,
    pub unlocked: u8,
}

impl InPlay {
    #[inline]
    pub fn eff(&self) -> CardId {
        self.copy.unwrap_or(self.card)
    }
    #[inline]
    pub fn def(&self) -> &'static Card {
        card(self.eff())
    }
    #[inline]
    pub fn is_base(&self) -> bool {
        card(self.card).is_base()
    }
    pub fn faction_bits(&self) -> u8 {
        card(self.card).faction.bit() | self.copy.map_or(0, |c| card(c).faction.bit())
    }
    pub fn defense(&self) -> u8 {
        card(self.card).defense
    }
    pub fn outpost(&self) -> bool {
        card(self.card).outpost
    }
    pub fn primary_available(&self) -> bool {
        self.is_base() && self.used & F_PRIMARY == 0 && !self.def().primary.is_empty()
    }
    pub fn ally_available(&self) -> bool {
        self.used & F_ALLY == 0 && self.unlocked & F_ALLY != 0 && !self.def().ally.is_empty()
    }
    pub fn ally2_available(&self) -> bool {
        self.used & F_ALLY2 == 0 && self.unlocked & F_ALLY2 != 0 && !self.def().ally2.is_empty()
    }
    pub fn scrap_available(&self) -> bool {
        !self.def().scrap.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Player {
    pub authority: i32,
    /// Draw pile; the top card is the last element.
    pub deck: Vec<CardId>,
    pub hand: Vec<CardId>,
    pub discard: Vec<CardId>,
    /// Bases, then (during this player's turn) ships played this turn, in play order.
    pub in_play: Vec<InPlay>,
    pub owed_discards: u8,
}

impl Player {
    pub fn bases(&self) -> impl Iterator<Item = &InPlay> {
        self.in_play.iter().filter(|e| e.is_base())
    }
    /// Index into `in_play` of the `b`-th base.
    pub fn base_slot(&self, b: usize) -> Option<usize> {
        self.in_play.iter().enumerate().filter(|(_, e)| e.is_base()).nth(b).map(|(i, _)| i)
    }
    pub fn has_outpost(&self) -> bool {
        self.in_play.iter().any(|e| e.is_base() && e.outpost())
    }
    pub fn num_bases(&self) -> usize {
        self.bases().count()
    }
    pub fn total_cards(&self) -> usize {
        self.deck.len() + self.hand.len() + self.discard.len() + self.in_play.len()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Turn {
    pub trade: i32,
    pub combat: i32,
    pub played_factions: u8,
    pub played_count: [u8; 4],
    /// Cards of this player scrapped this turn by any means (Reclamation Station).
    pub scrapped_own: u8,
    /// Cards scrapped from hand or discard pile this turn (The Incinerator).
    pub scrapped_hd: u8,
    pub next_acquire: Vec<NextAcquireSpec>,
    /// Cards to move from the scrap heap to the discard pile at end of turn (Mobile Market).
    pub returns: Vec<CardId>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Src {
    pub uid: u16,
    pub card: CardId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Job {
    Eff(&'static Effect, Src),
    Draw(u8),
    Combat(i32),
    TradeOrCombat(u8, u8),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pending {
    Choose { options: &'static [Vec<Effect>], src: Src },
    Scrap { spec: &'static ScrapSpec, picked: u8, src: Src },
    Discard { min: u8, max: u8, picked: u8, reward: DiscardReward, forced: bool },
    DestroyBase { src: Src },
    Acquire { spec: &'static AcquireSpec, src: Src },
    TopdeckBase { src: Src },
    CopyShip { src: Src },
    CopyBase { src: Src },
    TopdeckChoice { card: CardId },
    TradeOrCombat { amount: u8, remaining: u8 },
}

impl Pending {
    pub fn kind(&self) -> PendingKind {
        match self {
            Pending::Choose { .. } => PendingKind::Choose,
            Pending::Scrap { spec, .. } => match spec.from {
                Zone::Hand => PendingKind::ScrapHand,
                Zone::Discard => PendingKind::ScrapDiscard,
                Zone::HandOrDiscard => PendingKind::ScrapHandOrDiscard,
                Zone::TradeRow => PendingKind::ScrapRow,
            },
            Pending::Discard { forced: true, .. } => PendingKind::ForcedDiscard,
            Pending::Discard { .. } => PendingKind::Discard,
            Pending::DestroyBase { .. } => PendingKind::DestroyBase,
            Pending::Acquire { .. } => PendingKind::Acquire,
            Pending::TopdeckBase { .. } => PendingKind::TopdeckBase,
            Pending::CopyShip { .. } => PendingKind::CopyShip,
            Pending::CopyBase { .. } => PendingKind::CopyBase,
            Pending::TopdeckChoice { .. } => PendingKind::TopdeckChoice,
            Pending::TradeOrCombat { .. } => PendingKind::TradeOrCombat,
        }
    }

    pub fn source(&self) -> Option<CardId> {
        match self {
            Pending::Choose { src, .. }
            | Pending::Scrap { src, .. }
            | Pending::DestroyBase { src }
            | Pending::Acquire { src, .. }
            | Pending::TopdeckBase { src }
            | Pending::CopyShip { src }
            | Pending::CopyBase { src } => Some(src.card),
            Pending::TopdeckChoice { card } => Some(*card),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingKind {
    Choose,
    ScrapHand,
    ScrapDiscard,
    ScrapHandOrDiscard,
    ScrapRow,
    Discard,
    ForcedDiscard,
    DestroyBase,
    Acquire,
    TopdeckBase,
    CopyShip,
    CopyBase,
    TopdeckChoice,
    TradeOrCombat,
}

impl PendingKind {
    pub const COUNT: usize = 14;
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct LogEntry {
    pub turn: u16,
    pub seat: u8,
    pub text: String,
}

#[derive(Clone)]
pub struct Game {
    pub rng: Rng,
    pub max_turns: u16,
    pub auto_pure: bool,
    pub log_enabled: bool,
    pub turn: u16,
    pub active: u8,
    pub trade_deck: Vec<CardId>,
    pub trade_row: [Option<CardId>; ROW],
    pub explorers: u8,
    pub scrap_heap: Vec<CardId>,
    pub players: [Player; 2],
    pub t: Turn,
    pub stack: Vec<Job>,
    pub pending: Option<Pending>,
    pub next_uid: u16,
    pub outcome: Option<Outcome>,
    pub log: Vec<LogEntry>,
}

const SCOUTS: usize = 8;
const VIPERS: usize = 2;

impl Game {
    pub fn new(cfg: &Config, seed: u64) -> Result<Game, String> {
        let d = db();
        if cfg.sets.is_empty() {
            return Err("no sets selected".into());
        }
        let mut trade_deck = Vec::with_capacity(80 * cfg.sets.len());
        for s in &cfg.sets {
            let set = d.sets.get(s).ok_or_else(|| format!("unknown set {s:?}"))?;
            for e in &set.trade_deck {
                let id = d.by_id[&e.card];
                trade_deck.extend(std::iter::repeat(id).take(e.count as usize));
            }
        }
        let mut rng = Rng::new(seed);
        rng.shuffle(&mut trade_deck);
        let mut trade_row = [None; ROW];
        for slot in trade_row.iter_mut() {
            *slot = trade_deck.pop();
        }
        let mut players: [Player; 2] = Default::default();
        for p in players.iter_mut() {
            p.authority = cfg.starting_authority as i32;
            p.deck.extend(std::iter::repeat(d.scout).take(SCOUTS));
            p.deck.extend(std::iter::repeat(d.viper).take(VIPERS));
            rng.shuffle(&mut p.deck);
        }
        let mut g = Game {
            rng,
            max_turns: cfg.max_turns,
            auto_pure: cfg.auto_pure,
            log_enabled: cfg.log,
            turn: 1,
            active: 0,
            trade_deck,
            trade_row,
            explorers: cfg.explorers,
            scrap_heap: Vec::new(),
            players,
            t: Turn::default(),
            stack: Vec::new(),
            pending: None,
            next_uid: 0,
            outcome: None,
            log: Vec::new(),
        };
        g.draw(0, 3);
        g.draw(1, 5);
        g.settle();
        Ok(g)
    }

    // ------------------------------------------------------------ accessors

    #[inline]
    pub fn me(&self) -> &Player {
        &self.players[self.active as usize]
    }
    #[inline]
    pub fn opp(&self) -> &Player {
        &self.players[1 - self.active as usize]
    }
    #[inline]
    fn a(&self) -> usize {
        self.active as usize
    }
    #[inline]
    fn o(&self) -> usize {
        1 - self.active as usize
    }
    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    /// Damage dealt to the opponent if the turn ended now.
    pub fn end_turn_damage(&self) -> i32 {
        if self.t.combat > 0 && !self.opp().has_outpost() {
            self.t.combat
        } else {
            0
        }
    }

    // ------------------------------------------------------------ legal actions

    pub fn legal_actions(&self, out: &mut Vec<u16>) {
        out.clear();
        if self.outcome.is_some() {
            return;
        }
        match self.pending {
            Some(p) => self.legal_pending(p, out),
            None => self.legal_main(out),
        }
    }

    pub fn legal_vec(&self) -> Vec<u16> {
        let mut v = Vec::with_capacity(32);
        self.legal_actions(&mut v);
        v
    }

    pub fn legal_mask(&self, mask: &mut [u8]) {
        debug_assert_eq!(mask.len(), action::N_ACTIONS);
        mask.fill(0);
        let mut v = Vec::with_capacity(32);
        self.legal_actions(&mut v);
        for a in v {
            mask[a as usize] = 1;
        }
    }

    fn push_distinct(cards: &[CardId], base: u16, out: &mut Vec<u16>, filter: impl Fn(CardId) -> bool) {
        let mut seen = [false; N_CARDS];
        for &c in cards {
            if !seen[c as usize] && filter(c) {
                seen[c as usize] = true;
                out.push(base + c as u16);
            }
        }
    }

    fn legal_main(&self, out: &mut Vec<u16>) {
        let me = self.me();
        let opp = self.opp();
        out.push(action::END);
        Self::push_distinct(&me.hand, action::PLAY, out, |_| true);
        for (j, e) in me.in_play.iter().enumerate().take(MAX_OWN_SLOTS) {
            let j = j as u16;
            if e.primary_available() {
                out.push(action::PRIMARY + j);
            }
            if e.ally_available() {
                out.push(action::ALLY + j);
            }
            if e.ally2_available() {
                out.push(action::ALLY2 + j);
            }
            if e.scrap_available() {
                out.push(action::SCRAP + j);
            }
        }
        for (k, slot) in self.trade_row.iter().enumerate() {
            if let Some(c) = slot {
                if self.t.trade >= card(*c).cost as i32 {
                    out.push(action::BUY + k as u16);
                }
            }
        }
        if self.explorers > 0 && self.t.trade >= card(db().explorer).cost as i32 {
            out.push(action::BUY_EXPLORER);
        }
        if self.t.combat > 0 {
            let outposts = opp.has_outpost();
            if !outposts {
                out.push(action::ATTACK_PLAYER);
            }
            for (b, e) in opp.bases().enumerate().take(MAX_OPP_BASES) {
                if self.t.combat >= e.defense() as i32 && (e.outpost() || !outposts) {
                    out.push(action::ATTACK_BASE + b as u16);
                }
            }
        }
    }

    fn acquire_candidates(&self, spec: &AcquireSpec, mut f: impl FnMut(Option<usize>)) {
        let ok = |c: CardId| {
            let d = card(c);
            spec.filter.matches(d.kind) && spec.max_cost.map_or(true, |m| d.cost <= m)
        };
        for (k, slot) in self.trade_row.iter().enumerate() {
            if let Some(c) = slot {
                if ok(*c) {
                    f(Some(k));
                }
            }
        }
        if self.explorers > 0 && ok(db().explorer) {
            f(None);
        }
    }

    fn copy_ship_targets(&self, src: Src, mut f: impl FnMut(usize)) {
        for (j, e) in self.me().in_play.iter().enumerate().take(MAX_OWN_SLOTS) {
            if !e.is_base() && e.uid != src.uid && !e.def().primary.contains(&Effect::CopyShip) {
                f(j);
            }
        }
    }

    /// Calls `own(j)` for own in-play slots and `opp(b)` for opponent base indices.
    fn copy_base_targets(&self, src: Src, mut own: impl FnMut(usize), mut opp: impl FnMut(usize)) {
        for (j, e) in self.me().in_play.iter().enumerate().take(MAX_OWN_SLOTS) {
            if e.is_base() && e.uid != src.uid && !e.def().primary.contains(&Effect::CopyBase) {
                own(j);
            }
        }
        for (b, e) in self.opp().bases().enumerate().take(MAX_OPP_BASES) {
            if !e.def().primary.contains(&Effect::CopyBase) {
                opp(b);
            }
        }
    }

    fn legal_pending(&self, p: Pending, out: &mut Vec<u16>) {
        let me = self.me();
        match p {
            Pending::Choose { options, .. } => {
                for i in 0..options.len().min(MAX_CHOICES) {
                    out.push(action::CHOOSE + i as u16);
                }
            }
            Pending::Scrap { spec, picked, .. } => {
                if picked >= spec.min {
                    out.push(action::SKIP);
                }
                if matches!(spec.from, Zone::Hand | Zone::HandOrDiscard) {
                    Self::push_distinct(&me.hand, action::PICK_HAND, out, |_| true);
                }
                if matches!(spec.from, Zone::Discard | Zone::HandOrDiscard) {
                    Self::push_distinct(&me.discard, action::PICK_DISCARD, out, |_| true);
                }
                if spec.from == Zone::TradeRow {
                    for (k, s) in self.trade_row.iter().enumerate() {
                        if s.is_some() {
                            out.push(action::PICK_ROW + k as u16);
                        }
                    }
                }
            }
            Pending::Discard { min, picked, .. } => {
                if picked >= min {
                    out.push(action::SKIP);
                }
                Self::push_distinct(&me.hand, action::PICK_HAND, out, |_| true);
            }
            Pending::DestroyBase { .. } => {
                out.push(action::SKIP);
                for b in 0..self.opp().num_bases().min(MAX_OPP_BASES) {
                    out.push(action::PICK_OPP_BASE + b as u16);
                }
            }
            Pending::Acquire { spec, .. } => {
                out.push(action::SKIP);
                self.acquire_candidates(spec, |k| match k {
                    Some(k) => out.push(action::PICK_ROW + k as u16),
                    None => out.push(action::PICK_EXPLORER),
                });
            }
            Pending::TopdeckBase { .. } => {
                Self::push_distinct(&me.discard, action::PICK_DISCARD, out, |c| card(c).is_base());
            }
            Pending::CopyShip { src } => {
                self.copy_ship_targets(src, |j| out.push(action::PICK_OWN + j as u16));
            }
            Pending::CopyBase { src } => {
                let mut own = Vec::new();
                let mut opp = Vec::new();
                self.copy_base_targets(src, |j| own.push(j), |b| opp.push(b));
                out.extend(own.into_iter().map(|j| action::PICK_OWN + j as u16));
                out.extend(opp.into_iter().map(|b| action::PICK_OPP_BASE + b as u16));
            }
            Pending::TopdeckChoice { .. } | Pending::TradeOrCombat { .. } => {
                out.push(action::CHOOSE);
                out.push(action::CHOOSE + 1);
            }
        }
    }

    // ------------------------------------------------------------ stepping

    /// Apply `a` if it is legal.
    pub fn step(&mut self, a: u16) -> Result<(), String> {
        let legal = self.legal_vec();
        if !legal.contains(&a) {
            return Err(format!("illegal action {a} ({:?})", Action::decode(a)));
        }
        self.apply(a);
        Ok(())
    }

    /// Apply an action known to be legal (e.g. sampled from the mask).
    pub fn apply(&mut self, a: u16) {
        let act = Action::decode(a).expect("action out of range");
        if self.log_enabled {
            let text = self.describe(a);
            self.log.push(LogEntry { turn: self.turn, seat: self.active, text });
        }
        match self.pending.take() {
            Some(p) => self.resolve(p, act),
            None => self.main(act),
        }
        self.settle();
    }

    /// Apply forced moves (exactly one legal action) until a real decision is
    /// reached. Ending the turn is only auto-applied if `include_end`.
    /// Returns the actions applied.
    pub fn apply_forced(&mut self, include_end: bool) -> Vec<u16> {
        let mut applied = Vec::new();
        let mut legal = Vec::new();
        loop {
            self.legal_actions(&mut legal);
            if legal.len() != 1 || (legal[0] == action::END && !include_end) {
                return applied;
            }
            let a = legal[0];
            self.apply(a);
            applied.push(a);
        }
    }

    /// A `Play` action for a card in hand that resolves without asking the
    /// player anything (the UI's "play all"). `None` once every remaining card
    /// needs a decision. Checked on a copy, which only learns what playing the
    /// card would reveal anyway.
    pub fn next_simple_play(&self) -> Option<u16> {
        if self.pending.is_some() || self.outcome.is_some() {
            return None;
        }
        let mut hand = self.me().hand.clone();
        hand.sort_unstable();
        hand.dedup();
        hand.into_iter().map(|c| Action::Play(c).encode()).find(|&a| {
            let mut g = self.clone();
            g.log_enabled = false;
            g.apply(a);
            g.pending.is_none()
        })
    }

    fn main(&mut self, act: Action) {
        let (a, o) = (self.a(), self.o());
        match act {
            Action::EndTurn => self.end_turn(),
            Action::Play(c) => {
                let hand = &mut self.players[a].hand;
                let pos = hand.iter().position(|&x| x == c).expect("card in hand");
                hand.swap_remove(pos);
                self.put_in_play(c);
            }
            Action::Primary(j) => self.activate(j as usize, F_PRIMARY),
            Action::Ally(j) => self.activate(j as usize, F_ALLY),
            Action::Ally2(j) => self.activate(j as usize, F_ALLY2),
            Action::ScrapAbility(j) => {
                let e = self.players[a].in_play.remove(j as usize);
                let def = e.def();
                let phys = card(e.card);
                self.t.scrapped_own = self.t.scrapped_own.saturating_add(1);
                self.to_scrap(e.card);
                if phys.returns_from_scrap {
                    self.t.returns.push(e.card);
                }
                let src = Src { uid: e.uid, card: e.eff() };
                self.push_effects(&def.scrap, src);
            }
            Action::Buy(k) => {
                let c = self.trade_row[k as usize].take().expect("card in row");
                self.t.trade -= card(c).cost as i32;
                self.refill_row(k as usize);
                self.gain_card(c, Dest::Discard);
            }
            Action::BuyExplorer => {
                self.explorers -= 1;
                let c = db().explorer;
                self.t.trade -= card(c).cost as i32;
                self.gain_card(c, Dest::Discard);
            }
            Action::AttackPlayer => {
                let dmg = self.t.combat;
                self.t.combat = 0;
                self.damage_opp(dmg);
            }
            Action::AttackBase(b) => {
                let slot = self.players[o].base_slot(b as usize).expect("base");
                self.t.combat -= self.players[o].in_play[slot].defense() as i32;
                self.destroy_opp_slot(slot);
            }
            other => panic!("{other:?} is not a main-phase action"),
        }
    }

    fn resolve(&mut self, p: Pending, act: Action) {
        let (a, o) = (self.a(), self.o());
        match (p, act) {
            (Pending::Choose { options, src }, Action::Choose(i)) => {
                self.push_effects(&options[i as usize], src);
            }
            (Pending::Scrap { spec, picked, src }, act) => {
                let picked_card = match act {
                    Action::Skip => None,
                    Action::PickHand(c) => {
                        let h = &mut self.players[a].hand;
                        h.swap_remove(h.iter().position(|&x| x == c).unwrap());
                        self.t.scrapped_own = self.t.scrapped_own.saturating_add(1);
                        self.t.scrapped_hd = self.t.scrapped_hd.saturating_add(1);
                        self.to_scrap(c);
                        Some(c)
                    }
                    Action::PickDiscard(c) => {
                        let d = &mut self.players[a].discard;
                        d.swap_remove(d.iter().position(|&x| x == c).unwrap());
                        self.t.scrapped_own = self.t.scrapped_own.saturating_add(1);
                        self.t.scrapped_hd = self.t.scrapped_hd.saturating_add(1);
                        self.to_scrap(c);
                        Some(c)
                    }
                    Action::PickRow(k) => {
                        let c = self.trade_row[k as usize].take().unwrap();
                        self.scrap_heap.push(c);
                        self.refill_row(k as usize);
                        Some(c)
                    }
                    other => panic!("{other:?} during scrap"),
                };
                match picked_card {
                    None => self.finish_scrap(spec, picked),
                    Some(c) => {
                        if spec.reward == ScrapReward::CombatCost {
                            self.t.combat += card(c).cost as i32;
                        }
                        let picked = picked + 1;
                        if picked >= spec.max || !self.scrap_candidates(spec.from) {
                            self.finish_scrap(spec, picked);
                        } else {
                            self.pending = Some(Pending::Scrap { spec, picked, src });
                        }
                    }
                }
            }
            (Pending::Discard { min, max, picked, reward, forced }, act) => match act {
                Action::Skip => self.finish_discard(reward, picked),
                Action::PickHand(c) => {
                    let p = &mut self.players[a];
                    p.hand.swap_remove(p.hand.iter().position(|&x| x == c).unwrap());
                    p.discard.push(c);
                    let picked = picked + 1;
                    if picked >= max || self.players[a].hand.is_empty() {
                        self.finish_discard(reward, picked);
                    } else {
                        self.pending = Some(Pending::Discard { min, max, picked, reward, forced });
                    }
                }
                other => panic!("{other:?} during discard"),
            },
            (Pending::DestroyBase { .. }, act) => match act {
                Action::Skip => {}
                Action::PickOppBase(b) => {
                    let slot = self.players[o].base_slot(b as usize).unwrap();
                    self.destroy_opp_slot(slot);
                }
                other => panic!("{other:?} during destroy base"),
            },
            (Pending::Acquire { spec, .. }, act) => {
                let c = match act {
                    Action::Skip => return,
                    Action::PickRow(k) => {
                        let c = self.trade_row[k as usize].take().unwrap();
                        self.refill_row(k as usize);
                        c
                    }
                    Action::PickExplorer => {
                        self.explorers -= 1;
                        db().explorer
                    }
                    other => panic!("{other:?} during acquire"),
                };
                self.gain_card(c, spec.dest);
            }
            (Pending::TopdeckBase { .. }, Action::PickDiscard(c)) => {
                let p = &mut self.players[a];
                p.discard.swap_remove(p.discard.iter().position(|&x| x == c).unwrap());
                p.deck.push(c);
            }
            (Pending::CopyShip { src }, Action::PickOwn(j)) => {
                let target = self.players[a].in_play[j as usize].eff();
                let me = &mut self.players[a];
                let needle = me.in_play.iter_mut().find(|e| e.uid == src.uid).expect("copier in play");
                needle.copy = Some(target);
                let f = card(target).faction;
                self.t.played_factions |= f.bit();
                if let Some(i) = f.index() {
                    self.t.played_count[i] += 1;
                }
                self.push_effects(&card(target).primary, Src { uid: src.uid, card: target });
            }
            (Pending::CopyBase { src }, act) => {
                let target = match act {
                    Action::PickOwn(j) => self.players[a].in_play[j as usize].eff(),
                    Action::PickOppBase(b) => {
                        let slot = self.players[o].base_slot(b as usize).unwrap();
                        self.players[o].in_play[slot].eff()
                    }
                    other => panic!("{other:?} during copy base"),
                };
                let tower = self.players[a].in_play.iter_mut().find(|e| e.uid == src.uid).expect("copier in play");
                tower.copy = Some(target);
                tower.used = 0;
                tower.unlocked = 0;
            }
            (Pending::TopdeckChoice { card: c }, Action::Choose(i)) => {
                let p = &mut self.players[a];
                if i == 0 {
                    p.deck.push(c);
                } else {
                    p.discard.push(c);
                }
            }
            (Pending::TradeOrCombat { amount, remaining }, Action::Choose(i)) => {
                if i == 0 {
                    self.t.trade += amount as i32;
                } else {
                    self.t.combat += amount as i32;
                }
                if remaining > 1 {
                    self.pending = Some(Pending::TradeOrCombat { amount, remaining: remaining - 1 });
                }
            }
            (p, act) => panic!("{act:?} does not answer {p:?}"),
        }
    }

    fn finish_scrap(&mut self, spec: &ScrapSpec, picked: u8) {
        match spec.reward {
            ScrapReward::DrawEach if picked > 0 => self.stack.push(Job::Draw(picked)),
            ScrapReward::CombatEach(n) if picked > 0 => self.stack.push(Job::Combat(n as i32 * picked as i32)),
            _ => {}
        }
    }

    fn finish_discard(&mut self, reward: DiscardReward, picked: u8) {
        if picked == 0 {
            return;
        }
        match reward {
            DiscardReward::None => {}
            DiscardReward::DrawEach => self.stack.push(Job::Draw(picked)),
            DiscardReward::CombatEach(n) => self.stack.push(Job::Combat(n as i32 * picked as i32)),
            DiscardReward::TradeOrCombatEach(n) => self.stack.push(Job::TradeOrCombat(n, picked)),
        }
    }

    // ------------------------------------------------------------ effects

    fn push_effects(&mut self, list: &'static [Effect], src: Src) {
        for e in list.iter().rev() {
            self.stack.push(Job::Eff(e, src));
        }
    }

    fn put_in_play(&mut self, c: CardId) {
        let a = self.a();
        let def = card(c);
        let uid = self.next_uid;
        self.next_uid = self.next_uid.wrapping_add(1);
        let is_ship = !def.is_base();
        let mut entry = InPlay { card: c, copy: None, uid, used: 0, unlocked: 0 };
        if is_ship {
            entry.used |= F_PRIMARY;
        }
        self.players[a].in_play.push(entry);
        self.t.played_factions |= def.faction.bit();
        if let Some(i) = def.faction.index() {
            self.t.played_count[i] += 1;
        }
        if is_ship {
            // Triggers resolve after the ship's own effects (pushed first, popped last).
            let mut bonus = 0;
            for e in &self.players[a].in_play {
                if e.is_base() {
                    if let Some(tr) = e.def().on_ship_played {
                        if tr.faction.matches(def.faction) {
                            bonus += tr.combat as i32;
                        }
                    }
                }
            }
            if bonus > 0 {
                self.stack.push(Job::Combat(bonus));
            }
            self.push_effects(&def.primary, Src { uid, card: c });
        }
    }

    fn activate(&mut self, j: usize, flag: u8) {
        let a = self.a();
        let e = &mut self.players[a].in_play[j];
        e.used |= flag;
        let def = e.def();
        let src = Src { uid: e.uid, card: e.eff() };
        let list = match flag {
            F_PRIMARY => &def.primary,
            F_ALLY => &def.ally,
            _ => &def.ally2,
        };
        self.push_effects(list, src);
    }

    fn scrap_candidates(&self, z: Zone) -> bool {
        let me = self.me();
        match z {
            Zone::Hand => !me.hand.is_empty(),
            Zone::Discard => !me.discard.is_empty(),
            Zone::HandOrDiscard => !me.hand.is_empty() || !me.discard.is_empty(),
            Zone::TradeRow => self.trade_row.iter().any(Option::is_some),
        }
    }

    fn exec(&mut self, job: Job) {
        let (a, o) = (self.a(), self.o());
        let (e, src) = match job {
            Job::Draw(n) => return self.draw(a, n as usize),
            Job::Combat(n) => {
                self.t.combat += n;
                return;
            }
            Job::TradeOrCombat(amount, count) => {
                self.pending = Some(Pending::TradeOrCombat { amount, remaining: count });
                return;
            }
            Job::Eff(e, src) => (e, src),
        };
        match e {
            Effect::Trade(n) => self.t.trade += *n as i32,
            Effect::Combat(n) => self.t.combat += *n as i32,
            Effect::Authority(n) => self.players[a].authority += *n as i32,
            Effect::Draw(n) => self.draw(a, *n as usize),
            Effect::OppDiscard(n) => {
                let p = &mut self.players[o];
                p.owed_discards = p.owed_discards.saturating_add(*n);
            }
            Effect::Choose(options) => self.pending = Some(Pending::Choose { options, src }),
            Effect::Scrap(spec) => {
                if self.scrap_candidates(spec.from) {
                    self.pending = Some(Pending::Scrap { spec, picked: 0, src });
                }
            }
            Effect::Discard(spec) => {
                if !self.players[a].hand.is_empty() {
                    self.pending =
                        Some(Pending::Discard { min: spec.min, max: spec.max, picked: 0, reward: spec.reward, forced: false });
                }
            }
            Effect::DestroyBase => {
                if self.players[o].num_bases() > 0 {
                    self.pending = Some(Pending::DestroyBase { src });
                }
            }
            Effect::Acquire(spec) => {
                let mut any = false;
                self.acquire_candidates(spec, |_| any = true);
                if any {
                    self.pending = Some(Pending::Acquire { spec, src });
                }
            }
            Effect::NextAcquire(spec) => self.t.next_acquire.push(*spec),
            Effect::When(w) => {
                let ok = match w.cond {
                    Cond::OppHasBase => self.players[o].num_bases() > 0,
                    Cond::OwnBasesAtLeast(n) => self.players[a].num_bases() >= n as usize,
                };
                if ok {
                    self.push_effects(&w.then, src);
                }
            }
            Effect::DrawPerPlayed(f) => {
                let n = f.index().map_or(0, |i| self.t.played_count[i]);
                self.draw(a, n as usize);
            }
            Effect::CombatPerScrappedOwn(n) => self.t.combat += *n as i32 * self.t.scrapped_own as i32,
            Effect::CombatPerScrapped(n) => self.t.combat += *n as i32 * self.t.scrapped_hd as i32,
            Effect::TopdeckBase => {
                if self.players[a].discard.iter().any(|&c| card(c).is_base()) {
                    self.pending = Some(Pending::TopdeckBase { src });
                }
            }
            Effect::CopyShip => {
                let mut any = false;
                self.copy_ship_targets(src, |_| any = true);
                if any {
                    self.pending = Some(Pending::CopyShip { src });
                }
            }
            Effect::CopyBase => {
                let (mut own, mut opp) = (false, false);
                self.copy_base_targets(src, |_| own = true, |_| opp = true);
                if own || opp {
                    self.pending = Some(Pending::CopyBase { src });
                }
            }
        }
    }

    /// Run queued effects until a decision is needed or nothing is left.
    pub(crate) fn settle(&mut self) {
        loop {
            if self.outcome.is_some() {
                self.stack.clear();
                self.pending = None;
                return;
            }
            if self.pending.is_some() {
                return;
            }
            if let Some(job) = self.stack.pop() {
                self.exec(job);
                continue;
            }
            self.update_unlocks();
            if self.auto_pure && self.fire_one_pure() {
                continue;
            }
            return;
        }
    }

    /// Ally abilities, once triggered, stay usable for the turn even if the ally leaves play.
    fn update_unlocks(&mut self) {
        let a = self.a();
        let ip = &self.players[a].in_play;
        let n = ip.len();
        let mut unlock = [0u8; 64];
        for i in 0..n.min(64) {
            let d = ip[i].def();
            if d.ally.is_empty() && d.ally2.is_empty() {
                continue;
            }
            let f = d.faction.bit();
            let mut count = 0;
            for (k, other) in ip.iter().enumerate() {
                if k != i && (other.faction_bits() & f != 0 || other.def().ally_all) {
                    count += 1;
                }
            }
            if count >= 1 {
                unlock[i] |= F_ALLY;
            }
            if count >= 2 {
                unlock[i] |= F_ALLY2;
            }
        }
        for (i, e) in self.players[a].in_play.iter_mut().enumerate().take(64) {
            e.unlocked |= unlock[i];
        }
    }

    fn fire_one_pure(&mut self) -> bool {
        let a = self.a();
        for j in 0..self.players[a].in_play.len() {
            let e = &self.players[a].in_play[j];
            let d = e.def();
            let flag = if e.primary_available() && d.pure_primary {
                F_PRIMARY
            } else if e.ally_available() && d.pure_ally {
                F_ALLY
            } else if e.ally2_available() && d.pure_ally2 {
                F_ALLY2
            } else {
                continue;
            };
            self.activate(j, flag);
            return true;
        }
        false
    }

    // ------------------------------------------------------------ zones

    pub(crate) fn draw(&mut self, p: usize, n: usize) {
        for _ in 0..n {
            let pl = &mut self.players[p];
            if pl.deck.is_empty() {
                if pl.discard.is_empty() {
                    return;
                }
                std::mem::swap(&mut pl.deck, &mut pl.discard);
                self.rng.shuffle(&mut self.players[p].deck);
            }
            let pl = &mut self.players[p];
            let c = pl.deck.pop().unwrap();
            pl.hand.push(c);
        }
    }

    fn refill_row(&mut self, k: usize) {
        if self.trade_row[k].is_none() {
            self.trade_row[k] = self.trade_deck.pop();
        }
    }

    fn to_scrap(&mut self, c: CardId) {
        if c == db().explorer {
            self.explorers += 1;
        } else {
            self.scrap_heap.push(c);
        }
    }

    /// Put an acquired card where it belongs. `dest` is the effect's explicit
    /// destination; `Dest::Discard` means "default", which next-acquire
    /// effects may redirect.
    fn gain_card(&mut self, c: CardId, dest: Dest) {
        let a = self.a();
        let def = card(c);
        if def.acquire_to_hand && self.t.played_factions & def.faction.bit() != 0 {
            self.players[a].hand.push(c);
            return;
        }
        let mut dest = dest;
        if dest == Dest::Discard {
            if let Some(i) = self.t.next_acquire.iter().position(|s| s.filter.matches(def.kind)) {
                let spec = self.t.next_acquire.remove(i);
                if spec.may && spec.dest == Dest::TopDeck {
                    self.pending = Some(Pending::TopdeckChoice { card: c });
                    return;
                }
                dest = spec.dest;
            }
        }
        let p = &mut self.players[a];
        match dest {
            Dest::Discard => p.discard.push(c),
            Dest::TopDeck => p.deck.push(c),
            Dest::Hand => p.hand.push(c),
        }
    }

    fn destroy_opp_slot(&mut self, slot: usize) {
        let o = self.o();
        let e = self.players[o].in_play.remove(slot);
        self.players[o].discard.push(e.card);
    }

    fn damage_opp(&mut self, dmg: i32) {
        let o = self.o();
        self.players[o].authority -= dmg;
        if self.players[o].authority <= 0 {
            self.outcome = Some(Outcome::Win(self.active));
        }
    }

    fn end_turn(&mut self) {
        let dmg = self.end_turn_damage();
        if dmg > 0 {
            self.t.combat = 0;
            self.damage_opp(dmg);
            if self.outcome.is_some() {
                return;
            }
        }
        let a = self.a();
        let p = &mut self.players[a];
        let mut kept = Vec::with_capacity(p.in_play.len());
        for mut e in p.in_play.drain(..) {
            if e.is_base() {
                e.used = 0;
                e.unlocked = 0;
                e.copy = None;
                kept.push(e);
            } else {
                p.discard.push(e.card);
            }
        }
        p.in_play = kept;
        p.discard.append(&mut p.hand);
        for c in std::mem::take(&mut self.t.returns) {
            if let Some(i) = self.scrap_heap.iter().rposition(|&x| x == c) {
                self.scrap_heap.remove(i);
                self.players[a].discard.push(c);
            }
        }
        self.t = Turn::default();
        self.stack.clear();
        self.draw(a, 5);

        if self.turn >= self.max_turns {
            self.outcome = Some(Outcome::Draw);
            return;
        }
        self.turn += 1;
        self.active = 1 - self.active;
        let n = self.a();
        let p = &mut self.players[n];
        let k = (p.owed_discards as usize).min(p.hand.len()) as u8;
        p.owed_discards = 0;
        if k > 0 {
            self.pending = Some(Pending::Discard { min: k, max: k, picked: 0, reward: DiscardReward::None, forced: true });
        }
    }

    // ------------------------------------------------------------ hashing

    /// Stable hash of the full state, used by golden-replay regression tests.
    pub fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut put = |x: u64| {
            for b in x.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        put(self.turn as u64);
        put(self.active as u64);
        put(self.explorers as u64);
        put(match self.outcome {
            None => 0,
            Some(Outcome::Draw) => 1,
            Some(Outcome::Win(s)) => 2 + s as u64,
        });
        put(self.t.trade as u64);
        put(self.t.combat as u64);
        for p in &self.players {
            put(p.authority as u64);
            put(p.owed_discards as u64);
            for zone in [&p.deck, &p.hand, &p.discard] {
                put(zone.len() as u64);
                for &c in zone.iter() {
                    put(c as u64);
                }
            }
            for e in &p.in_play {
                put(e.card as u64 | (e.copy.map_or(255, |c| c as u64) << 8) | ((e.used as u64) << 16));
            }
        }
        for s in &self.trade_row {
            put(s.map_or(255, |c| c as u64));
        }
        put(self.trade_deck.len() as u64);
        for &c in &self.trade_deck {
            put(c as u64);
        }
        for &c in &self.scrap_heap {
            put(c as u64);
        }
        h
    }

    /// Total number of physical cards in the game (conserved by every action).
    pub fn card_count(&self) -> usize {
        self.players.iter().map(Player::total_cards).sum::<usize>()
            + self.trade_deck.len()
            + self.trade_row.iter().filter(|s| s.is_some()).count()
            + self.scrap_heap.len()
            + self.explorers as usize
            + match self.pending {
                Some(Pending::TopdeckChoice { .. }) => 1,
                _ => 0,
            }
    }
}
