//! Per-seat JSON views for user interfaces. A view contains only what that
//! seat is allowed to see, plus the legal actions if it is that seat's move.

use crate::action::{Action, MAX_OPP_BASES};
use crate::cards::*;
use crate::game::{Game, InPlay, LogEntry, Outcome, PendingKind, Player};
use crate::text::effects_text;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct SlotView {
    pub card: CardId,
    pub copy: Option<CardId>,
    pub is_base: bool,
    pub defense: u8,
    pub outpost: bool,
    pub primary: bool,
    pub ally: bool,
    pub ally2: bool,
    pub scrap: bool,
}

impl SlotView {
    fn new(e: &InPlay) -> SlotView {
        SlotView {
            card: e.card,
            copy: e.copy,
            is_base: e.is_base(),
            defense: e.defense(),
            outpost: e.outpost(),
            primary: e.primary_available(),
            ally: e.ally_available(),
            ally2: e.ally2_available(),
            scrap: e.scrap_available(),
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct SeatView {
    pub authority: i32,
    /// Only present for the viewing seat.
    pub hand: Option<Vec<CardId>>,
    pub hand_count: usize,
    pub deck_count: usize,
    /// Sorted contents of the draw pile (known to its owner, not its order).
    pub deck_contents: Option<Vec<CardId>>,
    pub discard: Vec<CardId>,
    pub in_play: Vec<SlotView>,
    pub owed_discards: u8,
}

impl SeatView {
    fn new(p: &Player, own: bool) -> SeatView {
        let mut deck = p.deck.clone();
        deck.sort_unstable();
        SeatView {
            authority: p.authority,
            hand: own.then(|| {
                let mut h = p.hand.clone();
                h.sort_unstable();
                h
            }),
            hand_count: p.hand.len(),
            deck_count: p.deck.len(),
            deck_contents: own.then_some(deck),
            discard: p.discard.clone(),
            in_play: p.in_play.iter().map(SlotView::new).collect(),
            owed_discards: p.owed_discards,
        }
    }
}

/// What a UI element an action belongs to.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "zone", rename_all = "snake_case")]
pub enum Target {
    None,
    Hand { card: CardId },
    Discard { card: CardId },
    Own { slot: u8 },
    Row { slot: u8 },
    Explorer,
    OppBase { slot: u8 },
    Opponent,
}

#[derive(Serialize, Clone, Debug)]
pub struct ActionView {
    pub id: u16,
    pub label: String,
    pub group: &'static str,
    pub target: Target,
}

#[derive(Serialize, Clone, Debug)]
pub struct View {
    pub seat: u8,
    pub active: u8,
    pub turn: u16,
    pub outcome: Option<Outcome>,
    pub you: SeatView,
    pub opponent: SeatView,
    pub trade_row: Vec<Option<CardId>>,
    pub explorers: u8,
    pub trade_deck: usize,
    pub scrap_heap: Vec<CardId>,
    pub trade: i32,
    pub combat: i32,
    pub prompt: Option<String>,
    pub pending: Option<PendingKind>,
    pub pending_source: Option<CardId>,
    pub actions: Vec<ActionView>,
    pub log: Vec<LogEntry>,
}

pub fn target_of(a: Action) -> Target {
    match a {
        Action::Play(c) | Action::PickHand(c) => Target::Hand { card: c },
        Action::PickDiscard(c) => Target::Discard { card: c },
        Action::Primary(j) | Action::Ally(j) | Action::Ally2(j) | Action::ScrapAbility(j) | Action::PickOwn(j) => {
            Target::Own { slot: j }
        }
        Action::Buy(k) | Action::PickRow(k) => Target::Row { slot: k },
        Action::BuyExplorer | Action::PickExplorer => Target::Explorer,
        Action::AttackBase(b) | Action::PickOppBase(b) => Target::OppBase { slot: b },
        Action::AttackPlayer => Target::Opponent,
        _ => Target::None,
    }
}

impl Game {
    pub fn view(&self, seat: u8, log_tail: usize) -> View {
        let s = seat as usize;
        let actions = if seat == self.active && self.outcome.is_none() {
            self.legal_vec()
                .into_iter()
                .map(|a| {
                    let act = Action::decode(a).unwrap();
                    ActionView { id: a, label: self.describe(a), group: act.group(), target: target_of(act) }
                })
                .collect()
        } else {
            Vec::new()
        };
        let mine = seat == self.active;
        let _ = MAX_OPP_BASES;
        View {
            seat,
            active: self.active,
            turn: self.turn,
            outcome: self.outcome,
            you: SeatView::new(&self.players[s], true),
            opponent: SeatView::new(&self.players[1 - s], false),
            trade_row: self.trade_row.to_vec(),
            explorers: self.explorers,
            trade_deck: self.trade_deck.len(),
            scrap_heap: self.scrap_heap.clone(),
            trade: self.t.trade,
            combat: self.t.combat,
            prompt: if mine { self.prompt() } else { None },
            pending: self.pending.map(|p| p.kind()),
            pending_source: self.pending.and_then(|p| p.source()),
            actions,
            log: self.log[self.log.len().saturating_sub(log_tail)..].to_vec(),
        }
    }

    /// Seat 0's view with nothing private to either seat: no hands, no deck contents, no actions.
    pub fn spectator_view(&self, log_tail: usize) -> View {
        let mut v = self.view(0, log_tail);
        v.you = SeatView::new(&self.players[0], false);
        v.prompt = None;
        v.actions.clear();
        v
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct CardInfo {
    pub idx: CardId,
    pub id: String,
    pub name: String,
    pub set: String,
    pub faction: Faction,
    pub kind: Kind,
    pub cost: u8,
    pub defense: u8,
    pub outpost: bool,
    pub primary: String,
    pub ally: String,
    pub ally2: String,
    pub scrap: String,
    pub image: String,
}

/// Card catalog for clients: names, stats, ability text, image path (relative to the assets root).
pub fn catalog() -> Vec<CardInfo> {
    db().cards
        .iter()
        .map(|c| CardInfo {
            idx: c.idx,
            id: c.id.clone(),
            name: c.name.clone(),
            set: c.set.clone(),
            faction: c.faction,
            kind: c.kind,
            cost: c.cost,
            defense: c.defense,
            outpost: c.outpost,
            primary: {
                let mut s = effects_text(&c.primary);
                if c.ally_all {
                    s = "Counts as an ally for all factions".into();
                }
                if let Some(t) = c.on_ship_played {
                    let who = match t.faction {
                        TriggerFaction::Any => "a ship".to_string(),
                        _ => "a Star Empire ship".to_string(),
                    };
                    let extra = format!("Whenever you play {who}, +{} Combat", t.combat);
                    s = if s.is_empty() { extra } else { format!("{s}. {extra}") };
                }
                if c.acquire_to_hand {
                    s = format!(
                        "{s}. When acquired, put into your hand if you've played a {} card this turn",
                        c.faction.name()
                    );
                }
                s
            },
            ally: effects_text(&c.ally),
            ally2: effects_text(&c.ally2),
            scrap: {
                let s = effects_text(&c.scrap);
                if c.returns_from_scrap {
                    format!("{s}. Returns to your discard pile at end of turn")
                } else {
                    s
                }
            },
            image: format!("cards/{}.webp", c.id),
        })
        .collect()
}
