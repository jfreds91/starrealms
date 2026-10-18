//! Card schema and the static card database.
//!
//! This file is also compiled by `build.rs`, so it must not reference other
//! modules of the crate.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

pub const CARDS_TOML: &str = include_str!("../../../data/cards.toml");
pub const SETS_TOML: &str = include_str!("../../../data/sets.toml");

/// Number of distinct cards across the three sets. Fixed so the action and
/// observation spaces are constant; checked when the database loads.
pub const N_CARDS: usize = 137;

pub type CardId = u8;

#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Faction {
    Blob,
    MachineCult,
    StarEmpire,
    TradeFederation,
    Unaligned,
}

impl Faction {
    pub const ALL: [Faction; 4] = [Faction::Blob, Faction::MachineCult, Faction::StarEmpire, Faction::TradeFederation];

    pub fn bit(self) -> u8 {
        match self {
            Faction::Blob => 1,
            Faction::MachineCult => 2,
            Faction::StarEmpire => 4,
            Faction::TradeFederation => 8,
            Faction::Unaligned => 0,
        }
    }

    pub fn index(self) -> Option<usize> {
        match self {
            Faction::Blob => Some(0),
            Faction::MachineCult => Some(1),
            Faction::StarEmpire => Some(2),
            Faction::TradeFederation => Some(3),
            Faction::Unaligned => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Faction::Blob => "Blob",
            Faction::MachineCult => "Machine Cult",
            Faction::StarEmpire => "Star Empire",
            Faction::TradeFederation => "Trade Federation",
            Faction::Unaligned => "Unaligned",
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Ship,
    Base,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Zone {
    Hand,
    Discard,
    HandOrDiscard,
    TradeRow,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScrapReward {
    #[default]
    None,
    DrawEach,
    CombatCost,
    CombatEach(u8),
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum DiscardReward {
    #[default]
    None,
    DrawEach,
    CombatEach(u8),
    TradeOrCombatEach(u8),
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    #[default]
    Any,
    Ship,
    Base,
    ShipOrBase,
}

impl Filter {
    pub fn matches(self, kind: Kind) -> bool {
        match self {
            Filter::Any | Filter::ShipOrBase => true,
            Filter::Ship => kind == Kind::Ship,
            Filter::Base => kind == Kind::Base,
        }
    }
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum Dest {
    #[default]
    Discard,
    TopDeck,
    Hand,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Cond {
    OppHasBase,
    OwnBasesAtLeast(u8),
}

fn one() -> u8 {
    1
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(deny_unknown_fields)]
pub struct ScrapSpec {
    pub from: Zone,
    #[serde(default)]
    pub min: u8,
    #[serde(default = "one")]
    pub max: u8,
    #[serde(default)]
    pub reward: ScrapReward,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(deny_unknown_fields)]
pub struct DiscardSpec {
    #[serde(default)]
    pub min: u8,
    #[serde(default = "one")]
    pub max: u8,
    #[serde(default)]
    pub reward: DiscardReward,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(deny_unknown_fields)]
pub struct AcquireSpec {
    #[serde(default)]
    pub max_cost: Option<u8>,
    #[serde(default)]
    pub filter: Filter,
    #[serde(default)]
    pub dest: Dest,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(deny_unknown_fields)]
pub struct NextAcquireSpec {
    pub filter: Filter,
    pub dest: Dest,
    #[serde(default)]
    pub may: bool,
}

#[derive(Deserialize, Clone, PartialEq, Eq, Debug)]
#[serde(deny_unknown_fields)]
pub struct WhenSpec {
    pub cond: Cond,
    pub then: Vec<Effect>,
}

#[derive(Deserialize, Clone, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Trade(u8),
    Combat(u8),
    Authority(u8),
    Draw(u8),
    OppDiscard(u8),
    Choose(Vec<Vec<Effect>>),
    Scrap(ScrapSpec),
    Discard(DiscardSpec),
    DestroyBase,
    Acquire(AcquireSpec),
    NextAcquire(NextAcquireSpec),
    When(WhenSpec),
    DrawPerPlayed(Faction),
    CombatPerScrappedOwn(u8),
    CombatPerScrapped(u8),
    TopdeckBase,
    CopyShip,
    CopyBase,
}

impl Effect {
    /// Effects that never need a decision and are never worse to take now.
    pub fn is_pure(&self) -> bool {
        match self {
            Effect::Trade(_) | Effect::Combat(_) | Effect::Authority(_) | Effect::OppDiscard(_) => true,
            Effect::When(w) => w.then.iter().all(Effect::is_pure),
            _ => false,
        }
    }
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TriggerFaction {
    Any,
    Blob,
    MachineCult,
    StarEmpire,
    TradeFederation,
}

impl TriggerFaction {
    pub fn matches(self, f: Faction) -> bool {
        match self {
            TriggerFaction::Any => true,
            TriggerFaction::Blob => f == Faction::Blob,
            TriggerFaction::MachineCult => f == Faction::MachineCult,
            TriggerFaction::StarEmpire => f == Faction::StarEmpire,
            TriggerFaction::TradeFederation => f == Faction::TradeFederation,
        }
    }
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(deny_unknown_fields)]
pub struct ShipTrigger {
    pub faction: TriggerFaction,
    pub combat: u8,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct Card {
    pub id: String,
    pub name: String,
    pub set: String,
    pub faction: Faction,
    pub kind: Kind,
    pub cost: u8,
    #[serde(default)]
    pub defense: u8,
    #[serde(default)]
    pub outpost: bool,
    #[serde(default)]
    pub primary: Vec<Effect>,
    #[serde(default)]
    pub ally: Vec<Effect>,
    #[serde(default)]
    pub ally2: Vec<Effect>,
    #[serde(default)]
    pub scrap: Vec<Effect>,
    #[serde(default)]
    pub ally_all: bool,
    #[serde(default)]
    pub on_ship_played: Option<ShipTrigger>,
    #[serde(default)]
    pub acquire_to_hand: bool,
    #[serde(default)]
    pub returns_from_scrap: bool,

    #[serde(skip)]
    pub idx: CardId,
    #[serde(skip)]
    pub pure_primary: bool,
    #[serde(skip)]
    pub pure_ally: bool,
    #[serde(skip)]
    pub pure_ally2: bool,
}

impl Card {
    pub fn is_base(&self) -> bool {
        self.kind == Kind::Base
    }
}

#[derive(Deserialize, Clone, Debug)]
struct CardsFile {
    card: Vec<Card>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct SetEntry {
    pub card: String,
    pub count: u8,
}

#[derive(Deserialize, Clone, Debug)]
pub struct SetDef {
    pub name: String,
    pub trade_deck: Vec<SetEntry>,
}

#[derive(Debug)]
pub struct CardDb {
    pub cards: Vec<Card>,
    pub by_id: HashMap<String, CardId>,
    pub sets: BTreeMap<String, SetDef>,
    pub scout: CardId,
    pub viper: CardId,
    pub explorer: CardId,
}

impl CardDb {
    pub fn parse(cards_toml: &str, sets_toml: &str) -> Result<CardDb, String> {
        let file: CardsFile = toml::from_str(cards_toml).map_err(|e| format!("cards.toml: {e}"))?;
        let sets: BTreeMap<String, SetDef> = toml::from_str(sets_toml).map_err(|e| format!("sets.toml: {e}"))?;
        let mut cards = file.card;
        if cards.len() != N_CARDS {
            return Err(format!("expected {N_CARDS} cards, found {}", cards.len()));
        }
        let mut by_id = HashMap::new();
        for (i, c) in cards.iter_mut().enumerate() {
            c.idx = i as CardId;
            c.pure_primary = !c.primary.is_empty() && c.primary.iter().all(Effect::is_pure);
            c.pure_ally = !c.ally.is_empty() && c.ally.iter().all(Effect::is_pure);
            c.pure_ally2 = !c.ally2.is_empty() && c.ally2.iter().all(Effect::is_pure);
            if by_id.insert(c.id.clone(), i as CardId).is_some() {
                return Err(format!("duplicate card id {}", c.id));
            }
            if c.is_base() && c.defense == 0 {
                return Err(format!("{}: base without defense", c.id));
            }
            if (!c.ally.is_empty() || !c.ally2.is_empty()) && c.faction == Faction::Unaligned {
                return Err(format!("{}: unaligned card with ally ability", c.id));
            }
        }
        for (name, s) in &sets {
            for e in &s.trade_deck {
                if !by_id.contains_key(&e.card) {
                    return Err(format!("sets.toml [{name}]: unknown card {}", e.card));
                }
            }
        }
        let get = |id: &str| by_id.get(id).copied().ok_or_else(|| format!("missing basic card {id}"));
        let (scout, viper, explorer) = (get("scout")?, get("viper")?, get("explorer")?);
        Ok(CardDb { cards, by_id, sets, scout, viper, explorer })
    }

    pub fn get(&self, id: &str) -> Option<CardId> {
        self.by_id.get(id).copied()
    }
}

static DB: OnceLock<CardDb> = OnceLock::new();

pub fn db() -> &'static CardDb {
    DB.get_or_init(|| CardDb::parse(CARDS_TOML, SETS_TOML).expect("card data validated by build.rs"))
}

#[inline]
pub fn card(id: CardId) -> &'static Card {
    &db().cards[id as usize]
}
