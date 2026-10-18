//! Flat discrete action space.
//!
//! Every decision in the game is one `u16` from a fixed menu. Which entries are
//! legal is given by `Game::legal_actions`. Hand and discard cards are
//! addressed by card id (cards with the same id are interchangeable there);
//! cards in play and trade-row cards are addressed by slot.

use crate::cards::{CardId, N_CARDS};

pub const MAX_CHOICES: usize = 4;
pub const MAX_OWN_SLOTS: usize = 32;
pub const MAX_OPP_BASES: usize = 16;
pub const ROW: usize = 5;

pub const END: u16 = 0;
pub const SKIP: u16 = 1;
pub const CHOOSE: u16 = 2;
pub const PLAY: u16 = CHOOSE + MAX_CHOICES as u16;
pub const PRIMARY: u16 = PLAY + N_CARDS as u16;
pub const ALLY: u16 = PRIMARY + MAX_OWN_SLOTS as u16;
pub const ALLY2: u16 = ALLY + MAX_OWN_SLOTS as u16;
pub const SCRAP: u16 = ALLY2 + MAX_OWN_SLOTS as u16;
pub const BUY: u16 = SCRAP + MAX_OWN_SLOTS as u16;
pub const BUY_EXPLORER: u16 = BUY + ROW as u16;
pub const ATTACK_PLAYER: u16 = BUY_EXPLORER + 1;
pub const ATTACK_BASE: u16 = ATTACK_PLAYER + 1;
pub const PICK_HAND: u16 = ATTACK_BASE + MAX_OPP_BASES as u16;
pub const PICK_DISCARD: u16 = PICK_HAND + N_CARDS as u16;
pub const PICK_ROW: u16 = PICK_DISCARD + N_CARDS as u16;
pub const PICK_EXPLORER: u16 = PICK_ROW + ROW as u16;
pub const PICK_OPP_BASE: u16 = PICK_EXPLORER + 1;
pub const PICK_OWN: u16 = PICK_OPP_BASE + MAX_OPP_BASES as u16;
pub const N_ACTIONS: usize = (PICK_OWN + MAX_OWN_SLOTS as u16) as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    EndTurn,
    Skip,
    Choose(u8),
    Play(CardId),
    Primary(u8),
    Ally(u8),
    Ally2(u8),
    ScrapAbility(u8),
    Buy(u8),
    BuyExplorer,
    AttackPlayer,
    AttackBase(u8),
    PickHand(CardId),
    PickDiscard(CardId),
    PickRow(u8),
    PickExplorer,
    PickOppBase(u8),
    PickOwn(u8),
}

impl Action {
    pub fn encode(self) -> u16 {
        match self {
            Action::EndTurn => END,
            Action::Skip => SKIP,
            Action::Choose(i) => CHOOSE + i as u16,
            Action::Play(c) => PLAY + c as u16,
            Action::Primary(j) => PRIMARY + j as u16,
            Action::Ally(j) => ALLY + j as u16,
            Action::Ally2(j) => ALLY2 + j as u16,
            Action::ScrapAbility(j) => SCRAP + j as u16,
            Action::Buy(k) => BUY + k as u16,
            Action::BuyExplorer => BUY_EXPLORER,
            Action::AttackPlayer => ATTACK_PLAYER,
            Action::AttackBase(b) => ATTACK_BASE + b as u16,
            Action::PickHand(c) => PICK_HAND + c as u16,
            Action::PickDiscard(c) => PICK_DISCARD + c as u16,
            Action::PickRow(k) => PICK_ROW + k as u16,
            Action::PickExplorer => PICK_EXPLORER,
            Action::PickOppBase(b) => PICK_OPP_BASE + b as u16,
            Action::PickOwn(j) => PICK_OWN + j as u16,
        }
    }

    pub fn decode(a: u16) -> Option<Action> {
        let r = |base: u16| (a - base) as u8;
        Some(match a {
            END => Action::EndTurn,
            SKIP => Action::Skip,
            _ if a < PLAY => Action::Choose(r(CHOOSE)),
            _ if a < PRIMARY => Action::Play(r(PLAY)),
            _ if a < ALLY => Action::Primary(r(PRIMARY)),
            _ if a < ALLY2 => Action::Ally(r(ALLY)),
            _ if a < SCRAP => Action::Ally2(r(ALLY2)),
            _ if a < BUY => Action::ScrapAbility(r(SCRAP)),
            _ if a < BUY_EXPLORER => Action::Buy(r(BUY)),
            BUY_EXPLORER => Action::BuyExplorer,
            ATTACK_PLAYER => Action::AttackPlayer,
            _ if a < PICK_HAND => Action::AttackBase(r(ATTACK_BASE)),
            _ if a < PICK_DISCARD => Action::PickHand(r(PICK_HAND)),
            _ if a < PICK_ROW => Action::PickDiscard(r(PICK_DISCARD)),
            _ if a < PICK_EXPLORER => Action::PickRow(r(PICK_ROW)),
            PICK_EXPLORER => Action::PickExplorer,
            _ if a < PICK_OWN => Action::PickOppBase(r(PICK_OPP_BASE)),
            _ if (a as usize) < N_ACTIONS => Action::PickOwn(r(PICK_OWN)),
            _ => return None,
        })
    }

    /// Coarse group name, used by the UI and for logging action statistics.
    pub fn group(self) -> &'static str {
        match self {
            Action::EndTurn => "end",
            Action::Skip => "skip",
            Action::Choose(_) => "choose",
            Action::Play(_) => "play",
            Action::Primary(_) | Action::Ally(_) | Action::Ally2(_) => "ability",
            Action::ScrapAbility(_) => "scrap_ability",
            Action::Buy(_) | Action::BuyExplorer => "buy",
            Action::AttackPlayer | Action::AttackBase(_) => "attack",
            _ => "pick",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_every_action() {
        for a in 0..N_ACTIONS as u16 {
            let act = Action::decode(a).unwrap();
            assert_eq!(act.encode(), a, "{act:?}");
        }
        assert!(Action::decode(N_ACTIONS as u16).is_none());
    }
}
