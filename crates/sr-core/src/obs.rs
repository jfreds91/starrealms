//! Fixed-size observation from the perspective of the player to move.
//!
//! Layout (see `LAYOUT`):
//! - `scalars`: normalized counters and flags
//! - `pending`: one-hot of the pending decision kind (all zero in the main phase)
//! - `counts`: per-card-id counts for each zone in `COUNT_ZONES`
//! - `own_slots` / `opp_bases` / `row`: per-slot flags aligned with the action
//!   slots of the same name
//! - `ids`: card ids (`id + 1`, `0` = empty) for the slot segments and the
//!   pending decision's source card, meant for an embedding lookup
//!
//! Hidden information is never encoded: the opponent's hand and deck appear
//! only as their combined multiset (`opp_unseen`), and draw-pile order is never
//! exposed. The trade deck appears as its remaining multiset, which any player
//! can compute from the visible history.

use crate::action::{MAX_OPP_BASES, MAX_OWN_SLOTS, ROW};
use crate::cards::*;
use crate::game::{Game, Pending};

pub const N_SCALARS: usize = 32;
pub const N_PENDING: usize = crate::game::PendingKind::COUNT + 1;
pub const COUNT_ZONES: [&str; 9] = [
    "my_hand",
    "my_deck",
    "my_discard",
    "my_in_play",
    "opp_unseen",
    "opp_discard",
    "opp_bases",
    "trade_row",
    "trade_deck",
];
pub const N_COUNTS: usize = COUNT_ZONES.len() * N_CARDS;
pub const OWN_SLOT_FEATS: usize = 6;
pub const OPP_BASE_FEATS: usize = 4;
pub const ROW_FEATS: usize = 3;
/// own slot card, own slot copy, opponent base card, trade row card, pending source.
pub const N_IDS: usize = MAX_OWN_SLOTS * 2 + MAX_OPP_BASES + ROW + 1;

pub const OFF_SCALARS: usize = 0;
pub const OFF_PENDING: usize = OFF_SCALARS + N_SCALARS;
pub const OFF_COUNTS: usize = OFF_PENDING + N_PENDING;
pub const OFF_OWN: usize = OFF_COUNTS + N_COUNTS;
pub const OFF_OPP: usize = OFF_OWN + MAX_OWN_SLOTS * OWN_SLOT_FEATS;
pub const OFF_ROW: usize = OFF_OPP + MAX_OPP_BASES * OPP_BASE_FEATS;
pub const OFF_IDS: usize = OFF_ROW + ROW * ROW_FEATS;
pub const OBS_DIM: usize = OFF_IDS + N_IDS;

/// (name, offset, length) of each segment.
pub const LAYOUT: [(&str, usize, usize); 7] = [
    ("scalars", OFF_SCALARS, N_SCALARS),
    ("pending", OFF_PENDING, N_PENDING),
    ("counts", OFF_COUNTS, N_COUNTS),
    ("own_slots", OFF_OWN, MAX_OWN_SLOTS * OWN_SLOT_FEATS),
    ("opp_bases", OFF_OPP, MAX_OPP_BASES * OPP_BASE_FEATS),
    ("row", OFF_ROW, ROW * ROW_FEATS),
    ("ids", OFF_IDS, N_IDS),
];

impl Game {
    /// Write the observation for the player to move into `out` (length `OBS_DIM`).
    pub fn observe(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), OBS_DIM);
        out.fill(0.0);
        let me = self.me();
        let opp = self.opp();
        let t = &self.t;

        let s = &mut out[OFF_SCALARS..OFF_SCALARS + N_SCALARS];
        s[0] = me.authority as f32 / 50.0;
        s[1] = opp.authority as f32 / 50.0;
        s[2] = t.trade as f32 / 10.0;
        s[3] = t.combat as f32 / 10.0;
        s[4] = self.turn as f32 / 100.0;
        s[5] = me.deck.len() as f32 / 20.0;
        s[6] = me.discard.len() as f32 / 20.0;
        s[7] = me.hand.len() as f32 / 10.0;
        s[8] = opp.hand.len() as f32 / 10.0;
        s[9] = opp.deck.len() as f32 / 20.0;
        s[10] = opp.discard.len() as f32 / 20.0;
        s[11] = self.explorers as f32 / 10.0;
        s[12] = self.trade_deck.len() as f32 / 80.0;
        s[13] = opp.owed_discards as f32 / 3.0;
        for (i, f) in Faction::ALL.iter().enumerate() {
            s[14 + i] = (t.played_factions & f.bit() != 0) as u8 as f32;
        }
        s[18] = opp.has_outpost() as u8 as f32;
        s[19] = me.num_bases() as f32 / 5.0;
        s[20] = opp.num_bases() as f32 / 5.0;
        s[21] = t.scrapped_own as f32 / 5.0;
        s[22] = t.scrapped_hd as f32 / 5.0;
        s[23] = t.next_acquire.len() as f32 / 2.0;
        s[24] = (self.active == 0) as u8 as f32;
        if let Some(Pending::Scrap { spec, picked, .. }) = self.pending {
            s[25] = picked as f32 / 3.0;
            s[26] = spec.min as f32 / 3.0;
            s[27] = spec.max as f32 / 3.0;
        }
        if let Some(Pending::Discard { min, max, picked, .. }) = self.pending {
            s[25] = picked as f32 / 3.0;
            s[26] = min as f32 / 3.0;
            s[27] = (max.min(12)) as f32 / 3.0;
        }
        if let Some(Pending::TradeOrCombat { amount, remaining }) = self.pending {
            s[25] = remaining as f32 / 3.0;
            s[26] = amount as f32 / 3.0;
        }
        s[28] = self.end_turn_damage() as f32 / 10.0;
        s[29] = (me.authority - opp.authority) as f32 / 50.0;

        let pk = self.pending.map_or(0, |p| p.kind().index() + 1);
        out[OFF_PENDING + pk] = 1.0;

        let counts = &mut out[OFF_COUNTS..OFF_COUNTS + N_COUNTS];
        let mut add = |zone: usize, c: CardId| counts[zone * N_CARDS + c as usize] += 1.0;
        for &c in &me.hand {
            add(0, c);
        }
        for &c in &me.deck {
            add(1, c);
        }
        for &c in &me.discard {
            add(2, c);
        }
        for e in &me.in_play {
            add(3, e.card);
        }
        for &c in opp.hand.iter().chain(&opp.deck) {
            add(4, c);
        }
        for &c in &opp.discard {
            add(5, c);
        }
        for e in opp.bases() {
            add(6, e.card);
        }
        for c in self.trade_row.iter().flatten() {
            add(7, *c);
        }
        for &c in &self.trade_deck {
            add(8, c);
        }

        let outposts = opp.has_outpost();
        for (j, e) in me.in_play.iter().enumerate().take(MAX_OWN_SLOTS) {
            let f = &mut out[OFF_OWN + j * OWN_SLOT_FEATS..OFF_OWN + (j + 1) * OWN_SLOT_FEATS];
            f[0] = e.is_base() as u8 as f32;
            f[1] = e.primary_available() as u8 as f32;
            f[2] = e.ally_available() as u8 as f32;
            f[3] = e.ally2_available() as u8 as f32;
            f[4] = e.scrap_available() as u8 as f32;
            f[5] = e.outpost() as u8 as f32;
            out[OFF_IDS + j] = e.card as f32 + 1.0;
            if let Some(c) = e.copy {
                out[OFF_IDS + MAX_OWN_SLOTS + j] = c as f32 + 1.0;
            }
        }
        for (b, e) in opp.bases().enumerate().take(MAX_OPP_BASES) {
            let f = &mut out[OFF_OPP + b * OPP_BASE_FEATS..OFF_OPP + (b + 1) * OPP_BASE_FEATS];
            f[0] = e.defense() as f32 / 8.0;
            f[1] = e.outpost() as u8 as f32;
            f[2] = (e.outpost() || !outposts) as u8 as f32;
            f[3] = (t.combat >= e.defense() as i32 && (e.outpost() || !outposts)) as u8 as f32;
            out[OFF_IDS + 2 * MAX_OWN_SLOTS + b] = e.card as f32 + 1.0;
        }
        for (k, slot) in self.trade_row.iter().enumerate() {
            if let Some(c) = slot {
                let f = &mut out[OFF_ROW + k * ROW_FEATS..OFF_ROW + (k + 1) * ROW_FEATS];
                let cost = card(*c).cost as i32;
                f[0] = cost as f32 / 8.0;
                f[1] = (t.trade >= cost) as u8 as f32;
                f[2] = card(*c).is_base() as u8 as f32;
                out[OFF_IDS + 2 * MAX_OWN_SLOTS + MAX_OPP_BASES + k] = *c as f32 + 1.0;
            }
        }
        if let Some(src) = self.pending.and_then(|p| p.source()) {
            out[OFF_IDS + N_IDS - 1] = src as f32 + 1.0;
        }
    }

    pub fn observation(&self) -> Vec<f32> {
        let mut v = vec![0.0; OBS_DIM];
        self.observe(&mut v);
        v
    }
}
