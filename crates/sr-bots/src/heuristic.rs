//! Hand-written strategy: draw first, play everything, thin the deck early,
//! buy for value and faction synergy, clear outposts, race when lethal.

use crate::Agent;
use sr_core::cards::{Cond, DiscardReward, Effect, ScrapReward, Zone};
use sr_core::{action, card, db, Action, CardId, Game, Pending, Rng};

#[derive(Clone, Debug)]
pub struct Heuristic {
    /// Minimum value for a trade-row card to be worth buying.
    pub buy_threshold: f32,
}

impl Default for Heuristic {
    fn default() -> Self {
        Heuristic { buy_threshold: 2.0 }
    }
}

struct Ctx<'a> {
    g: &'a Game,
    /// Own cards per faction across all own zones.
    factions: [u32; 4],
    /// Factions the opponent has shown (discard pile and bases).
    opp_factions: [u32; 4],
    my_cards: usize,
    early: bool,
    late: bool,
}

impl<'a> Ctx<'a> {
    fn new(g: &'a Game) -> Ctx<'a> {
        let me = g.me();
        let mut factions = [0; 4];
        let count = |c: CardId, f: &mut [u32; 4]| {
            if let Some(i) = card(c).faction.index() {
                f[i] += 1;
            }
        };
        let mut my_cards = 0;
        for &c in me.deck.iter().chain(&me.hand).chain(&me.discard) {
            count(c, &mut factions);
            my_cards += 1;
        }
        for e in &me.in_play {
            count(e.card, &mut factions);
            my_cards += 1;
        }
        let mut opp_factions = [0; 4];
        let opp = g.opp();
        for &c in &opp.discard {
            count(c, &mut opp_factions);
        }
        for e in &opp.in_play {
            count(e.card, &mut opp_factions);
        }
        Ctx { g, factions, opp_factions, my_cards, early: g.turn < 14, late: g.turn > 26 }
    }

    fn junk(&self, c: CardId) -> bool {
        c == db().scout || c == db().viper
    }

    fn lethal_open(&self) -> bool {
        !self.g.opp().has_outpost()
    }

    /// Value of owning one more copy of `c`.
    fn card_value(&self, c: CardId) -> f32 {
        let d = card(c);
        let d0 = db();
        if c == d0.scout {
            return if self.early { 0.5 } else { 0.3 };
        }
        if c == d0.viper {
            return if self.early { 0.3 } else { 0.5 };
        }
        if c == d0.explorer {
            return if self.g.turn < 12 { 1.3 } else { 0.2 };
        }
        let mut v = d.cost as f32;
        if let Some(i) = d.faction.index() {
            v += 0.5 * (self.factions[i].min(4) as f32);
        }
        let all: Vec<&Effect> = d.primary.iter().chain(&d.ally).chain(&d.ally2).chain(&d.scrap).collect();
        let any = |p: &dyn Fn(&Effect) -> bool| all.iter().any(|e| p(e));
        if any(&|e| matches!(e, Effect::Draw(_) | Effect::DrawPerPlayed(_))) {
            v += 0.5;
        }
        if self.early
            && any(&|e| matches!(e, Effect::Scrap(s) if matches!(s.from, Zone::Hand | Zone::Discard | Zone::HandOrDiscard)))
        {
            v += 1.0;
        }
        if any(&|e| matches!(e, Effect::OppDiscard(_))) {
            v += 0.3;
        }
        if d.is_base() {
            v += 0.3;
            if d.outpost {
                v += 0.4;
            }
        }
        if self.g.me().authority < 25 && any(&|e| matches!(e, Effect::Authority(n) if *n >= 3)) {
            v += 0.8;
        }
        if self.late && d.cost <= 2 {
            v -= 1.0;
        }
        v
    }

    /// Value of denying `c` to the opponent (for trade-row scrapping).
    fn deny_value(&self, c: CardId) -> f32 {
        let d = card(c);
        let syn = d.faction.index().map_or(0, |i| self.opp_factions[i].min(4));
        d.cost as f32 + 0.7 * syn as f32
    }

    fn effects_score(&self, list: &[Effect]) -> f32 {
        let g = self.g;
        let (wt, wc) = if self.early { (1.2, 0.8) } else if self.late { (0.7, 1.4) } else { (1.0, 1.1) };
        let wa = if g.me().authority < 20 { 0.8 } else { 0.25 };
        let mut s = 0.0;
        for e in list {
            s += match e {
                Effect::Trade(n) => *n as f32 * wt,
                Effect::Combat(n) => {
                    if self.lethal_open() && g.t.combat + *n as i32 >= g.opp().authority {
                        100.0
                    } else {
                        *n as f32 * wc
                    }
                }
                Effect::Authority(n) => *n as f32 * wa,
                Effect::Draw(n) => *n as f32 * 2.0,
                Effect::OppDiscard(n) => *n as f32,
                Effect::Scrap(_) => 1.0,
                Effect::Discard(_) => 0.8,
                Effect::DestroyBase => {
                    if g.opp().num_bases() > 0 {
                        3.5
                    } else {
                        0.0
                    }
                }
                Effect::Acquire(_) => 4.0,
                Effect::NextAcquire(_) => 0.8,
                Effect::When(w) => {
                    let ok = match w.cond {
                        Cond::OppHasBase => g.opp().num_bases() > 0,
                        Cond::OwnBasesAtLeast(n) => g.me().num_bases() >= n as usize,
                    };
                    if ok {
                        self.effects_score(&w.then)
                    } else {
                        0.0
                    }
                }
                Effect::DrawPerPlayed(f) => 2.0 * f.index().map_or(0, |i| g.t.played_count[i]) as f32,
                Effect::TopdeckBase => {
                    if g.me().discard.iter().any(|&c| card(c).is_base()) {
                        2.5
                    } else {
                        0.0
                    }
                }
                Effect::CombatPerScrappedOwn(n) => *n as f32 * g.t.scrapped_own as f32 * wc,
                Effect::CombatPerScrapped(n) => *n as f32 * g.t.scrapped_hd as f32 * wc,
                Effect::Choose(opts) => opts.iter().map(|o| self.effects_score(o)).fold(0.0, f32::max),
                Effect::CopyShip | Effect::CopyBase => 3.0,
            };
        }
        s
    }

    /// Whether every mandatory scrap in `list` has a junk card to hit.
    fn mandatory_scraps_ok(&self, list: &[Effect]) -> bool {
        let me = self.g.me();
        list.iter().all(|e| match e {
            Effect::Scrap(s) if s.min > 0 => {
                let hand = me.hand.iter().any(|&c| self.junk(c));
                let disc = me.discard.iter().any(|&c| self.junk(c));
                match s.from {
                    Zone::Hand => hand,
                    Zone::Discard => disc,
                    Zone::HandOrDiscard => hand || disc,
                    Zone::TradeRow => true,
                }
            }
            _ => true,
        })
    }
}

fn draws(list: &[Effect]) -> bool {
    list.iter().any(|e| match e {
        Effect::Draw(_) | Effect::DrawPerPlayed(_) => true,
        Effect::When(w) => draws(&w.then),
        _ => false,
    })
}

impl Agent for Heuristic {
    fn name(&self) -> &str {
        "heuristic"
    }

    fn act(&mut self, g: &Game, legal: &[u16], _rng: &mut Rng) -> u16 {
        let ctx = Ctx::new(g);
        let a = match g.pending {
            Some(p) => self.pending(&ctx, p, legal),
            None => self.main(&ctx, legal),
        };
        if legal.contains(&a) {
            a
        } else {
            legal[0]
        }
    }
}

impl Heuristic {
    fn main(&self, ctx: &Ctx, legal: &[u16]) -> u16 {
        let g = ctx.g;
        let me = g.me();
        let opp = g.opp();
        let acts: Vec<(u16, Action)> = legal.iter().map(|&a| (a, Action::decode(a).unwrap())).collect();
        let find = |pred: &dyn Fn(&Action) -> bool| acts.iter().find(|(_, x)| pred(x)).map(|(a, _)| *a);

        // Lethal.
        if ctx.lethal_open() && g.t.combat >= opp.authority {
            if let Some(a) = find(&|x| matches!(x, Action::AttackPlayer)) {
                return a;
            }
        }
        let needle = db().get("stealthNeedle").unwrap();
        // Cards and abilities that draw come first: more information for every later choice.
        if let Some(a) = find(&|x| matches!(x, Action::Play(c) if *c != needle && draws(&card(*c).primary))) {
            return a;
        }
        let ability_list = |x: &Action| -> Option<&'static [Effect]> {
            match x {
                Action::Primary(j) => Some(&me.in_play[*j as usize].def().primary),
                Action::Ally(j) => Some(&me.in_play[*j as usize].def().ally),
                Action::Ally2(j) => Some(&me.in_play[*j as usize].def().ally2),
                _ => None,
            }
        };
        if let Some(a) = find(&|x| ability_list(x).is_some_and(|l| draws(l) && ctx.mandatory_scraps_ok(l))) {
            return a;
        }
        if let Some(a) = find(&|x| matches!(x, Action::Play(c) if *c != needle)) {
            return a;
        }
        if let Some(a) = find(&|x| matches!(x, Action::Play(_))) {
            return a;
        }
        if let Some(a) = find(&|x| ability_list(x).is_some_and(|l| ctx.mandatory_scraps_ok(l))) {
            return a;
        }

        // Scrap abilities: Explorers once the economy is built, anything that gives lethal, Mobile Market.
        for (a, x) in &acts {
            if let Action::ScrapAbility(j) = x {
                let e = &me.in_play[*j as usize];
                let def = e.def();
                let combat: i32 = def.scrap.iter().map(|ef| if let Effect::Combat(n) = ef { *n as i32 } else { 0 }).sum();
                let lethal = ctx.lethal_open() && combat > 0 && g.t.combat + combat >= opp.authority;
                let explorer = e.card == db().explorer && g.turn >= 10;
                if lethal || explorer || card(e.card).returns_from_scrap {
                    return *a;
                }
            }
        }

        // Buying: best-value affordable subset of the row.
        if let Some(a) = self.choose_buy(ctx, &acts) {
            return a;
        }

        // Combat.
        if g.t.combat > 0 {
            let outposts = opp.has_outpost();
            let mut best: Option<(f32, u16)> = None;
            for (a, x) in &acts {
                if let Action::AttackBase(b) = x {
                    let slot = opp.base_slot(*b as usize).unwrap();
                    let e = &opp.in_play[slot];
                    let d = card(e.card);
                    let worth = if outposts { true } else { d.cost >= 4 || d.primary.iter().any(|ef| matches!(ef, Effect::Combat(_))) };
                    if worth {
                        let score = d.cost as f32 + if e.outpost() { 2.0 } else { 0.0 };
                        if best.map_or(true, |(s, _)| score > s) {
                            best = Some((score, *a));
                        }
                    }
                }
            }
            if let Some((_, a)) = best {
                return a;
            }
            if let Some(a) = find(&|x| matches!(x, Action::AttackPlayer)) {
                return a;
            }
        }
        action::END
    }

    fn choose_buy(&self, ctx: &Ctx, acts: &[(u16, Action)]) -> Option<u16> {
        let g = ctx.g;
        let trade = g.t.trade;
        let row: Vec<(usize, CardId, i32, f32)> = g
            .trade_row
            .iter()
            .enumerate()
            .filter_map(|(k, s)| s.map(|c| (k, c, card(c).cost as i32, ctx.card_value(c))))
            .filter(|&(_, _, cost, v)| cost <= trade && v >= self.buy_threshold)
            .collect();
        let mut best = (0.0f32, 0u32);
        for mask in 1u32..(1 << row.len()) {
            let (mut cost, mut val) = (0, 0.0);
            for (i, r) in row.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    cost += r.2;
                    val += r.3;
                }
            }
            if cost <= trade && val > best.0 {
                best = (val, mask);
            }
        }
        if best.1 != 0 {
            let pick = row.iter().enumerate().filter(|(i, _)| best.1 & (1 << i) != 0).max_by_key(|(_, r)| r.2).unwrap().1;
            let a = action::BUY + pick.0 as u16;
            if acts.iter().any(|(x, _)| *x == a) {
                return Some(a);
            }
        }
        if g.turn < 12 && acts.iter().any(|(a, _)| *a == action::BUY_EXPLORER) {
            return Some(action::BUY_EXPLORER);
        }
        None
    }

    fn pending(&self, ctx: &Ctx, p: Pending, legal: &[u16]) -> u16 {
        let g = ctx.g;
        let me = g.me();
        let opp = g.opp();
        let pick_max = |score: &dyn Fn(u16) -> Option<f32>| -> Option<u16> {
            legal
                .iter()
                .filter_map(|&a| score(a).map(|s| (s, a)))
                .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                .map(|(_, a)| a)
        };
        let hand_card = |a: u16| match Action::decode(a) {
            Some(Action::PickHand(c)) => Some(c),
            _ => None,
        };
        // Lower value = better to lose.
        let junkiness = |c: CardId| -ctx.card_value(c);

        match p {
            Pending::Choose { options, .. } => {
                let mut best = (f32::MIN, action::CHOOSE);
                for (i, o) in options.iter().enumerate() {
                    let s = ctx.effects_score(o);
                    if s > best.0 {
                        best = (s, action::CHOOSE + i as u16);
                    }
                }
                best.1
            }
            Pending::Scrap { spec, picked, .. } => {
                let mandatory = picked < spec.min;
                if spec.from == Zone::TradeRow {
                    let score = |a: u16| match Action::decode(a) {
                        Some(Action::PickRow(k)) => g.trade_row[k as usize].map(|c| {
                            if spec.reward == ScrapReward::CombatCost {
                                card(c).cost as f32
                            } else {
                                ctx.deny_value(c)
                            }
                        }),
                        _ => None,
                    };
                    let best = pick_max(&score);
                    let best_score = best.and_then(score).unwrap_or(0.0);
                    if mandatory || best_score >= 6.0 {
                        return best.unwrap_or(legal[0]);
                    }
                    return action::SKIP;
                }
                // Prefer the discard pile (the card is not usable this turn anyway), then hand.
                let score = |a: u16| match Action::decode(a) {
                    Some(Action::PickDiscard(c)) => Some(junkiness(c) + 0.05),
                    Some(Action::PickHand(c)) => Some(junkiness(c)),
                    _ => None,
                };
                let best = pick_max(&score);
                let bc = best.and_then(|a| match Action::decode(a) {
                    Some(Action::PickDiscard(c)) | Some(Action::PickHand(c)) => Some(c),
                    _ => None,
                });
                let thin_ok = ctx.my_cards > 9;
                match bc {
                    Some(c) if mandatory || (ctx.junk(c) && thin_ok) => best.unwrap(),
                    _ if mandatory => best.unwrap_or(legal[0]),
                    _ => action::SKIP,
                }
            }
            Pending::Discard { min, picked, reward, .. } => {
                let worst = pick_max(&|a| hand_card(a).map(junkiness));
                if picked < min {
                    return worst.unwrap_or(legal[0]);
                }
                let rewarded = reward != DiscardReward::None;
                match worst.and_then(hand_card) {
                    Some(c) if rewarded && ctx.junk(c) => worst.unwrap(),
                    _ => action::SKIP,
                }
            }
            Pending::DestroyBase { .. } => pick_max(&|a| match Action::decode(a) {
                Some(Action::PickOppBase(b)) => {
                    let e = &opp.in_play[opp.base_slot(b as usize)?];
                    Some(card(e.card).cost as f32 + if e.outpost() { 2.0 } else { 0.0 })
                }
                _ => None,
            })
            .unwrap_or(action::SKIP),
            Pending::Acquire { .. } => {
                let score = |a: u16| match Action::decode(a) {
                    Some(Action::PickRow(k)) => g.trade_row[k as usize].map(|c| ctx.card_value(c)),
                    Some(Action::PickExplorer) => Some(ctx.card_value(db().explorer)),
                    _ => None,
                };
                match pick_max(&score) {
                    Some(a) if score(a).unwrap_or(0.0) > 0.5 => a,
                    _ => action::SKIP,
                }
            }
            Pending::TopdeckBase { .. } => pick_max(&|a| match Action::decode(a) {
                Some(Action::PickDiscard(c)) => Some(card(c).cost as f32),
                _ => None,
            })
            .unwrap_or(legal[0]),
            Pending::CopyShip { .. } | Pending::CopyBase { .. } => pick_max(&|a| match Action::decode(a) {
                Some(Action::PickOwn(j)) => Some(card(me.in_play[j as usize].eff()).cost as f32),
                Some(Action::PickOppBase(b)) => {
                    let e = &opp.in_play[opp.base_slot(b as usize)?];
                    Some(card(e.eff()).cost as f32)
                }
                _ => None,
            })
            .unwrap_or(legal[0]),
            Pending::TopdeckChoice { .. } => action::CHOOSE,
            Pending::TradeOrCombat { amount, .. } => {
                let lethal = ctx.lethal_open() && g.t.combat + amount as i32 >= opp.authority;
                if lethal || !ctx.early {
                    action::CHOOSE + 1
                } else {
                    action::CHOOSE
                }
            }
        }
    }
}