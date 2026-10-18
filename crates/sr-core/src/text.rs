//! Human-readable descriptions of effects, actions, and pending decisions.

use crate::action::Action;
use crate::cards::*;
use crate::game::{Game, Pending};

fn zone_text(z: Zone) -> &'static str {
    match z {
        Zone::Hand => "your hand",
        Zone::Discard => "your discard pile",
        Zone::HandOrDiscard => "your hand or discard pile",
        Zone::TradeRow => "the trade row",
    }
}

fn filter_text(f: Filter) -> &'static str {
    match f {
        Filter::Any => "card",
        Filter::Ship => "ship",
        Filter::Base => "base",
        Filter::ShipOrBase => "ship or base",
    }
}

fn plural(n: u8, word: &str) -> String {
    if n == 1 {
        format!("a {word}")
    } else {
        format!("{n} {word}s")
    }
}

pub fn effect_text(e: &Effect) -> String {
    match e {
        Effect::Trade(n) => format!("+{n} Trade"),
        Effect::Combat(n) => format!("+{n} Combat"),
        Effect::Authority(n) => format!("+{n} Authority"),
        Effect::Draw(n) => format!("Draw {}", plural(*n, "card")),
        Effect::OppDiscard(n) => format!("Opponent discards {}", plural(*n, "card")),
        Effect::Choose(opts) => opts.iter().map(|o| effects_text(o)).collect::<Vec<_>>().join(" OR "),
        Effect::Scrap(s) => {
            let what = if s.max == 1 { "a card".to_string() } else { format!("up to {} cards", s.max) };
            let may = if s.min == 0 { "You may scrap" } else { "Scrap" };
            let reward = match s.reward {
                ScrapReward::None => String::new(),
                ScrapReward::DrawEach => ", draw a card for each".into(),
                ScrapReward::CombatCost => ", gain Combat equal to its cost".into(),
                ScrapReward::CombatEach(n) => format!(", +{n} Combat for each"),
            };
            format!("{may} {what} from {}{reward}", zone_text(s.from))
        }
        Effect::Discard(s) => {
            let what = match (s.min, s.max) {
                (1, 1) => "Discard a card".to_string(),
                (0, 1) => "You may discard a card".to_string(),
                (_, m) if m >= 12 => "Discard any number of cards".to_string(),
                (_, m) => format!("Discard up to {m} cards"),
            };
            let reward = match s.reward {
                DiscardReward::None => String::new(),
                DiscardReward::DrawEach => ", draw a card for each".into(),
                DiscardReward::CombatEach(n) => format!(", +{n} Combat for each"),
                DiscardReward::TradeOrCombatEach(n) => format!(", +{n} Trade or +{n} Combat for each"),
            };
            format!("{what}{reward}")
        }
        Effect::DestroyBase => "You may destroy target base".into(),
        Effect::Acquire(s) => {
            let cost = s.max_cost.map_or(String::new(), |m| format!(" of cost {m} or less"));
            let dest = match s.dest {
                Dest::Discard => "",
                Dest::TopDeck => " and put it on top of your deck",
                Dest::Hand => " and put it into your hand",
            };
            format!("Acquire a {}{cost} for free{dest}", filter_text(s.filter))
        }
        Effect::NextAcquire(s) => {
            let dest = match s.dest {
                Dest::Hand => "into your hand",
                _ => "on top of your deck",
            };
            let may = if s.may { "You may put" } else { "Put" };
            format!("{may} the next {} you acquire this turn {dest}", filter_text(s.filter))
        }
        Effect::When(w) => {
            let cond = match w.cond {
                Cond::OppHasBase => "If an opponent controls a base".to_string(),
                Cond::OwnBasesAtLeast(n) => format!("If you have {n} or more bases in play"),
            };
            format!("{cond}: {}", effects_text(&w.then))
        }
        Effect::DrawPerPlayed(f) => format!("Draw a card for each {} card you've played this turn", f.name()),
        Effect::CombatPerScrappedOwn(n) => format!("+{n} Combat for each of your cards scrapped this turn"),
        Effect::CombatPerScrapped(n) => format!("+{n} Combat for each card scrapped from your hand or discard pile this turn"),
        Effect::TopdeckBase => "Put a base from your discard pile on top of your deck".into(),
        Effect::CopyShip => "Copy another ship you've played this turn".into(),
        Effect::CopyBase => "Become a copy of any base in play until end of turn".into(),
    }
}

pub fn effects_text(list: &[Effect]) -> String {
    list.iter().map(effect_text).collect::<Vec<_>>().join(", ")
}

impl Game {
    /// Description of action `a` in the current state, from the mover's point of view.
    pub fn describe(&self, a: u16) -> String {
        let Some(act) = Action::decode(a) else { return format!("invalid action {a}") };
        let me = self.me();
        let opp = self.opp();
        let name = |c: CardId| card(c).name.as_str();
        let slot_name = |j: u8| me.in_play.get(j as usize).map_or("?", |e| card(e.eff()).name.as_str());
        let opp_base = |b: u8| opp.base_slot(b as usize).map(|s| &opp.in_play[s]);
        match act {
            Action::EndTurn => {
                let d = self.end_turn_damage();
                if d > 0 {
                    format!("End turn (deal {d} damage)")
                } else {
                    "End turn".into()
                }
            }
            Action::Skip => match self.pending {
                Some(Pending::Scrap { picked, .. }) | Some(Pending::Discard { picked, .. }) if picked > 0 => "Done".into(),
                _ => "Skip".into(),
            },
            Action::Choose(i) => match self.pending {
                Some(Pending::Choose { options, .. }) => {
                    options.get(i as usize).map_or("?".into(), |o| effects_text(o))
                }
                Some(Pending::TopdeckChoice { card: c }) => {
                    if i == 0 {
                        format!("Put {} on top of your deck", name(c))
                    } else {
                        format!("Put {} in your discard pile", name(c))
                    }
                }
                Some(Pending::TradeOrCombat { amount, .. }) => {
                    if i == 0 {
                        format!("+{amount} Trade")
                    } else {
                        format!("+{amount} Combat")
                    }
                }
                _ => format!("Choose {i}"),
            },
            Action::Play(c) => format!("Play {}", name(c)),
            Action::Primary(j) => {
                let e = &me.in_play[j as usize];
                format!("{}: {}", slot_name(j), effects_text(&e.def().primary))
            }
            Action::Ally(j) => {
                let e = &me.in_play[j as usize];
                format!("{} (ally): {}", slot_name(j), effects_text(&e.def().ally))
            }
            Action::Ally2(j) => {
                let e = &me.in_play[j as usize];
                format!("{} (double ally): {}", slot_name(j), effects_text(&e.def().ally2))
            }
            Action::ScrapAbility(j) => {
                let e = &me.in_play[j as usize];
                format!("Scrap {}: {}", slot_name(j), effects_text(&e.def().scrap))
            }
            Action::Buy(k) => {
                let c = self.trade_row[k as usize].unwrap_or(0);
                format!("Buy {} ({})", name(c), card(c).cost)
            }
            Action::BuyExplorer => "Buy Explorer (2)".into(),
            Action::AttackPlayer => format!("Attack opponent for {}", self.t.combat),
            Action::AttackBase(b) => match opp_base(b) {
                Some(e) => format!("Destroy {} ({} combat)", name(e.card), e.defense()),
                None => "?".into(),
            },
            Action::PickHand(c) => match self.pending {
                Some(Pending::Scrap { .. }) => format!("Scrap {} from hand", name(c)),
                _ => format!("Discard {}", name(c)),
            },
            Action::PickDiscard(c) => match self.pending {
                Some(Pending::TopdeckBase { .. }) => format!("Put {} on top of your deck", name(c)),
                _ => format!("Scrap {} from discard pile", name(c)),
            },
            Action::PickRow(k) => {
                let c = self.trade_row[k as usize].unwrap_or(0);
                match self.pending {
                    Some(Pending::Acquire { .. }) => format!("Acquire {}", name(c)),
                    _ => format!("Scrap {} from the trade row", name(c)),
                }
            }
            Action::PickExplorer => "Acquire an Explorer".into(),
            Action::PickOppBase(b) => match (self.pending, opp_base(b)) {
                (Some(Pending::CopyBase { .. }), Some(e)) => format!("Copy {}", name(e.eff())),
                (_, Some(e)) => format!("Destroy {}", name(e.card)),
                _ => "?".into(),
            },
            Action::PickOwn(j) => format!("Copy {}", slot_name(j)),
        }
    }

    /// Plain-text board from `seat`'s perspective (hidden zones shown as counts).
    pub fn render(&self, seat: u8) -> String {
        use std::fmt::Write;
        let names = |v: &[CardId]| v.iter().map(|&c| card(c).name.as_str()).collect::<Vec<_>>().join(", ");
        let mut s = String::new();
        let _ = writeln!(s, "Turn {} | seat {} to act | trade {} combat {}", self.turn, self.active, self.t.trade, self.t.combat);
        for (i, p) in [seat as usize, 1 - seat as usize].into_iter().enumerate() {
            let who = if i == 0 { "You" } else { "Opponent" };
            let _ = writeln!(
                s,
                "{who} (seat {p}): {} authority, deck {}, discard {}, hand {}",
                self.players[p].authority,
                self.players[p].deck.len(),
                self.players[p].discard.len(),
                self.players[p].hand.len()
            );
            let pl = &self.players[p];
            if i == 0 {
                let _ = writeln!(s, "  hand: {}", names(&pl.hand));
            }
            let play: Vec<String> = pl
                .in_play
                .iter()
                .map(|e| {
                    let n = &card(e.eff()).name;
                    if e.is_base() {
                        format!("{n} [{}{}]", e.defense(), if e.outpost() { " outpost" } else { "" })
                    } else {
                        n.clone()
                    }
                })
                .collect();
            let _ = writeln!(s, "  in play: {}", play.join(", "));
        }
        let row: Vec<String> = self
            .trade_row
            .iter()
            .map(|c| c.map_or("-".into(), |c| format!("{} ({})", card(c).name, card(c).cost)))
            .collect();
        let _ = writeln!(s, "Trade row: {} | explorers {}", row.join(", "), self.explorers);
        if let Some(p) = self.prompt() {
            let _ = writeln!(s, "> {p}");
        }
        if self.active == seat && self.outcome.is_none() {
            for a in self.legal_vec() {
                let _ = writeln!(s, "  [{a}] {}", self.describe(a));
            }
        }
        s
    }

    /// Prompt text for the current pending decision, if any.
    pub fn prompt(&self) -> Option<String> {
        let src = |p: &Pending| p.source().map_or(String::new(), |c| format!("{}: ", card(c).name));
        let p = self.pending?;
        Some(match p {
            Pending::Choose { .. } => format!("{}Choose one", src(&p)),
            Pending::Scrap { spec, picked, .. } => {
                let left = spec.max - picked;
                let what = if left == 1 { "a card".to_string() } else { format!("up to {left} cards") };
                let verb = if picked < spec.min { "Scrap" } else { "You may scrap" };
                format!("{}{verb} {what} from {}", src(&p), zone_text(spec.from))
            }
            Pending::Discard { min, max, picked, forced: true, .. } => {
                let _ = (min, picked);
                format!("Your opponent made you discard: discard {} ", plural(max - picked, "card"))
            }
            Pending::Discard { min, max, picked, .. } => {
                if picked < min {
                    format!("Discard {}", plural(min - picked, "card"))
                } else if max >= 12 {
                    "Discard any number of cards".into()
                } else {
                    format!("You may discard up to {}", plural(max - picked, "card"))
                }
            }
            Pending::DestroyBase { .. } => format!("{}You may destroy a base", src(&p)),
            Pending::Acquire { spec, .. } => format!("{}{}", src(&p), effect_text(&Effect::Acquire(*spec))),
            Pending::TopdeckBase { .. } => format!("{}Choose a base from your discard pile", src(&p)),
            Pending::CopyShip { .. } => format!("{}Choose a ship to copy", src(&p)),
            Pending::CopyBase { .. } => format!("{}Choose a base to copy", src(&p)),
            Pending::TopdeckChoice { card: c } => format!("Put {} on top of your deck?", card(c).name),
            Pending::TradeOrCombat { amount, remaining } => {
                format!("Gain {amount} Trade or {amount} Combat ({remaining} left)")
            }
        })
    }
}
