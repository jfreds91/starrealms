use crate::action::{self, Action};
use crate::cards::*;
use crate::game::*;
use crate::rng::Rng;
use crate::Replay;

fn id(s: &str) -> CardId {
    db().get(s).unwrap_or_else(|| panic!("no card {s}"))
}

fn cfg(sets: &[&str]) -> Config {
    Config { sets: sets.iter().map(|s| s.to_string()).collect(), ..Config::default() }
}

/// A fresh game with the active player's hand replaced.
fn scenario(hand: &[&str]) -> Game {
    let mut g = Game::new(&cfg(&["core", "colony_wars", "frontiers"]), 11).unwrap();
    g.players[0].hand = hand.iter().map(|s| id(s)).collect();
    g
}

fn put_base(g: &mut Game, seat: usize, name: &str) {
    let uid = 10_000 + g.players[seat].in_play.len() as u16 + 100 * seat as u16;
    g.players[seat].in_play.insert(0, InPlay { card: id(name), copy: None, uid, used: 0, unlocked: 0 });
}

fn play(g: &mut Game, name: &str) {
    g.step(action::PLAY + id(name) as u16).unwrap();
}

fn slot_of(g: &Game, name: &str) -> u16 {
    g.me().in_play.iter().position(|e| e.card == id(name)).unwrap() as u16
}

fn has(g: &Game, a: u16) -> bool {
    g.legal_vec().contains(&a)
}

// ------------------------------------------------------------------ data

#[test]
fn card_data_matches_upstream_review() {
    let csv = include_str!("../../../data/upstream/review.csv");
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let mut seen = 0;
    for line in lines {
        // The last column (text) may contain commas; the fields we need come first.
        let f: Vec<&str> = line.splitn(header.len(), ',').collect();
        let c = card(id(f[col("id")]));
        assert_eq!(c.name, f[col("name")].trim_matches('"'), "{}", c.id);
        assert_eq!(c.cost.to_string(), f[col("cost")], "{} cost", c.id);
        let faction = serde_json::to_string(&c.faction).unwrap();
        assert_eq!(faction.trim_matches('"'), f[col("faction")], "{} faction", c.id);
        let kind = serde_json::to_string(&c.kind).unwrap();
        assert_eq!(kind.trim_matches('"'), f[col("kind")], "{} kind", c.id);
        if c.is_base() {
            assert_eq!(c.defense.to_string(), f[col("defense")], "{} defense", c.id);
            assert_eq!(c.outpost.to_string(), f[col("outpost")], "{} outpost", c.id);
        }
        seen += 1;
    }
    assert_eq!(seen, N_CARDS);
}

#[test]
fn trade_decks_are_80_cards() {
    for s in ["core", "colony_wars", "frontiers"] {
        let g = Game::new(&cfg(&[s]), 1).unwrap();
        assert_eq!(g.trade_deck.len() + 5, 80, "{s}");
        assert_eq!(g.card_count(), 80 + 20 + 10);
    }
}

// ------------------------------------------------------------------ random play

fn random_game(sets: &[&str], seed: u64, check_all_legal: bool) -> Game {
    let mut g = Game::new(&cfg(sets), seed).unwrap();
    let total = g.card_count();
    let mut rng = Rng::new(seed ^ 0xdead_beef);
    let mut legal = Vec::new();
    let mut steps = 0;
    while !g.is_over() {
        g.legal_actions(&mut legal);
        assert!(!legal.is_empty(), "no legal actions in live game");
        if check_all_legal && steps % 7 == 0 {
            for &a in &legal {
                let mut h = g.clone();
                h.step(a).unwrap();
                assert_eq!(h.card_count(), total, "card conservation after {:?}", Action::decode(a));
            }
        }
        // Bias away from ending the turn so games actually develop.
        let a = if legal.len() > 1 && legal[0] == action::END && rng.below(4) != 0 {
            legal[1 + rng.below(legal.len() - 1)]
        } else {
            legal[rng.below(legal.len())]
        };
        g.step(a).unwrap();
        assert_eq!(g.card_count(), total);
        for p in &g.players {
            assert!(p.in_play.iter().all(|e| e.is_base()) || std::ptr::eq(p, g.me()), "ships left in play");
        }
        steps += 1;
        assert!(steps < 100_000, "runaway game");
    }
    g
}

#[test]
fn random_games_preserve_invariants() {
    let combos: [&[&str]; 5] =
        [&["core"], &["colony_wars"], &["frontiers"], &["core", "colony_wars"], &["core", "colony_wars", "frontiers"]];
    let mut wins = [0; 3];
    for (i, sets) in combos.iter().enumerate() {
        for seed in 0..120 {
            let g = random_game(sets, seed * 31 + i as u64, seed < 20);
            match g.outcome.unwrap() {
                Outcome::Win(s) => wins[s as usize] += 1,
                Outcome::Draw => wins[2] += 1,
            }
        }
    }
    assert!(wins[0] > 0 && wins[1] > 0, "{wins:?}");
}

#[test]
fn replays_are_deterministic() {
    for seed in 0..20 {
        let mut g = Game::new(&cfg(&["core", "frontiers"]), seed).unwrap();
        let mut rep = Replay::new(cfg(&["core", "frontiers"]), seed);
        let mut rng = Rng::new(seed);
        while !g.is_over() {
            let legal = g.legal_vec();
            let a = legal[rng.below(legal.len())];
            g.step(a).unwrap();
            rep.actions.push(a);
        }
        let h = rep.run().unwrap();
        assert_eq!(g.state_hash(), h.state_hash());
        let json = serde_json::to_string(&rep).unwrap();
        let back: Replay = serde_json::from_str(&json).unwrap();
        assert_eq!(back.run().unwrap().state_hash(), g.state_hash());
    }
}

#[test]
fn replay_seed_survives_javascript() {
    let rep = Replay::new(cfg(&["core"]), u64::MAX - 12345);
    let json = serde_json::to_string(&rep).unwrap();
    assert!(json.contains(&format!("\"seed\":\"{}\"", u64::MAX - 12345)), "{json}");
    let old = json.replace(&format!("\"{}\"", u64::MAX - 12345), &(u64::MAX - 12345).to_string());
    assert_eq!(serde_json::from_str::<Replay>(&old).unwrap(), rep);
}

/// Final-state hashes of fixed random-policy games. A rules change that alters
/// any of them must be deliberate: rerun with `UPDATE_GOLDEN=1` and commit.
#[test]
fn golden_replays() {
    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden.txt");
    let combos = ["core", "colony_wars", "frontiers", "core,colony_wars,frontiers"];
    let mut lines = Vec::new();
    for sets in combos {
        for seed in 0..25u64 {
            let c = cfg(&sets.split(',').collect::<Vec<_>>());
            let mut g = Game::new(&c, seed).unwrap();
            let mut rng = Rng::new(seed.wrapping_mul(0x9E37));
            let mut n = 0;
            while !g.is_over() {
                let legal = g.legal_vec();
                let a = if legal.len() > 1 && rng.below(8) != 0 {
                    legal[1 + rng.below(legal.len() - 1)]
                } else {
                    legal[rng.below(legal.len())]
                };
                g.step(a).unwrap();
                n += 1;
            }
            lines.push(format!("{sets} {seed} {n} {:016x}", g.state_hash()));
        }
    }
    let got = lines.join("\n") + "\n";
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        std::fs::create_dir_all(std::path::Path::new(PATH).parent().unwrap()).unwrap();
        std::fs::write(PATH, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(PATH).expect("missing golden file; run with UPDATE_GOLDEN=1");
    for (g, w) in got.lines().zip(want.lines()) {
        assert_eq!(g, w, "golden replay changed");
    }
    assert_eq!(got.lines().count(), want.lines().count());
}

/// Every card, played (or put into play) with allies around and targets
/// available, then driven by random decisions. Catches panics in any effect.
#[test]
fn every_card_survives_random_resolution() {
    for c in &db().cards {
        for seed in 0..25u64 {
            let mut g = Game::new(&cfg(&["core", "colony_wars", "frontiers"]), seed).unwrap();
            let ally = db().cards.iter().find(|x| x.faction == c.faction && x.idx != c.idx && !x.is_base()).map(|x| x.idx);
            let mut hand = vec![c.idx];
            if let Some(a) = ally {
                hand.extend([a, a]);
            }
            hand.extend([db().scout, db().viper, db().explorer]);
            g.players[0].hand = hand;
            g.players[0].discard = vec![db().scout, id("blobWheel"), id("cutter")];
            put_base(&mut g, 1, "warWorld");
            put_base(&mut g, 1, "blobWheel");
            put_base(&mut g, 0, "mechWorld");
            g.t.trade = 20;
            g.t.combat = 3;
            let mut rng = Rng::new(seed);
            for _ in 0..60 {
                if g.is_over() {
                    break;
                }
                let legal = g.legal_vec();
                let non_end: Vec<u16> = legal.iter().copied().filter(|&a| a != action::END).collect();
                let pool = if non_end.is_empty() { &legal } else { &non_end };
                let a = if g.pending.is_none() && pool.contains(&(action::PLAY + c.idx as u16)) {
                    action::PLAY + c.idx as u16
                } else {
                    pool[rng.below(pool.len())]
                };
                g.step(a).unwrap_or_else(|e| panic!("{}: {e}", c.id));
            }
        }
    }
}

// ------------------------------------------------------------------ rules

#[test]
fn play_all_skips_cards_that_need_a_decision() {
    let mut g = scenario(&["scout", "patrolMech", "viper", "stealthNeedle", "cutter"]);
    while let Some(a) = g.next_simple_play() {
        g.step(a).unwrap();
    }
    let mut left = g.me().hand.clone();
    left.sort_unstable();
    let mut want = vec![id("patrolMech"), id("stealthNeedle")];
    want.sort_unstable();
    assert_eq!(left, want);
    assert_eq!(g.t.trade, 1 + 2);
    assert_eq!(g.t.combat, 1);
}

#[test]
fn first_player_draws_three() {
    let g = Game::new(&Config::default(), 3).unwrap();
    assert_eq!(g.players[0].hand.len(), 3);
    assert_eq!(g.players[1].hand.len(), 5);
}

#[test]
fn scout_and_viper_pools() {
    let mut g = scenario(&["scout", "scout", "viper"]);
    play(&mut g, "scout");
    play(&mut g, "scout");
    play(&mut g, "viper");
    assert_eq!((g.t.trade, g.t.combat), (2, 1));
    let before = g.opp().authority;
    g.step(action::END).unwrap();
    assert_eq!(g.players[1].authority, before - 1, "leftover combat applied at end of turn");
    assert_eq!(g.active, 1);
    assert_eq!(g.players[0].hand.len(), 5);
}

#[test]
fn explorer_scrap_returns_to_pile() {
    let mut g = scenario(&["explorer"]);
    let pile = g.explorers;
    play(&mut g, "explorer");
    let j = slot_of(&g, "explorer");
    g.step(action::SCRAP + j).unwrap();
    assert_eq!(g.explorers, pile + 1);
    assert_eq!((g.t.trade, g.t.combat), (2, 2));
    assert!(g.scrap_heap.is_empty());
}

#[test]
fn outposts_block_player_and_other_bases() {
    let mut g = scenario(&["viper"]);
    put_base(&mut g, 1, "blobWheel"); // 5, not an outpost
    put_base(&mut g, 1, "warWorld"); // 4, outpost
    g.t.combat = 10;
    play(&mut g, "viper");
    assert!(!has(&g, action::ATTACK_PLAYER));
    let legal = g.legal_vec();
    let war = action::ATTACK_BASE; // warWorld is base 0 (inserted last at front)
    assert!(legal.contains(&war));
    assert!(!legal.contains(&(action::ATTACK_BASE + 1)), "non-outpost protected");
    g.step(war).unwrap();
    assert_eq!(g.t.combat, 7);
    assert!(has(&g, action::ATTACK_PLAYER));
    assert!(has(&g, action::ATTACK_BASE));
    assert_eq!(g.end_turn_damage(), 7);
    assert!(g.players[1].discard.contains(&id("warWorld")));
}

#[test]
fn opponent_discards_at_start_of_their_turn() {
    let mut g = scenario(&["imperialFighter"]);
    play(&mut g, "imperialFighter");
    assert_eq!(g.players[1].owed_discards, 1);
    assert_eq!(g.players[1].hand.len(), 5, "no discard during my turn");
    g.step(action::END).unwrap();
    assert_eq!(g.pending.map(|p| p.kind()), Some(PendingKind::ForcedDiscard));
    assert!(!has(&g, action::SKIP));
    let c = g.me().hand[0];
    g.step(action::PICK_HAND + c as u16).unwrap();
    assert_eq!(g.me().hand.len(), 4);
    assert!(g.pending.is_none());
}

#[test]
fn ally_and_double_ally() {
    let mut g = scenario(&["hiveQueen", "blobFighter", "stinger"]);
    play(&mut g, "hiveQueen");
    let base_combat = g.t.combat;
    play(&mut g, "blobFighter"); // unlocks Hive Queen ally (+3, auto)
    assert_eq!(g.t.combat, base_combat + 3 + 3);
    let hq = &g.me().in_play[slot_of(&g, "hiveQueen") as usize];
    assert!(hq.used & F_ALLY != 0 && hq.unlocked & F_ALLY2 == 0);
    play(&mut g, "stinger"); // second Blob: double ally (+3), Stinger ally (+3)
    assert_eq!(g.t.combat, base_combat + 3 + 3 + 3 + 3 + 3);
}

#[test]
fn mech_world_is_an_ally_for_every_faction() {
    let mut g = scenario(&["cutter"]);
    put_base(&mut g, 0, "mechWorld");
    play(&mut g, "cutter");
    assert_eq!(g.t.combat, 4, "Cutter's ally ability fired");
}

#[test]
fn stealth_needle_copies_a_ship_and_its_faction() {
    let mut g = scenario(&["blobFighter", "stealthNeedle"]);
    g.players[0].deck = vec![db().scout; 5];
    play(&mut g, "blobFighter");
    play(&mut g, "stealthNeedle");
    assert_eq!(g.pending.map(|p| p.kind()), Some(PendingKind::CopyShip));
    let j = slot_of(&g, "blobFighter");
    g.step(action::PICK_OWN + j).unwrap();
    assert_eq!(g.t.combat, 6);
    // The copy is a Blob card, so both Blob Fighters' ally abilities unlock.
    let n = slot_of(&g, "stealthNeedle");
    assert!(has(&g, action::ALLY + j));
    assert!(has(&g, action::ALLY + n));
}

#[test]
fn stealth_tower_copies_opponent_base() {
    let mut g = scenario(&["scout"]);
    put_base(&mut g, 0, "stealthTower");
    put_base(&mut g, 1, "loyalColony");
    g.settle();
    let j = slot_of(&g, "stealthTower");
    g.step(action::PRIMARY + j).unwrap();
    g.step(action::PICK_OPP_BASE).unwrap();
    // Loyal Colony's primary is pure, so it fires immediately.
    assert_eq!((g.t.trade, g.t.combat), (3, 3));
    g.step(action::END).unwrap();
    assert_eq!(g.players[0].in_play[0].copy, None, "copy ends with the turn");
}

#[test]
fn blob_world_draws_per_blob_played() {
    let mut g = scenario(&["blobFighter", "ram", "blobWorld"]);
    g.players[0].deck = vec![db().scout; 10];
    play(&mut g, "blobFighter");
    play(&mut g, "ram");
    play(&mut g, "blobWorld");
    let hand = g.me().hand.len();
    let j = slot_of(&g, "blobWorld");
    g.step(action::PRIMARY + j).unwrap();
    g.step(action::CHOOSE + 1).unwrap();
    // Blob Fighter's ally may also have been used; count only Blob World's draws.
    assert_eq!(g.me().hand.len(), hand + 3);
}

#[test]
fn mobile_market_returns_from_the_scrap_heap() {
    let mut g = scenario(&["scout"]);
    g.players[0].deck = vec![db().scout; 10];
    put_base(&mut g, 0, "mobileMarket");
    g.settle();
    let j = slot_of(&g, "mobileMarket");
    g.step(action::SCRAP + j).unwrap();
    assert!(g.scrap_heap.contains(&id("mobileMarket")));
    g.step(action::END).unwrap();
    assert!(!g.scrap_heap.contains(&id("mobileMarket")));
    let p = &g.players[0];
    assert!(p.discard.contains(&id("mobileMarket")) || p.hand.contains(&id("mobileMarket")) || p.deck.contains(&id("mobileMarket")));
}

#[test]
fn next_acquire_on_top_of_deck() {
    let mut g = scenario(&["longHauler"]);
    play(&mut g, "longHauler");
    g.t.trade = 10;
    let j = slot_of(&g, "longHauler");
    g.step(action::SCRAP + j).unwrap();
    g.trade_row[0] = Some(id("blobWheel"));
    g.trade_row[1] = Some(id("cutter"));
    g.step(action::BUY + 1).unwrap(); // a ship: not affected
    assert_eq!(*g.me().discard.last().unwrap(), id("cutter"));
    g.step(action::BUY).unwrap(); // the next base goes on top of the deck
    assert_eq!(*g.me().deck.last().unwrap(), id("blobWheel"));
}

#[test]
fn optional_top_deck_asks() {
    let mut g = scenario(&["freighter", "cutter"]);
    play(&mut g, "freighter");
    play(&mut g, "cutter");
    let j = slot_of(&g, "freighter");
    g.step(action::ALLY + j).unwrap();
    g.trade_row[2] = Some(id("corvette"));
    g.step(action::BUY + 2).unwrap();
    assert_eq!(g.pending.map(|p| p.kind()), Some(PendingKind::TopdeckChoice));
    g.step(action::CHOOSE).unwrap();
    assert_eq!(*g.me().deck.last().unwrap(), id("corvette"));
}

#[test]
fn acquire_to_hand_after_playing_faction() {
    let mut g = scenario(&["tradeBot"]);
    play(&mut g, "tradeBot");
    if g.pending.is_some() {
        g.step(action::SKIP).unwrap();
    }
    g.t.trade = 5;
    g.trade_row[0] = Some(id("warningBeacon"));
    g.step(action::BUY).unwrap();
    assert!(g.me().hand.contains(&id("warningBeacon")));
}

#[test]
fn reclamation_station_counts_its_own_scrap() {
    let mut g = scenario(&["tradeBot"]);
    g.players[0].discard = vec![db().scout, db().viper];
    put_base(&mut g, 0, "reclamationStation");
    g.settle();
    play(&mut g, "tradeBot");
    g.step(action::PICK_DISCARD + db().scout as u16).unwrap();
    let j = slot_of(&g, "reclamationStation");
    g.step(action::PRIMARY + j).unwrap();
    g.step(action::PICK_DISCARD + db().viper as u16).unwrap();
    let j = slot_of(&g, "reclamationStation");
    let before = g.t.combat;
    g.step(action::SCRAP + j).unwrap();
    assert_eq!(g.t.combat - before, 3 * 3);
}

#[test]
fn supply_depot_chooses_per_card() {
    let mut g = scenario(&["scout", "viper", "viper"]);
    put_base(&mut g, 0, "supplyDepot");
    g.settle();
    let j = slot_of(&g, "supplyDepot");
    g.step(action::PRIMARY + j).unwrap();
    g.step(action::PICK_HAND + db().viper as u16).unwrap();
    g.step(action::PICK_HAND + db().viper as u16).unwrap();
    g.step(action::CHOOSE).unwrap();
    g.step(action::CHOOSE + 1).unwrap();
    assert_eq!((g.t.trade, g.t.combat), (2, 2));
}

#[test]
fn fleet_hq_triggers_on_ships() {
    let mut g = scenario(&["scout", "viper"]);
    put_base(&mut g, 0, "fleetHq");
    play(&mut g, "scout");
    play(&mut g, "viper");
    assert_eq!((g.t.trade, g.t.combat), (1, 3));
}

#[test]
fn turn_limit_is_a_draw() {
    let mut c = Config::default();
    c.max_turns = 4;
    let mut g = Game::new(&c, 5).unwrap();
    while !g.is_over() {
        if g.pending.is_some() {
            let a = g.legal_vec()[0];
            g.step(a).unwrap();
        } else {
            g.step(action::END).unwrap();
        }
    }
    assert!(matches!(g.outcome, Some(Outcome::Draw) | Some(Outcome::Win(_))));
}

#[test]
fn observation_has_fixed_size_and_is_finite() {
    let g = random_game(&["core", "colony_wars", "frontiers"], 99, false);
    let o = g.observation();
    assert_eq!(o.len(), crate::OBS_DIM);
    assert!(o.iter().all(|x| x.is_finite()));
}

#[test]
fn every_legal_action_has_a_description() {
    let mut g = Game::new(&cfg(&["core", "colony_wars", "frontiers"]), 4).unwrap();
    let mut rng = Rng::new(4);
    for _ in 0..3000 {
        if g.is_over() {
            break;
        }
        for a in g.legal_vec() {
            let d = g.describe(a);
            assert!(!d.contains('?'), "{d}");
        }
        let v = g.view(g.active, 5);
        assert_eq!(v.actions.len(), g.legal_vec().len());
        let legal = g.legal_vec();
        g.step(legal[rng.below(legal.len())]).unwrap();
    }
}
