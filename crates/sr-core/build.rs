// Parses data/cards.toml and data/sets.toml with the engine's own schema so a
// malformed card file fails the build instead of the first game.

#[allow(dead_code)]
#[path = "src/cards.rs"]
mod cards;

fn main() {
    println!("cargo:rerun-if-changed=../../data/cards.toml");
    println!("cargo:rerun-if-changed=../../data/sets.toml");
    println!("cargo:rerun-if-changed=src/cards.rs");
    if let Err(e) = cards::CardDb::parse(cards::CARDS_TOML, cards::SETS_TOML) {
        panic!("invalid card data: {e}");
    }
}
