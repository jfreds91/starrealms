import { createContext, useContext } from "react";
import type { CardId, CardInfo, Faction } from "./types";

export interface Catalog {
  cards: CardInfo[];
  explorer: CardId;
}

export function makeCatalog(cards: CardInfo[]): Catalog {
  return { cards, explorer: cards.find((c) => c.id === "explorer")?.idx ?? 2 };
}

export const CatalogContext = createContext<Catalog | null>(null);

export function useCatalog(): Catalog {
  const c = useContext(CatalogContext);
  if (!c) throw new Error("catalog not loaded");
  return c;
}

export const FACTION_NAMES: Record<Faction, string> = {
  blob: "Blob",
  machine_cult: "Machine Cult",
  star_empire: "Star Empire",
  trade_federation: "Trade Federation",
  unaligned: "Unaligned",
};

/** Group identical cards: [[id, count], ...] sorted by cost then name. */
export function groupCards(cat: Catalog, ids: CardId[]): [CardId, number][] {
  const m = new Map<CardId, number>();
  for (const id of ids) m.set(id, (m.get(id) ?? 0) + 1);
  return [...m.entries()].sort(([a], [b]) => {
    const ca = cat.cards[a], cb = cat.cards[b];
    return cb.cost - ca.cost || ca.name.localeCompare(cb.name);
  });
}
