// Mirrors the JSON produced by sr-core's `view` module and sr-server.

export type CardId = number;
export type Faction = "blob" | "machine_cult" | "star_empire" | "trade_federation" | "unaligned";

export interface CardInfo {
  idx: CardId;
  id: string;
  name: string;
  set: string;
  faction: Faction;
  kind: "ship" | "base";
  cost: number;
  defense: number;
  outpost: boolean;
  primary: string;
  ally: string;
  ally2: string;
  scrap: string;
  image: string;
}

export interface SlotView {
  card: CardId;
  copy: CardId | null;
  is_base: boolean;
  defense: number;
  outpost: boolean;
  primary: boolean;
  ally: boolean;
  ally2: boolean;
  scrap: boolean;
}

export interface SeatView {
  authority: number;
  hand: CardId[] | null;
  hand_count: number;
  deck_count: number;
  /** Sorted draw pile; pooled with the hand when the hand is hidden. */
  deck_contents: CardId[] | null;
  discard: CardId[];
  in_play: SlotView[];
  owed_discards: number;
}

export type Target =
  | { zone: "none" }
  | { zone: "hand"; card: CardId }
  | { zone: "discard"; card: CardId }
  | { zone: "own"; slot: number }
  | { zone: "row"; slot: number }
  | { zone: "explorer" }
  | { zone: "opp_base"; slot: number }
  | { zone: "opponent" };

export type ActionGroup = "end" | "skip" | "choose" | "play" | "ability" | "scrap_ability" | "buy" | "attack" | "pick";

export interface ActionView {
  id: number;
  label: string;
  group: ActionGroup;
  target: Target;
}

export type Outcome = { Win: number } | "Draw";

export interface LogEntry {
  turn: number;
  seat: number;
  text: string;
}

export interface View {
  seat: number;
  active: number;
  turn: number;
  outcome: Outcome | null;
  you: SeatView;
  opponent: SeatView;
  trade_row: (CardId | null)[];
  explorers: number;
  trade_deck: number;
  scrap_heap: CardId[];
  trade: number;
  combat: number;
  prompt: string | null;
  pending: string | null;
  pending_source: CardId | null;
  actions: ActionView[];
  log: LogEntry[];
}

export type Seat =
  | { kind: "open" }
  | { kind: "player"; id: string; name: string }
  | { kind: "bot"; spec: string; name: string };

export interface GameMeta {
  id: string;
  seats: [Seat & { online: boolean }, Seat & { online: boolean }];
  status: "waiting" | "active" | "finished";
  winner: number | null;
  resigned: number | null;
  sets: string[];
  created: number;
}

export interface GameSummary {
  id: string;
  seat: number;
  seats: [Seat, Seat];
  status: "waiting" | "active" | "finished";
  active: number;
  turn: number;
  winner: number | null;
  sets: string[];
  updated: number;
}

export interface Replay {
  config: { sets: string[]; [k: string]: unknown };
  /** A u64, kept as a string so it survives JSON.parse. */
  seed: string;
  actions: number[];
}

export const SET_NAMES: Record<string, string> = {
  core: "Core Set",
  colony_wars: "Colony Wars",
  frontiers: "Frontiers",
};

const BOT_NAMES: Record<string, string> = {
  "model:phase_f": "Uncle Jesse",
};

/** Player-facing name. `phase_f` is shown as Uncle Jesse; the model file keeps its id. */
export function opponentName(spec: string, name: string): string {
  return BOT_NAMES[spec] ?? name;
}

export function seatName(s: Seat | undefined, fallback: string): string {
  if (!s || s.kind === "open") return fallback;
  if (s.kind === "bot") return opponentName(s.spec, s.name);
  return s.name;
}
