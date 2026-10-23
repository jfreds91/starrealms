import { SET_NAMES, seatName, type GameSummary, type Seat } from "./types";

/** phase_f with every set, anchored at 0. Same Bradley-Terry fit the trainer uses. */
const ANCHOR = "bot:model:phase_f|colony_wars+core+frontiers";
const PRIOR = 1;

export interface RatingRow {
  key: string;
  name: string;
  elo: number;
  wins: number;
  draws: number;
  losses: number;
}

function setsKey(sets: string[]): string {
  return [...sets].sort().join("+");
}

function side(seat: Seat, sets: string[]): { key: string; name: string } {
  if (seat.kind === "bot") {
    const label = sets.map((s) => SET_NAMES[s] ?? s).join(" + ");
    return { key: `bot:${seat.spec}|${setsKey(sets)}`, name: `${seatName(seat, seat.name)} · ${label}` };
  }
  if (seat.kind === "player") return { key: `player:${seat.id}`, name: seat.name };
  return { key: "open", name: "Open seat" };
}

/** Ratings for the signed-in player and everyone they have finished a game against. */
export function ratings(games: GameSummary[]): { you: RatingRow; opponents: RatingRow[] } | null {
  const finished = games.filter((g) => g.status === "finished" && g.winner != null && g.seats[g.seat]);
  if (finished.length === 0) return null;
  const you = side(finished[0].seats[finished[0].seat], finished[0].sets);
  const names = new Map<string, string>([[you.key, you.name], [ANCHOR, "Uncle Jesse · Core Set + Colony Wars + Frontiers"]]);
  const score = new Map<string, number>();
  const record = new Map<string, { wins: number; draws: number; losses: number }>();
  const add = (a: string, b: string, points: number) => score.set(`${a}\0${b}`, (score.get(`${a}\0${b}`) ?? 0) + points);
  for (const g of finished) {
    const me = side(g.seats[g.seat], g.sets);
    const opp = side(g.seats[1 - g.seat], g.sets);
    names.set(me.key, me.name);
    names.set(opp.key, opp.name);
    const rec = record.get(opp.key) ?? { wins: 0, draws: 0, losses: 0 };
    if (g.winner === -1) {
      add(me.key, opp.key, 0.5);
      add(opp.key, me.key, 0.5);
      rec.draws += 1;
    } else if (g.winner === g.seat) {
      add(me.key, opp.key, 1);
      rec.wins += 1;
    } else {
      add(opp.key, me.key, 1);
      rec.losses += 1;
    }
    record.set(opp.key, rec);
  }
  const keys = [...names.keys()];
  const index = new Map(keys.map((k, i) => [k, i]));
  const n = keys.length;
  const s = Array.from({ length: n }, () => Array(n).fill(0));
  for (const [pair, points] of score) {
    const [a, b] = pair.split("\0");
    s[index.get(a)!][index.get(b)!] = points;
  }
  const played = s.map((row, i) => row.map((v, j) => v + s[j][i]));
  const points = s.map((row) => row.reduce((a, b) => a + b, 0) + PRIOR / 2);
  let g = Array(n).fill(1);
  for (let iter = 0; iter < 20000; iter++) {
    const next = g.map((gi, i) => {
      let denom = PRIOR / (gi + 1);
      for (let j = 0; j < n; j++) if (played[i][j]) denom += played[i][j] / (gi + g[j]);
      return points[i] / denom;
    });
    const done = next.every((v, i) => Math.abs(Math.log(v) - Math.log(g[i])) < 1e-10);
    g = next;
    if (done) break;
  }
  const anchor = index.get(ANCHOR) ?? 0;
  const eloOf = (i: number) => Math.round((400 * (Math.log10(g[i]) - Math.log10(g[anchor]))) * 10) / 10;
  const row = (key: string): RatingRow => {
    const rec = record.get(key) ?? { wins: 0, draws: 0, losses: 0 };
    return { key, name: names.get(key) ?? key, elo: eloOf(index.get(key)!), ...rec };
  };
  return {
    you: row(you.key),
    opponents: keys.filter((k) => k !== you.key && record.has(k)).map(row).sort((a, b) => b.elo - a.elo),
  };
}
