import type { CardInfo, GameSummary, Replay, Seat } from "./types";

const TOKEN = "sr.token";
const NAME = "sr.name";

export interface Me {
  token: string;
  name: string;
}

export function loadMe(): Me | null {
  const token = localStorage.getItem(TOKEN);
  const name = localStorage.getItem(NAME);
  return token && name ? { token, name } : null;
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  const me = loadMe();
  if (me) headers.Authorization = `Bearer ${me.token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(`/api${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? `${res.status} ${res.statusText}`);
  return data as T;
}

/** Create a player identity (or rename the current one). */
export async function setName(name: string): Promise<Me> {
  const me = loadMe();
  if (me) {
    try {
      const r = await call<{ name: string }>("PUT", "/me", { name });
      localStorage.setItem(NAME, r.name);
      return { token: me.token, name: r.name };
    } catch {
      // Unknown token (e.g. the server database was reset): fall through and register again.
    }
  }
  const r = await call<{ token: string; name: string }>("POST", "/players", { name });
  localStorage.setItem(TOKEN, r.token);
  localStorage.setItem(NAME, r.name);
  return { token: r.token, name: r.name };
}

export const api = {
  catalog: () => call<CardInfo[]>("GET", "/catalog"),
  opponents: () => call<{ spec: string; name: string }[]>("GET", "/opponents"),
  games: () => call<GameSummary[]>("GET", "/games"),
  createGame: (sets: string[], opponent: string, seat: string) =>
    call<{ id: string; seat: number }>("POST", "/games", { sets, opponent, seat }),
  join: (id: string) => call<{ seat: number }>("POST", `/games/${id}/join`),
  replay: (id: string) =>
    call<{ id: string; replay: Replay; seats: [Seat, Seat]; resigned: number | null }>("GET", `/games/${id}/replay`),
};

export function socketUrl(id: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const me = loadMe();
  const q = me ? `?token=${encodeURIComponent(me.token)}` : "";
  return `${proto}://${location.host}/api/games/${id}/ws${q}`;
}
