import { useEffect, useState } from "react";
import { api, loadMe, setName, type Me } from "../api";
import { ratings } from "../elo";
import { Link, navigate } from "../router";
import { SET_NAMES, opponentName, seatName, type GameSummary } from "../types";
import { AboutAI } from "./AboutAI";
import { HowToPlay } from "./HowToPlay";

const ALL_SETS = ["core", "colony_wars", "frontiers"];

export function NameForm({ me, onDone, cta = "Save" }: { me: Me | null; onDone(me: Me): void; cta?: string }) {
  const [name, setNameInput] = useState(me?.name ?? "");
  const [err, setErr] = useState<string | null>(null);
  return (
    <form
      className="name-form"
      onSubmit={async (e) => {
        e.preventDefault();
        try {
          onDone(await setName(name));
        } catch (x) {
          setErr(String((x as Error).message));
        }
      }}
    >
      <input value={name} maxLength={24} placeholder="Your name" autoFocus={!me} onChange={(e) => setNameInput(e.target.value)} />
      <button className="btn primary" disabled={!name.trim()}>{cta}</button>
      {err && <span className="error">{err}</span>}
    </form>
  );
}

function SetPicker({ sets, onChange }: { sets: string[]; onChange(s: string[]): void }) {
  return (
    <div className="set-picker">
      {ALL_SETS.map((s) => (
        <label key={s} className={`toggle ${sets.includes(s) ? "on" : ""}`}>
          <input
            type="checkbox"
            checked={sets.includes(s)}
            onChange={(e) => {
              const next = e.target.checked ? ALL_SETS.filter((x) => x === s || sets.includes(x)) : sets.filter((x) => x !== s);
              if (next.length) onChange(next);
            }}
          />
          {SET_NAMES[s]}
        </label>
      ))}
    </div>
  );
}

function timeAgo(t: number): string {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

function GameList() {
  const [games, setGames] = useState<GameSummary[] | null>(null);
  const [tab, setTab] = useState<"turn" | "waiting" | "finished">("turn");
  useEffect(() => {
    let alive = true;
    const load = () => api.games().then((g) => alive && setGames(g), () => alive && setGames([]));
    load();
    const t = window.setInterval(load, 15000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, []);
  if (!games) return null;
  const yourTurn = (g: GameSummary) => g.status === "active" && g.active === g.seat;
  const lists = {
    turn: games.filter(yourTurn),
    waiting: games.filter((g) => g.status !== "finished" && !yourTurn(g)),
    finished: games.filter((g) => g.status === "finished"),
  };
  const shown = lists[tab];
  const rated = ratings(games);
  return (
    <section className="panel">
      <h2>Elo</h2>
      {rated ? (
        <div className="ratings">
          <div className="rating-row"><span>{rated.you.name}</span><b>{rated.you.elo > 0 ? `+${rated.you.elo}` : rated.you.elo}</b></div>
          {rated.opponents.map((o) => (
            <div className="rating-row" key={o.key}>
              <span>{o.name}</span>
              <span className="game-sets">{o.wins}–{o.draws}–{o.losses}</span>
              <b>{o.elo > 0 ? `+${o.elo}` : o.elo}</b>
            </div>
          ))}
          <p className="empty-hint">Fit from finished games. Uncle Jesse with all three sets is 0.</p>
        </div>
      ) : (
        <p className="empty-hint">Finished games record you, the opponent, and the winner. Elo shows up here.</p>
      )}
      <div className="tabs">
        {(["turn", "waiting", "finished"] as const).map((t) => (
          <button key={t} className={`tab ${tab === t ? "on" : ""}`} onClick={() => setTab(t)}>
            {t === "turn" ? "Your turn" : t === "waiting" ? "Their turn" : "Finished"}
            <span className="tab-count">{lists[t].length}</span>
          </button>
        ))}
      </div>
      {shown.length === 0 && <div className="empty-hint">Nothing here.</div>}
      <ul className="game-list">
        {shown.map((g) => {
          const opp = g.seats[1 - g.seat];
          const result =
            g.winner == null ? null : g.winner === -1 ? "Draw" : g.winner === g.seat ? "Won" : "Lost";
          return (
            <li key={g.id}>
              <Link to={g.status === "finished" ? `/replay/${g.id}` : `/g/${g.id}`} className="game-row">
                <span className="game-opp">vs {seatName(opp, "open seat")}</span>
                <span className="game-sets">{g.sets.map((s) => SET_NAMES[s]).join(" + ")}</span>
                <span className="game-state">
                  {result ? <b className={`result ${result.toLowerCase()}`}>{result}</b> : g.status === "waiting" ? "Waiting for a friend" : `Turn ${Math.ceil(g.turn / 2)}`}
                </span>
                <span className="game-time">{timeAgo(g.updated)}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export function Home() {
  const [me, setMe] = useState<Me | null>(loadMe);
  const [opponents, setOpponents] = useState<{ spec: string; name: string }[]>([]);
  const [sets, setSets] = useState<string[]>(ALL_SETS);
  const [opponent, setOpponent] = useState("model:phase_f");
  const [seat, setSeat] = useState("random");
  const [err, setErr] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [about, setAbout] = useState(false);
  const [rules, setRules] = useState(false);

  useEffect(() => {
    api.opponents().then(setOpponents, () => {});
  }, []);

  const create = async () => {
    setErr(null);
    try {
      const g = await api.createGame(sets, opponent, seat);
      navigate(`/g/${g.id}`);
    } catch (e) {
      setErr(String((e as Error).message));
    }
  };

  return (
    <div className="home">
      <header className="home-head">
        <h1>Star Realms</h1>
        <div className="me">
          {me && !editing && (
            <>
              Playing as <b>{me.name}</b>
              <button className="btn ghost small" onClick={() => setEditing(true)}>Rename</button>
            </>
          )}
          <button className={`btn small ${rules ? "" : "ghost"}`} onClick={() => { setRules((v) => !v); setAbout(false); }}>
            How to play
          </button>
          <button className={`btn small ${about ? "" : "ghost"}`} onClick={() => { setAbout((v) => !v); setRules(false); }}>
            How does the AI work?
          </button>
        </div>
      </header>

      {rules && <HowToPlay onClose={() => setRules(false)} />}
      {about && <AboutAI onClose={() => setAbout(false)} />}

      {(!me || editing) && (
        <section className="panel">
          <h2>{me ? "Change your name" : "What should your friends call you?"}</h2>
          <NameForm me={me} cta={me ? "Save" : "Continue"} onDone={(m) => { setMe(m); setEditing(false); }} />
        </section>
      )}

      {me && (
        <div className="home-grid">
          <section className="panel">
            <h2>New online game</h2>
            <label className="field">
              <span>Opponent</span>
              <select value={opponent} onChange={(e) => setOpponent(e.target.value)}>
                <option value="human">A friend (get an invite link)</option>
                {opponents.every((o) => o.spec !== "model:phase_f") && <option value="model:phase_f">Uncle Jesse</option>}
                {opponents.map((o) => (
                  <option key={o.spec} value={o.spec}>{opponentName(o.spec, o.name)}</option>
                ))}
              </select>
            </label>
            <div className="field"><span>Trade deck</span><SetPicker sets={sets} onChange={setSets} /></div>
            <label className="field">
              <span>You go</span>
              <select value={seat} onChange={(e) => setSeat(e.target.value)}>
                <option value="random">Random</option>
                <option value="first">First</option>
                <option value="second">Second</option>
              </select>
            </label>
            <button className="btn primary wide" onClick={create}>Create game</button>
            {err && <div className="error">{err}</div>}
          </section>

          <GameList />
        </div>
      )}
    </div>
  );
}
