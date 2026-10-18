import { useCallback, useEffect, useRef, useState } from "react";
import { Link, navigate } from "../router";
import { Table } from "../table/Table";
import type { View } from "../types";
import { loadEngine, randomSeed } from "../wasm";

const KEY = "sr.local";
export const LOCAL_REPLAY_KEY = "sr.local.replay";
const BOT_DELAY = 450;

interface Saved {
  replay: string;
  /** Per seat: "" for a human, else a bot name. */
  bots: [string, string];
  names: [string, string];
}

export function savedLocal(): Saved | null {
  try {
    const s = JSON.parse(localStorage.getItem(KEY) ?? "null") as Saved | null;
    return s && Array.isArray(JSON.parse(s.replay).actions) ? s : null;
  } catch {
    return null;
  }
}

/** Create a local game and open it. With `shuffleSeats`, who goes first is random. */
export async function startLocal(sets: string[], bots: [string, string], names: [string, string], shuffleSeats: boolean) {
  const wasm = await loadEngine();
  if (shuffleSeats && Math.random() < 0.5) {
    bots = [bots[1], bots[0]];
    names = [names[1], names[0]];
  }
  const e = new wasm.Engine(JSON.stringify({ sets }), randomSeed());
  const saved: Saved = { replay: e.replay(), bots, names };
  e.free();
  localStorage.setItem(KEY, JSON.stringify(saved));
  window.dispatchEvent(new Event("sr-local-start"));
  navigate("/local");
}

type Engine = InstanceType<Awaited<ReturnType<typeof loadEngine>>["Engine"]>;

export function LocalGame() {
  const [saved, setSaved] = useState(savedLocal);
  const engine = useRef<Engine | null>(null);
  const [view, setView] = useState<View | null>(null);
  const [err, setErr] = useState<string | null>(null);
  /** Hot-seat: whose screen is showing; a hand-off screen hides it when the turn passes. */
  const [shown, setShown] = useState(0);
  const [handoff, setHandoff] = useState(false);

  const humans = saved ? [0, 1].filter((s) => !saved.bots[s]) : [];
  const hotseat = humans.length === 2;

  useEffect(() => {
    const on = () => setSaved(savedLocal());
    window.addEventListener("sr-local-start", on);
    return () => window.removeEventListener("sr-local-start", on);
  }, []);

  const refresh = useCallback(() => {
    const e = engine.current;
    if (!e || !saved) return;
    localStorage.setItem(KEY, JSON.stringify({ ...saved, replay: e.replay() }));
    const seat = hotseat ? e.active() : humans[0] ?? 0;
    if (hotseat) {
      setShown((prev) => {
        if (prev !== seat && !e.is_over()) setHandoff(true);
        return seat;
      });
    }
    setView(JSON.parse(e.view(seat, 300)));
  }, [saved, hotseat, humans[0]]);

  useEffect(() => {
    if (!saved) return;
    let alive = true;
    loadEngine().then((wasm) => {
      if (!alive) return;
      try {
        const e = wasm.Engine.from_replay(saved.replay);
        saved.bots.forEach((b, s) => e.set_bot(s, b));
        engine.current = e;
        setShown(hotseat ? e.active() : humans[0] ?? 0);
        setHandoff(false);
        refresh();
      } catch (x) {
        setErr(String(x));
      }
    });
    return () => {
      alive = false;
      engine.current?.free();
      engine.current = null;
    };
  }, [saved]);

  // Bots move one action at a time so their turn can be followed.
  useEffect(() => {
    const e = engine.current;
    if (!e || !view || !e.bot_to_move()) return;
    const t = window.setTimeout(() => {
      try {
        e.bot_step();
      } catch (x) {
        setErr(String(x));
      }
      refresh();
    }, BOT_DELAY);
    return () => window.clearTimeout(t);
  }, [view, refresh]);

  if (!saved) {
    return (
      <div className="center-msg">
        No local game in progress. <Link to="/">Start one</Link>
      </div>
    );
  }
  if (!view) return <div className="center-msg">{err ?? "Loading engine…"}</div>;

  const act = (id: number) => {
    try {
      engine.current?.step(id);
      setErr(null);
    } catch (x) {
      setErr(String(x));
    }
    refresh();
  };
  const playAll = () => {
    const e = engine.current;
    if (!e) return;
    for (let a = e.next_simple_play(); a >= 0; a = e.next_simple_play()) e.step(a);
    refresh();
  };
  const myTurn = !view.outcome && !saved.bots[view.active];

  return (
    <>
      <Table
        view={view}
        names={saved.names}
        hotseat={hotseat}
        interactive={myTurn && !handoff}
        onAct={act}
        onPlayAll={playAll}
        notice={err && <span className="error">{err}</span>}
        menu={
          <>
            <Link to="/" className="btn ghost small">Home</Link>
            <button
              className="btn ghost small"
              onClick={() => {
                if (confirm("Abandon this local game?")) {
                  localStorage.removeItem(KEY);
                  navigate("/");
                }
              }}
            >
              Abandon
            </button>
          </>
        }
        endActions={
          <>
            <button
              className="btn primary"
              onClick={() => {
                sessionStorage.setItem(
                  LOCAL_REPLAY_KEY,
                  JSON.stringify({ replay: JSON.parse(engine.current!.replay()), names: saved.names }),
                );
                navigate("/replay/local");
              }}
            >
              Watch replay
            </button>
            <button
              className="btn"
              onClick={() => startLocal(JSON.parse(saved.replay).config.sets, saved.bots, saved.names, !hotseat)}
            >
              Play again
            </button>
            <Link to="/" className="btn ghost">Home</Link>
          </>
        }
      />
      {hotseat && handoff && (
        <div className="handoff">
          <div className="handoff-box">
            <div className="handoff-title">Pass to {saved.names[shown]}</div>
            <button className="btn primary" onClick={() => setHandoff(false)}>
              I'm {saved.names[shown]}: show my hand
            </button>
          </div>
        </div>
      )}
    </>
  );
}
