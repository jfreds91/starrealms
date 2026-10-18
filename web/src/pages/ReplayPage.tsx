import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { Link } from "../router";
import { Table } from "../table/Table";
import { seatName, type Replay, type View } from "../types";
import { loadEngine } from "../wasm";
import { LOCAL_REPLAY_KEY } from "./LocalGame";

type Viewer = InstanceType<Awaited<ReturnType<typeof loadEngine>>["ReplayViewer"]>;

interface Loaded {
  viewer: Viewer;
  names: [string, string];
  resigned: number | null;
  turns: number[];
}

/** Step through a finished game with both hands visible from either seat. */
export function ReplayPage({ id }: { id: string | null }) {
  const [data, setData] = useState<Loaded | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [n, setN] = useState(0);
  const [seat, setSeat] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    let alive = true;
    let viewer: Viewer | null = null;
    (async () => {
      let replay: Replay, names: [string, string], resigned: number | null = null;
      if (id) {
        const r = await api.replay(id);
        replay = r.replay;
        names = [seatName(r.seats[0], "Player 1"), seatName(r.seats[1], "Player 2")];
        resigned = r.resigned;
      } else {
        const s = JSON.parse(sessionStorage.getItem(LOCAL_REPLAY_KEY) ?? "null");
        if (!s) throw new Error("no local replay");
        replay = s.replay;
        names = s.names;
      }
      const wasm = await loadEngine();
      viewer = new wasm.ReplayViewer(JSON.stringify(replay));
      if (!alive) return viewer.free();
      setData({ viewer, names, resigned, turns: Array.from(viewer.turn_starts()) });
      setN(viewer.len());
    })().catch((e) => alive && setErr(String((e as Error).message ?? e)));
    return () => {
      alive = false;
      viewer?.free();
    };
  }, [id]);

  const view: View | null = useMemo(() => (data ? JSON.parse(data.viewer.view_at(n, seat, 400)) : null), [data, n, seat]);

  const len = data?.viewer.len() ?? 0;
  useEffect(() => {
    if (!playing) return;
    if (n >= len) return setPlaying(false);
    const t = window.setTimeout(() => setN(n + 1), 350);
    return () => window.clearTimeout(t);
  }, [playing, n, len]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") setN((x) => Math.min(len, x + 1));
      if (e.key === "ArrowLeft") setN((x) => Math.max(0, x - 1));
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [len]);

  if (err) return <div className="center-msg">{err} · <Link to="/">Home</Link></div>;
  if (!data || !view) return <div className="center-msg">Loading replay…</div>;

  const prevTurn = [...data.turns].reverse().find((t) => t < n) ?? 0;
  const nextTurn = data.turns.find((t) => t > n) ?? len;

  return (
    <div className="replay">
      <Table
        view={view}
        names={data.names}
        spectator
        interactive={false}
        onAct={() => {}}
        onPlayAll={() => {}}
        winner={n === len ? (data.resigned != null ? 1 - data.resigned : undefined) : null}
        notice={<span className="muted">Replay · {data.names[seat]}'s view</span>}
        menu={<Link to="/" className="btn ghost small">Home</Link>}
        endActions={<button className="btn" onClick={() => setN(0)}>Restart replay</button>}
      />
      <div className="replay-bar">
        <button className="btn small" onClick={() => setN(0)} title="Start">⏮</button>
        <button className="btn small" onClick={() => setN(prevTurn)} title="Previous turn">⏪</button>
        <button className="btn small" onClick={() => setN(Math.max(0, n - 1))} title="Back one action (←)">◀</button>
        <button className="btn small primary" onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
        <button className="btn small" onClick={() => setN(Math.min(len, n + 1))} title="Forward one action (→)">▶</button>
        <button className="btn small" onClick={() => setN(nextTurn)} title="Next turn">⏩</button>
        <button className="btn small" onClick={() => setN(len)} title="End">⏭</button>
        <input type="range" min={0} max={len} value={n} onChange={(e) => setN(Number(e.target.value))} />
        <span className="replay-pos">{n}/{len}</span>
        <button className="btn small ghost" onClick={() => setSeat(1 - seat)}>View as {data.names[1 - seat]}</button>
      </div>
    </div>
  );
}
