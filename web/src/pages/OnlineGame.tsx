import { useEffect, useRef, useState } from "react";
import { api, loadMe, socketUrl, type Me } from "../api";
import { Link, navigate } from "../router";
import { Table } from "../table/Table";
import { seatName, type GameMeta, type View } from "../types";
import { NameForm } from "./Home";

interface StateMsg {
  type: "state";
  game: GameMeta;
  seat: number | null;
  n: number;
  view: View;
  /** The viewer's chance to win, per the server's meter model. */
  meter: number | null;
  can_undo: boolean;
}

type Conn = "connecting" | "open" | "lost";

/** Live connection to a server game. Reconnects with backoff; every message is a full state. */
function useGameSocket(id: string, me: Me | null) {
  const [state, setState] = useState<StateMsg | null>(null);
  const [conn, setConn] = useState<Conn>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    let closed = false;
    let retry = 0;
    let timer: number | undefined;
    const connect = () => {
      const s = new WebSocket(socketUrl(id));
      ws.current = s;
      s.onopen = () => {
        retry = 0;
        setConn("open");
      };
      s.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.type === "state") {
          setState(m);
          setBusy(false);
        } else if (m.type === "error") {
          setError(m.message);
          setBusy(false);
        }
      };
      s.onclose = () => {
        if (closed) return;
        setBusy(false);
        setConn("lost");
        timer = window.setTimeout(connect, Math.min(8000, 500 * 2 ** retry++));
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      ws.current?.close();
    };
  }, [id, me?.token]);

  const send = (msg: object) => {
    if (ws.current?.readyState !== WebSocket.OPEN) return;
    setError(null);
    setBusy(true);
    ws.current.send(JSON.stringify(msg));
  };
  return { state, conn, error, busy, send, clearError: () => setError(null) };
}

export function OnlineGame({ id, join }: { id: string; join: boolean }) {
  const [me, setMe] = useState<Me | null>(loadMe);
  const { state, conn, error, busy, send } = useGameSocket(id, me);
  const [joinErr, setJoinErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const game = state?.game;
  const openSeat = game?.seats.some((s) => s.kind === "open");
  const mySeat = state?.seat ?? null;

  const doJoin = async () => {
    try {
      await api.join(id);
      navigate(`/g/${id}`, true);
    } catch (e) {
      setJoinErr(String((e as Error).message));
    }
  };

  // Invite links (?join) take the open seat automatically once we have a name.
  useEffect(() => {
    if (join && me && state && openSeat && mySeat == null) doJoin();
    else if (join && state && (!openSeat || mySeat != null)) navigate(`/g/${id}`, true);
  }, [join, me, state?.n, openSeat, mySeat]);

  const myTurn = !!state && mySeat != null && state.game.status === "active" && state.view.active === mySeat;
  useEffect(() => {
    document.title = myTurn ? "● Your turn · Star Realms" : "Star Realms";
    return () => {
      document.title = "Star Realms";
    };
  }, [myTurn]);

  if (!state) {
    return <div className="center-msg">{conn === "lost" ? "Can't open this game (check the link). Retrying…" : "Connecting…"}</div>;
  }
  if (join && !me) {
    return (
      <div className="center-msg">
        <div className="panel narrow">
          <h2>You're invited to a game of Star Realms</h2>
          <p className="muted">Pick a name to take your seat.</p>
          <NameForm me={null} cta="Join game" onDone={setMe} />
        </div>
      </div>
    );
  }

  const names = game!.seats.map((s, i) =>
    s.kind === "open" ? "Waiting…" : i === mySeat ? s.name : seatName(s, "Opponent"),
  ) as [string, string];
  const online = game!.seats.map((s) => s.online) as [boolean, boolean];
  const invite = `${location.origin}/g/${id}?join`;

  const notice = (
    <>
      {conn !== "open" && <span className="warn">Reconnecting…</span>}
      {error && <span className="error">{error}</span>}
      {joinErr && <span className="error">{joinErr}</span>}
      {openSeat && mySeat != null && (
        <span className="invite">
          Waiting for your friend.
          <button
            className="btn small"
            onClick={() => {
              navigator.clipboard?.writeText(invite);
              setCopied(true);
            }}
          >
            {copied ? "Copied!" : "Copy invite link"}
          </button>
        </span>
      )}
      {openSeat && mySeat == null && me && (
        <button className="btn primary small" onClick={doJoin}>Take the open seat</button>
      )}
      {mySeat == null && !openSeat && <span className="muted">Spectating</span>}
    </>
  );

  return (
    <Table
      view={state.view}
      names={names}
      online={online}
      spectator={mySeat == null}
      interactive={myTurn && !busy && conn === "open"}
      onAct={(action) => send({ type: "act", action, n: state.n })}
      onPlayAll={() => send({ type: "play_all", n: state.n })}
      onUndo={state.can_undo ? () => send({ type: "undo", n: state.n }) : undefined}
      meter={state.meter}
      notice={notice}
      menu={
        <>
          <Link to="/" className="btn ghost small">Home</Link>
          {mySeat != null && game!.status !== "finished" && (
            <button
              className="btn ghost small"
              onClick={() => confirm("Resign this game?") && send({ type: "resign" })}
            >
              Resign
            </button>
          )}
        </>
      }
      endActions={
        <>
          {game!.resigned != null && (
            <div className="muted">{game!.resigned === mySeat ? "You resigned." : `${names[game!.resigned]} resigned.`}</div>
          )}
          <Link to={`/replay/${id}`} className="btn primary">Watch replay</Link>
          <Link to="/" className="btn ghost">Home</Link>
        </>
      }
      winner={game!.winner}
    />
  );
}
