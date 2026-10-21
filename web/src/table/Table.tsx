import { useEffect, useMemo, useRef, useState } from "react";
import { groupCards, useCatalog } from "../catalog";
import { AuthorityIcon, CombatIcon, DeckIcon, HandIcon, TradeIcon } from "../icons";
import type { ActionView, CardId, LogEntry, SeatView, Target, View } from "../types";
import { Card, CardBack, CardFace } from "./Card";

export interface TableProps {
  view: View;
  /** Display names by seat. */
  names: [string, string];
  online?: [boolean, boolean];
  /** False while waiting on the server, or when it is not the viewer's move. */
  interactive: boolean;
  onAct(id: number): void;
  onPlayAll(): void;
  /** Take back the action that opened the current choice, when the server allows it. */
  onUndo?: () => void;
  /** The viewer's chance to win (0..1), when the server runs a meter model. */
  meter?: number | null;
  spectator?: boolean;
  /** Shown under the turn indicator (connection state, invite link, errors). */
  notice?: React.ReactNode;
  /** Buttons for the menu bar (resign, leave). */
  menu?: React.ReactNode;
  /** Buttons for the game-over panel (replay, rematch). */
  endActions?: React.ReactNode;
  /** Result decided outside the engine (resignation): winning seat, or -1 for a draw. */
  winner?: number | null;
}

type Pile = { title: string; cards: CardId[]; picks?: ActionView[] };

function sameTarget(a: Target, b: Target): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** Cards that just left the trade row (or the Explorer pile) on someone else's move. */
type Taken = { slots: Map<number, CardId>; explorer: boolean; key: number };

const TAKEN_MS = 600;

/** While it isn't the viewer's move, keeps a card taken from the trade row on screen briefly so it can be seen leaving. */
function useTaken(v: View, watching: boolean): Taken | null {
  const [taken, setTaken] = useState<Taken | null>(null);
  const prev = useRef(v);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => {
    const before = prev.current;
    prev.current = v;
    if (!watching || before === v) return;
    const slots = new Map<number, CardId>();
    before.trade_row.forEach((c, k) => {
      if (c != null && c !== v.trade_row[k]) slots.set(k, c);
    });
    const explorer = v.explorers < before.explorers;
    if (!slots.size && !explorer) return;
    setTaken({ slots, explorer, key: Date.now() });
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setTaken(null), TAKEN_MS);
  }, [v, watching]);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return taken;
}

export function Table(p: TableProps) {
  const { view: v } = p;
  const cat = useCatalog();
  const [pile, setPile] = useState<Pile | null>(null);
  const winner = p.winner ?? (v.outcome == null ? null : v.outcome === "Draw" ? -1 : v.outcome.Win);
  const over = winner != null;
  const actions = p.interactive && !over ? v.actions : [];
  const at = (t: Target) => actions.filter((a) => sameTarget(a.target, t));
  const me = v.seat, them = 1 - v.seat;
  const myTurn = v.active === me && !over && !p.spectator;
  const taken = useTaken(v, v.active !== me || !!p.spectator);
  const act = (id: number) => {
    setPile(null);
    p.onAct(id);
  };
  const undo = p.interactive && !over && v.prompt ? p.onUndo : undefined;
  useEffect(() => {
    if (!undo) return;
    const k = (e: KeyboardEvent) => e.key === "Escape" && !pile && undo();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [undo, pile]);

  // Opponent bases are addressed by base index, not by in-play position.
  const oppSlots = useMemo(() => {
    let b = 0;
    return v.opponent.in_play.map((s) => (s.is_base ? b++ : -1));
  }, [v.opponent.in_play]);

  const promptButtons = actions.filter((a) => a.target.zone === "none" && a.group !== "end");
  const discardPicks = actions.filter((a) => a.target.zone === "discard");
  const playable = (v.you.hand ?? []).filter((c) => actions.some((a) => a.group === "play" && sameTarget(a.target, { zone: "hand", card: c })));
  const endTurn = actions.find((a) => a.group === "end");
  const attackFace = actions.find((a) => a.target.zone === "opponent");

  const outcomeText =
    winner == null ? null : winner === -1 ? "Draw" : !p.spectator && winner === me ? "You win!" : `${p.names[winner]} wins`;
  const ownName = p.spectator ? p.names[me] : "You";

  const choosing = myTurn && !!v.prompt;

  return (
    <div className="table-layout">
      <main className={`board ${choosing ? "choosing" : ""}`}>
        <section className="zone opp-zone">
          <PlayerPanel
            name={p.names[them]}
            seat={v.opponent}
            online={p.online?.[them]}
            active={v.active === them && !over}
            onPile={setPile}
          />
          <div className="in-play opp-play">
            {v.opponent.in_play.length === 0 && <div className="empty-hint">No bases</div>}
            {v.opponent.in_play.map((s, i) => (
              <Card
                key={i}
                id={s.card}
                copy={s.copy}
                landscape={s.is_base}
                actions={s.is_base ? at({ zone: "opp_base", slot: oppSlots[i] }) : []}
                onAct={act}
                badge={s.is_base ? <DefenseBadge def={s.defense} outpost={s.outpost} /> : null}
              />
            ))}
          </div>
          <LastMoves log={v.log} seat={them} turn={v.turn} name={p.names[them]} show={v.active === them} />
        </section>

        <section className="zone row-zone">
          <Scoreboard
            view={v}
            names={[p.names[me], p.names[them]]}
            ownName={ownName}
            over={over}
            meter={p.meter ?? null}
            attack={attackFace}
            onAct={act}
          />
          <div className="pile-stack">
            <Card
              key={taken?.explorer ? taken.key : "explorer"}
              className={`pile ${taken?.explorer ? "taken" : ""}`}
              id={cat.explorer}
              actions={at({ zone: "explorer" })}
              onAct={act}
              dim={v.explorers === 0}
            />
            <div className="pile-label">Explorer</div>
          </div>
          <div className="trade-row">
            {v.trade_row.map((c, k) => {
              const gone = taken?.slots.get(k);
              if (gone != null) return <Card key={`taken-${taken!.key}-${k}`} id={gone} className="taken" />;
              return c == null ? (
                <div key={k} className="card-wrap"><div className="card empty-slot" /></div>
              ) : (
                <Card key={`${k}-${c}`} id={c} className="row-card" actions={at({ zone: "row", slot: k })} onAct={act} />
              );
            })}
          </div>
          <button className="pile-stack pile-button" onClick={() => setPile({ title: "Scrap heap", cards: v.scrap_heap })}>
            {v.scrap_heap.length ? (
              <Card id={v.scrap_heap[v.scrap_heap.length - 1]} landscape={false} className="pile" dim />
            ) : (
              <div className="card-wrap pile"><div className="card empty-slot" /></div>
            )}
            <div className="pile-label">Scrap heap · {v.scrap_heap.length}</div>
          </button>
        </section>

        <section className="zone my-zone">
          <div className="in-play my-play">
            {v.you.in_play.length === 0 && <div className="empty-hint">{myTurn ? "Play cards from your hand" : "No bases"}</div>}
            {v.you.in_play.map((s, j) => (
              <Card
                key={j}
                id={s.card}
                copy={s.copy}
                landscape={s.is_base}
                actions={at({ zone: "own", slot: j })}
                buttons
                onAct={act}
                badge={s.is_base ? <DefenseBadge def={s.defense} outpost={s.outpost} /> : null}
              />
            ))}
          </div>

          <div className="controls">
            <div className="turn-info">
              {outcomeText ? (
                <span className="turn-pill over">{outcomeText}</span>
              ) : (
                <span className={`turn-pill ${myTurn ? "mine" : ""}`}>
                  {p.spectator ? `${p.names[v.active]}'s turn` : myTurn ? "Your turn" : `${p.names[them]}'s turn`}
                </span>
              )}
              <span className="turn-number">Turn {Math.ceil(v.turn / 2)}</span>
              {v.prompt && myTurn && <div className="notice small">{p.notice}</div>}
            </div>
            <div className="center-slot">
              {v.prompt && myTurn ? (
                <div className="prompt">
                  {v.pending_source != null && (
                    <div className="prompt-source"><CardFace id={v.pending_source} /></div>
                  )}
                  <div className="prompt-body">
                    <div className="prompt-text">{v.prompt}</div>
                    {discardPicks.length > 0 && (
                      <div className="prompt-picks">
                        {discardPicks.map((a) => (
                          <Card key={a.id} id={(a.target as { card: CardId }).card} actions={[a]} onAct={act} className="small" />
                        ))}
                      </div>
                    )}
                    {(promptButtons.length > 0 || undo) && (
                      <div className="prompt-buttons">
                        {promptButtons.map((a) => (
                          <button key={a.id} className={a.group === "skip" ? "btn" : "btn primary"} onClick={() => act(a.id)}>
                            {a.group === "skip" && a.label === "Skip" ? "Choose none" : a.label}
                          </button>
                        ))}
                        {undo && (
                          <button className="btn ghost" onClick={undo} title="Take back the card or ability that asked for this choice (Esc)">
                            Cancel
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="notice">{p.notice}</div>
              )}
            </div>
            <div className="control-buttons">
              {playable.length > 1 && !v.pending && (
                <button className="btn" onClick={p.onPlayAll} title="Play every card that doesn't need a choice">
                  Play all
                </button>
              )}
              {attackFace && (
                <button className="btn danger" onClick={() => act(attackFace.id)}>
                  {attackFace.label}
                </button>
              )}
              {endTurn && <EndTurn action={endTurn} view={v} onAct={act} />}
            </div>
          </div>

          <div className="hand-row">
            <PlayerPanel
              name={ownName}
              seat={v.you}
              online={p.spectator ? p.online?.[me] : undefined}
              active={v.active === me && !over}
              onPile={(pl) => setPile(pl.title === "Discard pile" && discardPicks.length ? { ...pl, picks: discardPicks } : pl)}
            />
            <div className="hand">
              {v.you.hand
                ? v.you.hand.map((c, i) => <Card key={`${c}-${i}`} id={c} actions={at({ zone: "hand", card: c })} onAct={act} />)
                : Array.from({ length: v.you.hand_count }, (_, i) => <CardBack key={i} />)}
              {v.you.hand && v.you.hand.length === 0 && <div className="empty-hint">Hand empty</div>}
            </div>
          </div>
        </section>
      </main>

      <aside className="side">
        <div className="side-menu">{p.menu}</div>
        <Log entries={v.log} names={p.names} viewer={p.spectator ? null : me} />
      </aside>

      {outcomeText && (
        <div className="game-over">
          <div className="game-over-title">{outcomeText}</div>
          <div className="game-over-sub">
            {p.names[me]} {v.you.authority} · {p.names[them]} {v.opponent.authority}
          </div>
          <div className="game-over-actions">{p.endActions}</div>
        </div>
      )}

      {pile && <PileModal pile={pile} onClose={() => setPile(null)} onAct={act} />}
    </div>
  );
}

function PlayerPanel(props: {
  name: string;
  seat: SeatView;
  online?: boolean;
  active: boolean;
  onPile(p: Pile): void;
}) {
  const s = props.seat;
  const top = s.discard[s.discard.length - 1];
  const handHidden = s.hand == null;
  const deckTitle = handHidden ? `${props.name}: deck and hand (unordered)` : `${props.name}: deck (unordered)`;
  return (
    <div className={`player-panel ${props.active ? "active" : ""}`}>
      <div className="player-name">
        {props.online !== undefined && <span className={`dot ${props.online ? "on" : ""}`} title={props.online ? "Connected" : "Away"} />}
        {props.name}
      </div>
      <div className="player-stats">
        {handHidden && (
          <span title="Cards in hand"><HandIcon /> {s.hand_count}</span>
        )}
        <button
          className="stat-button"
          title={handHidden ? "Deck: click to see what's in the deck and hand together" : "Deck: click to see what's left in it"}
          disabled={!s.deck_contents}
          onClick={() => s.deck_contents && props.onPile({ title: deckTitle, cards: s.deck_contents })}
        >
          <DeckIcon /> {s.deck_count}
        </button>
        <button className="stat-button discard-stat" onClick={() => props.onPile({ title: "Discard pile", cards: s.discard })} title="Discard pile">
          {top != null ? <span className="mini-card"><CardFace id={top} /></span> : <span className="mini-card empty" />}
          {s.discard.length}
        </button>
      </div>
      {s.owed_discards > 0 && (
        <div className="owed" title="Discarded at the start of their next turn">
          Discards {s.owed_discards} next turn
        </div>
      )}
    </div>
  );
}

/** Both authorities, the win-chance meter, and the trade and combat of whoever is moving. */
function Scoreboard(props: {
  view: View;
  /** [viewer, opponent] */
  names: [string, string];
  ownName: string;
  over: boolean;
  meter: number | null;
  attack?: ActionView;
  onAct(id: number): void;
}) {
  const v = props.view;
  const me = v.seat, them = 1 - v.seat;
  const pools = (seat: number) =>
    v.active === seat && !props.over ? (
      <div className="pools">
        <Pool icon={<TradeIcon size={20} />} value={v.trade} label="Trade" />
        <Pool icon={<CombatIcon size={20} />} value={v.combat} label="Combat" />
      </div>
    ) : null;
  return (
    <div className="scoreboard">
      <div className={`score-row theirs ${v.active === them && !props.over ? "active" : ""}`}>
        <div className="score-name">{props.names[1]}</div>
        <button
          className={`authority ${props.attack ? "attackable" : ""}`}
          disabled={!props.attack}
          title={props.attack?.label ?? "Authority"}
          onClick={() => props.attack && props.onAct(props.attack.id)}
        >
          <AuthorityIcon size={24} />
          <AnimatedNumber value={v.opponent.authority} />
        </button>
        {pools(them)}
      </div>
      {props.meter != null && <WinMeter you={props.meter} />}
      <div className={`score-row mine ${v.active === me && !props.over ? "active" : ""}`}>
        <div className="score-name">{props.ownName}</div>
        <div className="authority" title="Authority">
          <AuthorityIcon size={24} />
          <AnimatedNumber value={v.you.authority} />
        </div>
        {pools(me)}
      </div>
    </div>
  );
}

/** Split bar: the viewer's chance to win (left, green) against the opponent's (right, red). */
function WinMeter({ you }: { you: number }) {
  const pct = Math.round(you * 100);
  return (
    <div className="win-meter" title="Chance to win, estimated by the strongest model's value head">
      <span className="win-pct mine">{pct}%</span>
      <div className="win-bar">
        <div className="win-fill" style={{ width: `${you * 100}%` }} />
      </div>
      <span className="win-pct theirs">{100 - pct}%</span>
    </div>
  );
}

function DefenseBadge({ def, outpost }: { def: number; outpost: boolean }) {
  return (
    <span className={`def-badge ${outpost ? "outpost" : ""}`} title={outpost ? "Outpost: must be destroyed first" : "Base defense"}>
      {def}
    </span>
  );
}

function Pool({ icon, value, label }: { icon: React.ReactNode; value: number; label: string }) {
  return (
    <div className={`pool ${value > 0 ? "has" : ""}`} title={label}>
      {icon}
      <AnimatedNumber value={value} />
    </div>
  );
}

/** A number that briefly shows its change (+3 / -5). */
function AnimatedNumber({ value }: { value: number }) {
  const prev = useRef(value);
  const [delta, setDelta] = useState<{ d: number; k: number } | null>(null);
  useEffect(() => {
    const d = value - prev.current;
    prev.current = value;
    if (d !== 0) {
      setDelta({ d, k: Date.now() });
      const t = window.setTimeout(() => setDelta(null), 1400);
      return () => window.clearTimeout(t);
    }
  }, [value]);
  return (
    <span className="num">
      {value}
      {delta && (
        <span key={delta.k} className={`delta ${delta.d > 0 ? "up" : "down"}`}>
          {delta.d > 0 ? `+${delta.d}` : delta.d}
        </span>
      )}
    </span>
  );
}

/** End turn, with a second click required when that would waste cards or trade. */
function EndTurn({ action, view, onAct }: { action: ActionView; view: View; onAct(id: number): void }) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = window.setTimeout(() => setArmed(false), 3000);
    return () => window.clearTimeout(t);
  }, [armed]);
  useEffect(() => setArmed(false), [view]);
  const unplayed = view.actions.filter((a) => a.group === "play").length;
  const canBuy = view.actions.some((a) => a.group === "buy");
  const warn =
    unplayed > 0 ? `You still have ${unplayed === 1 ? "a card" : "cards"} to play` : canBuy ? `${view.trade} trade unspent` : null;
  return (
    <button
      className={`btn end ${armed ? "armed" : ""}`}
      onClick={() => {
        if (warn && !armed) setArmed(true);
        else onAct(action.id);
      }}
    >
      {armed ? `${warn}: end anyway?` : action.label}
    </button>
  );
}

/** The opponent's actions this turn, so a bot's (or friend's) turn can be followed. */
function LastMoves({ log, seat, turn, name, show }: { log: LogEntry[]; seat: number; turn: number; name: string; show: boolean }) {
  const recent = log.filter((e) => e.seat === seat && e.turn >= turn - (show ? 0 : 1)).slice(-4);
  if (!recent.length) return <div className="last-moves" />;
  return (
    <div className="last-moves">
      <span className="last-moves-who">{name}:</span>
      {recent.map((e, i) => (
        <span key={i} className="last-move">{e.text}</span>
      ))}
    </div>
  );
}

function Log({ entries, names, viewer }: { entries: LogEntry[]; names: [string, string]; viewer: number | null }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => end.current?.scrollIntoView({ block: "end" }), [entries.length]);
  const turns: { turn: number; seat: number; items: string[] }[] = [];
  for (const e of entries) {
    const last = turns[turns.length - 1];
    if (last && last.turn === e.turn) last.items.push(e.text);
    else turns.push({ turn: e.turn, seat: e.seat, items: [e.text] });
  }
  return (
    <div className="log">
      <div className="log-title">Game log</div>
      <div className="log-body">
        {turns.map((t) => (
          <div key={t.turn} className={`log-turn ${t.seat === viewer ? "mine" : ""}`}>
            <div className="log-who">
              {t.seat === viewer ? "You" : names[t.seat]} <span>turn {Math.ceil(t.turn / 2)}</span>
            </div>
            {t.items.map((x, i) => (
              <div key={i} className="log-item">{x}</div>
            ))}
          </div>
        ))}
        <div ref={end} />
      </div>
    </div>
  );
}

function PileModal({ pile, onClose, onAct }: { pile: Pile; onClose(): void; onAct(id: number): void }) {
  const cat = useCatalog();
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  const groups = groupCards(cat, pile.cards);
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span>{pile.title} · {pile.cards.length}</span>
          <button className="btn ghost" onClick={onClose}>Close</button>
        </div>
        {groups.length === 0 ? (
          <div className="empty-hint">Empty</div>
        ) : (
          <div className="modal-grid">
            {groups.map(([id, n]) => (
              <Card
                key={id}
                id={id}
                landscape={cat.cards[id].kind === "base"}
                actions={pile.picks?.filter((a) => a.target.zone === "discard" && a.target.card === id)}
                onAct={onAct}
                badge={n > 1 ? <span className="count-badge">×{n}</span> : null}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
