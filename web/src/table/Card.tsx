import { useState } from "react";
import { FACTION_NAMES, useCatalog } from "../catalog";
import type { ActionView, CardId } from "../types";
import { usePreview } from "./Preview";

/** The printed card: art from /assets, or a text rendering when art is missing. */
export function CardFace({ id }: { id: CardId }) {
  const cat = useCatalog();
  const c = cat.cards[id];
  const [broken, setBroken] = useState(false);
  if (!broken) {
    return (
      <img
        className="card-art"
        src={`/assets/${c.image}`}
        alt={c.name}
        draggable={false}
        loading="lazy"
        onError={() => setBroken(true)}
      />
    );
  }
  return (
    <div className={`card-text faction-${c.faction}`}>
      <div className="card-text-head">
        <span className="card-text-name">{c.name}</span>
        {c.cost > 0 && <span className="card-text-cost">{c.cost}</span>}
      </div>
      <div className="card-text-faction">{FACTION_NAMES[c.faction]}</div>
      <div className="card-text-body">
        {c.primary && <p>{c.primary}</p>}
        {c.ally && <p><b>Ally:</b> {c.ally}</p>}
        {c.ally2 && <p><b>Double ally:</b> {c.ally2}</p>}
        {c.scrap && <p><b>Scrap:</b> {c.scrap}</p>}
      </div>
      {c.kind === "base" && <div className="card-text-def">{c.defense}{c.outpost ? " outpost" : ""}</div>}
    </div>
  );
}

export const GROUP_CLASS: Record<string, string> = {
  play: "act-play",
  buy: "act-buy",
  ability: "act-ability",
  scrap_ability: "act-scrap",
  attack: "act-attack",
  pick: "act-pick",
};

/** Short name for an ability button under an in-play card; the full text is the tooltip. */
export function abilityLabel(a: ActionView): string {
  if (a.group === "scrap_ability") return "Scrap";
  if (a.group === "attack") return "Attack";
  if (a.group === "pick") return a.label.split(" ")[0];
  if (a.label.includes("(double ally)")) return "Ally ×2";
  if (a.label.includes("(ally)")) return "Ally";
  return "Use";
}

interface CardProps {
  id: CardId;
  /** Engine actions that target this card. One action: clicking the card does it. More: buttons. */
  actions?: ActionView[];
  onAct?: (id: number) => void;
  /** Defaults to the card's printed orientation (bases are landscape). */
  landscape?: boolean;
  /** Card shown on top of `id`, e.g. what a Stealth Needle is copying. */
  copy?: CardId | null;
  /** Ability buttons are shown even for a single action (in-play cards). */
  buttons?: boolean;
  dim?: boolean;
  badge?: React.ReactNode;
  className?: string;
  /** Extra hover text. */
  note?: string;
}

export function Card(p: CardProps) {
  const preview = usePreview();
  const cat = useCatalog();
  const acts = p.actions ?? [];
  const single = acts.length === 1 && !p.buttons ? acts[0] : null;
  const glow = acts.length ? GROUP_CLASS[acts[0].group] ?? "act-pick" : "";
  const landscape = p.landscape ?? cat.cards[p.id].kind === "base";
  return (
    <div className={`card-wrap ${landscape ? "landscape" : ""} ${p.className ?? ""}`}>
      <div
        className={`card ${glow ? "actionable " + glow : ""} ${p.dim ? "dim" : ""}`}
        role={single ? "button" : undefined}
        tabIndex={single ? 0 : undefined}
        title={single ? single.label : p.note}
        onClick={single ? () => p.onAct?.(single.id) : undefined}
        onKeyDown={single ? (e) => (e.key === "Enter" || e.key === " ") && p.onAct?.(single.id) : undefined}
        onMouseEnter={() => preview.show(p.copy ?? p.id)}
        onMouseLeave={() => preview.hide()}
      >
        <CardFace id={p.id} />
        {p.copy != null && (
          <div className="card-copy" title="Copying">
            <CardFace id={p.copy} />
          </div>
        )}
        {p.badge}
      </div>
      {acts.length > 0 && !single && (
        <div className="card-buttons">
          {acts.map((a) => (
            <button key={a.id} className={`chip ${GROUP_CLASS[a.group] ?? ""}`} title={a.label} onClick={() => p.onAct?.(a.id)}>
              {abilityLabel(a)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Face-down card back. */
export function CardBack({ label, className }: { label?: React.ReactNode; className?: string }) {
  return (
    <div className={`card-wrap ${className ?? ""}`}>
      <div className="card card-back">{label != null && <span className="card-back-label">{label}</span>}</div>
    </div>
  );
}
