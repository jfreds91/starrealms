import { createContext, useContext, useMemo, useRef, useState } from "react";
import { FACTION_NAMES, useCatalog } from "../catalog";
import type { CardId } from "../types";
import { CardFace } from "./Card";

interface PreviewApi {
  show(id: CardId): void;
  hide(): void;
}

const Ctx = createContext<PreviewApi>({ show() {}, hide() {} });

export function usePreview(): PreviewApi {
  return useContext(Ctx);
}

/** Hovering any card shows it large, with its rules text, over the side panel. */
export function PreviewProvider({ children }: { children: React.ReactNode }) {
  const [id, setId] = useState<CardId | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const api = useMemo<PreviewApi>(
    () => ({
      show(c) {
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setId(c), 180);
      },
      hide() {
        window.clearTimeout(timer.current);
        setId(null);
      },
    }),
    [],
  );
  return (
    <Ctx.Provider value={api}>
      {children}
      {id != null && <PreviewPanel id={id} />}
    </Ctx.Provider>
  );
}

function PreviewPanel({ id }: { id: CardId }) {
  const c = useCatalog().cards[id];
  return (
    <div className="preview" aria-hidden>
      <div className={`preview-card ${c.kind === "base" ? "landscape" : ""}`}>
        <CardFace id={id} />
      </div>
      <div className="preview-text">
        <div className="preview-name">
          {c.name}
          {c.cost > 0 && <span className="preview-cost">{c.cost}</span>}
        </div>
        <div className={`preview-meta faction-fg-${c.faction}`}>
          {FACTION_NAMES[c.faction]} {c.kind}
          {c.kind === "base" && ` · defense ${c.defense}${c.outpost ? " · outpost" : ""}`}
        </div>
        {c.primary && <p>{c.primary}</p>}
        {c.ally && <p><b>Ally:</b> {c.ally}</p>}
        {c.ally2 && <p><b>Double ally:</b> {c.ally2}</p>}
        {c.scrap && <p><b>Scrap:</b> {c.scrap}</p>}
      </div>
    </div>
  );
}
