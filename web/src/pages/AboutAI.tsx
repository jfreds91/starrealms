import { useEffect, useRef, useState } from "react";
import { renderMarkdown } from "../markdown";

/** The note in `web/public/ai/uncle-jesse.md`, fetched so an edit shows up on refresh. */
export function AboutAI({ onClose }: { onClose(): void }) {
  const [body, setBody] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    let alive = true;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close.current();
    };
    window.addEventListener("keydown", onKey);
    fetch("/ai/uncle-jesse.md", { cache: "no-cache" })
      .then((r) => {
        if (!r.ok) throw new Error(`Could not load the note (${r.status})`);
        return r.text();
      })
      .then((t) => alive && setBody(t))
      .catch((e) => alive && setErr(String((e as Error).message)));
    return () => {
      alive = false;
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, []);
  return (
    <div className="splash-backdrop" onClick={() => close.current()}>
      <div className="splash" role="dialog" aria-modal="true" aria-label="How the AI works" onClick={(e) => e.stopPropagation()}>
        <div className="about-bar">
          <button className="btn ghost small" onClick={() => close.current()}>Close</button>
        </div>
        {err && <p className="error">{err}</p>}
        {body == null && !err && <p className="muted">Loading…</p>}
        {body != null && <article className="about-body">{renderMarkdown(body)}</article>}
      </div>
    </div>
  );
}
