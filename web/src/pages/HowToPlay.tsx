import { useEffect, useRef } from "react";

/** Short rules for the home and invite screens. */
export function HowToPlay({ onClose }: { onClose(): void }) {
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close.current();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, []);
  return (
    <div className="splash-backdrop" onClick={() => close.current()}>
      <div className="splash" role="dialog" aria-modal="true" aria-label="How to play" onClick={(e) => e.stopPropagation()}>
        <div className="about-bar">
          <button className="btn ghost small" onClick={() => close.current()}>Close</button>
        </div>
        <article className="about-body">
          <h1>How to play</h1>
          <p>Two players start at 50 Authority. Get your opponent to 0 and you win.</p>
          <p>
            Your deck starts as 8 Scouts (1 trade each) and 2 Vipers (1 combat each).
            The first player draws 3 cards; the second draws 5. Every later turn draws 5.
          </p>
          <h2>Your turn</h2>
          <ol>
            <li>Play ships from your hand and use bases you already have. Trade and combat pile up until you spend them.</li>
            <li>Spend trade to buy from the five-card trade row, or buy an Explorer for 2. Purchases go to your discard pile.</li>
            <li>Spend combat to destroy enemy bases. An outpost has to fall before you can hit the player or their other bases. Combat left over at the end of the turn hits their Authority, unless an outpost is still standing.</li>
            <li>End the turn. Played ships, and anything still in your hand, go to the discard. Bases stay. Draw 5.</li>
          </ol>
          <p>
            Two cards of the same faction in play switch on that faction's ally abilities.
            Scraping a card removes it from your deck, which is how the starter Scouts and Vipers leave the game.
            When your deck runs out, the discard is shuffled into a new one.
          </p>
        </article>
      </div>
    </div>
  );
}
