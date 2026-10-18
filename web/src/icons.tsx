// Small resource glyphs drawn for this project (the art assets are cards only).

export function TradeIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <circle cx="12" cy="12" r="10" fill="#e9b949" stroke="#8a6414" strokeWidth="1.5" />
      <circle cx="12" cy="12" r="6.2" fill="none" stroke="#8a6414" strokeWidth="1.3" />
      <path d="M12 7.5v9M9.5 10h5M9.5 14h5" stroke="#8a6414" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

export function CombatIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path
        d="M12 1.8l2.3 5.3 5.6-1.7-2.2 5.4 4.6 3.4-5.6 1.2.6 5.8-5.3-3.1-5.3 3.1.6-5.8-5.6-1.2 4.6-3.4-2.2-5.4 5.6 1.7z"
        fill="#e5533d"
        stroke="#7c1d12"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function AuthorityIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path
        d="M12 2l8 3.2v6.1c0 5-3.4 9.1-8 10.7-4.6-1.6-8-5.7-8-10.7V5.2z"
        fill="#3fb37f"
        stroke="#145c3c"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path d="M12 7v9M7.5 11.5h9" stroke="#e8fff2" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function DeckIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <rect x="6" y="3" width="13" height="17" rx="2" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M4 6.5V19a2 2 0 002 2h10" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

export function HandIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <rect x="3" y="6" width="9" height="13" rx="1.5" transform="rotate(-12 7.5 12.5)" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <rect x="11" y="5" width="9" height="13" rx="1.5" transform="rotate(10 15.5 11.5)" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}
