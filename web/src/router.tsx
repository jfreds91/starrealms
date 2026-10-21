import { useEffect, useState } from "react";

export type Route =
  | { page: "home" }
  | { page: "game"; id: string; join: boolean }
  | { page: "replay"; id: string };

export function parse(path: string, search: string): Route {
  const parts = path.split("/").filter(Boolean);
  if (parts[0] === "g" && parts[1]) return { page: "game", id: parts[1], join: new URLSearchParams(search).has("join") };
  if (parts[0] === "replay" && parts[1]) return { page: "replay", id: parts[1] };
  return { page: "home" };
}

export function navigate(to: string, replace = false) {
  if (replace) history.replaceState(null, "", to);
  else history.pushState(null, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(location.pathname, location.search));
  useEffect(() => {
    const on = () => setRoute(parse(location.pathname, location.search));
    window.addEventListener("popstate", on);
    return () => window.removeEventListener("popstate", on);
  }, []);
  return route;
}

export function Link(props: { to: string; className?: string; children: React.ReactNode }) {
  return (
    <a
      href={props.to}
      className={props.className}
      onClick={(e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
        e.preventDefault();
        navigate(props.to);
      }}
    >
      {props.children}
    </a>
  );
}
