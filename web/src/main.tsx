import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api";
import { CatalogContext, makeCatalog, type Catalog } from "./catalog";
import { Home } from "./pages/Home";
import { LocalGame } from "./pages/LocalGame";
import { OnlineGame } from "./pages/OnlineGame";
import { ReplayPage } from "./pages/ReplayPage";
import { useRoute } from "./router";
import { PreviewProvider } from "./table/Preview";
import "./styles.css";

function App() {
  const route = useRoute();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.catalog().then((c) => setCatalog(makeCatalog(c)), (e) => setError(String(e)));
  }, []);
  if (error) return <div className="center-msg">Could not reach the server: {error}</div>;
  if (!catalog) return <div className="center-msg">Loading…</div>;
  return (
    <CatalogContext.Provider value={catalog}>
      <PreviewProvider>
        {route.page === "home" && <Home />}
        {route.page === "game" && <OnlineGame key={route.id} id={route.id} join={route.join} />}
        {route.page === "local" && <LocalGame />}
        {route.page === "replay" && <ReplayPage key={route.id} id={route.id} />}
        {route.page === "local-replay" && <ReplayPage id={null} />}
      </PreviewProvider>
    </CatalogContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
