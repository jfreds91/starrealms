import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const server = process.env.SR_SERVER ?? "http://127.0.0.1:3030";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: server, ws: true },
      "/assets": server,
    },
  },
  // `/assets` is the server's card-art route.
  build: { target: "es2022", assetsDir: "static" },
});
