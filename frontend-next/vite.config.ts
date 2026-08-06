import { defineConfig } from "vite";

// The FastAPI app owns /api; Vite serves everything else and proxies across.
// Override with API_URL when the backend is not on its usual port.
const api = process.env.API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  build: { outDir: "dist", sourcemap: true },
  server: { port: 5173, proxy: { "/api": api } },
  preview: { port: 4173, proxy: { "/api": api } },
});
