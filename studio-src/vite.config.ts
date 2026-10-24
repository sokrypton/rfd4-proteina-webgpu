import { defineConfig } from "vite";

// Builds only the Studio's JS/CSS. The page markup lives in the repository's index.html / studio.html (with the
// RFD4-Proteina card), which tools/build-studio.mjs points at the emitted entry files.
export default defineConfig({
  // Relative URLs, so the site works from any sub-path (GitHub Pages serves it under /rfd4-proteina-webgpu/).
  base: "./",
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
    manifest: true,
    rollupOptions: {
      input: { studio: "src/main.ts" },
    },
  },
});
