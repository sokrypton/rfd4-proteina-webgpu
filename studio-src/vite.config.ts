import { defineConfig } from "vite";

// Builds only the Studio's JS/CSS. The page markup lives in the repository's index.html / studio.html (with the
// RFD4-Proteina card), which tools/build-studio.mjs points at the emitted entry files.
export default defineConfig({
  // Relative URLs, so the site works from any sub-path (GitHub Pages serves it under /rfd4-proteina-webgpu/).
  base: "./",
  plugins: [
    {
      // Safari < 16.4 cannot parse regex lookbehind, and this one sits at the top level of the Mol* chunk, so a
      // single literal stops the renderer from loading at all. findEmail already checks the preceding character
      // (previous(match, true)), so dropping the lookbehind keeps the same matches.
      name: "rfd4p-no-lookbehind",
      transform(code, id) {
        if (!/mdast-util-gfm-autolink-literal\/lib\/index\.js/.test(id)) return null;
        const out = code.replace("/(?<=^|\\s|\\p{P}|\\p{S})([-.\\w+]+)@", "/([-.\\w+]+)@");
        if (out === code) throw new Error("rfd4p-no-lookbehind: pattern not found; recheck for Safari 16.3");
        return out;
      },
    },
  ],
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
