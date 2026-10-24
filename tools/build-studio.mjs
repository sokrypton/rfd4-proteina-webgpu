#!/usr/bin/env node
/**
 * Build the vendored AtomWorks Studio (studio-src/) into assets/ and point index.html / studio.html at it.
 *
 * Vite emits content-hashed files, so each build replaces the previous Studio files (listed in
 * assets/studio-files.json) and rewrites the entry <script>/<link> tags. Run via `npm run build:studio`.
 */
import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = join(ROOT, 'studio-src');
const DIST = join(SRC, 'dist');
const ASSETS = join(ROOT, 'assets');
const LIST = join(ASSETS, 'studio-files.json');
const PAGES = ['index.html', 'studio.html'];

function run(cmd, args) {
  const r = spawnSync(cmd, args, { cwd: SRC, stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`${cmd} ${args.join(' ')} failed (${r.status})`);
}

if (!existsSync(join(SRC, 'node_modules'))) run('npm', ['ci', '--no-audit', '--no-fund']);
run(join(SRC, 'node_modules/.bin/tsc'), ['--noEmit']);
run(join(SRC, 'node_modules/.bin/vite'), ['build', '--logLevel', 'warn']);

const manifest = JSON.parse(readFileSync(join(DIST, '.vite/manifest.json'), 'utf8'));
const entry = manifest['src/main.ts'];
if (!entry?.file || !entry.css?.length) throw new Error('studio entry (src/main.ts) or its CSS missing from manifest');

const walk = (dir) => readdirSync(dir).flatMap((name) => {
  const p = join(dir, name);
  return statSync(p).isDirectory() ? walk(p) : [p];
});
const emitted = walk(join(DIST, 'assets')).map((p) => relative(DIST, p));

// Replace the previous build's files (never anything else in assets/).
const previous = existsSync(LIST) ? JSON.parse(readFileSync(LIST, 'utf8')) : [];
for (const file of previous) rmSync(join(ROOT, file), { force: true });
for (const file of emitted) {
  mkdirSync(dirname(join(ROOT, file)), { recursive: true });
  copyFileSync(join(DIST, file), join(ROOT, file));
}
writeFileSync(LIST, JSON.stringify(emitted.sort(), null, 2) + '\n');

// The Studio entry is the only module script / stylesheet under ./assets/ in each page.
for (const page of PAGES) {
  const path = join(ROOT, page);
  let html = readFileSync(path, 'utf8');
  const scripts = html.match(/<script type="module" crossorigin src="\.\/assets\/[^"]+"><\/script>/g) ?? [];
  const styles = html.match(/<link rel="stylesheet" crossorigin href="\.\/assets\/[^"]+"\s*\/?>/g) ?? [];
  if (scripts.length !== 1 || styles.length !== 1) {
    throw new Error(`${page}: expected one ./assets/ module script and stylesheet, found ${scripts.length}/${styles.length}`);
  }
  html = html
    .replace(scripts[0], `<script type="module" crossorigin src="./${entry.file}"></script>`)
    .replace(styles[0], entry.css.map((css) => `<link rel="stylesheet" crossorigin href="./${css}" />`).join('\n    '));
  writeFileSync(path, html);
}

console.log(`studio: ${entry.file} + ${entry.css.join(', ')} (${emitted.length} files)`);
