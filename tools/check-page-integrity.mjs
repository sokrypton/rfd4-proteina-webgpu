#!/usr/bin/env node
/**
 * Static checks that need no GPU, so CI can run them on every push.
 *
 * Each one exists because the matching failure actually happened:
 *   1. Every local asset the page references exists. A Studio bundle swap renames content-hashed files, and a
 *      missed reference serves a page whose editor never loads.
 *   2. Every stamped module URL (`?v=<hash>`) points at a file that exists.
 *   3. The page's bridge to Studio is real: every `window.atomworksStudio.<member>` the app or the tools call
 *      appears in the shipped bundle, and no code still reaches for a `window.__rfd4p*` global that no longer
 *      exists. `check-webapp.js` counted live frames through `__rfd4pRenderLiveFrame` for five days after the
 *      vendored Studio was dropped, and silently passed that check while the gate failed for the wrong reason.
 *
 *     node tools/check-page-integrity.mjs
 */
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const page = readFileSync(join(ROOT, 'index.html'), 'utf8');
const problems = [];

// 1 + 2. Everything index.html points at locally must be on disk.
const refs = new Set();
for (const m of page.matchAll(/(?:src|href)="(\.\/[^"]+)"/g)) refs.add(m[1]);
for (const m of page.matchAll(/"(\.\/[A-Za-z0-9._/-]+\.(?:js|css|wasm|svg|json|webmanifest))"/g)) refs.add(m[1]);
for (const ref of [...refs].sort()) {
  const clean = ref.replace(/^\.\//, '').split('?')[0];
  if (!clean || clean.endsWith('/')) continue;
  if (!existsSync(join(ROOT, clean))) problems.push(`missing asset referenced by index.html: ${clean}`);
}

// 3. The Studio bridge.
const bundles = execFileSync('git', ['ls-files', 'assets/*.js'], { cwd: ROOT, encoding: 'utf8' })
  .split('\n').filter(Boolean).map((f) => readFileSync(join(ROOT, f), 'utf8')).join('\n');
const ourFiles = execFileSync('git', ['ls-files', 'web/*.js', 'tools/*.mjs', 'tools/gpu/*.js'], { cwd: ROOT, encoding: 'utf8' })
  .split('\n').filter(Boolean);

// Comments are prose, not calls: a note naming a retired global must not read as a use of it.
const stripComments = (src) => src.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/^\s*\/\/.*$/gm, ' ');
const sources = new Map(ourFiles.map((f) => [f, stripComments(readFileSync(join(ROOT, f), 'utf8'))]));
const allOurs = [...sources.values()].join('\n');

// A `__rfd4p*` name this repo assigns, or defines as a property key, is the page's own marker rather than
// something Studio has to supply.
const writtenByUs = (name) => new RegExp(`(?:\\.|\\b)${name}\\s*=[^=]`).test(allOurs)
  || new RegExp(`['"\`]${name}['"\`]`).test(allOurs);

const called = new Map();
for (const [f, src] of sources) {
  for (const m of src.matchAll(/atomworksStudio\??\.\s*(\w+)/g)) if (!called.has(m[1])) called.set(m[1], f);
  for (const m of src.matchAll(/__rfd4p\w+/g)) {
    const name = m[0];
    if (bundles.includes(name) || writtenByUs(name)) continue;
    problems.push(`${f} reads ${name}, which neither this repo nor the shipped Studio bundle defines`);
  }
}
for (const [member, file] of called) {
  if (!bundles.includes(member)) {
    problems.push(`${file} calls atomworksStudio.${member}, which the shipped Studio bundle does not define`);
  }
}

if (problems.length) {
  for (const p of problems) console.error(`FAIL ${p}`);
  console.error(`\ncheck-page-integrity: ${problems.length} problem(s)`);
  process.exit(1);
}
console.log(`check-page-integrity: ok (${refs.size} asset refs, ${called.size} Studio API members)`);
