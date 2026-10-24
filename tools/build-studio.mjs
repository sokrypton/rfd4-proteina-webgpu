#!/usr/bin/env node
/**
 * Compose index.html and assets/ from a released AtomWorks Studio browser bundle.
 *
 * Studio used to be vendored under studio-src/ and built here, which meant re-applying local patches on every
 * import and keeping a fork of Studio's index.html. The fork was the worse half: Studio looks its elements up by id
 * and throws on a missing one, so a page that had drifted would stop Studio booting outright. Nothing is vendored
 * now — the bundle is a release download, and the page is composed from Studio's own index.html plus the two
 * fragments this repo owns (web/page/head.html, web/page/card.html).
 *
 *   npm run build:studio                      # the pinned release in studio.json
 *   npm run build:studio -- --from <dir>      # a bundle built locally, for trying an unreleased Studio
 *
 * Studio is driven through its documented embedding API (window.atomworksStudio); see web/app.js.
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ASSETS = join(ROOT, 'assets');
const LIST = join(ASSETS, 'studio-files.json');
const PIN = JSON.parse(readFileSync(join(ROOT, 'studio.json'), 'utf8'));

const args = process.argv.slice(2);
const fromIndex = args.indexOf('--from');
const localBundle = fromIndex === -1 ? null : args[fromIndex + 1];

function run(cmd, cmdArgs, cwd = ROOT) {
  const result = spawnSync(cmd, cmdArgs, { cwd, stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`${cmd} ${cmdArgs.join(' ')} failed (${result.status})`);
}

/** The release tarball, verified against the checksum in studio.json so a build cannot silently change bundle. */
function downloadBundle() {
  const url = PIN.bundleUrl.replaceAll('${version}', PIN.version);
  const work = mkdtempSync(join(tmpdir(), 'studio-bundle-'));
  const archive = join(work, 'bundle.tar.gz');
  run('curl', ['-fsSL', url, '-o', archive], work);
  const digest = createHash('sha256').update(readFileSync(archive)).digest('hex');
  if (PIN.sha256 && PIN.sha256 !== digest) {
    throw new Error(`${url}\n  expected sha256 ${PIN.sha256}\n  got      sha256 ${digest}`);
  }
  if (!PIN.sha256) console.warn(`studio: no sha256 pinned; downloaded bundle is ${digest}`);
  const out = join(work, 'bundle');
  mkdirSync(out);
  run('tar', ['-xzf', archive, '-C', out], work);
  return out;
}

const bundle = localBundle ? resolve(localBundle) : downloadBundle();
const studioPage = join(bundle, 'index.html');
if (!existsSync(studioPage)) throw new Error(`${bundle} is not a Studio bundle (no index.html)`);

// Studio's own files, replacing the previous build's and nothing else in assets/.
const walk = (dir) => readdirSync(dir).flatMap((name) => {
  const path = join(dir, name);
  return statSync(path).isDirectory() ? walk(path) : [path];
});
const emitted = walk(join(bundle, 'assets')).map((path) => relative(bundle, path));
for (const file of existsSync(LIST) ? JSON.parse(readFileSync(LIST, 'utf8')) : []) {
  rmSync(join(ROOT, file), { force: true });
}
for (const file of emitted) {
  mkdirSync(dirname(join(ROOT, file)), { recursive: true });
  cpSync(join(bundle, file), join(ROOT, file));
}
writeFileSync(LIST, JSON.stringify(emitted.sort(), null, 2) + '\n');

// Compose the page: Studio's markup, this repo's head additions, this repo's card ahead of Studio's shell.
const head = readFileSync(join(ROOT, 'web/page/head.html'), 'utf8').trimEnd();
const card = readFileSync(join(ROOT, 'web/page/card.html'), 'utf8').trimEnd();
let page = readFileSync(studioPage, 'utf8');

page = page.replace(/<title>[^<]*<\/title>/, `<title>${PIN.title}</title>`);
if (!page.includes('</head>')) throw new Error('Studio index.html has no </head>');
// Ahead of Studio's entry script, so app.js installs the /api/* interceptor before Studio boots against it.
const entryScript = page.match(/\s*<script type="module" crossorigin src="[^"]+"><\/script>/);
if (!entryScript) throw new Error('Studio index.html has no module entry script');
page = page.replace(entryScript[0], `\n${head}${entryScript[0]}`);
const body = page.match(/<body[^>]*>/);
if (!body) throw new Error('Studio index.html has no <body>');
page = page.replace(body[0], `${body[0]}\n${card}`);
// Studio's bundle is built to be served from the site root; this one lives under a project path on Pages.
page = page.replaceAll('="/assets/', '="./assets/').replaceAll('="/guide/', '="./guide/');
if (/(?:src|href)="\/[^"]/.test(page)) {
  throw new Error(`composed page still has root-absolute paths: ${page.match(/(?:src|href)="\/[^"]+"/g)}`);
}

writeFileSync(join(ROOT, 'index.html'), page);
console.log(`studio: ${PIN.version}${localBundle ? ` (local: ${localBundle})` : ''}, ${emitted.length} files`);
run('node', [join(ROOT, 'tools/stamp-modules.mjs')]);
