#!/usr/bin/env node
/**
 * Cache-bust the site's unbundled ES modules: write an import map into index.html that pins every module under
 * web/, shared/, webgpu/ and cpu/ to `?v=<content hash>`, and stamp the app.js / styles.css tags the same way.
 *
 * GitHub Pages serves files with max-age=600, so without this a browser can run a new web/app.js against a cached
 * older module it imports. With it, a module's URL changes exactly when its content does, and every importer still
 * resolves to that one URL (one module instance). Run after editing any of those files (`npm run stamp`;
 * `npm run build:studio` runs it too).
 */
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIRS = ['web', 'shared', 'webgpu', 'cpu'];
const PAGE = join(ROOT, 'index.html');
const START = '<!-- module-versions:start (tools/stamp-modules.mjs) -->';
const END = '<!-- module-versions:end -->';

const hash = (path) => createHash('sha256').update(readFileSync(path)).digest('hex').slice(0, 10);
const walk = (dir) => readdirSync(dir).flatMap((name) => {
  const p = join(dir, name);
  return statSync(p).isDirectory() ? walk(p) : [p];
});

const modules = DIRS.flatMap((d) => walk(join(ROOT, d))).filter((p) => p.endsWith('.js')).sort();
const imports = Object.fromEntries(modules.map((p) => {
  const url = `./${relative(ROOT, p).split('\\').join('/')}`;
  return [url, `${url}?v=${hash(p)}`];
}));

let html = readFileSync(PAGE, 'utf8');
const block = `${START}\n    <script type="importmap">\n${JSON.stringify({ imports }, null, 2)
  .split('\n').map((l) => `      ${l}`).join('\n')}\n    </script>\n    ${END}`;
if (html.includes(START)) {
  html = html.slice(0, html.indexOf(START)) + block + html.slice(html.indexOf(END) + END.length);
} else {
  // The import map must come before the first module script.
  const at = html.indexOf('<script type="module"');
  if (at < 0) throw new Error('index.html: no module script to place the import map before');
  html = `${html.slice(0, at)}${block}\n    ${html.slice(at)}`;
}
html = html
  .replace(/src="\.\/web\/app\.js(\?v=\w+)?"/, `src="./web/app.js?v=${hash(join(ROOT, 'web/app.js'))}"`)
  .replace(/href="\.\/web\/styles\.css(\?v=\w+)?"/, `href="./web/styles.css?v=${hash(join(ROOT, 'web/styles.css'))}"`);
writeFileSync(PAGE, html);
console.log(`stamped ${modules.length} modules`);
