/**
 * The guide's screenshots (help/img/*.webp), taken from the live page in headless Chrome: each scene is set up through
 * the page itself, then captured once it has drawn. Re-run after the interface changes:
 *
 *   node tools/guide-shots.mjs [scene ...]
 *
 * Needs network (PDB entries, model and OpenBind-0 parameters) and a WebGPU-capable Chrome; the designs scene runs
 * the model and a refold, so it takes a few minutes. Chrome is driven over its DevTools protocol (Node's WebSocket),
 * so there is nothing to install.
 */
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const OUT = join(ROOT, 'help', 'img');
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 4321;
const DEBUG_PORT = 9341;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const server = spawn(process.execPath, [join(ROOT, 'tools', 'serve.mjs'), String(PORT)], { stdio: 'ignore' });
const profile = mkdtempSync(join(tmpdir(), 'guide-shots-'));
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${DEBUG_PORT}`, `--user-data-dir=${profile}`, '--window-size=1440,1000',
  '--force-device-scale-factor=2', '--disable-gpu-sandbox', '--no-first-run', '--no-default-browser-check', 'about:blank',
], { stdio: 'ignore' });

let ws;
let nextId = 0;
const pending = new Map();
const send = (method, params = {}) => new Promise((ok, fail) => {
  const id = (nextId += 1);
  pending.set(id, { ok, fail });
  ws.send(JSON.stringify({ id, method, params }));
});

/** Run `fn` (source of an async function of `w`, the page's window) in the page and return its JSON result. */
async function page(fn, timeoutMs = 600000) {
  const expression = `(async () => { const w = window; const d = document; const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    return JSON.stringify(await (${fn})(w, d, sleep)); })()`;
  const res = await Promise.race([send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }),
    sleep(timeoutMs).then(() => { throw new Error('page step timed out'); })]);
  if (res.exceptionDetails) throw new Error(res.exceptionDetails.exception?.description ?? res.exceptionDetails.text);
  return JSON.parse(res.result.value ?? 'null');
}

/** Capture `rect` (CSS pixels, or the whole window) as help/img/<name>.webp, at `scale` of the device pixels. */
async function shot(name, rect = null, scale = 1) {
  const view = rect ? null : await page('async (w) => ({ width: w.innerWidth, height: w.innerHeight })');
  const clip = rect ? { ...rect, scale } : { x: 0, y: 0, ...view, scale };
  const { data } = await send('Page.captureScreenshot', { format: 'webp', quality: 86, clip, captureBeyondViewport: false });
  writeFileSync(join(OUT, `${name}.webp`), Buffer.from(data, 'base64'));
  console.log(`help/img/${name}.webp`);
}

const rectOf = (selector, pad = 0) => page(`async (w, d) => {
  // Shown: not hidden and laid out (popups are position: fixed, so offsetParent says nothing).
  const e = [...d.querySelectorAll(${JSON.stringify(selector)})].find((x) => !x.hidden && x.getClientRects().length);
  if (!e) return null;
  const r = e.getBoundingClientRect();
  return { x: Math.max(0, r.left - ${pad}), y: Math.max(0, r.top - ${pad}), width: Math.min(1440, r.width + 2 * ${pad}),
    height: Math.min(1000 - Math.max(0, r.top - ${pad}), r.height + 2 * ${pad}) };
}`);

/** Open the start screen and build an example from its card, as a visitor would. */
const openExample = (example) => page(`async (w, d, sleep) => {
  for (let i = 0; i < 600 && !d.querySelector('[data-example="${example}"]'); i++) await sleep(50);
  if (d.getElementById('rfd4p-start').hidden) { d.getElementById('rfd4p-new').click(); await sleep(400); }
  d.querySelector('[data-example="${example}"]').click(); await sleep(300);
  const panel = w.rfd4pApp.designPanel;
  for (let i = 0; i < 2400 && (panel.building || !d.getElementById('rfd4p-start').hidden); i++) await sleep(50);
  await sleep(3000);
  return d.getElementById('rfd4p-task-label').textContent;
}`);

const SCENES = {
  // The start screen: the parts to start from, then the examples.
  async start() {
    await page(`async (w, d, sleep) => { for (let i = 0; i < 600 && !d.querySelector('[data-example]'); i++) await sleep(50); await sleep(1500); }`);
    await shot('start', await rectOf('#rfd4p-studio-card', 6));
  },
  // The binder example: the card's pieces beside the target, hotspots in amber.
  async card() {
    await openExample('protein_binder');
    await shot('card', null, 0.5);
  },
  // A piece's popup, its crop bar dragged in: the residues to be cut fade in the structure.
  async piece() {
    await openExample('protein_binder');
    await page(`async (w, d, sleep) => {
      [...d.querySelectorAll('#rfd4p-chain-list .rfd4p-sub-open')].find((b) => /fixed/.test(b.innerText)).click(); await sleep(1500);
      const pop = [...d.querySelectorAll('.rfd4p-piece-pop')].find((x) => !x.hidden);
      const h = pop.querySelector('.rfd4p-crop-lo'); const t = pop.querySelector('.rfd4p-crop-track').getBoundingClientRect();
      const hb = h.getBoundingClientRect();
      const ev = (type, x) => h.dispatchEvent(new PointerEvent(type, { bubbles: true, button: 0, pointerId: 1, clientX: x, clientY: hb.top + 5 }));
      ev('pointerdown', hb.left + 4); ev('pointermove', t.left + t.width * 0.3); ev('pointerup', t.left + t.width * 0.3);
      await sleep(2500);
    }`);
    await shot('piece', null, 0.5);
    await shot('piece-popup', await rectOf('.rfd4p-piece-pop:not(.rfd4p-sel-pop)', 4));
  },
  // A few residues picked in the structure: the short popup, and the outline that marks a selection everywhere.
  async select() {
    await openExample('protein_binder');
    await page(`async (w, d, sleep) => {
      const a = w.rfd4pApp.designPanel._inputDoc().atoms; const pick = [];
      for (let i = 0; i < a.nAtoms; i++) if (a.chain_id[i] === 'A' && a.res_id[i] >= 50 && a.res_id[i] <= 60) pick.push(i);
      const r = d.getElementById('viewer').getBoundingClientRect();
      w.rfd4pApp.designPanel.lastPointer = { x: r.left + r.width * 0.7, y: r.top + r.height * 0.42 };
      w.atomworksStudio.setSelection(pick); await sleep(2500);
    }`);
    await shot('select', null, 0.5);
  },
  // Three designs from scratch, refolded with OpenBind-0 and ranked.
  async designs() {
    await page(`async (w, d, sleep) => {
      for (let i = 0; i < 600 && !d.getElementById('rfd4p-start-seq'); i++) await sleep(50);
      if (d.getElementById('rfd4p-start').hidden) { d.getElementById('rfd4p-new').click(); await sleep(400); }
      d.getElementById('rfd4p-start-seq').click(); await sleep(3000);
      d.getElementById('samples-input').value = '3';
      d.getElementById('predict').click();
      const app = w.rfd4pApp;
      for (let i = 0; i < 400 && !app.activeAbortController; i++) await sleep(50);
      for (let i = 0; i < 12000 && app.activeAbortController; i++) await sleep(100);
      await sleep(1500);
      d.getElementById('rfd4p-refold-model').value = 'openbind0';
      d.getElementById('rfd4p-refold-btn').click(); await sleep(500);
      for (let i = 0; i < 12000 && app.refoldAbort; i++) await sleep(100);
      await sleep(3000);
    }`);
    await shot('designs', null, 0.5);
    await shot('designs-table', await rectOf('#design-results', 6));
  },
};

try {
  mkdirSync(OUT, { recursive: true });
  let target = null;
  for (let i = 0; i < 100 && !target; i += 1) {
    await sleep(200);
    try { target = (await (await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`)).json()).find((t) => t.type === 'page'); } catch { /* not up yet */ }
  }
  if (!target) throw new Error('Chrome did not start');
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = fail; });
  ws.onmessage = (msg) => {
    const m = JSON.parse(msg.data);
    if (m.id && pending.has(m.id)) {
      const p = pending.get(m.id);
      pending.delete(m.id);
      if (m.error) p.fail(new Error(m.error.message)); else p.ok(m.result);
    }
  };
  await send('Page.enable');
  await send('Runtime.enable');
  const wanted = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(SCENES);
  for (const name of wanted) {
    if (!SCENES[name]) throw new Error(`no scene ${name} (have ${Object.keys(SCENES).join(', ')})`);
    // A fresh page for each scene: no state carried from the last.
    await send('Page.navigate', { url: `http://127.0.0.1:${PORT}/index.html` });
    await sleep(1500);
    // The faster-WebGPU notice shows on a GPU without Chrome's experimental features (headless Chrome's): not part
    // of what the guide shows.
    await page(`async (w, d, sleep) => { for (let i = 0; i < 600 && !w.rfd4pApp?.designPanel; i++) await sleep(50);
      await sleep(500); d.getElementById('rfd4p-speed-dismiss')?.click(); }`);
    await SCENES[name]();
  }
} finally {
  ws?.close();
  chrome.kill();
  server.kill();
  await sleep(500);
  rmSync(profile, { recursive: true, force: true });
}
// Chrome's helpers and the open socket would keep Node alive.
process.exit(0);
