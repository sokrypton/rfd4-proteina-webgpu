/**
 * Refolding a finished design, end to end, through Studio's Design card: start a chain from the start screen, press
 * the card's Generate, refold the design in the Refold panel, and check the result the page shows.
 *
 * Uses OpenBind-0 (Apache-2.0) rather than AlphaFold 3 on purpose. AF3's parameters carry Google DeepMind's terms,
 * which Studio's Refold panel puts behind a dialog a person has to accept; a gate must not click through that.
 * The panel settles terms only for a model that has them, so this path is the same one minus the agreement.
 *
 * The card and its Refold panel are Studio's (window.atomworksStudio.mountDesignCard, mounted by web/app.js, which
 * hosts the panel in the card's Outputs tab); the engine and the fold are this page's (web/app.js, web/refold-provider.js).
 * This drives the card and the panel the way a person does, by their controls, and reads what they kept.
 *
 * The weight bundle is ~265 MiB from HuggingFace on a cold Cache API, so give this a long timeout:
 *   LOCALFOLD_STOCK_FLAGS=1 LOCALFOLD_GPU_TIMEOUT_MS=2400000 node tools/gpu-chrome.mjs tools/gpu/check-refold.js
 * Add --nsteps= / --recycle= only to depart from the card's defaults (60 steps, Thinking `none`); `--seed=` the first
 * design's seed (42).
 *
 * `--gl=1`: the page's WebGL2 path (`?gl=1`), where the refold is LocalFold's WebGL2 fold.
 * `--weights=/runs/model-openbind0-int5`: the bundle from a local copy instead of HuggingFace (a link under /runs/,
 *   which is ignored, to LocalFold's checkout). `--refold-recycles=` / `--refold-steps=`: the Refold box's settings.
 */
export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  // The card's start screen: `+ new`, a free chain of 60-100 residues. (`--start=` another of its buttons.)
  const preset = opt.start ?? 'rfd4p-start-seq';
  const model = opt.model ?? 'openbind0';
  // Default to the card's own sampling: 60 steps, Thinking none. Forcing a cheaper schedule produces a worse design,
  // and a worse design refolds worse, which would read as a refold problem.
  const overrides = {};
  if (opt.nsteps !== undefined) overrides.nsteps = Number(opt.nsteps);
  if (opt.recycle !== undefined) overrides.recycleSchedule = opt.recycle;
  if (model === 'af3') throw new Error('refusing to run AF3 here: its weights need a person to accept the terms');

  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px'; iframe.style.height = '960px';
  document.body.appendChild(iframe);
  await new Promise((res, rej) => { iframe.onload = res; iframe.onerror = rej; iframe.src = opt.gl === '1' ? '/index.html?gl=1' : '/index.html'; });
  const win = iframe.contentWindow;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const errors = [];
  win.addEventListener('error', (e) => errors.push(`error: ${e.message}`));
  win.addEventListener('unhandledrejection', (e) => errors.push(`rejection: ${String(e.reason)}`));
  // ...and what the page logs as an error (a failed refold is logged, not thrown).
  const consoleError = win.console.error.bind(win.console);
  win.console.error = (...parts) => {
    errors.push(`console: ${parts.map((p) => (p instanceof Error ? `${p.message} @ ${(p.stack ?? '').split('\n').slice(0, 3).join(' < ')}` : String(p))).join(' ')}`);
    consoleError(...parts);
  };

  const t0 = performance.now();
  while (!win.rfd4pApp?.card) {
    if (performance.now() - t0 > 60000) throw new Error('app init timeout');
    await sleep(100);
  }
  const app = win.rfd4pApp;
  const card = app.card;
  const doc = win.document;
  if (opt.weights) {
    // The vendored LocalFold's own module instance (vendor/ is outside the import map), as its loader reads it.
    const pin = await (await fetch('/localfold.json')).json();
    const { MODEL_BUNDLES } = await win.eval(`import('/vendor/localfold-${pin.commit}/shared/bundles/manifests/index.js')`);
    MODEL_BUNDLES[model].remote = undefined;
    MODEL_BUNDLES[model].directory = opt.weights;
  }

  // The card reads its fields when they change, as a person's edit does, so set each one the way a person would.
  const choose = (id, value) => {
    const field = doc.getElementById(id);
    if (field.tagName === 'SELECT' && ![...field.options].some((o) => o.value === String(value))) {
      throw new Error(`${id} does not offer ${value} (have ${[...field.options].map((o) => o.value)})`);
    }
    field.value = String(value);
    field.dispatchEvent(new win.Event('input', { bubbles: true }));
    field.dispatchEvent(new win.Event('change', { bubbles: true }));
  };
  for (let i = 0; i < 400 && !doc.getElementById(preset); i += 1) await sleep(50);
  doc.getElementById(preset).click();
  for (let i = 0; i < 400 && !(card.inputSessionId() && card.spec()); i += 1) await sleep(50);
  if (!card.spec()) throw new Error(`${preset} did not open a job in the card`);
  await sleep(1500);
  choose('seed-input', opt.seed ?? 42);
  if (overrides.recycleSchedule) {
    const radio = doc.querySelector(`#thinking-radio-group input[value="${overrides.recycleSchedule}"]`);
    if (!radio) throw new Error(`no Thinking choice ${overrides.recycleSchedule}`);
    radio.click();
  }
  if (overrides.nsteps !== undefined) app.nsteps = overrides.nsteps;
  const tD = performance.now();
  await card.generate();
  const designMs = Math.round(performance.now() - tD);
  const design = card.designs()[0];
  if (!app.lastResult?.cif || !design) throw new Error(`design failed: ${doc.getElementById('rfd4p-status-message').textContent.slice(0, 200)}`);

  if (opt['refold-recycles']) choose('refold-recycles', opt['refold-recycles']);
  if (opt['refold-steps']) choose('refold-steps', opt['refold-steps']);
  const panel = doc.getElementById('refold-box');
  const select = doc.getElementById('refold-model');
  const result = doc.getElementById('refold-result');
  if (!panel || !select || !result) throw new Error('refold controls are missing from the page');
  if (panel.hidden) throw new Error('refold panel stayed hidden after a design finished');
  const offered = [...select.options].map((o) => o.value);
  if (!offered.includes(model)) throw new Error(`refold model ${model} is not offered (have ${offered})`);
  choose('refold-model', model);

  // The terms dialog must not appear for a non-AF3 model; if it does, this run would be agreeing to something.
  const terms = doc.getElementById('refold-terms');
  const tR = performance.now();
  // The longest the page's thread was held during the refold: a 20 ms timer's worst late tick.
  let lastTick = performance.now();
  let longestStallMs = 0;
  // ...and where the long ones fell: the refold's status line either side of each.
  const stalls = [];
  const statusNow = () => (doc.getElementById('refold-btn')?.textContent ?? '').trim().slice(0, 60);
  let lastStatus = statusNow();
  const ticker = win.setInterval(() => {
    const now = performance.now();
    const late = now - lastTick - 20;
    longestStallMs = Math.max(longestStallMs, late);
    const status = statusNow();
    if (late > 150) stalls.push({ atMs: Math.round(now - tR), ms: Math.round(late), before: lastStatus, after: status });
    lastStatus = status;
    lastTick = now;
  }, 20);
  const refolding = card.refold.refold();
  await sleep(500);
  if (terms?.open) { terms.close(); throw new Error(`the AF3 terms dialog opened for model ${model}`); }
  await refolding;
  win.clearInterval(ticker);
  const refoldMs = Math.round(performance.now() - tR);

  const text = (result.textContent ?? '').trim();
  // The status line can read well while nothing was recorded, so check the view the page actually keeps.
  const view = design.data ?? null;
  const toRecord = (r) => ({ ...r, ...r.data });   // the page's own extras (pdb, confidence, ...) ride in `data`
  const refolds = (card.refold.records(design.id) ?? []).map(toRecord);
  if (card.refold.session.isRunning) throw new Error('refold left a run behind');
  if (refolds.length !== 1) {
    throw new Error(`expected one recorded refold, got ${refolds.length}: ${text || '(no failure line)'}; errors: ${errors.slice(0, 4).join(' | ')}`);
  }
  const [refold] = refolds;
  if (!refold.cif) throw new Error('recorded refold carries no overlay');
  if (refold.model !== model) throw new Error(`recorded refold is ${refold.model}, not ${model}`);
  if (!refold.pdb || !refold.cif) throw new Error('recorded refold carries no structure');
  if (!Number.isFinite(refold.rmsd)) throw new Error(`recorded refold has no RMSD: ${refold.rmsd}`);
  // The refold is drawn over the design on show: Studio's viewer counts the atoms of each overlay it draws.
  const overlayAtoms = Number(doc.getElementById('viewer')?.dataset.overlay_refold ?? 0);
  if (!(overlayAtoms > 0)) throw new Error('the refold overlay was not drawn in the viewer');
  // The line under the Refold box is for failures only; the refold's own summary is its record's.
  if (!result.hidden && text) throw new Error(`refold failed: ${text}`);

  // `--again=1`: a second refold (another seed) with the weights resident and the programs built.
  let again;
  if (opt.again === '1') {
    choose('refold-seed', 1);
    longestStallMs = 0;
    lastTick = performance.now();
    const t2 = performance.now();
    const stalls2 = [];
    let lastStatus2 = statusNow();
    const tick2 = win.setInterval(() => {
      const now = performance.now();
      const late = now - lastTick - 20;
      longestStallMs = Math.max(longestStallMs, late);
      const status = statusNow();
      if (late > 150) stalls2.push({ atMs: Math.round(now - t2), ms: Math.round(late), before: lastStatus2, after: status });
      lastStatus2 = status;
      lastTick = now;
    }, 20);
    await card.refold.refold();
    win.clearInterval(tick2);
    again = { refoldMs: Math.round(performance.now() - t2), longestStallMs: Math.round(longestStallMs), stalls: stalls2.slice(0, 12), refolds: card.refold.records(design.id).length };
  }
  // Is either structure a chain at all? Consecutive Cα spacing (3.8 Å in a protein), design and refold.
  const caSpacing = (pdb) => {
    const ca = pdb.split('\n').filter((l) => l.startsWith('ATOM') && l.slice(12, 16).trim() === 'CA')
      .map((l) => ({ chain: l[21], x: +l.slice(30, 38), y: +l.slice(38, 46), z: +l.slice(46, 54) }));
    const d = [];
    for (let i = 1; i < ca.length; i += 1) {
      if (ca[i].chain !== ca[i - 1].chain) continue;
      d.push(Math.hypot(ca[i].x - ca[i - 1].x, ca[i].y - ca[i - 1].y, ca[i].z - ca[i - 1].z));
    }
    d.sort((a, b) => a - b);
    return { n: d.length, median: Number((d[d.length >> 1] ?? NaN).toFixed(2)), max: Number((d.at(-1) ?? NaN).toFixed(2)) };
  };
  const geometry = { design: caSpacing(view.pdb), refold: caSpacing(refold.pdb) };
  return {
    ok: true, geometry, again, preset, overrides, model, offered, designMs, refoldMs, longestStallMs: Math.round(longestStallMs), stalls: stalls.slice(0, 12), backend: doc.getElementById('webgpu-badge')?.textContent,
    resultText: String(refold.text ?? '').slice(0, 300),
    viewerBackend: doc.getElementById('viewer')?.dataset.backend ?? null,
    renderedAtoms: Number(doc.getElementById('viewer')?.dataset.renderedAtoms ?? 0),
    errors: errors.slice(0, 8),
    rmsd: refold.rmsd,
    // Summarised: the raw confidence carries a per-atom deviation map thousands of entries long.
    confidence: Object.fromEntries(Object.entries(refold.confidence ?? {})
      .filter(([, v]) => typeof v === 'number')),
    templated: refold.templated ?? null,
    overlayAtoms,
  };
}
