/**
 * Refolding a finished design, end to end: run a design, refold it, and check the result the page shows.
 *
 * Uses OpenBind-0 (Apache-2.0) rather than AlphaFold 3 on purpose. AF3's parameters carry Google DeepMind's terms,
 * which `_agreeRefoldTerms` puts behind a dialog a person has to accept; a gate must not click through that.
 * `_agreeRefoldTerms` returns any non-`af3` model untouched, so this path is the same one minus the agreement.
 *
 * The weight bundle is ~265 MiB from HuggingFace on a cold Cache API, so give this a long timeout:
 *   LOCALFOLD_STOCK_FLAGS=1 LOCALFOLD_GPU_TIMEOUT_MS=2400000 node tools/gpu-chrome.mjs tools/gpu/check-refold.js
 * Add --nsteps= / --recycle= only to depart from the preset's own sampling.
 *
 * `--gl=1`: the page's WebGL2 path (`?gl=1`), where the refold is LocalFold's WebGL2 fold.
 * `--weights=/runs/model-openbind0-int5`: the bundle from a local copy instead of HuggingFace (a link under /runs/,
 *   which is ignored, to LocalFold's checkout). `--refold-recycles=` / `--refold-steps=`: the Refold box's settings.
 */
export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const preset = opt.preset ?? 'uncond_short';
  const model = opt.model ?? 'openbind0';
  // Default to the preset's own sampling. `uncond_short` asks for 60 steps and `low` recycling; forcing a cheaper
  // schedule produces a worse design, and a worse design refolds worse, which would read as a refold problem.
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
  while (!win.rfd4pApp || win.rfd4pApp.els.predictBtn.disabled) {
    if (performance.now() - t0 > 60000) throw new Error('app init timeout');
    await sleep(100);
  }
  const app = win.rfd4pApp;
  const doc = win.document;
  if (opt.weights) {
    // The vendored LocalFold's own module instance (vendor/ is outside the import map), as its loader reads it.
    const pin = await (await fetch('/localfold.json')).json();
    const { MODEL_BUNDLES } = await win.eval(`import('/vendor/localfold-${pin.commit}/shared/bundles/manifests/index.js')`);
    MODEL_BUNDLES[model].remote = undefined;
    MODEL_BUNDLES[model].directory = opt.weights;
  }

  await app._loadPreset(preset);
  const tD = performance.now();
  await app.runDesign(overrides);
  const designMs = Math.round(performance.now() - tD);
  if (!app.lastResult?.cif) throw new Error(`design failed: ${app.els.statusMessage.textContent.slice(0, 200)}`);

  if (opt['refold-recycles']) doc.getElementById('rfd4p-refold-recycles').value = opt['refold-recycles'];
  if (opt['refold-steps']) {
    const steps = doc.getElementById('rfd4p-refold-steps');
    if (![...steps.options].some((o) => o.value === opt['refold-steps'])) steps.append(new win.Option(opt['refold-steps'], opt['refold-steps']));
    steps.value = opt['refold-steps'];
  }
  const panel = doc.getElementById('rfd4p-refold');
  const select = doc.getElementById('rfd4p-refold-model');
  const result = doc.getElementById('rfd4p-refold-result');
  if (!panel || !select || !result) throw new Error('refold controls are missing from the page');
  if (panel.hidden) throw new Error('refold panel stayed hidden after a design finished');
  const offered = [...select.options].map((o) => o.value);
  if (!offered.includes(model)) throw new Error(`refold model ${model} is not offered (have ${offered})`);
  select.value = model;

  // The terms dialog must not appear for a non-AF3 model; if it does, this run would be agreeing to something.
  const terms = doc.getElementById('rfd4p-refold-terms');
  const tR = performance.now();
  // The longest the page's thread was held during the refold: a 20 ms timer's worst late tick.
  let lastTick = performance.now();
  let longestStallMs = 0;
  // ...and where the long ones fell: the refold's status line either side of each.
  const stalls = [];
  const statusNow = () => (doc.getElementById('rfd4p-refold-btn')?.textContent ?? '').trim().slice(0, 60);
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
  const refolding = app._refoldDesigns(null);
  await sleep(500);
  if (terms?.open) { terms.close(); throw new Error(`the AF3 terms dialog opened for model ${model}`); }
  await refolding;
  win.clearInterval(ticker);
  const refoldMs = Math.round(performance.now() - tR);

  const text = (result.textContent ?? '').trim();
  // The status line can read well while nothing was recorded, so check the view the page actually keeps.
  const view = app.generatedViews?.get(app.studioSessionId) ?? null;
  const refolds = view?.refolds ?? [];
  if (app.refoldAbort !== null) throw new Error('refold left an abort controller behind');
  if (refolds.length !== 1) {
    throw new Error(`expected one recorded refold, got ${refolds.length}: ${text || '(no failure line)'}; errors: ${errors.slice(0, 4).join(' | ')}`);
  }
  const [refold] = refolds;
  if (refold.model !== model) throw new Error(`recorded refold is ${refold.model}, not ${model}`);
  if (!refold.pdb || !refold.cif) throw new Error('recorded refold carries no structure');
  if (!Number.isFinite(refold.rmsd)) throw new Error(`recorded refold has no RMSD: ${refold.rmsd}`);
  if (!view?.overlays?.refold?.cif) throw new Error('the refold overlay was not handed to the viewer');
  // The line under the Refold box is for failures only (`_syncRefoldUi`); the refold's own summary is its record's.
  if (!result.hidden && text) throw new Error(`refold failed: ${text}`);

  // `--again=1`: a second refold (another seed) with the weights resident and the programs built.
  let again;
  if (opt.again === '1') {
    doc.getElementById('rfd4p-refold-seed').value = '1';
    doc.getElementById('rfd4p-refold-seed').dispatchEvent(new win.Event('change', { bubbles: true }));
    longestStallMs = 0;
    lastTick = performance.now();
    const t2 = performance.now();
    const tick2 = win.setInterval(() => { const now = performance.now(); longestStallMs = Math.max(longestStallMs, now - lastTick - 20); lastTick = now; }, 20);
    await app._refoldDesigns(null);
    win.clearInterval(tick2);
    again = { refoldMs: Math.round(performance.now() - t2), longestStallMs: Math.round(longestStallMs), refolds: view.refolds.length };
  }
  return {
    ok: true, again, preset, overrides, model, offered, designMs, refoldMs, longestStallMs: Math.round(longestStallMs), stalls: stalls.slice(0, 12), backend: app.els.webgpuBadge?.textContent,
    resultText: String(refold.text ?? '').slice(0, 300),
    viewerBackend: doc.getElementById('viewer')?.dataset.backend ?? null,
    renderedAtoms: Number(doc.getElementById('viewer')?.dataset.renderedAtoms ?? 0),
    errors: errors.slice(0, 8),
    rmsd: refold.rmsd,
    // Summarised: the raw confidence carries a per-atom deviation map thousands of entries long.
    confidence: Object.fromEntries(Object.entries(refold.confidence ?? {})
      .filter(([, v]) => typeof v === 'number')),
    templated: refold.templated ?? null,
    overlayChars: view.overlays.refold.cif.length,
  };
}
