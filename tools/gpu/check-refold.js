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
  await new Promise((res, rej) => { iframe.onload = res; iframe.onerror = rej; iframe.src = '/index.html'; });
  const win = iframe.contentWindow;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const errors = [];
  win.addEventListener('error', (e) => errors.push(`error: ${e.message}`));
  win.addEventListener('unhandledrejection', (e) => errors.push(`rejection: ${String(e.reason)}`));

  const t0 = performance.now();
  while (!win.rfd4pApp || win.rfd4pApp.els.predictBtn.disabled) {
    if (performance.now() - t0 > 60000) throw new Error('app init timeout');
    await sleep(100);
  }
  const app = win.rfd4pApp;
  const doc = win.document;

  await app._loadPreset(preset);
  const tD = performance.now();
  await app.runDesign(overrides);
  const designMs = Math.round(performance.now() - tD);
  if (!app.lastResult?.cif) throw new Error(`design failed: ${app.els.statusMessage.textContent.slice(0, 200)}`);

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
  const refolding = app._refoldSelected();
  await sleep(500);
  if (terms?.open) { terms.close(); throw new Error(`the AF3 terms dialog opened for model ${model}`); }
  await refolding;
  const refoldMs = Math.round(performance.now() - tR);

  const text = (result.textContent ?? '').trim();
  // The status line can read well while nothing was recorded, so check the view the page actually keeps.
  const view = app.generatedViews?.get(app.studioSessionId) ?? null;
  const refolds = view?.refolds ?? [];
  if (app.refoldAbort !== null) throw new Error('refold left an abort controller behind');
  if (refolds.length !== 1) throw new Error(`expected one recorded refold, got ${refolds.length}`);
  const [refold] = refolds;
  if (refold.model !== model) throw new Error(`recorded refold is ${refold.model}, not ${model}`);
  if (!refold.pdb || !refold.cif) throw new Error('recorded refold carries no structure');
  if (!Number.isFinite(refold.rmsd)) throw new Error(`recorded refold has no RMSD: ${refold.rmsd}`);
  if (!view?.overlays?.refold?.cif) throw new Error('the refold overlay was not handed to the viewer');
  if (!text) throw new Error('refold produced no result text');
  if (/^Refold failed/i.test(text)) throw new Error(`refold failed: ${text}`);
  if (result.hidden) throw new Error(`refold result stayed hidden while reading: ${text}`);

  return {
    ok: true, preset, overrides, model, offered, designMs, refoldMs,
    resultText: text.slice(0, 300),
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
