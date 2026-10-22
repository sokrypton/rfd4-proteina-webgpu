/**
 * End-to-end browser verification of `index.html` + `web/app.js` in headless Chrome:
 *   1. Mounts `/index.html` in an iframe and waits for `window.rfd4pApp` to initialize.
 *   2. Verifies all 10 presets in `RFD4P_PRESETS` load and build batches cleanly in the UI.
 *   3. Runs a real WebGPU design (`nsteps=10`, `recycleSchedule='none'`) on `motif_unidx_1prw`
 *      and verifies that `py2Dmol` renders the trajectory frames + final all-atom structure
 *      and populates `lastResult.pdb` and `lastResult.cif`.
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-webapp.js
 */

export async function main() {
  const iframe = document.createElement('iframe');
  iframe.style.width = '1280px';
  iframe.style.height = '900px';
  document.body.appendChild(iframe);

  await new Promise((resolve, reject) => {
    iframe.onload = resolve;
    iframe.onerror = reject;
    iframe.src = '/index.html';
  });

  const win = iframe.contentWindow;
  // Wait for app initialization
  const t0 = performance.now();
  while (!win.rfd4pApp || win.rfd4pApp.els.predictBtn.disabled) {
    if (performance.now() - t0 > 15000) {
      throw new Error('Timed out waiting for window.rfd4pApp.init()');
    }
    await new Promise((r) => setTimeout(r, 50));
  }

  const app = win.rfd4pApp;
  const presetResults = [];
  const options = [...app.els.presetSelect.options].map((o) => o.value);

  for (const presetId of options) {
    await app._loadPreset(presetId);
    const pillText = app.els.activeConditionPills.textContent;
    const n = app.currentBatchResult?.n ?? 0;
    if (n <= 0) {
      throw new Error(`Preset ${presetId} failed to build batch (status=${app.els.statusMessage.textContent})`);
    }
    presetResults.push({ presetId, n, pills: pillText });
  }

  // Select `motif_unidx_1prw`, set 10 steps + 'none' recycles for fast E2E verification, and run Design
  app.els.presetSelect.value = 'motif_unidx_1prw';
  await app._loadPreset('motif_unidx_1prw');
  app.els.nstepsInput.value = '10';
  app.els.recycleSelect.value = 'none';

  await app.runDesign();

  const statusText = app.els.statusMessage.textContent;
  if (app.els.statusMessage.classList.contains('error') || !app.lastResult) {
    throw new Error(`runDesign failed: ${statusText}`);
  }

  const renderer = app.getPy2DmolRenderer();
  const objName = renderer?.currentObjectName;
  const frameCount = renderer?.objectsData?.[objName]?.frames?.length ?? 0;
  const coreTensorCount = Object.keys(app.activeWeights ?? {}).length;
  const encoderLoadedAfterCore = Boolean(app.shardedBundle?.encoderLoaded);

  // Now run `inverse_folding_1prw` to verify on-demand lazy loading of the 5 encoder shards
  app.els.presetSelect.value = 'inverse_folding_1prw';
  await app._loadPreset('inverse_folding_1prw');
  app.els.nstepsInput.value = '10';
  app.els.recycleSelect.value = 'none';
  await app.runDesign();

  const invStatusText = app.els.statusMessage.textContent;
  if (app.els.statusMessage.classList.contains('error') || !app.lastResult) {
    throw new Error(`inverse_folding_1prw runDesign failed: ${invStatusText}`);
  }
  const fullTensorCount = Object.keys(app.activeWeights ?? {}).length;
  const encoderLoadedAfterInv = Boolean(app.shardedBundle?.encoderLoaded);

  return {
    ok:
      presetResults.length === 10 &&
      coreTensorCount === 910 &&
      !encoderLoadedAfterCore &&
      fullTensorCount === 1257 &&
      encoderLoadedAfterInv &&
      app.lastResult.pdb.includes('ATOM  ') &&
      app.lastResult.cif.includes('_atom_site.Cartn_x') &&
      frameCount === 11,
    presetCount: presetResults.length,
    presets: presetResults,
    coreTensorCount,
    encoderLoadedAfterCore,
    fullTensorCount,
    encoderLoadedAfterInv,
    statusText,
    invStatusText,
    tokensMetric: app.els.metricTokens.textContent,
    atomsMetric: app.els.metricAtoms.textContent,
    latencyMetric: app.els.metricLatency.textContent,
    py2dmolFrames: frameCount,
    pdbLines: app.lastResult.pdb.split('\n').length,
    cifLines: app.lastResult.cif.split('\n').length,
  };
}
