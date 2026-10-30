/**
 * The second renderer. A visitor can switch the viewer from Mol* (WebGL2) to Patinae (WebGPU) from the page's
 * own dropdown, so that path needs a gate of its own: it shares the GPU with the model, and when it fails it
 * fails quietly - the Studio falls back to Mol* on a Patinae WASM error, and every adapter method the contract
 * marks optional degrades to a no-op rather than an error.
 *
 *   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/check-patinae.js
 */
export async function main() {
  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px'; iframe.style.height = '960px';
  document.body.appendChild(iframe);
  await new Promise((res, rej) => { iframe.onload = res; iframe.onerror = rej; iframe.src = '/index.html'; });
  const win = iframe.contentWindow;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const t0 = performance.now();
  while (!win.rfd4pApp || win.rfd4pApp.els.predictBtn.disabled) {
    if (performance.now() - t0 > 60000) throw new Error('app init timeout');
    await sleep(100);
  }
  const app = win.rfd4pApp;
  const doc = win.document;
  const viewerEl = doc.getElementById('viewer');
  const sel = doc.getElementById('viewer-mode');
  if (!sel) throw new Error('#viewer-mode is missing');
  const offered = [...sel.options].map((o) => o.value);
  if (!offered.includes('patinae')) return { skipped: 'this Studio build does not offer Patinae', offered };

  // Draw something real first: a design opens a Studio session, which is what renders.
  await app._loadPreset('motif_unidx_1prw');
  await app.runDesign({ nsteps: 8, recycleSchedule: 'none' });
  await sleep(1500);
  const molstarAtoms = Number(viewerEl.dataset.renderedAtoms ?? 0);
  if (viewerEl.dataset.backend !== 'molstar' || molstarAtoms <= 0) {
    throw new Error(`expected Mol* to be drawing before the switch: backend=${viewerEl.dataset.backend}, atoms=${molstarAtoms}`);
  }

  let liveFrames = 0;
  const api = win.atomworksStudio;
  const realRender = api.renderFrame;
  api.renderFrame = async (...a) => { liveFrames += 1; return realRender.apply(api, a); };

  sel.value = 'patinae';
  sel.dispatchEvent(new win.Event('change'));
  const tS = performance.now();
  while (viewerEl.dataset.backend !== 'patinae' && performance.now() - tS < 30000) await sleep(100);
  await sleep(3000);
  // A Patinae WASM failure is caught and silently switched back to Mol*, so "still molstar" is the failure mode.
  if (viewerEl.dataset.backend !== 'patinae') {
    throw new Error(`switch to Patinae did not stick (fell back to ${viewerEl.dataset.backend}); notice: ${doc.getElementById('renderer-notice')?.textContent?.slice(0, 200)}`);
  }
  const patinaeAtoms = Number(viewerEl.dataset.renderedAtoms ?? 0);
  if (patinaeAtoms <= 0) throw new Error('Patinae is active but drawing no atoms');
  // Patinae advertises PyMOL-style commands; if its capabilities never reached the UI it is not really up.
  const syntax = [...(doc.getElementById('selection-syntax')?.options ?? [])].map((o) => o.value);
  if (!syntax.includes('patinae')) throw new Error(`Patinae capabilities did not reach the UI: syntax=${syntax}`);

  liveFrames = 0;
  app.lastResult = null;
  const tD = performance.now();
  await app.runDesign({ nsteps: 10, recycleSchedule: 'none' });
  const designMs = Math.round(performance.now() - tD);
  await sleep(1500);
  if (!app.lastResult?.cif) throw new Error(`design failed on Patinae: ${app.els.statusMessage.textContent.slice(0, 200)}`);
  if (liveFrames < 2) throw new Error(`no live frames rendered on Patinae (got ${liveFrames})`);
  if (viewerEl.dataset.backend !== 'patinae') throw new Error('Patinae fell back to Mol* during the design');

  // And Mol* must still be reachable afterwards.
  sel.value = 'molstar';
  sel.dispatchEvent(new win.Event('change'));
  const tB = performance.now();
  while (viewerEl.dataset.backend !== 'molstar' && performance.now() - tB < 30000) await sleep(100);
  await sleep(2000);
  if (viewerEl.dataset.backend !== 'molstar') throw new Error('could not switch back to Mol*');
  if (Number(viewerEl.dataset.renderedAtoms ?? 0) <= 0) throw new Error('Mol* drew nothing after switching back');

  api.renderFrame = realRender;
  return {
    ok: true, offered, molstarAtoms, patinaeAtoms, syntax,
    liveFramesOnPatinae: liveFrames, designMsOnPatinae: designMs,
    backAtoms: Number(viewerEl.dataset.renderedAtoms ?? 0),
  };
}
