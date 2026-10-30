/**
 * Where Patinae's cost to a design comes from: drawing live frames, or merely holding the GPU.
 * Each backend is measured twice - once drawing every live frame, once with frame drawing stubbed out.
 *   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/diag-patinae-cost.js
 */
export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const nsteps = Number(opt.nsteps ?? 20);
  const rounds = Number(opt.rounds ?? 2);
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
  const api = win.atomworksStudio;
  const realRender = api.renderFrame;

  const switchTo = async (mode) => {
    if (viewerEl.dataset.backend === mode) return;
    sel.value = mode; sel.dispatchEvent(new win.Event('change'));
    const t = performance.now();
    while (viewerEl.dataset.backend !== mode && performance.now() - t < 30000) await sleep(100);
    await sleep(2000);
  };

  await app._loadPreset('motif_unidx_1prw');
  await app.runDesign({ nsteps, recycleSchedule: 'none' });
  await sleep(1500);

  const res = {};
  for (let r = 0; r < rounds; r += 1) {
    for (const mode of ['molstar', 'patinae']) {
      for (const frames of ['drawn', 'stubbed']) {
        await switchTo(mode);
        api.renderFrame = frames === 'drawn' ? realRender : async () => {};
        app.lastResult = null;
        const t = performance.now();
        await app.runDesign({ nsteps, recycleSchedule: 'none' });
        const ms = Math.round(performance.now() - t);
        api.renderFrame = realRender;
        await sleep(1200);
        (res[`${mode}/${frames}`] ??= []).push(ms);
      }
    }
  }
  const best = (k) => Math.min(...(res[k] ?? [Infinity]));
  return {
    nsteps, rounds, raw: res,
    best: Object.fromEntries(Object.keys(res).map((k) => [k, best(k)])),
    readout: {
      liveFrameCostMolstar: best('molstar/drawn') - best('molstar/stubbed'),
      liveFrameCostPatinae: best('patinae/drawn') - best('patinae/stubbed'),
      residencyCost: best('patinae/stubbed') - best('molstar/stubbed'),
    },
  };
}
