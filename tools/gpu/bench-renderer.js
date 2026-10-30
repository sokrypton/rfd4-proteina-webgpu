/**
 * What the viewer backend costs a design. Interleaved A/B in one process: Mol* (WebGL2) vs Patinae (WebGPU),
 * which shares the GPU with the model.
 *   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/bench-renderer.js --rounds=3 --nsteps=20
 */
export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const rounds = Number(opt.rounds ?? 3);
  const nsteps = Number(opt.nsteps ?? 20);
  const preset = opt.preset ?? 'motif_unidx_1prw';

  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px';
  iframe.style.height = '960px';
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

  const switchTo = async (mode) => {
    if (viewerEl.dataset.backend === mode) return;
    sel.value = mode;
    sel.dispatchEvent(new win.Event('change'));
    const t = performance.now();
    while (viewerEl.dataset.backend !== mode && performance.now() - t < 30000) await sleep(100);
    await sleep(2000);
    if (viewerEl.dataset.backend !== mode) throw new Error(`could not switch to ${mode}`);
  };

  await app._loadPreset(preset);
  // Warm up so neither arm pays for first-run compilation.
  await app.runDesign({ nsteps, recycleSchedule: 'none' });
  await sleep(1500);

  const times = { molstar: [], patinae: [] };
  for (let r = 0; r < rounds; r += 1) {
    for (const mode of ['molstar', 'patinae']) {
      await switchTo(mode);
      app.lastResult = null;
      const t = performance.now();
      await app.runDesign({ nsteps, recycleSchedule: 'none' });
      const ms = Math.round(performance.now() - t);
      await sleep(1200);
      times[mode].push({ ms, ok: Boolean(app.lastResult?.cif), atoms: Number(viewerEl.dataset.renderedAtoms ?? 0) });
    }
  }
  const best = (k) => Math.min(...times[k].map((x) => x.ms));
  const med = (k) => { const s = times[k].map((x) => x.ms).sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
  return {
    preset, nsteps, rounds, times,
    summary: {
      molstarBest: best('molstar'), patinaeBest: best('patinae'),
      molstarMedian: med('molstar'), patinaeMedian: med('patinae'),
      slowdownBest: +(best('patinae') / best('molstar')).toFixed(2),
      slowdownMedian: +(med('patinae') / med('molstar')).toFixed(2),
    },
  };
}
