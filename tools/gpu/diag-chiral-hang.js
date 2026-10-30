/**
 * SPEED-HANDOFF section 11: intermittent GPU reset on chiral-ligand inputs, and slow first forwards.
 *
 * Runs the chiral case (SAM) and a non-chiral control (the enzyme's ORO) repeatedly, recording per-step wall
 * time and whether the fused GPU step loop was taken, so the "first two forwards are 2-4x slow" pattern and any
 * device loss show up as data rather than an anecdote.
 *
 *   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/diag-chiral-hang.js --runs=3 --nsteps=12
 */
export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const runs = Number(opt.runs ?? 3);
  const nsteps = Number(opt.nsteps ?? 12);
  const presets = (opt.presets ?? 'sm_sam,enzyme_oro').split(',');

  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px'; iframe.style.height = '960px';
  document.body.appendChild(iframe);
  await new Promise((res, rej) => { iframe.onload = res; iframe.onerror = rej; iframe.src = '/index.html'; });
  const win = iframe.contentWindow;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let lost = null;
  device.lost?.then((i) => { lost = `${i.reason}: ${i.message}`; });

  const t0 = performance.now();
  while (!win.rfd4pApp || win.rfd4pApp.els.predictBtn.disabled) {
    if (performance.now() - t0 > 60000) throw new Error('app init timeout');
    await sleep(100);
  }
  const app = win.rfd4pApp;
  const { RFD4P_PRESETS } = await import('/webgpu/index.js');
  const known = new Set(RFD4P_PRESETS.map((p) => p.id));
  const out = { availablePresets: [...known], nsteps, runs, results: [] };

  for (const preset of presets) {
    if (!known.has(preset)) { out.results.push({ preset, skipped: 'no such preset' }); continue; }
    for (let r = 0; r < runs; r += 1) {
      await app._loadPreset(preset);
      const n = app.currentBatchResult?.n ?? 0;
      const stepMs = [];
      let last = performance.now();
      const onStepProbe = () => { const now = performance.now(); stepMs.push(Math.round(now - last)); last = now; };
      // Observe step boundaries through the status line the app already updates.
      const obs = new win.MutationObserver(() => {
        if (/step \d+\//.test(app.els.statusMessage.textContent)) onStepProbe();
      });
      obs.observe(app.els.statusMessage, { childList: true, characterData: true, subtree: true });
      app.lastResult = null;
      const t = performance.now();
      let threw = null;
      try {
        await app.runDesign({ nsteps, recycleSchedule: 'none' });
      } catch (e) { threw = String(e).slice(0, 300); }
      obs.disconnect();
      out.results.push({
        preset, n, run: r,
        totalMs: Math.round(performance.now() - t),
        ok: Boolean(app.lastResult?.cif),
        threw,
        status: app.els.statusMessage.textContent.slice(0, 180),
        stepMs,
        firstTwoVsMedian: (() => {
          if (stepMs.length < 6) return null;
          const tail = [...stepMs.slice(2)].sort((a, b) => a - b);
          const med = tail[Math.floor(tail.length / 2)] || 1;
          return { med, s0: +(stepMs[0] / med).toFixed(2), s1: +(stepMs[1] / med).toFixed(2) };
        })(),
        deviceLost: lost,
      });
      await sleep(1200);
      if (lost) return { ...out, lost, note: 'device lost; stopping' };
    }
  }
  return { ...out, lost };
}
