/**
 * WebGPU device-loss recovery in the studio: destroy the device mid-generation and between runs; the failed run
 * must say the GPU was reset, and the next Generate must succeed on a new device (web/app.js `_watchDevice`).
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-recovery.js
 */
export async function main() {
  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px';
  iframe.style.height = '960px';
  document.body.appendChild(iframe);
  await new Promise((resolve, reject) => { iframe.onload = resolve; iframe.onerror = reject; iframe.src = '/index.html'; });
  const win = iframe.contentWindow;
  const t0 = performance.now();
  while (!win.rfd4pApp || win.rfd4pApp.els.predictBtn.disabled) {
    if (performance.now() - t0 > 60000) throw new Error('app init timeout');
    await new Promise((r) => setTimeout(r, 100));
  }
  const app = win.rfd4pApp;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const run = async (nsteps = 8) => {
    await app.runDesign({ nsteps, recycleSchedule: 'none' });
    return { status: app.els.statusMessage.textContent.slice(0, 160), ok: Boolean(app.lastResult?.cif) && !app.els.statusMessage.classList.contains('error') };
  };
  // Destroy the device only once a design is genuinely mid-flight. This used to be a fixed 1.5 s sleep, which
  // raced the design itself: an 8-step design takes ~1.2 s on an M5, so the destroy landed after it had already
  // finished, nothing was interrupted, and every later assertion reported the wrong thing.
  const waitForStep = async (minStep) => {
    const t = performance.now();
    for (;;) {
      const m = /step (\d+)\/(\d+)/.exec(app.els.statusMessage.textContent);
      if (m && Number(m[1]) >= minStep) return;
      if (performance.now() - t > 30000) throw new Error('design never reached a mid-run step');
      await sleep(20);
    }
  };
  const out = {};
  out.first = await run();
  const dev1 = app.gpuDevice;
  // 1. Lose the device mid-generation.
  app.lastResult = null;
  // Long enough that there is a comfortable window to interrupt it on any hardware.
  const pending = run(40);
  await waitForStep(4);
  dev1.destroy();
  out.midRun = await pending;
  out.lostInfo = app.gpuDeviceLost;
  // 2. The next Generate must recover on a new device.
  app.lastResult = null;
  out.afterMidRun = await run();
  out.newDevice = app.gpuDevice && app.gpuDevice !== dev1;
  // 3. Lose it between runs.
  const dev2 = app.gpuDevice;
  if (!dev2) return { ...out, allOk: false, note: 'no device after the mid-run recovery', lostAfter: app.gpuDeviceLost };
  dev2.destroy();
  await sleep(200);
  app.lastResult = null;
  out.afterIdleLoss = await run();
  out.thirdDevice = app.gpuDevice && app.gpuDevice !== dev2;
  out.allOk = out.first.ok && !out.midRun.ok && /GPU was reset/.test(out.midRun.status) && out.afterMidRun.ok &&
    out.newDevice && out.afterIdleLoss.ok && out.thirdDevice;
  return out;
}
