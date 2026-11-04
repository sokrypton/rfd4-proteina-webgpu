export async function main(device, args) {
  const out = {};
  try {
    const iframe = document.createElement('iframe'); iframe.style.width = '1440px'; iframe.style.height = '1000px'; document.body.appendChild(iframe);
    await new Promise((r) => { iframe.onload = r; iframe.src = '/index.html'; });
    const w = iframe.contentWindow, d = w.document; const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const errs = []; out.errors = errs; w.addEventListener('error', (e) => errs.push(e.message));
    const app = () => w.rfd4pApp; const panel = () => app().designPanel;
    const ex = globalThis.__EX ?? 'protein_binder';
    for (let i = 0; i < 400 && !d.querySelector(`[data-example="${ex}"]`); i++) await sleep(50);
    d.querySelector(`[data-example="${ex}"]`).click(); await sleep(300);
    for (let i = 0; i < 1200 && (panel().building || !d.getElementById('rfd4p-start').hidden); i++) await sleep(50);
    await sleep(1500);
    // Warm the model once so weights load is not counted.
    const timeRun = async () => {
      for (let i = 0; i < 200 && !app().gpuModel; i++) await sleep(50);
      const model = app().gpuModel;
      const gen = model?.generate?.bind(model);
      const steps = [];
      if (gen) model.generate = async (batch, opts) => {
        const orig = opts.onStep; let last = performance.now();
        opts.onStep = async (info) => { const t0 = performance.now(); const gpu = t0 - last; await orig(info); const t1 = performance.now(); steps.push({ gpu, cb: t1 - t0, decoded: Boolean(info.decoded), depth: info.depth }); last = performance.now(); };
        return gen(batch, opts);
      };
      d.getElementById('seed-input').value = '7';
      const t = performance.now();
      d.getElementById('predict').click();
      for (let i = 0; i < 400 && !app().activeAbortController; i++) await sleep(25);
      for (let i = 0; i < 12000 && app().activeAbortController; i++) await sleep(25);
      const total = performance.now() - t;
      if (gen) model.generate = gen;
      const sum = (k) => steps.reduce((s, x) => s + x[k], 0);
      return { totalMs: Math.round(total), steps: steps.length, gpuMs: Math.round(sum('gpu')), callbackMs: Math.round(sum('cb')), perStepGpu: Math.round(sum('gpu') / steps.length), perStepCb: Math.round(sum('cb') / steps.length), n: app().lastResult?.structure?.atoms ? null : null };
    };
    out.warm = await timeRun();
    const variant = async (name, setup, undo) => { setup(); const r = await timeRun(); undo(); return [name, r.totalMs, r.steps, r.perStepGpu]; };
    const st = w.atomworksStudio; const rf = st.renderFrame;
    const res = [];
    for (let round = 0; round < 2; round++) {
      res.push(await variant('normal', () => {}, () => {}));
      res.push(await variant('noViewer', () => { st.renderFrame = async () => {}; }, () => { st.renderFrame = rf; }));
      const model = app().gpuModel; const gen = model.generate.bind(model);
      res.push(await variant('noDecode', () => { model.generate = (b, o) => gen(b, { ...o, decodeStep: () => false }); }, () => { model.generate = gen; }));
    }
    out.variants = res;
  } catch (e) { out.FAILED = String(e?.stack ?? e).split('\n').slice(0, 3).join(' '); }
  return out;
}
