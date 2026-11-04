export async function main() {
  const out = {};
  try {
    const iframe = document.createElement('iframe'); iframe.style.width = '1440px'; iframe.style.height = '1000px'; document.body.appendChild(iframe);
    await new Promise((r) => { iframe.onload = r; iframe.src = '/index.html'; });
    const w = iframe.contentWindow, d = w.document; const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const app = () => w.rfd4pApp; const panel = () => app().designPanel;
    for (let i = 0; i < 400 && !d.querySelector('[data-example="protein_binder"]'); i++) await sleep(50);
    d.querySelector('[data-example="protein_binder"]').click(); await sleep(300);
    for (let i = 0; i < 1200 && (panel().building || !d.getElementById('rfd4p-start').hidden); i++) await sleep(50);
    await sleep(1500);
    // A short run, enough to record the decoder.
    const model0 = () => app().gpuModel;
    d.getElementById('predict').click();
    for (let i = 0; i < 400 && !app().activeAbortController; i++) await sleep(25);
    for (let i = 0; i < 400 && !(model0()?.decoder?.decodeProgram); i++) await sleep(50);
    await sleep(3000);
    d.getElementById('predict').click();
    for (let i = 0; i < 400 && app().activeAbortController; i++) await sleep(50);
    const dec = model0().decoder; const e = dec.engine;
    out.hasTimestamp = e.device.features.has('timestamp-query');
    const prof = await e.profile(dec.decodeProgram);
    let total = 0, calls = 0; const rows = [];
    for (const [label, { ms, calls: c }] of prof) { total += ms; calls += c ?? 1; rows.push([label, +ms.toFixed(2), c]); }
    rows.sort((a, b) => b[1] - a[1]);
    out.dispatchMsSum = +total.toFixed(1); out.dispatches = calls; out.top = rows.slice(0, 10);
    // Wall time of the real decode program run.
    const t0 = performance.now(); for (let k = 0; k < 5; k++) await e.runAndRead([dec.decodeProgram], dec.buf.out, 16); out.wallMsPerDecode = +((performance.now() - t0) / 5).toFixed(1);
    out.nlayers = dec.nlayers;
    // And a trunk pass for scale.
    const tr = model0().trunk; if (tr?.passPrograms?.[0]) { const p2 = await tr.engine.profile(tr.passPrograms[0]); let t2 = 0, c2 = 0; for (const [, v] of p2) { t2 += v.ms; c2 += v.calls ?? 1; } out.trunkPassMs = +t2.toFixed(1); out.trunkDispatches = c2; }
  } catch (e) { out.FAILED = String(e?.stack ?? e).split('\n').slice(0, 3).join(' '); }
  return out;
}
