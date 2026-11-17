/**
 * Does a motif job draw its reference motif beside the design? Opens a Studio example through the card, generates,
 * and records every overlay the viewer is handed (window.atomworksStudio.setOverlay and frame options).
 *   node tools/gpu-chrome.mjs tools/gpu/probe-motif-overlay.js [--example=<id>]
 */
export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px'; iframe.style.height = '960px';
  document.body.appendChild(iframe);
  await new Promise((res, rej) => { iframe.onload = res; iframe.onerror = rej; iframe.src = '/index.html'; });
  const win = iframe.contentWindow;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const t0 = performance.now();
  while (!win.rfd4pApp?.card || !win.atomworksStudio?.openExample) { if (performance.now() - t0 > 30000) throw new Error('no card'); await sleep(50); }
  const app = win.rfd4pApp; const card = app.card; const api = win.atomworksStudio;
  const calls = [];
  for (const name of Object.keys(api)) {
    if (typeof api[name] !== 'function' || !/overlay|frame|render/i.test(name)) continue;
    const orig = api[name].bind(api);
    api[name] = (...a) => {
      const summary = a.map((x) => (typeof x === 'string' ? (x.length > 60 ? `<${x.length} chars>` : x)
        : x && typeof x === 'object' ? Object.fromEntries(Object.entries(x).map(([k, v]) => [k, typeof v === 'string' ? `<${v.length}>` : v && typeof v === 'object' ? Object.keys(v) : v])) : x));
      calls.push({ name, phase, args: summary, motifOverlay: JSON.stringify(a, (k, v) => (typeof v === 'string' && v.length > 200 ? `<${v.length}>` : v)).slice(0, 400) });
      return orig(...a);
    };
  }
  let phase = 'open';
  const examples = await (await win.fetch('/api/v1/examples')).json();
  const ex = examples.find((e) => e.id === opt.example) ?? examples.find((e) => /motif/i.test(`${e.id} ${e.label}`));
  const before = card.inputSessionId?.() ?? null;
  await api.openExample(ex.id);
  while (card.inputSessionId() === before || !card.spec()) await sleep(50);
  await sleep(1500);
  phase = 'generate';
  app.nsteps = 5;
  await card.generate();
  await sleep(1500);
  phase = 'after';
  const overlayCalls = calls.filter((c) => /motif/i.test(c.motifOverlay));
  return { examples: examples.map((e) => e.id), example: ex.id, task: card.spec()?.task?.key ?? null,
    callNames: [...new Set(calls.map((c) => `${c.phase}:${c.name}`))],
    lastByOverlay: Object.fromEntries(calls.filter((c) => c.name === 'setOverlay').map((c) => [c.args[0], c.args[1]])),
    motifCalls: overlayCalls.length,
    viewerStructures: await api.debugStructures?.() ?? null };
}
