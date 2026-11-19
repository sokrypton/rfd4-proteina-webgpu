/**
 * The card's "+ seq": a target typed as a sequence, from the start screen and under a job, then + new and Generate,
 * through the card's own buttons. The peptide keeps its sequence and is built by the model; the job reads as a binder.
 *   node tools/gpu-chrome.mjs tools/gpu/check-add-sequence.js [--seq=...] [--nsteps=60]
 */
export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const seq = opt.seq ?? 'GSSPEEDLRKLLEELKRRSG';
  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px'; iframe.style.height = '960px';
  document.body.appendChild(iframe);
  await new Promise((res, rej) => { iframe.onload = res; iframe.onerror = rej; iframe.src = '/index.html'; });
  const win = iframe.contentWindow; const d = win.document;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const until = async (fn, what, ms = 60000) => { const t = performance.now(); while (!fn()) { if (performance.now() - t > ms) throw new Error(`timed out: ${what}`); await sleep(50); } };
  await until(() => win.rfd4pApp?.card && d.getElementById('rfd4p-start-sq'), 'card');
  const app = win.rfd4pApp; const card = app.card;
  const submitForm = async (text) => {
    const form = [...d.querySelectorAll('form.rfd4p-add-structure')].find((f) => !f.hidden && /sequence/i.test(f.textContent));
    if (!form) throw new Error('no + seq form');
    form.querySelector('textarea, input').value = text;
    form.requestSubmit();
    return form;
  };
  // A bad sequence says why and adds nothing.
  d.getElementById('rfd4p-start-sq').click(); await sleep(200);
  const bad = await submitForm('GSXB'); await sleep(300);
  const badStatus = bad.querySelector('.rfd4p-add-status')?.textContent ?? '';
  // From the start screen: the peptide opens as a new job.
  const startForm = await submitForm(`>peptide\n${seq}`);
  await until(() => card.spec()?.chains?.length === 1, 'peptide job', 20000).catch((err) => {
    throw new Error(`${err.message}: form says "${startForm.querySelector('.rfd4p-add-status')?.textContent}",`
      + ` start "${d.getElementById('rfd4p-fetch-status')?.textContent}", chains ${JSON.stringify(card.spec()?.chains?.map((c) => c.role))}`);
  });
  const startSpec = card.spec();
  // + new beside it, then a second sequence under the job.
  d.getElementById('rfd4p-add-seq').click();
  await until(() => card.spec()?.chains?.length === 2, '+ new');
  await sleep(500);
  const spec = card.spec();
  d.getElementById('rfd4p-add-sq').click(); await sleep(200);
  await submitForm('GWK');
  await until(() => card.spec()?.chains?.length === 3, '+ seq under a job');
  const threeChains = card.spec().chains.map((c) => ({ id: c.id, role: c.role }));
  d.getElementById('rfd4p-undo').click();
  await until(() => card.spec()?.chains?.length === 2, 'undo');
  await sleep(500);
  app.nsteps = Number(opt.nsteps ?? 60);
  await card.generate();
  const pdb = app.lastResult?.pdb ?? '';
  const three = { ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V' };
  const ca = new Map();
  for (const l of pdb.split('\n')) {
    if (!l.startsWith('ATOM') || l.slice(12, 16).trim() !== 'CA') continue;
    if (!ca.has(l[21])) ca.set(l[21], []);
    ca.get(l[21]).push({ aa: three[l.slice(17, 20)] ?? 'X', xyz: [+l.slice(30, 38), +l.slice(38, 46), +l.slice(46, 54)] });
  }
  const pepId = startSpec.chains[0].id;
  const pep = ca.get(pepId) ?? []; const binder = [...ca].find(([k]) => k !== pepId)?.[1] ?? [];
  const near = pep.filter((p) => binder.some((b) => Math.hypot(...p.xyz.map((v, k) => v - b.xyz[k])) < 8)).length;
  const result = {
    badStatus,
    start: { task: startSpec.task.key, roles: startSpec.chains.map((c) => c.role) },
    job: { task: spec.task.key, label: spec.task.label, roles: spec.chains.map((c) => `${c.id}:${c.role}`) },
    threeChains,
    status: d.getElementById('rfd4p-status-message')?.textContent ?? '',
    peptide: pep.map((r) => r.aa).join(''), binderLength: binder.length, peptideResiduesWithin8A: near,
  };
  const ok = /twenty amino acids: X, B/.test(badStatus) && result.start.roles[0] === 'sequence' && spec.task.key === 'binder'
    && result.peptide === seq && binder.length >= 60 && near >= 3;
  if (!ok) throw new Error(`+ seq check failed: ${JSON.stringify(result)}`);
  return { ok, ...result };
}
