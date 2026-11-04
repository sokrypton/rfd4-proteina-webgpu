export async function main() {
  const out = {};
  try {
    const iframe = document.createElement('iframe'); iframe.style.width = '1440px'; iframe.style.height = '1000px'; document.body.appendChild(iframe);
    await new Promise((r) => { iframe.onload = r; iframe.src = '/index.html'; });
    const w = iframe.contentWindow, d = w.document; const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const errs = []; out.errors = errs; w.addEventListener('error', (e) => errs.push(e.message)); w.addEventListener('unhandledrejection', (e) => errs.push(String(e.reason)));
    const app = () => w.rfd4pApp; const panel = () => app().designPanel;
    for (let i = 0; i < 400 && !d.getElementById('rfd4p-start-pdb'); i++) await sleep(50);
    await sleep(800);
    d.getElementById('rfd4p-start-pdb').click(); await sleep(300);
    const form = [...d.querySelectorAll('#rfd4p-start .rfd4p-add-structure')].find((f) => !f.hidden);
    form.querySelector('input').value = '6MRR'; form.dispatchEvent(new w.Event('submit', { cancelable: true, bubbles: true }));
    for (let i = 0; i < 300 && !d.getElementById('rfd4p-start').hidden; i++) await sleep(100);
    await sleep(1500);
    const a = panel()._inputDoc().atoms; const pick = [];
    for (let i = 0; i < a.nAtoms; i++) if (a.chain_id[i] === 'A' && [20, 24, 28].includes(Number(a.res_id[i]))) pick.push(i);
    w.atomworksStudio.setSelection(pick); await sleep(1200);
    const pop = d.querySelector('.rfd4p-sel-pop');
    const hot = pop.querySelector('[aria-label="Hotspot"]'); hot.click(); hot.dispatchEvent(new w.Event('change', { bubbles: true }));
    [...pop.querySelectorAll('button')].find((b) => b.textContent === 'Apply').click(); await sleep(1500);
    const where = globalThis.__WHERE;
    if (where === 'seq') { d.getElementById('rfd4p-add-seq').click(); await sleep(2000); }
    else {
      const a2 = panel()._inputDoc().atoms; let first = -1, last = 0;
      for (let i = 0; i < a2.nAtoms; i++) if (!a2.hetero[i]) { if (first < 0) first = i; last = i; }
      await panel()._mutate('/segments', { minimum: 100, maximum: 100, atom_index: where === 'start' ? first : last, before: where === 'start', new_chain: false, chain_id: null, chain_type: 'polypeptide(L)' });
      await sleep(2000);
    }
    const hl = []; const origH = w.atomworksStudio.setHighlight;
    w.atomworksStudio.setHighlight = (key, res, opt) => { if (key === 'hotspots') hl.push(res); return origH.call(w.atomworksStudio, key, res, opt); };
    d.getElementById('predict').click();
    for (let i = 0; i < 2000 && !app().activeAbortController; i++) await sleep(50);
    for (let i = 0; i < 6000 && app().activeAbortController; i++) await sleep(100);
    await sleep(2500);
    const sid = app().studioSessionId; const v = app().generatedViews.get(sid);
    out.viewing = v?.label;
    const store = (await import('/web/studio-api.js')).getSharedStudioStore();
    const doc = store.getSession(sid); const b = doc.atoms;
    const last = hl.at(-1) ?? [];
    out.finalHighlight = last.map((r) => {
      const idx = Array.from({ length: b.nAtoms }, (_, i) => i).filter((i) => String(b.chain_id[i]) === r.chain && Number(b.auth_seq_id?.[i] ?? b.res_id[i]) === r.residue);
      return `${r.chain}${r.residue}:${[...new Set(idx.map((i) => b.res_name[i]))].join('/')}`;
    });
    out.designHot = [...new Set(Array.from({ length: b.nAtoms }, (_, i) => i).filter((i) => b.annotations.condition_hotspot_1_atom?.[i]).map((i) => `${b.chain_id[i]}${b.res_id[i]}/${b.auth_seq_id?.[i]}${b.res_name[i]}`))];
    // Where 6MRR's E20 N24 R28 ended up in the design: find the kept stretch by sequence.
    const seq = []; let lastKey = null;
    for (let i = 0; i < b.nAtoms; i++) { if (b.chain_id[i] !== 'A') continue; const k = `${b.res_id[i]}${b.ins_code[i] || ''}`; if (k !== lastKey) { seq.push([k, b.auth_seq_id?.[i], b.res_name[i]]); lastKey = k; } }
    const one = { GLY: 'G', TRP: 'W', SER: 'S', THR: 'T', GLU: 'E', LEU: 'L', LYS: 'K', HIS: 'H', ARG: 'R', PHE: 'F', ILE: 'I', ASN: 'N', VAL: 'V', ASP: 'D', TYR: 'Y', GLN: 'Q', ALA: 'A', MET: 'M', PRO: 'P', CYS: 'C' };
    const str = seq.map((x) => one[x[2]] ?? 'X').join('');
    const at = str.indexOf('GWSTELEKHREELKEF');
    out.keptStartsAt = at >= 0 ? seq[at] : 'not found';
    out.keptHotspotsInDesign = at >= 0 ? [19, 23, 27].map((o) => seq[at + o]?.join('/')) : null;
  } catch (e) { out.FAILED = String(e?.stack ?? e).split('\n').slice(0, 3).join(' '); }
  return out;
}
