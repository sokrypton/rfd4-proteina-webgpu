/**
 * A target given by its sequence alone (a disordered peptide, no structure): a chain whose residues have no
 * coordinates, opened as an entry's unresolved residues are (sequence fixed, structure free), a new chain beside it,
 * and a design. What the card's "+ seq" relies on from this engine.
 *   node tools/gpu-chrome.mjs tools/gpu/probe-sequence-target.js [--seq=...] [--nsteps=20]
 */
import { ATOM37_NAMES, UNIFIED_ATOM37_TOKENS, UNIFIED_RESTYPE_ATOM37_MASK } from '../../shared/rfd4p/encoding.js';

const ONE = { A: 'ALA', R: 'ARG', N: 'ASN', D: 'ASP', C: 'CYS', Q: 'GLN', E: 'GLU', G: 'GLY', H: 'HIS', I: 'ILE',
  L: 'LEU', K: 'LYS', M: 'MET', F: 'PHE', P: 'PRO', S: 'SER', T: 'THR', W: 'TRP', Y: 'TYR', V: 'VAL' };

/** An mmCIF of one protein chain with every heavy atom of each residue and no coordinates ('?'). */
export function sequenceChainCif(seq, chainId = 'B') {
  const rows = [];
  [...seq].forEach((aa, r) => {
    const res = ONE[aa];
    if (!res) throw new Error(`not an amino acid: ${aa}`);
    const mask = UNIFIED_RESTYPE_ATOM37_MASK[UNIFIED_ATOM37_TOKENS.indexOf(res)];
    ATOM37_NAMES.forEach((name, k) => {
      if (!mask[k]) return;
      rows.push(['ATOM', rows.length + 1, name[0], name, res, chainId, 1, r + 1, r + 1, res, chainId, name,
        '?', '?', '?', '1.00', '0.00', '?', 1, '?', '.'].join(' '));
    });
  });
  return ['data_seq', 'loop_', '_entity.id', '_entity.type', '1 polymer', 'loop_', '_entity_poly.entity_id', '_entity_poly.type',
    "1 'polypeptide(L)'", 'loop_',
    ...['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_comp_id', 'label_asym_id', 'label_entity_id',
      'label_seq_id', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id', 'Cartn_x', 'Cartn_y', 'Cartn_z',
      'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge', 'pdbx_PDB_model_num', 'pdbx_PDB_ins_code', 'label_alt_id']
      .map((c) => `_atom_site.${c}`), ...rows, ''].join('\n');
}

export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const seq = opt.seq ?? 'GSSPEEDLRKLLEELKRRSG';
  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px'; iframe.style.height = '960px';
  document.body.appendChild(iframe);
  await new Promise((res, rej) => { iframe.onload = res; iframe.onerror = rej; iframe.src = '/index.html'; });
  const win = iframe.contentWindow;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const t0 = performance.now();
  while (!win.rfd4pApp?.card || !win.atomworksStudio?.openSession) { if (performance.now() - t0 > 30000) throw new Error('no card'); await sleep(50); }
  const app = win.rfd4pApp; const card = app.card;
  const api = async (path, body, method = 'POST') => {
    const r = await win.fetch(`/api/v1${path}`, { method, ...(body instanceof win.FormData ? { body } : body ? { body: JSON.stringify(body), headers: { 'content-type': 'application/json' } } : {}) });
    if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
    return r.json();
  };
  const form = new win.FormData();
  form.append('file', new win.Blob([sequenceChainCif(seq)], { type: 'chemical/x-mmcif' }), 'peptide.cif');
  form.append('parse_preset', 'rcsb');
  form.append('keep_unresolved', 'true');
  form.append('fix_all', 'true');
  const { session_id: sid } = await api('/sessions', form);
  await api(`/sessions/${sid}/segments`, { minimum: 60, maximum: 80, atom_index: null, before: false, new_chain: true, chain_id: null, chain_type: 'polypeptide(L)' });
  await win.atomworksStudio.openSession(sid);
  while (card.inputSessionId?.() !== sid || !card.spec()) await sleep(50);
  await sleep(1000);
  const spec = card.spec();
  app.nsteps = Number(opt.nsteps ?? 20);
  await card.generate();
  const status = win.document.getElementById('rfd4p-status-message')?.textContent ?? '';
  const pdb = app.lastResult?.pdb ?? '';
  // The peptide's residues in the design: its sequence kept, and a chain of real geometry (CA-CA ~3.8 A).
  const ca = new Map();
  for (const l of pdb.split('\n')) {
    if (!l.startsWith('ATOM') || l.slice(12, 16).trim() !== 'CA') continue;
    const ch = l[21]; if (!ca.has(ch)) ca.set(ch, []);
    ca.get(ch).push({ res: l.slice(17, 20), xyz: [+l.slice(30, 38), +l.slice(38, 46), +l.slice(46, 54)] });
  }
  const chains = Object.fromEntries([...ca].map(([ch, rs]) => {
    const d = rs.slice(1).map((r, i) => Math.hypot(...r.xyz.map((v, k) => v - rs[i].xyz[k])));
    const three = Object.fromEntries(Object.entries(ONE).map(([a, b]) => [b, a]));
    return [ch, { n: rs.length, seq: rs.map((r) => three[r.res] ?? 'X').join(''),
      caCa: d.length ? [Math.min(...d).toFixed(2), Math.max(...d).toFixed(2)] : null }];
  }));
  // Contact: the peptide residues with a binder CA within 8 A, and the closest CA pair.
  let minD = Infinity; let touched = 0;
  for (const b of ca.get('B') ?? []) {
    let near = false;
    for (const c of ca.get('C') ?? []) { const d = Math.hypot(...b.xyz.map((v, k) => v - c.xyz[k])); minD = Math.min(minD, d); if (d < 8) near = true; }
    if (near) touched += 1;
  }
  return { contact: { peptideResiduesWithin8A: touched, minCaCa: +minD.toFixed(2) }, status, task: spec?.task?.key ?? spec?.task ?? null,
    pieces: spec?.chains?.map((c) => ({ id: c.id, role: c.role, pieces: c.pieces.map((p) => p.kind) })),
    chains, peptideKept: chains.B?.seq === seq };
}
