/**
 * Every block of new residues stays bonded to the residues before and after it in its chain, whatever edits come
 * after it was added: removing a neighbour once left a block hanging from one end, and the model read a chain break
 * there (a 100-residue insert came back 29 A from the residue after it).
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { readFileSync } from 'node:fs';

import { buildBatch } from '../shared/rfd4p/pipeline/index.js';
import { designSpec } from '../web/design-spec.js';
import { StudioDoc } from '../web/studio-api.js';

const atom = (k, name, el, seq, x) =>
  `ATOM ${k} ${el} ${name} ALA A 1 ${seq} ${seq} ALA A ${name} ${x}.000 0.000 0.000 1.00 0.00 0 1`;
const header = ['data_t', 'loop_', ...['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_comp_id', 'label_asym_id',
  'label_entity_id', 'label_seq_id', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id', 'Cartn_x', 'Cartn_y',
  'Cartn_z', 'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge', 'pdbx_PDB_model_num'].map((c) => `_atom_site.${c}`)];
const lines = [];
let n = 0;
for (let r = 1; r <= 30; r += 1) for (const [name, el] of [['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O']]) lines.push(atom(++n, name, el, r, r * 4 + (n % 4)));
const open = () => new StudioDoc([...header, ...lines, ''].join('\n'), 't', { preset: 'rcsb' });
const idx = (doc, lo, hi) => Array.from({ length: doc.atoms.nAtoms }, (_, i) => i)
  .filter((i) => doc.atoms.res_id[i] >= lo && doc.atoms.res_id[i] <= hi && !doc._isSegmentAtom(i));

/** Blocks of new residues not bonded to their neighbours in the chain. */
function loose(doc) {
  const a = doc.atoms;
  const res = [];
  let last = null;
  for (let i = 0; i < a.nAtoms; i += 1) {
    const k = `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}:${a.res_name[i]}`;
    if (k !== last) { res.push({ s: i, e: i + 1, chain: String(a.chain_id[i]), seg: doc._isSegmentAtom(i) }); last = k; } else res.at(-1).e = i + 1;
  }
  const bonded = (x, y) => (a.bonds || []).some(([u, v]) => (u >= x.s && u < x.e && v >= y.s && v < y.e) || (v >= x.s && v < x.e && u >= y.s && u < y.e));
  const out = [];
  res.forEach((r, k) => {
    if (!r.seg) return;
    for (const o of [res[k - 1], res[k + 1]]) if (o && o.chain === r.chain && !bonded(r, o)) out.push(`new@${a.res_id[r.s]}${a.ins_code[r.s]}~${a.res_id[o.s]}`);
  });
  return out;
}

const cases = {
  'new residues added, then the residues after them removed': (d) => {
    d.insertSegment({ minimum: 10, maximum: 10, atomIndex: idx(d, 9, 9).at(-1), before: false });
    d.deleteAtoms(idx(d, 10, 12));
  },
  'two blocks with one residue between, that residue removed': (d) => {
    d.insertSegment({ minimum: 5, maximum: 5, atomIndex: idx(d, 9, 9).at(-1), before: false });
    d.insertSegment({ minimum: 5, maximum: 5, atomIndex: idx(d, 11, 11)[0], before: true });
    d.deleteAtoms(idx(d, 10, 10));
  },
  'a motif, then one of its residues removed': (d) => {
    d.scaffoldMotif([...idx(d, 5, 5), ...idx(d, 20, 20), ...idx(d, 12, 12)], { ordered: true });
    const a = d.atoms;
    const first = Array.from({ length: a.nAtoms }, (_, i) => i).find((i) => !d._isSegmentAtom(i));
    d.deleteAtoms(Array.from({ length: a.nAtoms }, (_, i) => i).filter((i) => !d._isSegmentAtom(i) && a.res_id[i] === a.res_id[first]));
  },
  'residues replaced, their length changed, then restored': (d) => {
    d.replaceWithSegments(idx(d, 10, 13), 8, 8);
    const seg = () => Array.from({ length: d.atoms.nAtoms }, (_, i) => i).find((i) => d._isSegmentAtom(i));
    d.setSegmentRange(seg(), 20, 30);
    d.restoreSegment(seg());
  },
  'a piece grown back towards new residues beside it, on either side': (d) => {
    d.insertSegment({ minimum: 20, maximum: 20, atomIndex: idx(d, 15, 15).at(-1), before: false });
    d.deleteAtoms(idx(d, 13, 15));
    d.deleteAtoms(idx(d, 16, 18));
    // 1-12 grown to 14 (after 12, before the new residues); 19-30 grown back to 17 (before 19, after them).
    d.restoreFromSource('A', [13, 14], { atom: idx(d, 12, 12).at(-1), before: false });
    d.restoreFromSource('A', [17, 18], { atom: idx(d, 19, 19)[0], before: true });
    const a = d.atoms;
    const order = [];
    for (let i = 0; i < a.nAtoms; i += 1) { const k = d._isSegmentAtom(i) ? 'new' : String(a.res_id[i]); if (order.at(-1) !== k) order.push(k); }
    assert.deepEqual(order.slice(10, 17), ['11', '12', '13', '14', 'new', '17', '18']);
  },
  'new residues at the end, the chain cropped and dragged back out': (d) => {
    d.deleteAtoms(idx(d, 21, 30));
    d.insertSegment({ minimum: 5, maximum: 5, atomIndex: idx(d, 20, 20).at(-1), before: false });
    d.restoreFromSource('A', [21, 22, 23]);
  },
};

for (const [name, edit] of Object.entries(cases)) {
  test(`new residues stay joined: ${name}`, () => {
    const doc = open();
    edit(doc);
    assert.deepEqual(loose(doc), []);
  });
}

test('a motif residue removed leaves its two blocks of new residues joined end to end, as the model reads them', () => {
  // The blocks either side share a name and a number ("15-30" at 4 and 4A): the bonds between them are told apart by
  // the insertion code when the job is read back, or both landed on the second and the model saw two chain breaks.
  const doc = new StudioDoc(readFileSync(new URL('../web/studio/examples/motif_scaffolding.cif', import.meta.url), 'utf8'), 'motif');
  const before = designSpec(doc.atoms).contig;
  const a = doc.atoms;
  doc.removeAtoms(Array.from({ length: a.nAtoms }, (_, i) => i).filter((i) => String(a.chain_id[i]) === 'A' && a.res_id[i] === 3 && !doc._isSegmentAtom(i)));
  const res = buildBatch(doc.toCif(), { seed: 1 });
  const pl = res.batch.token_bond_path_length;
  const n = pl.shape.at(-1);
  const toks = [];
  for (let i = 0; i < res.atoms.res_id.length; i += 1) {
    const t = Number(res.atoms.token_id[i]);
    if (String(res.atoms.chain_id[i]) === 'A' && toks.at(-1) !== t) toks.push(t);
  }
  for (let k = 0; k + 1 < toks.length; k += 1) assert.equal(pl.data[toks[k] * n + toks[k + 1]], 1, `tokens ${k} and ${k + 1} joined`);
  // Put back between the two blocks (the "+" there), the motif is as it was, its residues still named by the entry.
  const plan = doc.chainPlan('A');
  const parts = plan.parts.map((q) => (q.kind === 'kept' ? { kind: 'kept', lo: q.lo, hi: q.hi } : { kind: 'new', atom: q.atom, min: q.min, max: q.max }));
  parts.splice(3, 0, { kind: 'kept', lo: 3, hi: 3 });
  doc.rebuildChain('A', parts);
  assert.equal(designSpec(doc.atoms).contig, before);
  const names = new Set(Array.from({ length: doc.atoms.nAtoms }, (_, i) => doc.atoms.annotations['annotation_src-component_1_atom']?.[i]).filter((x) => /^A\d/.test(x ?? '')));
  assert.deepEqual([...names], ['A38', 'A14', 'A99']);
});

test('new residues joining two parts of a chain are numbered straight through, no hole where residues were cut', () => {
  // A1-9, 5 new, A12-30 (10 and 11 cut): the job is numbered 1-9, 10-14, 15-33, as the design will be; a hole after 9
  // broke the viewer's cartoon there and told the model a residue was missing.
  const doc = open();
  const k = idx(doc, 10, 10)[0];
  doc.rebuildChain('A', [{ kind: 'kept', lo: 1, hi: 9 }, { kind: 'new', min: 5, max: 5 }, { kind: 'kept', lo: 12, hi: 30 }]);
  assert.ok(k >= 0);
  const res = buildBatch(doc.toCif(), { seed: 1 });
  const ids = [];
  for (let i = 0; i < res.atoms.res_id.length; i += 1) if (ids.at(-1) !== res.atoms.res_id[i]) ids.push(res.atoms.res_id[i]);
  assert.deepEqual(ids, Array.from({ length: 33 }, (_, i) => i + 1));
});
