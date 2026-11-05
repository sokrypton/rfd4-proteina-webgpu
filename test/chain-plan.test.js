/**
 * A chain as a plan (kept runs of the chain as loaded, blocks of new residues, in order) and rebuilt from one: the
 * order, numbering, source names and bonds all come from the plan (studio-api.js `chainPlan`, `rebuildChain`).
 */
import test from 'node:test';
import assert from 'node:assert/strict';

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

/** The chain as read back: kept runs as "lo-hi", new blocks as "new min-max". */
const plan = (doc) => doc.chainPlan('A').parts.map((p) => (p.kind === 'kept' ? `${p.lo}-${p.hi}` : `new ${p.min}-${p.max}`));
/** Residues in order: number (and insertion code), "*" for a new block, its source name when it has one. */
const residues = (doc) => {
  const a = doc.atoms;
  const src = a.annotations['annotation_src-component_1_atom'];
  const out = [];
  let last = null;
  for (let i = 0; i < a.nAtoms; i += 1) {
    const k = `${a.res_id[i]}${a.ins_code[i]}${a.res_name[i]}`;
    if (k === last) continue;
    last = k;
    out.push(`${a.res_id[i]}${a.ins_code[i] || ''}${doc._isSegmentAtom(i) ? '*' : ''}${src?.[i] ? `=${src[i]}` : ''}`);
  }
  return out;
};
/** Bonded residue pairs across the chain (backbone links and new-block links), as "x-y" of `residues`. */
const links = (doc) => {
  const a = doc.atoms;
  const name = residues(doc);
  const starts = [];
  let last = null;
  for (let i = 0; i < a.nAtoms; i += 1) { const k = `${a.res_id[i]}${a.ins_code[i]}${a.res_name[i]}`; if (k !== last) { starts.push(i); last = k; } }
  const resOf = (i) => starts.findLastIndex((s) => s <= i);
  const out = new Set();
  for (const [u, v] of a.bonds || []) { const x = resOf(u); const y = resOf(v); if (x !== y) out.add(`${name[Math.min(x, y)]}-${name[Math.max(x, y)]}`); }
  return [...out].sort();
};

test('a chain as loaded is one kept part', () => {
  assert.deepEqual(plan(open()), ['1-30']);
});

test('split around new residues: numbered in the gap, joined both sides, the gap itself not bonded across', () => {
  const doc = open();
  doc.rebuildChain('A', [{ kind: 'kept', lo: 1, hi: 12 }, { kind: 'new', min: 20, max: 20 }, { kind: 'kept', lo: 19, hi: 30 }]);
  assert.deepEqual(plan(doc), ['1-12', 'new 20-20', '19-30']);
  const r = residues(doc);
  assert.deepEqual(r.slice(10, 15), ['11', '12', '18*', '19', '20']);
  const l = links(doc);
  assert.ok(l.includes('12-18*') && l.includes('18*-19'), 'the new block joined to 12 and 19');
  assert.ok(!l.includes('12-19'), 'no bond across the block');
});

test('a piece grown towards new residues next to it: the residues go between it and them', () => {
  const doc = open();
  doc.rebuildChain('A', [{ kind: 'kept', lo: 1, hi: 12 }, { kind: 'new', min: 20, max: 20 }, { kind: 'kept', lo: 19, hi: 30 }]);
  const parts = doc.chainPlan('A').parts;
  doc.rebuildChain('A', [{ kind: 'kept', lo: 1, hi: 15 }, parts[1], parts[2]]);
  assert.deepEqual(plan(doc), ['1-15', 'new 20-20', '19-30']);
  const r = residues(doc);
  assert.deepEqual(r.slice(13, 18), ['14', '15', '18*', '19', '20']);
  const l = links(doc);
  assert.ok(l.includes('14-15') && l.includes('15-18*') && l.includes('18*-19'));
  // The same, from the other side: 19-30 grown back to 16 goes after the block.
  doc.rebuildChain('A', [doc.chainPlan('A').parts[0], doc.chainPlan('A').parts[1], { kind: 'kept', lo: 16, hi: 30 }]);
  assert.deepEqual(plan(doc), ['1-15', 'new 20-20', '16-30']);
  assert.ok(links(doc).includes('15-15A*') || links(doc).some((x) => /^15-15A?\*/.test(x)), 'the block still joined to 15');
  assert.ok(links(doc).some((x) => /\*-16$/.test(x)), 'and to 16');
});

test('two blocks side by side, and a block at each end, all joined', () => {
  const doc = open();
  doc.rebuildChain('A', [{ kind: 'new', min: 3, max: 3 }, { kind: 'kept', lo: 1, hi: 10 }, { kind: 'new', min: 5, max: 5 }, { kind: 'new', min: 6, max: 6 }, { kind: 'kept', lo: 11, hi: 30 }, { kind: 'new', min: 4, max: 4 }]);
  assert.deepEqual(plan(doc), ['new 3-3', '1-10', 'new 5-5', 'new 6-6', '11-30', 'new 4-4']);
  const l = links(doc);
  const r = residues(doc);
  const blocks = r.filter((x) => x.includes('*'));
  assert.equal(blocks.length, 4);
  assert.equal(new Set(r.map((x) => x.replace(/=.*$/, ''))).size, r.length, 'no number used twice');
  for (const b of blocks) assert.equal(l.filter((x) => x.split('-').includes(b)).length, b === blocks[0] || b === blocks[3] ? 1 : 2, `${b} joined to its neighbours`);
});

test('a motif in its own order: numbered 1, 2, 3... each residue named by its source, new blocks joined', () => {
  const doc = open();
  doc.rebuildChain('A', [{ kind: 'new', min: 5, max: 5 }, { kind: 'kept', lo: 20, hi: 20 }, { kind: 'new', min: 8, max: 8 }, { kind: 'kept', lo: 5, hi: 6 }, { kind: 'new', min: 5, max: 5 }]);
  assert.deepEqual(plan(doc), ['new 5-5', '20-20', 'new 8-8', '5-6', 'new 5-5']);
  assert.deepEqual(residues(doc), ['0*', '1=A20', '2*', '3=A5', '4=A6', '5*']);
  const l = links(doc);
  assert.ok(l.includes('0*-1=A20') && l.includes('1=A20-2*') && l.includes('2*-3=A5') && l.includes('3=A5-4=A6') && l.includes('4=A6-5*'));
});

test('kept residues keep their settings; residues brought back come in as loaded; one undo', () => {
  const doc = open();
  doc.rebuildChain('A', [{ kind: 'kept', lo: 1, hi: 10 }]);
  const a = doc.atoms;
  const crd = a.annotations.condition_coordinate_1_atom;
  // Free residue 5 (no coordinates held), then grow the chain back to 20.
  for (let i = 0; i < a.nAtoms; i += 1) if (a.res_id[i] === 5) crd[i] = [NaN, NaN, NaN];
  const steps = doc.undoStack.length;
  doc.rebuildChain('A', [{ kind: 'kept', lo: 1, hi: 20 }]);
  assert.equal(doc.undoStack.length, steps + 1);
  const b = doc.atoms;
  const held = (id) => Array.from({ length: b.nAtoms }, (_, i) => i).filter((i) => b.res_id[i] === id).some((i) => Number.isFinite(b.annotations.condition_coordinate_1_atom[i]?.[0]));
  assert.equal(held(5), false, 'residue 5 still free');
  assert.equal(held(15), true, 'residue 15 back, fixed as loaded');
  doc.undo();
  assert.deepEqual(plan(doc), ['1-10']);
});
