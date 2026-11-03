/**
 * New residues that replaced residues turn back into them (the design panel's Length set back, or a fixed Structure):
 * in place, numbered, bonded and conditioned as before, with their coordinates, in one Undo.
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
let k = 0;
for (let r = 1; r <= 12; r += 1) {
  for (const [name, el] of [['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O']]) lines.push(atom(++k, name, el, r, r * 4 + (k % 4)));
}
const cif = [...header, ...lines, ''].join('\n');
const open = () => new StudioDoc(cif, 't', { preset: 'rcsb' });
const atomsOf = (doc, test) => Array.from({ length: doc.atoms.nAtoms }, (_, i) => i).filter((i) => test(Number(doc.atoms.res_id[i])));
const segment = (doc) => Array.from({ length: doc.atoms.nAtoms }, (_, i) => i).find((i) => /^\d+-\d+$/.test(String(doc.atoms.res_name[i])));
const numbers = (doc) => [...new Set(Array.from(doc.atoms.res_id, Number))];
const peptide = (doc, from, to) => {
  const a = doc.atoms;
  return (a.bonds || []).some(([u, v]) => [[u, v], [v, u]].some(([p, q]) => a.res_id[p] === from && a.atom_name[p] === 'C' && a.res_id[q] === to && a.atom_name[q] === 'N'));
};

test('new residues turn back into the residues they replaced', () => {
  const doc = open();
  const coords = JSON.stringify(atomsOf(doc, (r) => r >= 5 && r <= 7).map((i) => doc.atoms.coord[i]));
  doc.setConditions([{ condition: 'coordinate', atom_indices: atomsOf(doc, (r) => r >= 5 && r <= 7), enabled: false }]);
  doc.replaceWithSegments(atomsOf(doc, (r) => r >= 5 && r <= 7), 4, 9);
  const seg = segment(doc);
  assert.equal(doc.segmentOrigin(seg)?.count, 3);
  const steps = doc.undoStack.length;
  doc.restoreSegment(seg);
  assert.equal(segment(doc), undefined);
  assert.deepEqual(numbers(doc), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  assert.ok(peptide(doc, 4, 5) && peptide(doc, 6, 7) && peptide(doc, 7, 8), 'bonded in place');
  const back = atomsOf(doc, (r) => r >= 5 && r <= 7);
  assert.equal(JSON.stringify(back.map((i) => doc.atoms.coord[i])), coords, 'with their coordinates');
  const held = (i) => Number.isFinite(doc.atoms.annotations.condition_coordinate_1_atom?.[i]?.[0]);
  assert.ok(back.every((i) => !held(i)), 'and their conditions');
  assert.ok(atomsOf(doc, (r) => r === 4).every(held));
  assert.equal(doc.undoStack.length, steps + 1);
  doc.undo();
  assert.equal(segment(doc), seg);
});

test('a segment with other neighbours, or never residues, is not restorable', () => {
  const doc = open();
  doc.replaceWithSegments(atomsOf(doc, (r) => r >= 11), 2, 3);
  doc.insertSegment({ minimum: 5, maximum: 10, atomIndex: 0, before: true });
  const segs = Array.from({ length: doc.atoms.nAtoms }, (_, i) => i).filter((i) => /^\d+-\d+$/.test(String(doc.atoms.res_name[i])));
  assert.equal(segs.length, 2);
  assert.equal(doc.segmentOrigin(segs[0]), null);
  assert.equal(doc.segmentOrigin(segs[1])?.count, 2);
});
