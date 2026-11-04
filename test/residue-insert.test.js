/**
 * Residues brought back from their source entry (the design panel's strip): placed next to their source neighbours,
 * numbered and bonded as if never removed, in one Undo.
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

/** An entry as the panel opens it: every residue labelled with where it is in the entry. */
const open = () => {
  const doc = new StudioDoc(cif, 't', { preset: 'rcsb' });
  const a = doc.atoms;
  a.annotations['annotation_src-component_1_atom'] = Array.from({ length: a.nAtoms }, (_, i) => `A${a.res_id[i]}`);
  return doc;
};
const numbers = (doc) => [...new Set(Array.from(doc.atoms.res_id, Number))];
const peptide = (doc, from, to) => {
  const a = doc.atoms;
  return (a.bonds || []).some(([u, v]) => [[u, v], [v, u]].some(([p, q]) => a.res_id[p] === from && a.atom_name[p] === 'C' && a.res_id[q] === to && a.atom_name[q] === 'N'));
};

test('residues come back between their neighbours, bonded, in one undo', () => {
  const doc = open();
  const a = doc.atoms;
  doc.deleteAtoms(Array.from({ length: a.nAtoms }, (_, i) => i).filter((i) => a.res_id[i] >= 4 && a.res_id[i] <= 9));
  assert.deepEqual(numbers(doc), [1, 2, 3, 10, 11, 12]);
  const steps = doc.undoStack.length;

  const back = open();
  const b = back.atoms;
  back.deleteAtoms(Array.from({ length: b.nAtoms }, (_, i) => i).filter((i) => b.res_id[i] < 5 || b.res_id[i] > 7));
  doc.insertResiduesFrom(back, 'A', 'A');
  assert.deepEqual(numbers(doc), [1, 2, 3, 5, 6, 7, 10, 11, 12]);
  assert.ok(peptide(doc, 5, 6) && peptide(doc, 6, 7), 'the residues brought back are bonded to each other');
  assert.ok(!peptide(doc, 3, 5) && !peptide(doc, 7, 10), 'and not across the gaps either side');
  assert.ok(peptide(doc, 2, 3) && peptide(doc, 10, 11), 'the chain keeps its own bonds');
  assert.equal(doc.undoStack.length, steps + 1);

  doc.undo();
  assert.deepEqual(numbers(doc), [1, 2, 3, 10, 11, 12]);
});

test('a residue next to a kept one joins it', () => {
  const doc = open();
  const a = doc.atoms;
  doc.deleteAtoms(Array.from({ length: a.nAtoms }, (_, i) => i).filter((i) => a.res_id[i] > 6));
  const back = open();
  const b = back.atoms;
  back.deleteAtoms(Array.from({ length: b.nAtoms }, (_, i) => i).filter((i) => b.res_id[i] !== 7));
  doc.insertResiduesFrom(back, 'A', 'A');
  assert.deepEqual(numbers(doc), [1, 2, 3, 4, 5, 6, 7]);
  assert.ok(peptide(doc, 6, 7));
});

test('residues brought back where the "+" was pressed: out of source order, the chain is renumbered and keeps their names', () => {
  const doc = open();
  const a = doc.atoms;
  doc.deleteAtoms(Array.from({ length: a.nAtoms }, (_, i) => i).filter((i) => a.res_id[i] >= 4 && a.res_id[i] <= 9));
  const back = open();
  const b = back.atoms;
  back.deleteAtoms(Array.from({ length: b.nAtoms }, (_, i) => i).filter((i) => b.res_id[i] !== 5));
  // Before residue 2, as a motif in its own order would have it.
  const atom = Array.from({ length: a.nAtoms }, (_, i) => i).find((i) => a.res_id[i] === 2);
  doc.insertResiduesFrom(back, 'A', 'A', { atom, before: true });
  const names = [];
  const src = doc.atoms.annotations['annotation_src-component_1_atom'];
  for (let i = 0; i < doc.atoms.nAtoms; i += 1) if (!names.length || names.at(-1) !== src[i]) names.push(src[i]);
  assert.deepEqual(names, ['A1', 'A5', 'A2', 'A3', 'A10', 'A11', 'A12']);
  assert.deepEqual(numbers(doc), [1, 2, 3, 4, 5, 6, 7]);
});

test('in source order where the "+" was pressed, the numbers stay the entry\'s', () => {
  const doc = open();
  const a = doc.atoms;
  doc.deleteAtoms(Array.from({ length: a.nAtoms }, (_, i) => i).filter((i) => a.res_id[i] >= 4 && a.res_id[i] <= 9));
  const back = open();
  const b = back.atoms;
  back.deleteAtoms(Array.from({ length: b.nAtoms }, (_, i) => i).filter((i) => b.res_id[i] !== 5));
  const atom = Array.from({ length: a.nAtoms }, (_, i) => i).find((i) => a.res_id[i] === 3);
  doc.insertResiduesFrom(back, 'A', 'A', { atom, before: false });
  assert.deepEqual(numbers(doc), [1, 2, 3, 5, 10, 11, 12]);
});
