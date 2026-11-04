/**
 * A plain structure as fetched: everything fixed, buffer ions dropped; "rigid" joins one rigid group.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { StudioDoc } from '../web/studio-api.js';
import { designSpec } from '../web/design-spec.js';

const atom = (k, group, el, name, comp, asym, seq, x) =>
  `${group} ${k} ${el} ${name} ${comp} ${asym} 1 ${seq} ${seq} ${comp} ${asym} ${name} ${x}.000 0.000 0.000 1.00 0.00 0 1`;
const header = ['data_t', 'loop_', ...['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_comp_id', 'label_asym_id',
  'label_entity_id', 'label_seq_id', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id', 'Cartn_x', 'Cartn_y',
  'Cartn_z', 'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge', 'pdbx_PDB_model_num'].map((c) => `_atom_site.${c}`)];
let k = 0;
const residues = [];
for (let r = 1; r <= 6; r += 1) {
  for (const [name, el] of [['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O'], ['CB', 'C']]) residues.push(atom(++k, 'ATOM', el, name, 'ALA', 'A', r, r * 4 + k % 5));
}
const cif = [...header, ...residues, atom(++k, 'HETATM', 'NA', 'NA', 'NA', 'B', '.', 40), atom(++k, 'HETATM', 'CL', 'CL', 'CL', 'C', '.', 44), ''].join('\n');

test('a fetched structure opens fixed, without buffer ions', () => {
  const doc = new StudioDoc(cif, 't', { preset: 'rcsb' });
  assert.deepEqual([...new Set(Array.from(doc.atoms.res_name, String))], ['ALA']);
  const spec = designSpec(doc.atoms);
  assert.equal(spec.task.key, 'nothing');
  assert.deepEqual(spec.chains[0].pieces.map((p) => [p.kind, p.keep]), [['kept', 'all']]);
});

test('"rigid" twice is one rigid group when the group is given', () => {
  const doc = new StudioDoc(cif, 't', { preset: 'rcsb' });
  const resAtoms = (lo, hi) => Array.from({ length: doc.atoms.nAtoms }, (_, i) => i).filter((i) => doc.atoms.res_id[i] >= lo && doc.atoms.res_id[i] <= hi);
  doc.setInternalCoordinateConditioning(resAtoms(1, 2));
  const group = doc.atoms.annotations.condition_CoordinateGroup_1_atom[0];
  doc.setInternalCoordinateConditioning(resAtoms(5, 6), true, 'atom', group);
  doc.setInternalCoordinateConditioning(resAtoms(3, 3));
  const groups = (lo, hi) => [...new Set(resAtoms(lo, hi).map((i) => doc.atoms.annotations.condition_CoordinateGroup_1_atom[i]))];
  assert.deepEqual(groups(5, 6), [group]);
  assert.notDeepEqual(groups(3, 3), [group]);
});

test('residues an entry did not resolve are dropped on loading, kept for the example recipes', () => {
  // Residues 1 and 4 of 6 with no coordinates (CIF "?"): an unmodelled end and loop.
  const unresolved = new Set([1, 4]);
  let n = 0;
  const rows = [];
  for (let r = 1; r <= 6; r += 1) {
    for (const [name, el] of [['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O']]) {
      const row = atom(++n, 'ATOM', el, name, 'ALA', 'A', r, r * 4 + n % 4);
      rows.push(unresolved.has(r) ? row.replace(/ -?\d+\.000 0\.000 0\.000 /, ' ? ? ? ') : row);
    }
  }
  const text = [...header, ...rows, ''].join('\n');
  const ids = (doc) => [...new Set(Array.from(doc.atoms.res_id, Number))];
  assert.deepEqual(ids(new StudioDoc(text, 't', { preset: 'rcsb' })), [2, 3, 5, 6]);
  assert.deepEqual(ids(new StudioDoc(text, 't', { preset: 'rcsb', keepUnresolved: true })), [1, 2, 3, 4, 5, 6]);
});
