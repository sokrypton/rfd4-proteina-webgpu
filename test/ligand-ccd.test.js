import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { isCcdCode, ligandCifFromCcd } from '../web/ligand-ccd.js';
import { designSpec } from '../web/design-spec.js';
import { StudioDoc } from '../web/studio-api.js';

// A three-atom stand-in for a dictionary entry: one hydrogen, one quoted name, one atom with only model coordinates.
const CCD = `data_XYZ
_chem_comp.id XYZ
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
XYZ C1 C 0 9 9 9 0.000 0.000 0.000
XYZ "C1'" C 0 9 9 9 2.000 0.000 0.000
XYZ N1 N 1 4.000 0.000 0.000 ? ? ?
XYZ H1 H 0 9 9 9 1.000 1.000 0.000
`;

test('a dictionary entry becomes a one-residue ligand: heavy atoms, ideal coordinates, centred where asked', () => {
  const { cif, atoms, code } = ligandCifFromCcd(CCD, { chainId: 'C', center: [10, 20, 30] });
  assert.equal(code, 'XYZ');
  assert.equal(atoms, 3);
  const doc = new StudioDoc(cif, 'XYZ');
  const a = doc.atoms;
  assert.deepEqual(a.atom_name.map(String), ['C1', "C1'", 'N1']);
  assert.deepEqual([...new Set(a.chain_id.map(String))], ['C']);
  assert.ok(a.hetero.every(Boolean));
  const mean = [0, 1, 2].map((k) => a.coord.reduce((s, c) => s + c[k], 0) / a.nAtoms);
  mean.forEach((v, k) => assert.ok(Math.abs(v - [10, 20, 30][k]) < 1e-3, `centre ${k}`));
  assert.equal(Number(a.charge[2]), 1);
});

test('a ligand merged into a job reads as a ligand chain with its pose free', () => {
  const job = new StudioDoc(readFileSync(new URL('../web/studio/examples/small_molecule_binder.cif', import.meta.url), 'utf8'), 'sm');
  const before = designSpec(job.atoms).chains.length;
  job.combine([new StudioDoc(ligandCifFromCcd(CCD, { chainId: 'B' }).cif, 'XYZ')]);
  const chains = designSpec(job.atoms).chains;
  assert.equal(chains.length, before + 1);
  const added = chains[chains.length - 1];
  assert.equal(added.role, 'ligand');
  assert.deepEqual(added.ligands.map((l) => [l.name, l.pose]), [['XYZ', 'predicted']]);
  assert.notEqual(added.id, 'B', 'a taken chain letter is renamed');
});

test('ligand codes', () => {
  for (const ok of ['SAM', 'ZN', 'HEM', 'A1AAA']) assert.ok(isCcdCode(ok), ok);
  for (const bad of ['', 'sam', 'TOOLONG', 'S;M', 'A B']) assert.ok(!isCcdCode(bad), bad);
});
