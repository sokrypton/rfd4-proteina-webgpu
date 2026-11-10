import assert from 'node:assert/strict';
import test from 'node:test';
import { foldInputFromDesign, isDictionaryCode, pdbWithParents, restoreLigandNames } from '../web/refold-input.js';

// A design as the page holds it: a chain with a carboxy-lysine (KCX, a modified residue) in it, a zinc, a ligand that is
// not in the PDB's dictionary ("L:0": three atoms and two bonds) and a one-atom "W:1" (a hydroxide).
const atom = (n, name, res, chain, num, x, serial = n) => `${res === 'ZN' || res === 'L:0' || res === 'W:1' ? 'HETATM' : 'ATOM  '}`
  + `${String(serial).padStart(5)} ${name.padEnd(3)}  ${res.padStart(3)} ${chain}${String(num).padStart(4)}    `
  + `${x.toFixed(3).padStart(8)}   0.000   0.000  1.00  0.00`;
const PDB = [
  atom(1, 'CA', 'MET', 'A', 1, 1), atom(2, 'CA', 'KCX', 'A', 2, 2), atom(3, 'CA', 'HIS', 'A', 3, 3),
  atom(4, 'ZN', 'ZN', 'B', 1, 4), atom(5, 'P1', 'L:0', 'C', 1, 5), atom(6, 'O5', 'L:0', 'C', 1, 6),
  atom(7, 'O4', 'L:0', 'C', 1, 7), atom(8, 'O3', 'W:1', 'D', 1, 8), 'TER', 'END'].join('\n');
const CIF = `data_x
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.pdbx_formal_charge
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
HETATM P P1 L:0 C C ? 1.0 0.0 0.0
HETATM O O5 L:0 C C ? 2.0 0.0 0.0
HETATM O O4 L:0 C C -1 1.0 1.5 0.0
HETATM O O3 W:1 D D -1 5.0 5.0 5.0
HETATM ZN ZN ZN B B +2 9.0 9.0 9.0
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
L:0 O5 P1 DOUB N
L:0 O4 P1 SING N
`;

test('a modified residue stays in its chain on its parent letter, as a modification', () => {
  const input = foldInputFromDesign(PDB, CIF);
  assert.equal(input.sequence, 'MKH');
  assert.deepEqual(input.chainKinds, ['protein']);
  assert.deepEqual(input.modifications, [{ chain: 0, position: 2, code: 'KCX' }]);
  assert.equal(input.needed, true);
});

test('a ligand outside the dictionary goes in under a stand-in code, built from the design atoms and bonds', () => {
  const input = foldInputFromDesign(PDB, CIF);
  assert.deepEqual(input.ligandCodes, ['ZN', 'ZZ0', 'ZZ1'], 'ZN is a dictionary code and is left to the dictionary');
  assert.deepEqual(input.standIns, { ZZ0: 'L:0', ZZ1: 'W:1' });
  const text = input.components.ZZ0;
  assert.match(text, /^data_ZZ0\n_chem_comp\.id ZZ0$/m);
  assert.equal((text.match(/^ZZ0 \S+ [A-Z] -?\d+ N /gm) ?? []).length, 3, 'three atoms');
  assert.equal((text.match(/^ZZ0 \S+ \S+ (?:SING|DOUB) [NY]$/gm) ?? []).length, 2, 'and two bonds');
  assert.match(text, /^ZZ0 O5 P1 DOUB N$/m);
  assert.match(text, /^ZZ0 O4 O -1 N /m, 'the formal charge comes with the atom');
  assert.match(input.components.ZZ1, /^ZZ1 O3 O -1 N 5\.000 5\.000 5\.000$/m, 'a lone atom is a component of one atom');
  assert.ok(!/ZN/.test(input.components.ZZ0), 'another ligand\'s atoms are not in it');
});

test('a design of standard residues and dictionary ligands needs nothing special', () => {
  const plain = PDB.split('\n').filter((l) => !/KCX|L:0|W:1/.test(l)).join('\n');
  const input = foldInputFromDesign(plain, '');
  assert.equal(input.needed, false);
  assert.deepEqual(input.ligandCodes, ['ZN']);
  assert.deepEqual(input.modifications, []);
});

test('a residue that is neither standard nor a known modification is refused by name', () => {
  assert.throws(() => foldInputFromDesign(PDB.replace(/KCX/g, 'QQQ'), CIF), /chain A has QQQ/);
});

test('templates read the chain with the modified residue as its parent, where it was', () => {
  const parents = pdbWithParents(PDB);
  assert.match(parents, / LYS A   2 /);
  assert.equal(parents.split('\n').length, PDB.split('\n').length);
});

test('the refold names its ligands by the stand-in codes and they are written back', () => {
  const refold = ['HETATM    1  P1  ZZ0 C   1       1.000   0.000   0.000', 'HETATM    2  CA  KCX A   2       2.000   0.000   0.000',
    'HETATM    3  O3  ZZ1 D   1       5.000   5.000   5.000'].join('\n');
  const back = restoreLigandNames(refold, { ZZ0: 'L:0', ZZ1: 'W:1' }).split('\n');
  assert.match(back[0], /^HETATM {4}1  P1  L:0 C/);
  assert.match(back[1], / KCX A/, 'a modified residue in the chain is not a ligand');
  assert.match(back[2], / W:1 D/);
});

test('which names are dictionary codes', () => {
  for (const code of ['ZN', 'SAM', 'A1AAA', 'KCX']) assert.ok(isDictionaryCode(code), code);
  for (const code of ['L:0', 'W:1', 'lig', 'TOOLONG']) assert.ok(!isDictionaryCode(code), code);
});
