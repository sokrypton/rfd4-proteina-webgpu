import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { foldInputFromRefoldCif, isDictionaryCode, restoreLigandNames } from '../web/refold-input.js';

// What RFD4's AF3 oracle (AF3Oracle._build_manifest, run by tools/oracle/af3_refold_input.py) makes of a PTE design:
// a chain of 19 residues with a carboxy-lysine (KCX) in it, two Zn, a ligand outside the PDB's dictionary ("L:0") and
// a one-atom "W:1" (a hydroxide). A trimmed copy of a design this page made, so the file is the oracle's, not ours.
const ORACLE = readFileSync(new URL('./fixtures/pte-refold-input.cif', import.meta.url), 'utf8');

test('the oracle\'s input reads as a fold input: a modified residue stays in its chain on its parent\'s letter', () => {
  const input = foldInputFromRefoldCif(ORACLE);
  assert.equal(input.sequence.length, 19);
  assert.deepEqual(input.chainKinds, ['protein']);
  assert.equal(input.modifications.length, 1);
  const [mod] = input.modifications;
  assert.deepEqual({ chain: mod.chain, code: mod.code }, { chain: 0, code: 'KCX' });
  assert.equal(input.sequence[mod.position - 1], 'K', 'the letter at the modification is the parent\'s');
});

test('a ligand the PDB does not have is the component its own block defines, under a code the fold can ask for', () => {
  const input = foldInputFromRefoldCif(ORACLE);
  assert.deepEqual(input.ligandCodes, ['ZN', 'ZN', 'ZZ0', 'ZZ1'], 'ZN is a dictionary code; L:0 and W:1 are not');
  assert.deepEqual(input.standIns, { ZZ0: 'L:0', ZZ1: 'W:1' });
  assert.deepEqual(Object.keys(input.components).sort(), ['KCX', 'ZN', 'ZZ0', 'ZZ1']);
  assert.match(input.components.ZZ0, /^data_ZZ0$/m);
  assert.match(input.components.ZZ0, /^_chem_comp\.id\s+ZZ0$/m, 'the block is renamed, not the atoms: every comp_id is the code the fold asks for');
  assert.ok(!input.components.ZZ0.includes('L:0'));
  assert.match(input.components.ZZ1, /^_chem_comp_atom\.charge\s+-1$/m, 'a lone atom keeps its charge');
});

test('a modified residue\'s block names its parent, which the fold takes its restype from', () => {
  const { components } = foldInputFromRefoldCif(ORACLE);
  assert.match(components.KCX, /_chem_comp\.mon_nstd_parent_comp_id\s+LYS/, 'the oracle leaves "?"; the canonical letter is K');
});

test('the peptide links of a modified residue are the polymer\'s own and are not declared bonds', () => {
  assert.deepEqual(foldInputFromRefoldCif(ORACLE).bonds, []);
});

test('any other covalent bond is declared to the fold, between the chains\' places in its order', () => {
  // (the oracle writes its one peptide-link row as unlooped pairs; here a ligand bond joins it, as a loop)
  const row = ['9', 'covale', 'sing', 'A', 'D', 'KCX', 'L:0', '117', '1', 'NZ', 'P1'];
  const looped = ORACLE.replace(/_struct_conn\.id[\s\S]*?\n#\n/, [
    'loop_', ...['id', 'conn_type_id', 'pdbx_value_order', 'ptnr1_label_asym_id', 'ptnr2_label_asym_id', 'ptnr1_label_comp_id',
      'ptnr2_label_comp_id', 'ptnr1_label_seq_id', 'ptnr2_label_seq_id', 'ptnr1_label_atom_id', 'ptnr2_label_atom_id'].map((c) => `_struct_conn.${c}`),
    row.join(' '), '#', ''].join('\n'));
  assert.notEqual(looped, ORACLE);
  const { bonds, chains, modifications } = foldInputFromRefoldCif(looped);
  assert.equal(bonds.length, 1);
  assert.deepEqual(bonds[0].from, { asym: 0, residue: modifications[0].position, atom: 'NZ' });
  assert.deepEqual(bonds[0].to, { asym: chains.indexOf('D'), residue: 1, atom: 'P1' });
});

test('templates are the blocks that carry a _refold_template: a whole chain, or the residues it keeps', () => {
  const block = (ids) => `data_template_chain_A\n#\n_entry.id template_chain_A\n#\n${ORACLE.match(/loop_\n_atom_site[\s\S]*?\n#\n/)[0]}`
    + `_refold_template.chain_id A\n_refold_template.query_indices "${ids}"\n_refold_template.template_indices "${ids}"\n#\n`;
  const whole = foldInputFromRefoldCif(ORACLE + block(Array.from({ length: 19 }, (_, i) => i).join(' ')));
  assert.equal(whole.templates.length, 1);
  assert.equal(whole.templates[0].chainId, 'A');
  assert.equal(whole.templates[0].chain, 0);
  assert.equal(whole.templates[0].mapping, undefined, 'residue for residue is the chain, templated');
  const part = foldInputFromRefoldCif(ORACLE + block('2 3 5'));
  assert.deepEqual(part.templates[0].mapping, [[2, 2], [3, 3], [5, 5]]);
  assert.ok(part.templates[0].text.includes('_atom_site.'), 'the template is the chain\'s atoms, as the fold reads a CIF');
});

test('a modified residue the input does not define is refused, by name and place', () => {
  const broken = ORACLE.replace(/^data_KCX[\s\S]*?(?=^data_ZN)/m, '');
  assert.throws(() => foldInputFromRefoldCif(broken), /chain A has KCX at \d+/);
});

test('a ligand with neither a block nor a dictionary code is refused', () => {
  const broken = ORACLE.replace(/^data_W:1[\s\S]*$/m, '');
  assert.throws(() => foldInputFromRefoldCif(broken), /ligand W:1/);
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
