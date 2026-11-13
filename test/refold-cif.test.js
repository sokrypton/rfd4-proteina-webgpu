import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { describeComponent, dictionaryCodes, parentsOf, pdbWithParents, refoldCifFromDesign } from '../web/refold-cif.js';
import { foldInputFromRefoldCif } from '../web/refold-input.js';

const read = (name) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8');
const DESIGN = read('pte-design.cif');       // a trimmed PTE design as this page holds it
const ORACLE = read('pte-refold-input.cif');  // RFD4's AF3 oracle's input for that design (tools/oracle/af3_refold_input.py)
const KCX = read('KCX.cif');                  // the PDB's entry for KCX, which is what the oracle's user CCD takes it from

// LocalFold's own reading of a dictionary block, from the pinned vendor copy
const { commit } = JSON.parse(readFileSync(new URL('../localfold.json', import.meta.url), 'utf8'));
const { parseCcdComponent } = await import(`../vendor/localfold-${commit}/shared/af3/featurise/ccd-component.js`);

const mine = foldInputFromRefoldCif(refoldCifFromDesign(DESIGN, { name: 'pte', dictionary: { KCX } }));
const theirs = foldInputFromRefoldCif(ORACLE);

test('written for a design, the refold input reads as the oracle\'s does: chains, sequence, modifications, ligands, bonds', () => {
  for (const key of ['sequence', 'chainKinds', 'ligandCodes', 'modifications', 'standIns', 'bonds', 'chains']) {
    assert.deepEqual(mine[key], theirs[key], key);
  }
  assert.deepEqual(Object.keys(mine.components).sort(), Object.keys(theirs.components).sort());
});

test('a ligand the dictionary lacks is defined by the design\'s own atoms and bonds, as the oracle defines it', () => {
  for (const code of ['ZZ0', 'ZZ1']) {
    const a = parseCcdComponent(mine.components[code]);
    const b = parseCcdComponent(theirs.components[code]);
    assert.deepEqual(a.atoms.map((t) => [t.name, t.element, t.charge]), b.atoms.map((t) => [t.name, t.element, t.charge]), code);
    const named = (c) => c.bonds.map((t) => `${[c.atoms[t.from].name, c.atoms[t.to].name].sort().join('-')}:${t.order}`).sort();
    assert.deepEqual(named(a), named(b), `${code} bonds`);
    a.atoms.forEach((t, i) => assert.ok(Math.hypot(t.x - b.atoms[i].x, t.y - b.atoms[i].y, t.z - b.atoms[i].z) < 2e-3, `${code} ${t.name}`));
  }
  assert.equal(parseCcdComponent(mine.components.ZZ0).atoms.length, 18);
});

test('a residue the dictionary has is its dictionary entry, as in the oracle, with the same atoms and bonds', () => {
  const a = parseCcdComponent(mine.components.KCX);
  const b = parseCcdComponent(theirs.components.KCX);
  assert.equal(a.parent, 'LYS');
  assert.equal(b.parent, 'LYS');
  assert.deepEqual(a.atoms.map((t) => t.name).sort(), b.atoms.map((t) => t.name).sort());
  const named = (c) => c.bonds.map((t) => [c.atoms[t.from].name, c.atoms[t.to].name].sort().join('-')).sort();
  assert.deepEqual(named(a), named(b));
});

test('a residue the dictionary has no entry for cannot be told from a ligand, and is refused by name', () => {
  assert.throws(() => refoldCifFromDesign(DESIGN, { name: 'pte' }), /chain A has KCX.*dictionary has no entry/);
});

test('any residue the dictionary describes as a modification of a standard one is read: nothing here names them', () => {
  // N-formylmethionine, which no list in this code has ever held: the entry alone says it is a peptide link on MET
  const fme = KCX.replace(/^data_KCX/m, 'data_FME').replace(/\bKCX\b/g, 'FME').replace(/(_chem_comp\.mon_nstd_parent_comp_id\s+)LYS/, '$1MET');
  const input = foldInputFromRefoldCif(refoldCifFromDesign(DESIGN.replaceAll('KCX', 'FME'), { name: 'pte', dictionary: { FME: fme } }));
  assert.deepEqual(input.modifications.map((m) => m.code), ['FME']);
  assert.equal(input.sequence[input.modifications[0].position - 1], 'M', 'the letter is its parent\'s, which the entry names');
  assert.match(input.components.FME, /_chem_comp\.mon_nstd_parent_comp_id\s+MET/);
  assert.deepEqual(parentsOf({ FME: fme, ZN: 'data_ZN\n_chem_comp.id ZN\n_chem_comp.type NON-POLYMER\n_chem_comp.mon_nstd_parent_comp_id ?\n' }), { FME: 'MET' });
});

test('what an entry says of a component: its link type, and the standard residue it modifies', () => {
  assert.deepEqual(describeComponent(KCX), { kind: 'protein', parent: 'LYS', type: 'L-PEPTIDE LINKING' });
  assert.equal(describeComponent('data_X\n_chem_comp.type "RNA LINKING"\n_chem_comp.mon_nstd_parent_comp_id C\n').kind, 'rna');
  assert.equal(describeComponent('data_X\n_chem_comp.type NON-POLYMER\n_chem_comp.mon_nstd_parent_comp_id ?\n').parent, null);
  assert.equal(describeComponent(undefined), null);
});

test('the components a design has that the dictionary may have an entry for', () => {
  assert.deepEqual(dictionaryCodes(DESIGN).sort(), ['KCX', 'ZN'], 'L:0 and W:1 are not dictionary codes');
});

test('atoms with no coordinates are left out, as the oracle leaves them', () => {
  const nan = DESIGN.replace(/^(ATOM .* KCX A .* CD .*?)\s+-?[\d.]+\s+-?[\d.]+\s+-?[\d.]+\s+1$/m, '$1 nan nan nan 1');
  assert.notEqual(nan, DESIGN);
  const structure = (text) => text.split(/^(?=data_)/m)[0];
  const has = (text) => /^ATOM \S+ CD \. KCX /m.test(structure(text));
  assert.ok(has(refoldCifFromDesign(DESIGN, { name: 'pte', dictionary: { KCX } })), 'present when it has coordinates');
  assert.ok(!has(refoldCifFromDesign(nan, { name: 'pte', dictionary: { KCX } })), 'and gone when it has none');
});

test('templates: the chain\'s atoms and the residues kept, read back as the fold takes them', () => {
  const kept = foldInputFromRefoldCif(refoldCifFromDesign(DESIGN, { name: 'pte', dictionary: { KCX },
    templates: [{ chainId: 'A', residues: [0, 1, 4] }] }));
  assert.equal(kept.templates.length, 1);
  assert.deepEqual(kept.templates[0].mapping, [[0, 0], [1, 1], [4, 4]]);
  const whole = foldInputFromRefoldCif(refoldCifFromDesign(DESIGN, { name: 'pte', dictionary: { KCX }, templates: [{ chainId: 'A' }] }));
  assert.equal(whole.templates[0].mapping, undefined);
  assert.throws(() => refoldCifFromDesign(DESIGN, { dictionary: { KCX }, templates: [{ chainId: 'Q' }] }), /chain Q/);
});

test('a modified residue it cannot name a parent for is refused, by name', () => {
  assert.throws(() => refoldCifFromDesign(DESIGN.replaceAll('KCX', 'QQQ')), /chain A has QQQ/);
});

test('templates are chosen by name, so a modified residue is read as its parent there', () => {
  const pdb = 'ATOM      1  CA  KCX A   2       2.000   0.000   0.000\nHETATM    2  CA  KCX B   1       2.000   0.000   0.000';
  const out = pdbWithParents(pdb, { KCX: 'LYS' }).split('\n');
  assert.match(out[0], / LYS A/);
  assert.match(out[1], / KCX B/, 'only an ATOM record is a residue of a chain');
  assert.equal(pdbWithParents(pdb), pdb, 'and with no parents named, nothing changes');
});
