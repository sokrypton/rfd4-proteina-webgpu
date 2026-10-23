import test from 'node:test';
import assert from 'node:assert/strict';

import { parseCifToAtomArray, parseCifBlock } from '../shared/rfd4p/pipeline/cif.js';
import { parseOperationExpression, prepareRcsbStructure } from '../shared/rfd4p/pipeline/prepare.js';
import { StudioDoc } from '../web/studio-api.js';

const HEADER = [
  'group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id', 'label_comp_id', 'label_asym_id',
  'label_entity_id', 'label_seq_id', 'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy',
  'B_iso_or_equiv', 'pdbx_formal_charge', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id',
  'pdbx_PDB_model_num',
];
const q = (s) => (/['\s]/.test(s) ? `"${s}"` : s);
/** [group, element, atom, alt, comp, asym, entity, seq, occ] rows -> mmCIF text (coordinates spread along x). */
function cif(rows, extra = '') {
  const lines = ['data_t', 'loop_', ...HEADER.map((h) => `_atom_site.${h}`)];
  rows.forEach(([g, el, atom, alt, comp, asym, ent, seq, occ = '1.00'], i) => {
    lines.push(`${g} ${i + 1} ${el} ${q(atom)} ${alt} ${comp} ${asym} ${ent} ${seq} ? ${(1.5 * i).toFixed(3)} 0.000 0.000 ${occ} 10.0 ? ${seq} ${comp} ${asym} ${q(atom)} 1`);
  });
  return `${lines.join('\n')}\n#\n${extra}`;
}
const GLY = (asym, seq, alt = '.', occ = '1.00') =>
  [['N', 'N'], ['C', 'CA'], ['C', 'C'], ['O', 'O']].map(([el, at]) => ['ATOM', el, at, alt, 'GLY', asym, 1, seq, occ]);

test('parseOperationExpression matches biotite', () => {
  assert.deepEqual(parseOperationExpression('1'), [['1']]);
  assert.deepEqual(parseOperationExpression('1,2,3'), [['1'], ['2'], ['3']]);
  assert.deepEqual(parseOperationExpression('(1-3)'), [['1'], ['2'], ['3']]);
  assert.deepEqual(parseOperationExpression('P,1-2'), [['P'], ['1'], ['2']]);
  // biotite: [('3','1'), ('3','2'), ('4','1'), ('4','2')] and [('1','X0'), ('2','X0')]
  assert.deepEqual(parseOperationExpression('(1,2)(3-4)'), [['3', '1'], ['3', '2'], ['4', '1'], ['4', '2']]);
  assert.deepEqual(parseOperationExpression('(X0)(1-2)'), [['1', 'X0'], ['2', 'X0']]);
});

test('primed atom names survive a Studio export round trip', () => {
  const rows = [
    ['ATOM', 'P', 'P', '.', 'DC', 'A', 1, 1], ['ATOM', 'O', "O5'", '.', 'DC', 'A', 1, 1],
    ['ATOM', 'C', "C5'", '.', 'DC', 'A', 1, 1], ['ATOM', 'C', "C4'", '.', 'DC', 'A', 1, 1],
  ];
  const doc = new StudioDoc(cif(rows), 'dna.cif');
  const back = parseCifToAtomArray(doc.toCif());
  assert.deepEqual(back.atom_name, ['P', "O5'", "C5'", "C4'"]);
});

test('segment insertion keeps numbering and bonds the placeholder to the chain', () => {
  const doc = new StudioDoc(cif([...GLY('A', 1), ...GLY('A', 2), ...GLY('A', 3)]), 'gly.cif');
  doc.insertSegment({ minimum: 5, maximum: 10, atomIndex: 0, before: true });
  const a = doc.atoms;
  // N-terminal placeholder takes residue 0; residues 1..3 keep their numbers (AtomWorks `_compute_insert_res_id`).
  assert.equal(a.res_name[0], '5-10');
  assert.equal(a.res_id[0], 0);
  assert.deepEqual([...new Set(a.res_id.slice(1))], [1, 2, 3]);
  // ...and is bonded to residue 1's N.
  const n1 = a.atom_name.findIndex((nm, i) => nm === 'N' && a.res_id[i] === 1);
  assert.ok(a.bonds.some(([u, v]) => (u === 0 && v === n1) || (u === n1 && v === 0)));
  // Inserting after residue 2 (no numbering gap) takes 3 and shifts residue 3 to 4, bonded C(2) - sentinel - N(4).
  const c2 = a.atom_name.findIndex((nm, i) => nm === 'C' && a.res_id[i] === 2);
  doc.insertSegment({ minimum: 3, maximum: 3, atomIndex: c2, before: false });
  const b = doc.atoms;
  const sentinel = b.res_name.indexOf('3-3');
  assert.equal(b.res_id[sentinel], 3);
  assert.equal(Math.max(...b.res_id), 4);
  const c2b = b.atom_name.findIndex((nm, i) => nm === 'C' && b.res_id[i] === 2);
  const n4 = b.atom_name.findIndex((nm, i) => nm === 'N' && b.res_id[i] === 4);
  const bonded = (x, y) => b.bonds.some(([u, v]) => (u === x && v === y) || (u === y && v === x));
  assert.ok(bonded(c2b, sentinel) && bonded(sentinel, n4) && !bonded(c2b, n4));
  assert.match(doc.toCif(), /_struct_conn\.id[\s\S]*covale sing A A GLY 3-3 2 3 C UNK/);
});

test('rcsb preparation: waters, additives, MSE, per-chain altloc, unmodeled residues', () => {
  const scheme = [
    'loop_', '_pdbx_poly_seq_scheme.asym_id', '_pdbx_poly_seq_scheme.entity_id', '_pdbx_poly_seq_scheme.seq_id',
    '_pdbx_poly_seq_scheme.mon_id', '_pdbx_poly_seq_scheme.pdb_seq_num', '_pdbx_poly_seq_scheme.pdb_mon_id',
    'A 1 1 MET 1 ?', 'A 1 2 GLY 2 GLY', 'A 1 3 GLY 3 GLY', 'A 1 4 MSE 4 MSE', 'A 1 5 GLY 5 GLY', '#', '',
  ].join('\n');
  const rows = [
    ...GLY('A', 2, 'A', '0.50'), ...GLY('A', 3),
    ['HETATM', 'N', 'N', '.', 'MSE', 'A', 1, 4], ['HETATM', 'C', 'CA', '.', 'MSE', 'A', 1, 4],
    ['HETATM', 'C', 'C', '.', 'MSE', 'A', 1, 4], ['HETATM', 'O', 'O', '.', 'MSE', 'A', 1, 4],
    ['HETATM', 'SE', 'SE', '.', 'MSE', 'A', 1, 4],
    ...GLY('A', 5, 'B', '0.50'), // only in conformer B: dropped, then re-added as unresolved
    ['HETATM', 'P', 'P', '.', 'PO4', 'B', 2, '.'], ['HETATM', 'O', 'O', '.', 'HOH', 'C', 3, '.'],
  ];
  const text = cif(rows, scheme);
  const { atoms } = prepareRcsbStructure(parseCifToAtomArray(text), parseCifBlock(text));
  const res = [...new Set(atoms.res_name.map((r, i) => `${r}${atoms.res_id[i]}`))];
  assert.deepEqual(res, ['MET1', 'GLY2', 'GLY3', 'MET4', 'GLY5']);
  assert.ok(!atoms.res_name.includes('HOH') && !atoms.res_name.includes('PO4'));
  const sd = atoms.atom_name.indexOf('SD');
  assert.ok(sd >= 0 && atoms.element[sd] === 'S' && atoms.hetero[sd] === false);
  const unresolved = (resId) => atoms.coord.filter((c, i) => atoms.res_id[i] === resId).every((c) => Number.isNaN(c[0]));
  assert.ok(unresolved(1), 'unmodeled MET1 is added without coordinates');
  assert.ok(unresolved(5), 'conformer-B-only GLY5 becomes unresolved');
  assert.ok(!unresolved(2), 'GLY2 (conformer A) keeps its coordinates');
  // MET4 (from MSE) is completed with its missing side chain.
  assert.deepEqual(atoms.atom_name.filter((_, i) => atoms.res_id[i] === 4).slice(0, 8), ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE']);
});

test('CCD components are fetched only when needed, merged, and failures are tolerated', async () => {
  const { componentsNeedingCcd, fetchCcdComponents, mergeCcdIntoBlock } = await import('../shared/rfd4p/pipeline/ccd.js');
  const fileBlock = parseCifBlock([
    'data_x', 'loop_', '_chem_comp_bond.comp_id', '_chem_comp_bond.atom_id_1', '_chem_comp_bond.atom_id_2',
    '_chem_comp_bond.value_order', 'SAM N CA SING', '#',
  ].join('\n'));
  // Standard residues, components the file defines, segments and masks are skipped.
  assert.deepEqual(componentsNeedingCcd(['GLY', 'SAM', 'PTH', '5-10', '<M>', 'UNK'], fileBlock), ['PTH']);
  const ptH = [
    'data_PTH', 'loop_', '_chem_comp_atom.comp_id', '_chem_comp_atom.atom_id', '_chem_comp_atom.type_symbol',
    '_chem_comp_atom.pdbx_aromatic_flag', 'PTH P P N', 'PTH O1P O N', '#',
    'loop_', '_chem_comp_bond.comp_id', '_chem_comp_bond.atom_id_1', '_chem_comp_bond.atom_id_2',
    '_chem_comp_bond.value_order', '_chem_comp_bond.pdbx_aromatic_flag', 'PTH P O1P DOUB N', '#',
  ].join('\n');
  const calls = [];
  const fetchFn = async (url) => {
    calls.push(url);
    return url.includes('PTH') ? new Response(ptH) : new Response('', { status: 404 });
  };
  const { blocks, failed } = await fetchCcdComponents(['PTH', 'NOPE'], { fetchFn });
  assert.deepEqual([...blocks.keys()], ['PTH']);
  assert.deepEqual(failed, ['NOPE']);
  mergeCcdIntoBlock(fileBlock, blocks);
  assert.deepEqual(fileBlock.chem_comp_bond.comp_id, ['SAM', 'PTH']);
  assert.deepEqual(fileBlock.chem_comp_bond.pdbx_aromatic_flag, ['?', 'N']);
  assert.ok(calls.every((u) => u.startsWith('https://files.rcsb.org/ligands/view/')));
});
