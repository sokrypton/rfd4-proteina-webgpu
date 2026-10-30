import test from 'node:test';
import assert from 'node:assert/strict';

import { describeRefold, designedChains, refoldOverlayCif } from '../web/refold-view.js';

const columns = (rows) => ({
  chain_id: rows.map((r) => r[0]), res_id: rows.map((r) => r[1]), res_name: rows.map((r) => r[2]),
  hetero: rows.map((r) => Boolean(r[3])),
});
const records = (rows) => rows.map(([chainId, resId, resName, isHetero]) => ({ chainId, resId, resName, isHetero }));

test('designedChains: a kept target is templated, a new or changed chain is designed', () => {
  const input = columns([['A', 1, 'ALA'], ['A', 2, 'GLY'], ['C', 1, 'SAM', true]]);
  const result = records([['A', 1, 'ALA'], ['A', 1, 'ALA'], ['A', 2, 'GLY'], ['B', 1, 'LEU'], ['B', 2, 'LYS'],
    ['C', 1, 'SAM', true]]);
  assert.deepEqual(designedChains(input, result), ['B']);

  const redesigned = records([['A', 1, 'SER'], ['A', 2, 'GLY'], ['B', 1, 'LEU']]);
  assert.deepEqual(designedChains(input, redesigned), ['A', 'B']);
});

test('designedChains: nothing changed means every polymer chain is designed (fitted on itself)', () => {
  const input = columns([['A', 1, 'ALA'], ['A', 2, 'GLY']]);
  assert.deepEqual(designedChains(input, records([['A', 1, 'ALA'], ['A', 2, 'GLY']])), ['A']);
});

test('refoldOverlayCif keeps only the designed chains', () => {
  const atom = (serial, name, res, chain, resId, x) => `ATOM  ${String(serial).padStart(5)} ${name.padEnd(4)} ${res} ${chain}${String(resId).padStart(4)}    ${x.toFixed(3).padStart(8)}${'0.000'.padStart(8)}${'0.000'.padStart(8)}  1.00 90.00           ${name[0]}`;
  const pdb = [
    atom(1, 'N', 'ALA', 'A', 1, 1), atom(2, 'CA', 'ALA', 'A', 1, 2),
    atom(3, 'N', 'LEU', 'B', 1, 3), atom(4, 'CA', 'LEU', 'B', 1, 4), atom(5, 'CB', 'LEU', 'B', 1, 5),
    atom(6, 'C', 'LEU', 'B', 1, 6), atom(7, 'O', 'LEU', 'B', 1, 7),
  ].join('\n');
  const cif = refoldOverlayCif(pdb, ['B']);
  const rows = cif.split('\n').filter((l) => l.startsWith('ATOM'));
  assert.equal(rows.length, 5);
  assert.ok(rows.every((l) => / LEU B /.test(l)));
  assert.equal(refoldOverlayCif(pdb, ['Z']), null);
});

test('describeRefold reads RMSD and confidence', () => {
  const line = describeRefold('openbind0', { rmsd: 0.331, confidence: { meanPlddt: 87.04, ptm: 0.851, iptm: 0.703 } }, ['B']);
  assert.equal(line, 'OpenBind-0 refold: 0.33 Å RMSD (chain B) · pLDDT 87 · pTM 0.85 · ipTM 0.70');
});

test('pdbToCif keeps every chain, ligands and pLDDT when asked for the whole refold', async () => {
  const { pdbToCif } = await import('../web/refold-view.js');
  const pdb = [
    'ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 91.50           C',
    'ATOM      2  CB  ALA A   1       2.000   0.000   0.000  1.00 91.50           C',
    'ATOM      3  CA  LEU B   1       3.000   0.000   0.000  1.00 64.25           C',
    'HETATM    4  C1  SAM C   1       4.000   0.000   0.000  1.00 80.00           C',
  ].join('\n');
  const rows = pdbToCif(pdb).split('\n').filter((l) => /^(ATOM|HETATM) /.test(l));
  assert.equal(rows.length, 4);
  assert.ok(rows[0].includes(' 91.5 '));
  assert.ok(rows[3].startsWith('HETATM'));
});

test('refoldLigandCif keeps only the refold\'s ligands', async () => {
  const { refoldLigandCif } = await import('../web/refold-view.js');
  const pdb = [
    'ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 91.50           C',
    'HETATM    2  C1  SAM B   1       4.000   0.000   0.000  1.00 80.00           C',
    'HETATM    3  N1  SAM B   1       5.000   0.000   0.000  1.00 80.00           N',
  ].join('\n');
  const rows = refoldLigandCif(pdb).split('\n').filter((l) => /^(ATOM|HETATM) /.test(l));
  assert.equal(rows.length, 2);
  assert.ok(rows.every((l) => l.startsWith('HETATM') && / SAM B /.test(l)));
  assert.equal(refoldLigandCif(pdb.split('\n')[0]), null);
});

test('motif side chains: positions from the design pick the same residues in a renumbered refold', async () => {
  const { motifSlots, refoldMotifSideChainAtoms } = await import('../web/refold-view.js');
  const design = records([['A', 10, 'ALA'], ['A', 11, 'HIS'], ['A', 12, 'GLY'], ['B', 1, 'SAM', true]]);
  const slots = motifSlots(design, [{ chainId: 'A', resId: 11 }]);
  assert.deepEqual(slots, [{ chain: 'A', index: 1 }]);
  const line = (serial, name, res, resId, el) => `ATOM  ${String(serial).padStart(5)} ${name.padEnd(4)} ${res} A${String(resId).padStart(4)}    ${'1.000'.padStart(8)}${'2.000'.padStart(8)}${'3.000'.padStart(8)}  1.00 90.00          ${el.padStart(2)}`;
  const refold = [
    line(1, 'CA', 'ALA', 1, 'C'), line(2, 'CB', 'ALA', 1, 'C'),
    line(3, 'N', 'HIS', 2, 'N'), line(4, 'CA', 'HIS', 2, 'C'), line(5, 'CB', 'HIS', 2, 'C'), line(6, 'ND1', 'HIS', 2, 'N'),
    line(7, 'CA', 'GLY', 3, 'C'),
  ].join('\n');
  const atoms = refoldMotifSideChainAtoms(refold, slots);
  assert.deepEqual(atoms.map((a) => `${a.resName}${a.resId}:${a.atomName}`), ['HIS2:CB', 'HIS2:ND1']);
  assert.ok(atoms.every((a) => a.isHetero));
});

test('partialTemplates: a mostly held chain is templated on its held residues, a small motif is not', async () => {
  const { partialTemplates } = await import('../web/refold-view.js');
  const atom = (chainId, resId, held) => ({ chainId, resId, ...(held ? { crd: [1, 2, 3] } : {}) });
  // H: 8 residues, 6 held (framework), 2 redesigned (a loop). A: 8 residues, 2 held (a motif).
  const design = [
    ...[1, 2, 3, 4, 5, 6, 7, 8].map((r) => atom('H', r, ![4, 5].includes(r))),
    ...[1, 2, 3, 4, 5, 6, 7, 8].map((r) => atom('A', r, [2, 7].includes(r))),
  ];
  assert.deepEqual(partialTemplates(design, ['H', 'A']), [{ chain: 'H', residues: [0, 1, 2, 5, 6, 7], of: 8 }]);
  // internal-coordinate conditioning (an antibody framework) counts as held too
  const framework = [1, 2, 3, 4].map((r) => ({ chainId: 'H', resId: r, ...(r !== 3 ? { crdSrcComponent: 'InternalCoordinate_1' } : {}) }));
  assert.deepEqual(partialTemplates(framework, ['H']), [{ chain: 'H', residues: [0, 1, 3], of: 4 }]);
  // a fully held chain is not partial (it would not have been designed)
  assert.deepEqual(partialTemplates(design.map((a) => ({ ...a, crd: [0, 0, 0] })), ['H']), []);
});

