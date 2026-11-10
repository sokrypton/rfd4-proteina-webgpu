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

test('refoldOverlayCif keeps every chain of the refold', () => {
  const atom = (serial, name, res, chain, resId, x) => `ATOM  ${String(serial).padStart(5)} ${name.padEnd(4)} ${res} ${chain}${String(resId).padStart(4)}    ${x.toFixed(3).padStart(8)}${'0.000'.padStart(8)}${'0.000'.padStart(8)}  1.00 90.00           ${name[0]}`;
  const pdb = [
    atom(1, 'N', 'ALA', 'A', 1, 1), atom(2, 'CA', 'ALA', 'A', 1, 2),
    atom(3, 'N', 'LEU', 'B', 1, 3), atom(4, 'CA', 'LEU', 'B', 1, 4), atom(5, 'CB', 'LEU', 'B', 1, 5),
    atom(6, 'C', 'LEU', 'B', 1, 6), atom(7, 'O', 'LEU', 'B', 1, 7),
  ].join('\n');
  const rows = refoldOverlayCif(pdb).split('\n').filter((l) => l.startsWith('ATOM'));
  assert.equal(rows.length, 7);
  assert.deepEqual([...new Set(rows.map((l) => l.split(/\s+/)[6]))], ['A', 'B']);
  assert.equal(refoldOverlayCif(''), null);
});

test('describeRefold reads RMSD and confidence', () => {
  const line = describeRefold('openbind0', { rmsd: 0.331, confidence: { meanPlddt: 87.04, ptm: 0.851, iptm: 0.703 } }, ['B']);
  assert.equal(line, 'OpenBind-0 refold: 0.33 Å RMSD (chain B) · pLDDT 87 · pTM 0.85 · ipTM 0.70');
});

test('describeRefold says so when the RMSD misses the self-consistency threshold', () => {
  // A confident fold that sits far from the design is the case most likely to be misread: high pLDDT beside a
  // bare number reads as success. 9.04 A at pLDDT 85 was measured on `uncond_short` and is a failed design.
  const line = describeRefold('openbind0', { rmsd: 9.035, confidence: { meanPlddt: 85.2, ptm: 0.851 } }, ['A']);
  assert.equal(line, 'OpenBind-0 refold: 9.04 Å RMSD (chain A) — above the 2 Å a self-consistent design is'
    + ' usually held to · pLDDT 85 · pTM 0.85');

  const onThreshold = describeRefold('openbind0', { rmsd: 2, confidence: {} }, ['A']);
  assert.equal(onThreshold, 'OpenBind-0 refold: 2.00 Å RMSD (chain A)', 'exactly 2 A still counts as consistent');
});

test('describeRefold explains an RMSD it cannot compute, rather than dropping it', () => {
  // selfConsistency returns NaN with comparable:false on a length mismatch; without this the line would read
  // as a clean result that simply has no RMSD in it.
  const line = describeRefold('af3', { rmsd: Number.NaN, comparable: false, confidence: { meanPlddt: 70 } }, ['A']);
  assert.equal(line, 'AlphaFold 3 refold: RMSD not comparable — the design and the refold have different'
    + ' lengths · pLDDT 70');
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


test('a refold is scored as upstream\'s pass criteria read it', async () => {
  const { polymerChains, refoldMetrics, passes, criteriaKind, rankKey } = await import('../web/refold-view.js');
  const atom = (k, chain, res, het = false) => `${het ? 'HETATM' : 'ATOM  '}${String(k).padStart(5)}  CA  ALA ${chain}${String(res).padStart(4)}    `
    + '   0.000   0.000   0.000  1.00  0.00           C';
  // binder B (2 residues) on target A (3 residues), a ligand after
  const pdb = [atom(1, 'B', 1), atom(2, 'B', 2), atom(3, 'A', 1), atom(4, 'A', 2), atom(5, 'A', 3), atom(6, 'L', 1, true)].join('\n');
  assert.deepEqual(polymerChains(pdb), [{ letter: 'B', residues: 2 }, { letter: 'A', residues: 3 }]);
  const n = 6;
  const pae = new Float32Array(n * n).fill(20);
  pae[0 * n + 3] = 1.2; // binder residue 1 to target residue 2
  pae[4 * n + 1] = 0.9; // target residue 3 to binder residue 2
  pae[1 * n + 5] = 0.1; // to the ligand: not an interface between chains of protein
  const result = {
    rmsd: 0.8, comparable: true,
    deviations: [{ chain: 'B', distance: 1 }, { chain: 'B', distance: 3 }, { chain: 'A', distance: 9 }],
    confidence: { meanPlddt: 90, ptm: 0.8, iptm: 0.7, chainPtm: { 1: 0.85, 2: 0.9 }, predictedAlignedError: pae },
  };
  const m = refoldMetrics(result, ['B'], pdb);
  assert.equal(m.minIpae.toFixed(2), '0.90');
  assert.equal(m.placedRmsd.toFixed(3), Math.sqrt(5).toFixed(3));
  assert.equal(m.chainPtm, 0.85);
  assert.equal(passes(m, 'binder'), true, 'min iPAE 0.9, pTM 0.85, placed 2.24 Å: all within');
  assert.equal(passes({ ...m, placedRmsd: 3.0 }, 'binder'), false, 'a binder that moved on its target fails');
  assert.equal(passes(m, 'antibody'), false, 'antibodies are held to 2.0 Å');
  assert.equal(passes({ ...m, minIpae: NaN }, 'binder'), null, 'a missing number is no verdict');
  assert.equal(criteriaKind('binder'), 'binder');
  assert.equal(criteriaKind('unconditional', { designedAll: true }), 'monomer');
  assert.equal(criteriaKind('motif', { designedAll: true }), null);
  assert.deepEqual(rankKey('antibody'), { key: 'minIpae', dir: 1 });
  // a monomer has no interface
  const mono = refoldMetrics({ rmsd: 1.1, deviations: [], confidence: { predictedAlignedError: new Float32Array(4) } }, ['A'], atom(1, 'A', 1) + '\n' + atom(2, 'A', 2));
  assert.ok(Number.isNaN(mono.minIpae));
  assert.equal(passes(mono, 'monomer'), true);
});

test('describeRefold says how the motif was put back, fitted on itself and fitted on the backbone', () => {
  const line = describeRefold('openbind0', { rmsd: 1.1, confidence: { meanPlddt: 92 } }, ['A'], [], { motifRmsd: 0.62, motifBackboneRmsd: 1.9 });
  assert.match(line, /motif 0\.62 Å \(1\.90 Å on the backbone\)/);
  assert.ok(!/motif/.test(describeRefold('openbind0', { rmsd: 1.1, confidence: {} }, ['A'])), 'no motif, no words about one');
});
