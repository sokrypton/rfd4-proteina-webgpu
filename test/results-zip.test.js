import test from 'node:test';
import assert from 'node:assert/strict';

import { pdbSequences, resultsFiles } from '../web/results-zip.js';

const atom = (serial, name, res, chain, resId) => `ATOM  ${String(serial).padStart(5)} ${name.padEnd(4)} ${res} ${chain}${String(resId).padStart(4)}    ${'1.000'.padStart(8)}${'2.000'.padStart(8)}${'3.000'.padStart(8)}  1.00 90.00           ${name[0]}`;
const pdb = [atom(1, 'N', 'MET', 'A', 1), atom(2, 'CA', 'MET', 'A', 1), atom(3, 'CA', 'LYS', 'A', 2), atom(4, 'CA', 'GLY', 'B', 1)].join('\n');

test('pdbSequences reads one letter per residue, per chain', () => {
  assert.deepEqual(pdbSequences(pdb), { A: 'MK', B: 'G' });
});

test('resultsFiles packages input, designs, trajectories, every refold and a summary, as mmCIF only', () => {
  const refold = (model, rmsd) => ({ model, pdb, cif: 'data_r', rmsd, confidence: { meanPlddt: 80.123, ptm: 0.8, iptm: 0.61 },
    templated: ['A'], partial: [{ chain: 'H', residues: 109 }], text: 'line' });
  const files = resultsFiles({
    input: { cif: 'data_in', name: 'antibody', revision: 3 },
    designs: [
      { index: 1, label: 'Design 1', seed: 7, stepsRun: 50, nsteps: 60, cif: 'data_d1', pdb, trajectoryCif: 'data_t1',
        designedChains: ['B'], refolds: [refold('af3', 0.5), refold('openbind0', 1.2), refold('af3', 0.7)] },
      { index: 2, label: 'Design 2', seed: 8, cif: 'data_d2', pdb: null },
    ],
    generatedAt: '2026-10-30T00:00:00Z',
  });
  const names = files.map((f) => f.name);
  assert.deepEqual(names, [
    'input.cif',
    'design-1.cif', 'design-1-trajectory.cif',
    'design-1-refold-af3.cif', 'design-1-refold-openbind0.cif', 'design-1-refold-af3-2.cif',
    'design-2.cif',
    'summary.json', 'summary.csv', 'README.txt',
  ]);
  const summary = JSON.parse(files.find((f) => f.name === 'summary.json').text);
  assert.equal(summary.input.revision, 3);
  assert.equal(summary.designs[0].refolds.length, 3);
  assert.deepEqual(summary.designs[0].sequences, { A: 'MK', B: 'G' });
  assert.equal(summary.designs[0].refolds[0].plddt, 80.12);
  const csv = files.find((f) => f.name === 'summary.csv').text.trim().split('\n');
  assert.equal(csv.length, 1 + 3 + 1);
  // no metrics on these refolds: RMSD on target, designed-chain pTM, min iPAE, the motif RMSDs and the verdict are left empty
  assert.ok(csv[1].startsWith('1,7,50,B,A:MK B:G,af3,0,,,0.5,,80.12,0.8,0.61,,,,,,A,H:109'), 'refold seed 0 when none was given');
  assert.ok(csv[0].includes('rmsd_on_target') && csv[0].includes('min_ipae') && csv[0].includes('passes') && csv[0].includes('motif_rmsd') && csv[0].includes('motif_backbone_aligned_rmsd'));
  assert.match(files.find((f) => f.name === 'README.txt').text, /WEIGHTS_TERMS_OF_USE/);
  assert.ok(!names.some((n) => n.endsWith('.pdb')));
});

test('without AlphaFold 3 refolds the README carries no AlphaFold 3 terms', () => {
  const files = resultsFiles({ input: null, designs: [{ index: 1, label: 'Design 1', cif: 'x', pdb: null }] });
  assert.ok(!/AlphaFold 3/.test(files.find((f) => f.name === 'README.txt').text));
  assert.ok(!files.some((f) => f.name === 'input.cif'));
});

test('a refold kept by Studio\'s Refold panel is saved as the page always saved one', async () => {
  const { refoldRecordForZip } = await import('../web/refold-provider.js');
  // What the panel hands back for a design (`panel.records(id)`): its settings and result, the page's extras in `data`.
  const record = {
    model: 'openbind0', modelLabel: 'OpenBind-0', seed: 3, recycles: 3, steps: 25, cif: 'data_overlay', pass: true,
    metrics: { rmsd: 1.1, placedRmsd: NaN, plddt: 91, minIpae: 1.4 }, text: 'OpenBind-0 refold: 1.10 Å',
    data: { pdb: 'ATOM', cif: 'data_refold', rmsd: 1.1, confidence: { meanPlddt: 91, ptm: 0.9 }, templated: ['A'], partial: [] },
  };
  const files = resultsFiles({
    input: null,
    designs: [{ index: 1, label: 'seed 3', seed: 3, cif: 'data_d', pdb: null, designedChains: ['B'], refolds: [refoldRecordForZip(record)] }],
    generatedAt: '2026-11-11T00:00:00Z',
  });
  assert.ok(files.some((f) => f.name === 'design-1-refold-openbind0.cif' && f.text === 'data_refold'), 'the refold itself, not the overlay');
  const summary = JSON.parse(files.find((f) => f.name === 'summary.json').text);
  const [refold] = summary.designs[0].refolds;
  assert.deepEqual([refold.model_name, refold.seed, refold.recycles, refold.steps], ['OpenBind-0', 3, 3, 25]);
  assert.deepEqual([refold.rmsd_designed_chain, refold.plddt, refold.ptm, refold.min_ipae, refold.passes], [1.1, 91, 0.9, 1.4, true]);
  assert.deepEqual(refold.templated_chains, ['A']);
});
