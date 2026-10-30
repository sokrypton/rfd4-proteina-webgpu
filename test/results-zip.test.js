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
  assert.ok(csv[1].startsWith('1,7,50,B,A:MK B:G,af3,0.5,80.12,0.8,0.61,A,H:109'));
  assert.match(files.find((f) => f.name === 'README.txt').text, /WEIGHTS_TERMS_OF_USE/);
  assert.ok(!names.some((n) => n.endsWith('.pdb')));
});

test('without AlphaFold 3 refolds the README carries no AlphaFold 3 terms', () => {
  const files = resultsFiles({ input: null, designs: [{ index: 1, label: 'Design 1', cif: 'x', pdb: null }] });
  assert.ok(!/AlphaFold 3/.test(files.find((f) => f.name === 'README.txt').text));
  assert.ok(!files.some((f) => f.name === 'input.cif'));
});
