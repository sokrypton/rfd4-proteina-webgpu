import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { parseCifToAtomArray } from '../shared/rfd4p/pipeline/cif.js';
import { designSpec } from '../web/design-spec.js';
import { StudioDoc } from '../web/studio-api.js';

const exampleCif = (file) => readFileSync(new URL(`../web/studio/examples/${file}`, import.meta.url), 'utf8');
const specOf = (file, opts) => designSpec(parseCifToAtomArray(exampleCif(file)), opts);

test('designSpec reads task, roles, lengths and contig from the bundled examples', () => {
  const cases = [
    ['protein_binder.cif', 'binder', 'A17-131/0 50-120', [165, 235], { A: 'fixed', B: 'new' }],
    ['antibody_design.cif', 'antibody', null, [280, 280], { L: 'sequence', H: 'motif', T: 'fixed' }],
    ['enzyme_design.cif', 'enzyme', '180/[A1] + ORO, ZN', [180, 180], { A: 'motif', B: 'ligand', C: 'ligand' }],
    ['motif_scaffolding.cif', 'motif', '10-20/A1/15-30/A3/15-30/A5/10-20', [53, 103], { A: 'motif' }],
    ['nanobody.cif', 'antibody', null, [223, 246], { A: 'fixed', H: 'motif' }],
    ['small_molecule_binder.cif', 'sm_binder', '150 + SAM', [150, 150], { A: 'new', B: 'ligand' }],
    ['unconditional.cif', 'unconditional', '250', [250, 250], { A: 'new' }],
    ['inverse_folding.cif', 'redesign_all', '59', [59, 59], { A: 'free' }],
  ];
  for (const [file, task, contig, [lo, hi], roles] of cases) {
    const spec = specOf(file);
    assert.equal(spec.task.key, task, file);
    if (contig) assert.equal(spec.contig, contig, file);
    assert.deepEqual([spec.totalMin, spec.totalMax], [lo, hi], file);
    assert.deepEqual(Object.fromEntries(spec.chains.map((c) => [c.id, c.role])), roles, file);
  }
  assert.equal(specOf('inverse_folding.cif', { partialDiffusion: true }).task.key, 'partial');
});

test('batch condition edits and segment ranges change the spec in one undo step', () => {
  const doc = new StudioDoc(exampleCif('inverse_folding.cif'), '2kl8.cif');
  const chain = designSpec(doc.atoms).chains[0];
  const depth = doc.undoStack.length;

  // "Redesign the sequence": backbone atoms fixed, sequence free.
  doc.setConditions([
    { condition: 'coordinate', atom_indices: chain.polymerAtomIndices, enabled: false },
    { condition: 'coordinate', atom_indices: chain.backboneAtomIndices, enabled: true },
    { condition: 'sequence', atom_indices: chain.polymerAtomIndices, enabled: false },
  ]);
  assert.equal(doc.undoStack.length, depth + 1);
  let spec = designSpec(doc.atoms);
  assert.equal(spec.task.key, 'inverse');
  assert.equal(spec.chains[0].role, 'backbone');
  assert.equal(spec.chains[0].backboneOnly, 59);

  // "Design a binder": keep it fixed, add a new chain, then resize it.
  doc.setConditions([
    { condition: 'coordinate', atom_indices: chain.polymerAtomIndices, enabled: true },
    { condition: 'sequence', atom_indices: chain.polymerAtomIndices, enabled: true },
  ]);
  doc.insertSegment({ minimum: 60, maximum: 100, newChain: true });
  spec = designSpec(doc.atoms);
  assert.equal(spec.task.key, 'binder');
  const seg = spec.chains.find((c) => c.role === 'new').segments[0];
  doc.setSegmentRange(seg.atomIndex, 70, 80);
  spec = designSpec(doc.atoms);
  assert.deepEqual([spec.designedMin, spec.designedMax], [70, 80]);
  assert.match(spec.contig, /\/0 70-80$/);
  assert.throws(() => doc.setSegmentRange(0, 5, 6), /not an expandable segment/);
});
