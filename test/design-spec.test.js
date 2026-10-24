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

test('scaffoldMotif keeps the selection and rebuilds each gap as a +/-25% segment in one undo step', () => {
  const doc = new StudioDoc(exampleCif('inverse_folding.cif'), '2kl8.cif');
  const a = doc.atoms;
  const motif = [];
  for (let i = 0; i < a.nAtoms; i += 1) {
    const r = a.res_id[i];
    if (((r >= 30 && r <= 40) || (r >= 60 && r <= 65)) && a.atom_name[i] === 'CA') motif.push(i);
  }
  const depth = doc.undoStack.length;
  doc.scaffoldMotif(motif);
  assert.equal(doc.undoStack.length, depth + 1);
  const spec = designSpec(doc.atoms);
  assert.equal(spec.task.key, 'motif');
  // 2KL8 numbers 1-7 and 28-79: gaps of 9, 19 and 14 residues around the two motif pieces.
  assert.equal(spec.contig, '7-11/A30-40/14-24/A60-65/11-18');
  doc.undo();
  assert.equal(designSpec(doc.atoms).contig, '59');
});

test('review fixes: sequence-only chains still generate; failed compound edits leave no trace', () => {
  const doc = new StudioDoc(exampleCif('inverse_folding.cif'), '2kl8.cif');
  const chain = designSpec(doc.atoms).chains[0];
  doc.setConditions([
    { condition: 'coordinate', atom_indices: chain.polymerAtomIndices, enabled: false },
    { condition: 'sequence', atom_indices: chain.polymerAtomIndices, enabled: true },
  ]);
  const spec = designSpec(doc.atoms);
  assert.equal(spec.task.key, 'fold');
  assert.equal(spec.predicted, 59);

  const depth = doc.undoStack.length;
  const revision = doc.revision;
  const before = designSpec(doc.atoms).contig;
  // Replacing every residue would delete every atom: refused, with no half-applied state or empty undo step.
  assert.throws(() => doc.replaceWithSegments(chain.polymerAtomIndices, 5, 6), /cannot delete every atom/);
  assert.equal(doc.undoStack.length, depth);
  assert.equal(doc.revision, revision);
  assert.equal(designSpec(doc.atoms).contig, before);
  // A batch whose second edit is invalid applies nothing.
  assert.throws(() => doc.setConditions([
    { condition: 'hotspot', atom_indices: [0] },
    { condition: 'no-such-condition', atom_indices: [0] },
  ]));
  assert.equal(designSpec(doc.atoms).chains[0].hotspots, 0);
  assert.equal(doc.undoStack.length, depth);
});

test('review fixes: scaffolding with a ligand selected, re-scaffolding, modified residues', () => {
  // Enzyme: catalytic Asp + the ligand selected, motif placed anywhere.
  const enz = new StudioDoc(exampleCif('enzyme_design.cif'), 'enzyme.cif');
  enz.realizeSegments(1);
  delete enz.atoms.annotations.annotation_expsegmin_1_residue;
  delete enz.atoms.annotations.annotation_expsegmax_1_residue;
  enz.atoms.res_name.forEach((name, i) => {
    if (name === '<M>') { enz.atoms.hetero[i] = false; if (enz.atoms.is_polymer) enz.atoms.is_polymer[i] = true; }
  });
  const pick = [];
  for (let i = 0; i < enz.atoms.nAtoms; i += 1) if (['ASP', 'ORO'].includes(enz.atoms.res_name[i])) pick.push(i);
  enz.scaffoldMotif(pick, { unindexed: true, totalMin: 180, totalMax: 180 });
  const spec = designSpec(enz.atoms);
  assert.equal(spec.task.key, 'enzyme');
  assert.match(spec.contig, /^180\/\[A\d+\] \+ /);
  assert.ok(spec.chains.find((c) => c.ligands.some((l) => l.name === 'ORO')).ligands[0].fixed);

  // Scaffolding an already scaffolded chain replaces its segments instead of keeping them.
  const doc = new StudioDoc(exampleCif('inverse_folding.cif'), '2kl8.cif');
  const ca = (lo, hi) => {
    const out = [];
    for (let i = 0; i < doc.atoms.nAtoms; i += 1) {
      const r = doc.atoms.res_id[i];
      if (r >= lo && r <= hi && doc.atoms.atom_name[i] === 'CA' && !/^\d+-\d+$/.test(doc.atoms.res_name[i])) out.push(i);
    }
    return out;
  };
  doc.scaffoldMotif(ca(30, 40));
  doc.scaffoldMotif(ca(33, 36), { unindexed: true, totalMin: 50, totalMax: 60 });
  const re = designSpec(doc.atoms);
  assert.equal(re.chains[0].segments.length, 1);
  assert.match(re.contig, /^50-60\/\[A\d+-\d+\]$/);

  // A modified residue stored as HETATM is still a polymer residue (is_polymer), not a ligand.
  const mod = new StudioDoc(exampleCif('inverse_folding.cif'), '2kl8.cif');
  for (let i = 0; i < mod.atoms.nAtoms; i += 1) if (mod.atoms.res_id[i] === 30) mod.atoms.hetero[i] = true;
  const m = designSpec(mod.atoms);
  assert.equal(m.chains.length, 1);
  assert.equal(m.chains[0].ligands.length, 0);
  assert.equal(m.chains[0].residues, 59);
});
