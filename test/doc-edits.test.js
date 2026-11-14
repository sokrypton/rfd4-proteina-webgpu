/**
 * Compound edits of a document (the page's in-browser Studio API, web/studio-api.js) as the design card makes them: a
 * batch of conditions, a segment's range, scaffolding a motif. Each is one Undo step, a refused edit leaves no trace, and
 * what the model reads (test/model-read.js, new-residue blocks by `segmentsOf`) is what the edit says. (The card's own
 * reading of a document and its recipes are Studio's, tested there.)
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { StudioDoc } from '../web/studio-api.js';
import { modelRead, segmentsOf } from './model-read.js';

const exampleCif = (file) => readFileSync(new URL(`../web/studio/examples/${file}`, import.meta.url), 'utf8');
const open = (file = 'inverse_folding.cif') => new StudioDoc(exampleCif(file), file);
const atomsWhere = (doc, pred) => {
  const a = doc.atoms;
  const out = [];
  for (let i = 0; i < a.nAtoms; i += 1) if (pred(a, i)) out.push(i);
  return out;
};
const polymer = (doc) => atomsWhere(doc, (a, i) => !a.hetero[i] && !/^\d+-\d+$/.test(String(a.res_name[i])));
const backbone = (doc) => atomsWhere(doc, (a, i) => !a.hetero[i] && ['N', 'CA', 'C', 'O'].includes(a.atom_name[i]));
const ranges = (doc) => segmentsOf(doc).map((s) => `${s.min}-${s.max}`);

test('batch condition edits and segment ranges change what the model reads, one undo step each', () => {
  const doc = open();
  const depth = doc.undoStack.length;
  // The example redesigns everything: nothing held, every residue's sequence free.
  assert.deepEqual(modelRead(doc).A, { tokens: 59, designed: 59, held: 0, hotspots: 0, ligand: false });

  // "Redesign the sequence": backbone atoms fixed, sequence free.
  doc.setConditions([
    { condition: 'coordinate', atom_indices: polymer(doc), enabled: false },
    { condition: 'coordinate', atom_indices: backbone(doc), enabled: true },
    { condition: 'sequence', atom_indices: polymer(doc), enabled: false },
  ]);
  assert.equal(doc.undoStack.length, depth + 1);
  assert.deepEqual(modelRead(doc).A, { tokens: 59, designed: 59, held: 59, hotspots: 0, ligand: false });

  // "Design a binder": keep it fixed, add a new chain, then resize it.
  doc.setConditions([
    { condition: 'coordinate', atom_indices: polymer(doc), enabled: true },
    { condition: 'sequence', atom_indices: polymer(doc), enabled: true },
  ]);
  doc.insertSegment({ minimum: 60, maximum: 100, newChain: true });
  assert.deepEqual(ranges(doc), ['60-100']);
  assert.deepEqual(modelRead(doc).A, { tokens: 59, designed: 0, held: 59, hotspots: 0, ligand: false });
  doc.setSegmentRange(segmentsOf(doc)[0].atomIndex, 70, 80);
  assert.deepEqual(ranges(doc), ['70-80']);
  assert.throws(() => doc.setSegmentRange(0, 5, 6), /not an expandable segment/);
});

test('scaffoldMotif keeps the selection and rebuilds each gap as a +/-25% segment in one undo step', () => {
  const doc = open();
  const a = doc.atoms;
  const motif = [];
  for (let i = 0; i < a.nAtoms; i += 1) {
    const r = a.res_id[i];
    if (((r >= 30 && r <= 40) || (r >= 60 && r <= 65)) && a.atom_name[i] === 'CA') motif.push(i);
  }
  const depth = doc.undoStack.length;
  doc.scaffoldMotif(motif);
  assert.equal(doc.undoStack.length, depth + 1);
  // 2KL8 numbers 1-7 and 28-79: gaps of 9, 19 and 14 residues around the two motif pieces.
  assert.deepEqual(ranges(doc), ['7-11', '14-24', '11-18']);
  doc.undo();
  assert.deepEqual(ranges(doc), []);
});

test('review fixes: sequence-only chains still generate; failed compound edits leave no trace', () => {
  const doc = open();
  doc.setConditions([
    { condition: 'coordinate', atom_indices: polymer(doc), enabled: false },
    { condition: 'sequence', atom_indices: polymer(doc), enabled: true },
  ]);
  assert.deepEqual(modelRead(doc).A, { tokens: 59, designed: 0, held: 0, hotspots: 0, ligand: false });

  const depth = doc.undoStack.length;
  const revision = doc.revision;
  // Replacing every residue would delete every atom: refused, with no half-applied state or empty undo step.
  assert.throws(() => doc.replaceWithSegments(polymer(doc), 5, 6), /cannot delete every atom/);
  assert.equal(doc.undoStack.length, depth);
  assert.equal(doc.revision, revision);
  assert.deepEqual(modelRead(doc).A, { tokens: 59, designed: 0, held: 0, hotspots: 0, ligand: false });
  // A batch whose second edit is invalid applies nothing.
  assert.throws(() => doc.setConditions([
    { condition: 'hotspot', atom_indices: [0] },
    { condition: 'no-such-condition', atom_indices: [0] },
  ]));
  assert.equal(modelRead(doc).A.hotspots, 0);
  assert.equal(doc.undoStack.length, depth);
});

test('review fixes: scaffolding with a ligand selected, re-scaffolding, modified residues', () => {
  // Enzyme: catalytic Asp + the ligand selected, motif placed anywhere.
  const enz = open('enzyme_design.cif');
  enz.realizeSegments(1);
  delete enz.atoms.annotations.annotation_expsegmin_1_residue;
  delete enz.atoms.annotations.annotation_expsegmax_1_residue;
  enz.atoms.res_name.forEach((name, i) => {
    if (name === '<M>') { enz.atoms.hetero[i] = false; if (enz.atoms.is_polymer) enz.atoms.is_polymer[i] = true; }
  });
  enz.scaffoldMotif(atomsWhere(enz, (a, i) => ['ASP', 'ORO'].includes(a.res_name[i])), { unindexed: true, totalMin: 180, totalMax: 180 });
  assert.deepEqual(ranges(enz), ['180-180']);
  // The ligand keeps its pose: every ORO atom held at its coordinates.
  const held = enz.atoms.annotations.condition_coordinate_1_atom;
  const oro = atomsWhere(enz, (a, i) => a.res_name[i] === 'ORO');
  assert.ok(oro.length > 0 && oro.every((i) => Number.isFinite(held[i]?.[0])));

  // Scaffolding an already scaffolded chain replaces its segments instead of keeping them.
  const doc = open();
  const ca = (lo, hi) => atomsWhere(doc, (a, i) => a.res_id[i] >= lo && a.res_id[i] <= hi && a.atom_name[i] === 'CA' && !/^\d+-\d+$/.test(a.res_name[i]));
  doc.scaffoldMotif(ca(30, 40));
  doc.scaffoldMotif(ca(33, 36), { unindexed: true, totalMin: 50, totalMax: 60 });
  assert.deepEqual(ranges(doc), ['50-60']);

  // A modified residue stored as HETATM is still a polymer residue (is_polymer), not a ligand.
  const mod = open();
  for (let i = 0; i < mod.atoms.nAtoms; i += 1) if (mod.atoms.res_id[i] === 30) mod.atoms.hetero[i] = true;
  const read = modelRead(mod);
  assert.deepEqual(Object.keys(read), ['A']);
  assert.equal(read.A.tokens, 59);
});
