/**
 * Every bundled Studio example can be rebuilt from a bare structure with the design panel's operations alone: strip
 * the example (segments realized into plain residues, all conditions reset), replay what a user would click, and
 * compare the design spec with the original's.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { designSpec } from '../web/design-spec.js';
import { StudioDoc } from '../web/studio-api.js';

const cifOf = (file) => readFileSync(new URL(`../web/studio/examples/${file}`, import.meta.url), 'utf8');
const summary = (spec) => ({
  task: spec.task.key,
  contig: spec.contig,
  roles: Object.fromEntries(spec.chains.map((c) => [c.id, c.role])),
  hotspots: Object.fromEntries(spec.chains.map((c) => [c.id, c.hotspots])),
});

/** The spec without residue numbers (realizing a bare copy renumbers residues; the model does not use numbering). */
const unnumbered = (spec) => ({ ...summary(spec), contig: spec.contig.replace(/\b([A-Z]+)\d+(-\d+)?\b/g, '$1#') });

/** The example as shipped, and a bare copy: segments realized into residues, conditions removed. */
function load(file, { realize = true } = {}) {
  const original = new StudioDoc(cifOf(file), file);
  const bare = new StudioDoc(cifOf(file), file);
  if (realize) bare.realizeSegments(1);
  bare.resetConditions();
  // Realized residues are plain residues now, not segments.
  delete bare.atoms.annotations.annotation_expsegmin_1_residue;
  delete bare.atoms.annotations.annotation_expsegmax_1_residue;
  // ...and polymer residues, as in a structure the user would fetch.
  bare.atoms.res_name.forEach((name, i) => { if (name === '<M>') bare.atoms.hetero[i] = false; });
  return { original, bare };
}

const atomsWhere = (doc, pred) => {
  const a = doc.atoms;
  const out = [];
  for (let i = 0; i < a.nAtoms; i += 1) if (pred(a, i)) out.push(i);
  return out;
};
const chainAtoms = (doc, chain) => atomsWhere(doc, (a, i) => String(a.chain_id[i]) === chain && !a.hetero[i]);
const hotspotAtoms = (doc) => atomsWhere(doc, (a, i) => Boolean(a.annotations.condition_hotspot_1_atom?.[i]));
/** Atoms of the original's residues whose sequence is masked (designed) in `chain`. */
const maskedResidues = (doc, chain) => atomsWhere(doc, (a, i) => String(a.chain_id[i]) === chain && !a.hetero[i] &&
  a.annotations.condition_sequence_1_residue?.[i] === '<M>');
/** The panel's chain roles as condition edits. */
const role = (doc, chain, r) => {
  const all = chainAtoms(doc, chain);
  if (r === 'fixed') return [{ condition: 'coordinate', atom_indices: all }, { condition: 'sequence', atom_indices: all }];
  if (r === 'sequence') return [{ condition: 'coordinate', atom_indices: all, enabled: false }, { condition: 'sequence', atom_indices: all }];
  return [{ condition: 'coordinate', atom_indices: all, enabled: false }, { condition: 'sequence', atom_indices: all, enabled: false }];
};

test('protein binder: fix the target, mark hotspots, add a 50-120 chain', () => {
  const { original, bare } = load('protein_binder.cif', { realize: false });
  // Bare copy of a binder example: drop the realized binder chain, as if the user fetched only the target.
  bare.deleteAtoms(chainAtoms(bare, 'B').concat(atomsWhere(bare, (a, i) => String(a.chain_id[i]) === 'B')));
  bare.setConditions(role(bare, 'A', 'fixed'));
  bare.insertSegment({ minimum: 50, maximum: 120, newChain: true });
  bare.setCondition('hotspot', hotspotAtoms(original));
  assert.deepEqual(summary(designSpec(bare.atoms)), summary(designSpec(original.atoms)));
});

test('antibody: keep L/H sequences, redesign CDR H3, fix the target with hotspots', () => {
  const { original, bare } = load('antibody_design.cif');
  bare.setConditions([...role(bare, 'L', 'sequence'), ...role(bare, 'H', 'sequence'), ...role(bare, 'T', 'fixed')]);
  // Selection action "Redesign" on the CDR H3 residues.
  const h3 = maskedResidues(original, 'H');
  bare.setConditions([
    { condition: 'coordinate', atom_indices: h3, enabled: false, scope: 'residue' },
    { condition: 'sequence', atom_indices: h3, enabled: false, scope: 'residue' },
  ]);
  bare.setCondition('hotspot', hotspotAtoms(original));
  assert.deepEqual(summary(designSpec(bare.atoms)), summary(designSpec(original.atoms)));
});

test('enzyme: catalytic side chain placed anywhere in a 180-residue scaffold around a fixed ligand', () => {
  const { original, bare } = load('enzyme_design.cif');
  const asp = atomsWhere(bare, (a, i) => a.res_name[i] === 'ASP');
  bare.scaffoldMotif(asp, { sidechainsOnly: true, unindexed: true, totalMin: 180, totalMax: 180 });
  // Ligand rows: "Fix pose" on ORO and ZN.
  bare.setCondition('coordinate', atomsWhere(bare, (a, i) => ['ORO', 'ZN'].includes(a.res_name[i])));
  // Realizing renumbered the Asp; an unindexed motif's number does not matter, so compare without it.
  assert.deepEqual(unnumbered(designSpec(bare.atoms)), unnumbered(designSpec(original.atoms)));
});

test('motif scaffolding: keep three residues, rebuild the gaps, then set each segment range', () => {
  const { original, bare } = load('motif_scaffolding.cif');
  const fixedRes = new Set(atomsWhere(original, (a, i) => {
    const c = a.annotations.condition_coordinate_1_atom?.[i];
    return Array.isArray(c) && Number.isFinite(c[0]);
  }).map((i) => original.atoms.res_name[i]));
  bare.scaffoldMotif(atomsWhere(bare, (a, i) => fixedRes.has(a.res_name[i]) && !a.hetero[i]));
  const want = designSpec(original.atoms).chains[0].segments;
  const have = designSpec(bare.atoms).chains[0].segments;
  assert.equal(have.length, want.length);
  // Segment rows: type the original ranges (atom indices are re-read after every edit, as the panel re-renders).
  for (let k = 0; k < want.length; k += 1) {
    const seg = designSpec(bare.atoms).chains[0].segments[k];
    bare.setSegmentRange(seg.atomIndex, want[k].min, want[k].max);
  }
  assert.deepEqual(unnumbered(designSpec(bare.atoms)), unnumbered(designSpec(original.atoms)));
});

test('nanobody: keep the framework sequence, replace each CDR with a new-residue segment, fix the target', () => {
  const { original, bare } = load('nanobody.cif');
  bare.setConditions([...role(bare, 'A', 'fixed'), ...role(bare, 'H', 'sequence')]);
  bare.setCondition('hotspot', hotspotAtoms(original));
  // Selection action "Replace with new residues" on each realized CDR (the residues between framework pieces).
  const want = designSpec(original.atoms).chains.find((c) => c.id === 'H').segments;
  for (const seg of want) {
    const cdr = atomsWhere(bare, (a, i) => String(a.chain_id[i]) === 'H' && a.res_name[i] === '<M>');
    // Realized residues of the first remaining CDR: the first contiguous run of placeholders.
    const run = [];
    for (const i of cdr) {
      if (run.length && bare.atoms.res_id[i] !== bare.atoms.res_id[run[run.length - 1]] + 1 &&
        bare.atoms.res_id[i] !== bare.atoms.res_id[run[run.length - 1]]) break;
      run.push(i);
    }
    bare.replaceWithSegments(run, seg.min, seg.max);
  }
  assert.deepEqual(summary(designSpec(bare.atoms)), summary(designSpec(original.atoms)));
});

test('small-molecule binder: keep the ligand (pose predicted) and design a 150-residue protein around it', () => {
  const { original, bare } = load('small_molecule_binder.cif');
  bare.designAroundLigand({ removeProtein: true, poseFixed: false, minimum: 150, maximum: 150 });
  const got = summary(designSpec(bare.atoms));
  const want = summary(designSpec(original.atoms));
  assert.equal(got.task, want.task);
  assert.equal(got.contig, want.contig);
  assert.deepEqual(Object.values(got.roles).sort(), Object.values(want.roles).sort());
});
