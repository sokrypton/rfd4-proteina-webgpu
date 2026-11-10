import assert from 'node:assert/strict';
import test from 'node:test';
import { motifAtomsOf, motifRmsds, superpose } from '../web/motif-metrics.js';

const atom = (serial, name, res, chain, num, [x, y, z], hetero = false) => `${hetero ? 'HETATM' : 'ATOM  '}${String(serial).padStart(5)} `
  + `${name.padEnd(4)} ${res.padStart(3)} ${chain}${String(num).padStart(4)}    ${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00  0.00`;

// A 6-residue chain with a backbone and a Cb on each, the motif being residues 1 and 4 (their side chains)
const chain = (move = (p) => p, bend = 0) => {
  const lines = [];
  let serial = 0;
  for (let i = 0; i < 6; i += 1) {
    const base = [3.8 * i, 2 * Math.sin(i), 1.5 * Math.cos(i)];
    for (const [name, off] of [['N', [-1.2, 0.3, 0]], ['CA', [0, 0, 0]], ['C', [1.2, 0.4, 0.2]], ['O', [1.5, 1.4, 0.2]], ['CB', [0, -1.3, 0.9 + (i === 4 ? bend : 0)]]]) {
      serial += 1;
      lines.push(atom(serial, name, i === 1 ? 'KCX' : 'HIS', 'A', i + 1, move([base[0] + off[0], base[1] + off[1], base[2] + off[2]]), i === 1));
    }
  }
  // a ligand chain, which no metric reads
  lines.push(atom(99, 'ZN', 'ZN', 'B', 1, move([1, 1, 1]), true));
  return lines.join('\n');
};
const rotate = ([x, y, z]) => [-y + 10, x - 5, z + 2];   // a quarter turn about z, and a shift
const MOTIF = [1, 4].flatMap((index) => ['CB', 'O', 'N'].map((name) => ({ chain: 'A', index, name, group: 0 })));

test('a refold that is the design moved as a body has no motif error of either kind', () => {
  const r = motifRmsds(chain(), chain(rotate), MOTIF, ['A']);
  assert.ok(r.motifRmsd < 1e-6, String(r.motifRmsd));
  assert.ok(r.motifBackboneRmsd < 1e-6, String(r.motifBackboneRmsd));
  assert.deepEqual([r.atoms, r.residues], [6, 2]);
});

test('a motif atom that moved relative to the scaffold shows in the backbone-aligned number', () => {
  // residue 4's Cb is 2 A off in the refold, the backbone is not: the backbone fit leaves the whole 2 A on that one atom
  // (RMS over six atoms: 2/sqrt(6)), where fitting the motif itself spreads it over all of them
  const r = motifRmsds(chain(), chain(rotate, 2), MOTIF, ['A']);
  assert.ok(Math.abs(r.motifBackboneRmsd - 2 / Math.sqrt(6)) < 0.05, String(r.motifBackboneRmsd));
  assert.ok(r.motifRmsd < r.motifBackboneRmsd, `${r.motifRmsd} < ${r.motifBackboneRmsd}`);
});

test('the motif is found by its residue\'s place in its chain, so a refold that numbers its own way, and writes the modified residue as HETATM, still matches', () => {
  const renumbered = chain(rotate).split('\n').map((l) => (l.startsWith('ATOM') || l.startsWith('HETATM') ? `${l.slice(0, 22)}${String(Number(l.slice(22, 26)) + 100).padStart(4)}${l.slice(26)}` : l)).join('\n');
  const r = motifRmsds(chain(), renumbered, MOTIF, ['A']);
  assert.ok(r.motifRmsd < 1e-6);
});

test('rigid groups are each fitted on their own', () => {
  // residue 5 (index 4) sits 5 A further along x in the refold than the design has it, as a body: fitted as one motif that leaves
  // an error, as two groups of three atoms each it leaves none
  const moved = chain().split('\n').map((l) => (l.slice(22, 26).trim() === '5' ? l.replace(/^(.{30})(.{8})/, (_, a, x) => `${a}${(Number(x) + 5).toFixed(3).padStart(8)}`) : l)).join('\n');
  const grouped = MOTIF.map((m) => ({ ...m, group: m.index }));
  assert.ok(motifRmsds(chain(), moved, MOTIF, ['A']).motifRmsd > 0.5);
  assert.ok(motifRmsds(chain(), moved, grouped, ['A']).motifRmsd < 1e-6);
});

test('no motif, or too little of it, is no number', () => {
  assert.equal(motifRmsds(chain(), chain(), [], ['A']), null);
  assert.equal(motifRmsds(chain(), chain(), [{ chain: 'A', index: 1, name: 'CB', group: 0 }, { chain: 'A', index: 4, name: 'CB', group: 0 }], ['A']), null);
});

test('the motif atoms are those held to coordinates in the motif\'s residues, protein only', () => {
  const atoms = [
    { chainId: 'A', resId: 10, atomName: 'CB', crd: [1, 2, 3], coordGroup: 2 },
    { chainId: 'A', resId: 10, atomName: 'CG', crd: [NaN, 0, 0], coordGroup: 2 },
    { chainId: 'A', resId: 11, atomName: 'CB', crd: [1, 2, 3], coordGroup: 0 },
    { chainId: 'A', resId: 12, atomName: 'CB', crd: [1, 2, 3], coordGroup: 0 },
    { chainId: 'B', resId: 1, atomName: 'P1', crd: [1, 2, 3], isHetero: true },
  ];
  assert.deepEqual(motifAtomsOf(atoms, [{ chain: 'A', index: 0 }, { chain: 'A', index: 2 }]), [
    { chain: 'A', index: 0, name: 'CB', group: 2 }, { chain: 'A', index: 2, name: 'CB', group: 0 }]);
});

test('superpose recovers a rotation and a shift', () => {
  const points = [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1]];
  const moved = points.map(rotate);
  const back = superpose(moved, points);
  moved.forEach((p, i) => back(p).forEach((v, k) => assert.ok(Math.abs(v - points[i][k]) < 1e-9)));
});
