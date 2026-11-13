/**
 * How well a refold puts the motif back, as RFD4's own refolding metrics ask it (configs/metrics/refolding_metrics:
 * ground truth is the design, predicted is the refold):
 *
 * - `motifRmsd` is `refolded_motif_allatom_rmsd`: the refold is superposed on the design over the motif's atoms
 *   (each rigid coordinate group on its own, since the groups of a motif are free to sit apart), and the RMSD of those
 *   atoms is taken. It asks whether the motif's own geometry survives the refold, wherever the refold puts it.
 * - `motifBackboneRmsd` is the same atoms after the refold is superposed on the design's backbone (N, CA, C, O of the
 *   designed chains), the way `refolded_target_backbone_aligned_binder_rmsd` aligns. It asks whether the motif ends up
 *   where the design put it relative to the scaffold, so it is never smaller than the first for a rigid motif.
 *
 * The motif's atoms are the ones the run held to coordinates (`crd`) in the residues `motifSlots` found, protein only;
 * a refold numbers its residues its own way but keeps each chain's residues in the design's order, so an atom is
 * found by its residue's place along its chain and its name. A modified residue the refold writes as HETATM (KCX)
 * counts as the residue it is. Pure: strings and plain objects.
 */

/** The motif's atoms, from the design's atoms and the slots: those held to coordinates in the motif's residues. */
export function motifAtomsOf(designAtoms, slots) {
  if (!slots?.length) return [];
  const wanted = new Set(slots.map((s) => `${s.chain}:${s.index}`));
  const order = new Map();
  const atoms = [];
  for (const a of designAtoms) {
    if (a.isHetero) continue;
    if (!order.has(a.chainId)) order.set(a.chainId, new Map());
    const residues = order.get(a.chainId);
    if (!residues.has(a.resId)) residues.set(a.resId, residues.size);
    const index = residues.get(a.resId);
    if (!wanted.has(`${a.chainId}:${index}`)) continue;
    if (!(a.crd && Number.isFinite(a.crd[0]))) continue;
    atoms.push({ chain: a.chainId, index, name: a.atomName, group: a.coordGroup > 0 ? a.coordGroup : 0 });
  }
  return atoms;
}

/** The polymer chains of a PDB (any ATOM record; water out), residues in file order, each atom's coordinates by name. */
function residuesOf(pdb) {
  const chains = new Map();
  let previous = null;
  for (const line of String(pdb ?? '').split('\n')) {
    if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) continue;
    const name = line.slice(17, 20).trim();
    if (name === 'HOH' || name === 'WAT' || name === 'DOD') continue;
    const letter = line[21];
    if (!chains.has(letter)) chains.set(letter, { polymer: false, residues: [] });
    const chain = chains.get(letter);
    if (line.startsWith('ATOM  ')) chain.polymer = true;
    const key = line.slice(21, 27);
    if (key !== previous) { chain.residues.push(new Map()); previous = key; }
    chain.residues[chain.residues.length - 1].set(line.slice(12, 16).trim(),
      [Number(line.slice(30, 38)), Number(line.slice(38, 46)), Number(line.slice(46, 54))]);
  }
  return new Map([...chains].filter(([, c]) => c.polymer).map(([letter, c]) => [letter, c.residues]));
}

const eigenSymmetric = (matrix) => {
  // Jacobi rotations on a small symmetric matrix: eigenvalues, and eigenvectors as columns
  const n = matrix.length;
  const a = matrix.map((row) => [...row]);
  const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep += 1) {
    let off = 0;
    for (let i = 0; i < n; i += 1) for (let j = i + 1; j < n; j += 1) off += a[i][j] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        if (Math.abs(a[p][q]) < 1e-30) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k += 1) {
          const akp = a[k][p]; const akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k += 1) {
          const apk = a[p][k]; const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k += 1) {
          const vkp = v[k][p]; const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: a.map((row, i) => row[i]), vectors: v };
};

/**
 * The rigid motion that lays `moving` on `fixed` with least squared error (Horn's quaternion method), as a function.
 * @param {number[][]} moving
 * @param {number[][]} fixed
 */
export function superpose(moving, fixed) {
  const n = moving.length;
  const centre = (points) => [0, 1, 2].map((k) => points.reduce((s, p) => s + p[k], 0) / n);
  const cm = centre(moving);
  const cf = centre(fixed);
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i += 1) {
    for (let r = 0; r < 3; r += 1) for (let c = 0; c < 3; c += 1) S[r][c] += (moving[i][r] - cm[r]) * (fixed[i][c] - cf[c]);
  }
  const N = [
    [S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]],
    [S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]],
    [S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]],
    [S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]],
  ];
  const { values, vectors } = eigenSymmetric(N);
  const best = values.indexOf(Math.max(...values));
  const [w, x, y, z] = vectors.map((row) => row[best]);
  const R = [
    [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
  ];
  return (p) => [0, 1, 2].map((r) => R[r][0] * (p[0] - cm[0]) + R[r][1] * (p[1] - cm[1]) + R[r][2] * (p[2] - cm[2]) + cf[r]);
}

const sumSquares = (a, b) => a.reduce((s, p, i) => s + (p[0] - b[i][0]) ** 2 + (p[1] - b[i][1]) ** 2 + (p[2] - b[i][2]) ** 2, 0);

/**
 * @param {string} designPdb
 * @param {string} refoldPdb the refold, in the design's chain letters (any frame)
 * @param {{chain: string, index: number, name: string, group: number}[]} motif from `motifAtomsOf`
 * @param {string[]} [designed] the designed chains, whose backbone the second metric aligns on (default: every chain)
 * @returns {{motifRmsd: number, motifBackboneRmsd: number, atoms: number, residues: number}|null} null with no
 *   motif, or none of its atoms in both structures
 */
export function motifRmsds(designPdb, refoldPdb, motif, designed) {
  if (!motif?.length) return null;
  const design = residuesOf(designPdb);
  const refold = residuesOf(refoldPdb);
  const pairs = [];
  for (const m of motif) {
    const d = design.get(m.chain)?.[m.index]?.get(m.name);
    const r = refold.get(m.chain)?.[m.index]?.get(m.name);
    if (d && r && [...d, ...r].every(Number.isFinite)) pairs.push({ d, r, group: m.group, key: `${m.chain}:${m.index}` });
  }
  if (pairs.length < 3) return null;

  // On the motif: each rigid group by itself; atoms with no group together. A group of fewer than three atoms has no
  // fit of its own (any two points superpose exactly), so its atoms join the ungrouped ones.
  let squared = 0;
  const size = new Map();
  for (const p of pairs) size.set(p.group, (size.get(p.group) ?? 0) + 1);
  const groups = new Map();
  for (const p of pairs) {
    const group = size.get(p.group) >= 3 ? p.group : 0;
    groups.set(group, [...(groups.get(group) ?? []), p]);
  }
  for (const part of groups.values()) {
    if (part.length < 3) return null;
    const move = superpose(part.map((p) => p.r), part.map((p) => p.d));
    squared += sumSquares(part.map((p) => move(p.r)), part.map((p) => p.d));
  }

  // On the backbone of the designed chains
  const chains = (designed?.length ? designed : [...design.keys()]).filter((c) => design.has(c) && refold.has(c));
  const dBack = [];
  const rBack = [];
  for (const chain of chains) {
    const a = design.get(chain);
    const b = refold.get(chain);
    for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
      for (const name of ['N', 'CA', 'C', 'O']) {
        const p = a[i].get(name);
        const q = b[i].get(name);
        if (p && q) { dBack.push(p); rBack.push(q); }
      }
    }
  }
  let backbone = NaN;
  if (dBack.length >= 3) {
    const move = superpose(rBack, dBack);
    backbone = Math.sqrt(sumSquares(pairs.map((p) => move(p.r)), pairs.map((p) => p.d)) / pairs.length);
  }
  return { motifRmsd: Math.sqrt(squared / pairs.length), motifBackboneRmsd: backbone, atoms: pairs.length,
    residues: new Set(pairs.map((p) => p.key)).size };
}
