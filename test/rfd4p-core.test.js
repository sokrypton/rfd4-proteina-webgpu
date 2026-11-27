import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  ALLOWED_RESIDUE_TYPES,
  DEFAULT_SAMPLING,
  RFD4P_PRESETS,
  STANDARD_POLYMER_RESIDUES,
  UNIFIED_ATOM37_ENCODING,
  UNIFIED_RESTYPE_ATOM37_MASK,
  buildBatch,
  buildPartialSchedule,
  corruptBatch,
  flowEmitMasks,
  linearSumAssignmentMaximize,
  noiseSchedule,
  pairCondFeatures,
  pairFeatures,
  parseAtomSiteCif,
  processSampleToAtoms,
  readShardedTensor,
  recycleDepths,
  resolvePartialT,
  sampleInitialNoise,
  sampleNoisedTargetBackbone,
  sampleSchedule,
  scheduleTsrSafe,
  seededRng,
  superimposeCoords3D,
  toCifString,
  toPdbString,
  trunkCondFeatures,
  trunkSeqFeatures,
} from '../webgpu/index.js';
import { applyContig, evalSelection } from '../shared/rfd4p/pipeline/index.js';
import { parseCathCode } from '../shared/rfd4p/conditions.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EXAMPLES_DIR = path.resolve(__dirname, '../web/examples');

test('UNIFIED_ATOM37_ENCODING has 147 tokens, 37 atom slots, and 28 standard polymers', () => {
  assert.equal(UNIFIED_ATOM37_ENCODING.n_tokens, 147);
  assert.equal(UNIFIED_ATOM37_ENCODING.n_atoms, 37);
  assert.equal(STANDARD_POLYMER_RESIDUES.length, 28);
  assert.equal(UNIFIED_RESTYPE_ATOM37_MASK.length, 147);
  assert.equal(ALLOWED_RESIDUE_TYPES.length, 7);

  const alaIdx = UNIFIED_ATOM37_ENCODING.token_to_idx.ALA;
  const alaMask = UNIFIED_RESTYPE_ATOM37_MASK[alaIdx];
  const alaAtomCount = alaMask.reduce((acc, v) => acc + v, 0);
  assert.equal(alaAtomCount, 6);
  assert.equal(alaMask[1], 1); // CA at slot 1
  assert.equal(alaMask[36], 1); // OXT at slot 36
});

test('Pipeline contig parser, selection DSL, and CATH code parser match RFD4-Proteina semantics', () => {
  // Conflicting chain types in one run without /0 should throw
  assert.throws(() => applyContig(null, 'X:5P, Y:5R'), /conflicting chain types/i);

  // Selection DSL + CATH code triple parser
  const { atoms } = buildBatch(
    {
      contig: 'h1:10-10P,/0,r1:6-6R',
      conditions: {
        C_CATH: { '3.40.50': [{ select: "sel('A/*/*/*')" }] },
        C_SEC: { HELIX: [{ select: "sel('A/*/*/*')" }] },
      },
    },
    { seed: 42 }
  );
  assert.equal(atoms.res_name.length, 16);
  const maskA = evalSelection(atoms, "sel('A/*/*/*')");
  assert.equal(maskA.filter(Boolean).length, 10);
  const maskRna = evalSelection(atoms, 'is_rna');
  assert.equal(maskRna.filter(Boolean).length, 6);

  const [cIdx, aIdx, tIdx] = parseCathCode('3.40.50');
  assert.ok(cIdx >= 0 && aIdx >= 0 && tIdx >= 0);
  const [cWild, aWild, tWild] = parseCathCode('3.x.x');
  assert.ok(cWild >= 0);
  assert.equal(aWild, -1);
  assert.equal(tWild, -1);
});

test('buildBatch builds all 10 interactive presets (.cif.zst and specMode) and populates 80 batch keys', () => {
  for (const preset of RFD4P_PRESETS) {
    let res;
    if (preset.cifUrl && !preset.specMode) {
      const bytes = fs.readFileSync(path.resolve(__dirname, '..', preset.cifUrl));
      res = buildBatch(bytes, { seed: 42 });
    } else {
      const inputCif = preset.cifUrl ? fs.readFileSync(path.resolve(__dirname, '..', preset.cifUrl)) : null;
      res = buildBatch({ contig: preset.contig, conditions: preset.conditions }, { inputCif, seed: 42 });
    }
    assert.ok(res.n > 0, `Preset ${preset.id} produced n=${res.n}`);
    assert.equal(Object.keys(res.batch).length, 80, `Preset ${preset.id} should have 80 batch keys`);
    assert.ok(res.batch.__atoms, `Preset ${preset.id} should attach non-enumerable batch.__atoms`);
  }
});

test('pairFeatures static cache matches uncached pairFeatures bit-for-bit', () => {
  const bytes = fs.readFileSync(path.join(EXAMPLES_DIR, 'ppi_pdl1.cif.zst'));
  const { batch, n } = buildBatch(bytes, { seed: 42 });
  const cfg = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../weights/config.json'), 'utf8')).config.nn;
  const distCfg = { dim: cfg.xt_pair_dist_dim, min: cfg.xt_pair_dist_min, max: cfg.xt_pair_dist_max };

  const ca0 = new Float32Array(n * 3);
  const ca1 = new Float32Array(n * 3);
  for (let i = 0; i < ca0.length; i += 1) {
    ca0[i] = Math.sin(i * 0.1);
    ca1[i] = Math.cos(i * 0.13);
  }

  const cache = {};
  pairFeatures(batch, { n, caNm: ca0, feats: cfg.feats_pair_repr, dist: distCfg, seqSepDim: cfg.seq_sep_dim, cache });
  const cached1 = pairFeatures(batch, {
    n,
    caNm: ca1,
    feats: cfg.feats_pair_repr,
    dist: distCfg,
    seqSepDim: cfg.seq_sep_dim,
    cache,
  }).data;
  const fresh1 = pairFeatures(batch, {
    n,
    caNm: ca1,
    feats: cfg.feats_pair_repr,
    dist: distCfg,
    seqSepDim: cfg.seq_sep_dim,
  }).data;

  assert.equal(cached1.length, fresh1.length);
  for (let i = 0; i < cached1.length; i += 1) {
    assert.equal(cached1[i], fresh1[i]);
  }
});

test('Flow-matching schedule, recycleDepths, and partial diffusion corruption match RFD4-Proteina', () => {
  const tLog = scheduleTsrSafe({ mode: 'log', p: 2.0 }, 10);
  assert.equal(tLog.length, 11);
  assert.ok(tLog[0] > 0 && tLog[0] < 1e-2);
  assert.equal(tLog[10], 1.0);
  for (let i = 1; i <= 10; i += 1) {
    assert.ok(tLog[i] > tLog[i - 1]);
  }

  const dMed60 = recycleDepths('medium', 60, 60);
  assert.equal(dMed60.reduce((a, b) => a + b, 0), 180);
  const dLow60 = recycleDepths('low', 60, 60);
  assert.equal(dLow60.reduce((a, b) => a + b, 0), 120);
  // `configs/recycle_strategy/high/front_loaded.yaml`: 300 trunk passes.
  assert.equal(recycleDepths('high', 60, 60).reduce((a, b) => a + b, 0), 300);
  const dPartial30 = recycleDepths('low', 30, 60);
  assert.deepEqual([...dPartial30], [...dLow60.slice(30)]);

  const sched = sampleSchedule({ ...DEFAULT_SAMPLING, nsteps: 20 });
  assert.equal(sched.bb_ca.ts.length, 21);
  assert.equal(sched.local_latents.ts.length, 21);

  const partialTDict = resolvePartialT({ partialTrajectoryFractionSteps: 0.5, nsteps: 60 }, DEFAULT_SAMPLING);
  const pSched = buildPartialSchedule(partialTDict, 30, DEFAULT_SAMPLING, { fullNsteps: 60 });
  assert.equal(pSched.bb_ca.ts.length, 31);
  assert.ok(Math.abs(pSched.bb_ca.ts[0] - partialTDict.bb_ca) < 1e-6);
  assert.equal(pSched.bb_ca.ts[30], 1.0);

  const rng = seededRng(123);
  const n = 12;
  const emit = { bb_ca: new Uint8Array(n).fill(1), local_latents: new Uint8Array(n).fill(1) };
  const coordCondMask = new Uint8Array(n);
  const moleculeIid = new Int32Array(n);
  const x0 = sampleInitialNoise({ n, emit, coordCondMask, moleculeIid, rng });
  assert.equal(x0.bb_ca.length, n * 3);
  assert.equal(x0.local_latents.length, n * 32);

  const corrupted = corruptBatch({
    n,
    x1: { bb_ca: new Float32Array(n * 3), local_latents: new Float32Array(n * 32) },
    partialTDict: { bb_ca: 0.5, local_latents: 0.5 },
    emit,
    coordCondMask,
    moleculeIid,
    rng,
  });
  assert.equal(corrupted.xt.bb_ca.length, n * 3);
  const resampled = sampleNoisedTargetBackbone({
    n,
    targetBackbone: corrupted.x1.bb_ca,
    tBb: 0.5,
    emitBbCa: emit.bb_ca,
    coordCondMask,
    moleculeIid,
    rng,
  });
  assert.equal(resampled.length, n * 3);
});

test('Kabsch 3D superimposer, Hungarian assignment, and processSampleToAtoms export valid PDB/mmCIF', () => {
  const fixed = new Float32Array([0.0, 0.0, 0.0, 3.8, 0.0, 0.0, 1.9, 3.2, 0.0, 1.9, 1.1, 2.8]);
  const mobile = new Float32Array(fixed.length);
  for (let i = 0; i < fixed.length / 3; i += 1) {
    mobile[i * 3 + 0] = -fixed[i * 3 + 1] + 10.0;
    mobile[i * 3 + 1] = fixed[i * 3 + 0] - 5.0;
    mobile[i * 3 + 2] = fixed[i * 3 + 2] + 3.0;
  }
  const { aligned, rmsd } = superimposeCoords3D(fixed, mobile);
  assert.ok(rmsd < 1e-5);
  for (let i = 0; i < fixed.length; i += 1) {
    assert.ok(Math.abs(aligned[i] - fixed[i]) < 1e-4);
  }

  const score = new Float64Array([10, 1, 20, 2, 3, 9, 15, 19, 4, 1, 2, 3, 5, 12, 11]);
  const { colInd } = linearSumAssignmentMaximize(score, 3, 5);
  assert.deepEqual([...colInd], [2, 1, 3]);

  const bytes = fs.readFileSync(path.join(EXAMPLES_DIR, 'motif_unidx_1prw.cif.zst'));
  const { batch, n, atoms } = buildBatch(bytes, { seed: 42 });
  const coors = new Float32Array(n * 37 * 3);
  const atomMask = new Uint8Array(n * 37);
  const residueType = new Int32Array(n).fill(1); // ALA
  for (let i = 0; i < n; i += 1) {
    for (const s of [0, 1, 2, 3, 4]) {
      atomMask[i * 37 + s] = 1;
      coors[(i * 37 + s) * 3 + 0] = i * 3.8 + s * 0.5;
    }
  }
  const structure = processSampleToAtoms({
    coors,
    atomMask,
    residueType,
    paddingMask: batch.padding_mask.data,
    insertionLogits: new Float32Array(n * n),
    inputAtoms: atoms,
    batch,
  });
  const pdb = toPdbString(structure);
  const cif = toCifString(structure);
  assert.ok(pdb.includes('ATOM  '));
  assert.ok(cif.includes('_atom_site.Cartn_x'));
  const parsedCif = parseAtomSiteCif(cif);
  assert.equal(parsedCif.length, structure.atoms.length);
});

test('readShardedTensor dequantizes block-64 INT8 and ./weights/int8/manifest.json indexes all 1,257 tensors', () => {
  // Synthetic 2-block (128-element) int8 tensor
  const blockSize = 64;
  const n = 128;
  const nBlocks = 2;
  const buf = new ArrayBuffer(n + nBlocks * 2);
  const q = new Int8Array(buf, 0, n);
  const scales = new Uint16Array(buf, n, nBlocks);
  // 0x3c00 = 1.0 in IEEE-754 binary16, 0x3800 = 0.5 in IEEE-754 binary16
  scales[0] = 0x3c00;
  scales[1] = 0x3800;
  for (let i = 0; i < 64; i += 1) q[i] = i - 32;
  for (let i = 64; i < 128; i += 1) q[i] = i - 64;

  const data = readShardedTensor(
    {
      dtype: 'int8',
      shape: [2, 64],
      byteOffset: 0,
      scaleOffset: n,
      block: blockSize,
    },
    buf
  );
  assert.equal(data.length, 128);
  assert.equal(data[0], -32.0);
  assert.equal(data[32], 0.0);
  assert.equal(data[64], 0.0);
  assert.equal(data[66], 1.0);

  const manifestPath = path.resolve(__dirname, '../weights/int8/manifest.json');
  if (fs.existsSync(manifestPath)) {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    assert.equal(manifest.format, 'rfd4p-sharded-v1');
    assert.equal(manifest.bundle.format, 'int8');
    assert.equal(Object.keys(manifest.tensors).length, 1257);
  }
});

test('processSampleToAtoms inputFrame aligns atomized ligands and unindexed motifs onto inputAtoms.coord', () => {
  // 90 deg rotation around Z + translation [25, -14, 8]
  const rotTrans = ([x, y, z]) => [-y + 25, x - 14, z + 8];
  const alaIdx = UNIFIED_ATOM37_ENCODING.token_to_idx.ALA;
  const alaSlots = UNIFIED_ATOM37_ENCODING.token_slot_map.ALA; // N, CA, C, O, CB

  // Input atom table:
  //   tid 0: designed polymer residue (<M>, group -1) — receives unindexed motif from tid 4
  //   tid 1..3: atomized ligand atoms (group 1)
  //   tid 4: unindexed motif residue ALA (group 1, condition_unindexed=1)
  const ligCoords = [
    [10.0, 20.0, 30.0],
    [11.5, 20.0, 30.0],
    [10.0, 21.6, 30.2],
  ];
  const motifAtoms = [
    ['N',  [12.0, 18.0, 29.0]],
    ['CA', [13.2, 18.6, 29.4]],
    ['C',  [13.8, 19.8, 28.7]],
    ['O',  [13.2, 20.8, 28.4]],
    ['CB', [14.3, 17.6, 29.8]],
  ];

  const inputAtoms = {
    token_id: [],
    atom_name: [],
    res_name: [],
    chain_id: [],
    res_id: [],
    ins_code: [],
    element: [],
    hetero: [],
    atomize: [],
    coord: [],
    condition_CoordinateGroup_1_atom: [],
    condition_index_1_residue: [],
    condition_coordinate_1_atom: [],
    condition_InternalCoordinate_1_atom: [],
  };
  const pushAtom = (tid, name, res, ch, resId, el, het, atomize, coord, grp, unidx) => {
    inputAtoms.token_id.push(tid);
    inputAtoms.atom_name.push(name);
    inputAtoms.res_name.push(res);
    inputAtoms.chain_id.push(ch);
    inputAtoms.res_id.push(resId);
    inputAtoms.ins_code.push('');
    inputAtoms.element.push(el);
    inputAtoms.hetero.push(het);
    inputAtoms.atomize.push(atomize);
    inputAtoms.coord.push(coord);
    inputAtoms.condition_CoordinateGroup_1_atom.push(grp);
    inputAtoms.condition_index_1_residue.push(!unidx);
    inputAtoms.condition_coordinate_1_atom.push(null);
    inputAtoms.condition_InternalCoordinate_1_atom.push(grp > 0 ? coord : null);
  };

  // tid 0: designed scaffold residue
  pushAtom(0, 'CA', '<M>', 'A', 1, 'C', false, false, [0, 0, 0], -1, false);
  // tid 1..3: atomized ligand DHZ in group 1
  ligCoords.forEach((c, k) => pushAtom(1 + k, `C${k + 1}`, 'DHZ', 'B', 1, 'C', true, true, c, 1, false));
  // tid 4: unindexed motif ALA in group 1
  motifAtoms.forEach(([nm, c]) => pushAtom(4, nm, 'ALA', 'M', 102, nm[0], false, false, c, 1, true));

  const n = 5;
  const coors = new Float32Array(n * 37 * 3);
  const atomMask = new Uint8Array(n * 37);
  const residueType = new Int32Array([alaIdx, -1, -1, -1, alaIdx]);
  const paddingMask = new Uint8Array([1, 1, 1, 1, 1]);

  // Place tid 0 (designed residue) at rotated+translated motif coords
  for (const [nm, c] of motifAtoms) {
    const slot = alaSlots[nm];
    atomMask[0 * 37 + slot] = 1;
    const rt = rotTrans(c);
    coors[(0 * 37 + slot) * 3 + 0] = rt[0];
    coors[(0 * 37 + slot) * 3 + 1] = rt[1];
    coors[(0 * 37 + slot) * 3 + 2] = rt[2];
  }
  // Place atomized ligand atoms (tid 1..3) at slot 1 (atomized default slot)
  ligCoords.forEach((c, k) => {
    const tid = 1 + k;
    atomMask[tid * 37 + 1] = 1;
    const rt = rotTrans(c);
    coors[(tid * 37 + 1) * 3 + 0] = rt[0];
    coors[(tid * 37 + 1) * 3 + 1] = rt[1];
    coors[(tid * 37 + 1) * 3 + 2] = rt[2];
  });
  // Insertion logits strongly map unindexed tid 4 -> target tid 0
  const insertionLogits = new Float32Array(n * n).fill(-10);
  insertionLogits[4 * n + 0] = 20;

  const res = processSampleToAtoms({
    coors,
    atomMask,
    residueType,
    paddingMask,
    insertionLogits,
    inputAtoms,
    inputFrame: true,
  });

  assert.equal(res.inputFrame, true);
  // Every group-1 atom (both unindexed motif and atomized ligand) must align back to inputAtoms.coord
  const grpAtoms = res.atoms.filter((a) => a.coordGroup === 1);
  assert.equal(grpAtoms.length, motifAtoms.length + ligCoords.length);
  let sumSq = 0;
  for (const a of grpAtoms) {
    assert.ok(a.inCoord, `missing inCoord on ${a.resName}:${a.atomName}`);
    const d2 = (a.x - a.inCoord[0]) ** 2 + (a.y - a.inCoord[1]) ** 2 + (a.z - a.inCoord[2]) ** 2;
    sumSq += d2;
  }
  const rmsd = Math.sqrt(sumSq / grpAtoms.length);
  assert.ok(rmsd < 1e-3, `expected RMSD < 1e-3 Å, got ${rmsd}`);

  // NaN guard: a single NaN generated atom in the rigid group must not poison the alignment
  const coorsWithNan = new Float32Array(coors);
  coorsWithNan[(1 * 37 + 1) * 3 + 0] = NaN;
  const resNan = processSampleToAtoms({
    coors: coorsWithNan,
    atomMask,
    residueType,
    paddingMask,
    insertionLogits,
    inputAtoms,
    inputFrame: true,
  });
  assert.equal(resNan.inputFrame, true);
  const finiteMotif = resNan.atoms.filter((a) => a.coordGroup === 1 && Number.isFinite(a.x));
  assert.equal(finiteMotif.length, motifAtoms.length + ligCoords.length - 1);
  for (const a of finiteMotif) {
    const d = Math.hypot(a.x - a.inCoord[0], a.y - a.inCoord[1], a.z - a.inCoord[2]);
    assert.ok(d < 1e-3, `NaN guard failed for ${a.atomName}: dist=${d}`);
  }

  // Unindexed-only (no ligand): aligns onto unindexed motif alone when insertionLogits are supplied,
  // and does NOT trust geometric fallback when insertionLogits are absent.
  const uOnlyAtoms = {
    token_id: [0, ...motifAtoms.map(() => 1)],
    atom_name: ['CA', ...motifAtoms.map(([nm]) => nm)],
    res_name: ['<M>', ...motifAtoms.map(() => 'ALA')],
    chain_id: ['A', ...motifAtoms.map(() => 'M')],
    res_id: [1, ...motifAtoms.map(() => 102)],
    ins_code: Array(1 + motifAtoms.length).fill(''),
    element: ['C', ...motifAtoms.map(([nm]) => nm[0])],
    hetero: Array(1 + motifAtoms.length).fill(false),
    atomize: Array(1 + motifAtoms.length).fill(false),
    coord: [[0, 0, 0], ...motifAtoms.map(([, c]) => c)],
    condition_CoordinateGroup_1_atom: [-1, ...motifAtoms.map(() => 1)],
    condition_index_1_residue: [true, ...motifAtoms.map(() => false)],
    condition_coordinate_1_atom: Array(1 + motifAtoms.length).fill(null),
    condition_InternalCoordinate_1_atom: [null, ...motifAtoms.map(([, c]) => c)],
  };
  const uCoors = new Float32Array(2 * 37 * 3);
  const uMask = new Uint8Array(2 * 37);
  for (const [nm, c] of motifAtoms) {
    const slot = alaSlots[nm];
    uMask[0 * 37 + slot] = 1;
    const rt = rotTrans(c);
    uCoors[(0 * 37 + slot) * 3 + 0] = rt[0];
    uCoors[(0 * 37 + slot) * 3 + 1] = rt[1];
    uCoors[(0 * 37 + slot) * 3 + 2] = rt[2];
  }
  const uLogits = new Float32Array([-10, -10, 20, -10]); // tid 1 -> tid 0
  const resUOnly = processSampleToAtoms({
    coors: uCoors,
    atomMask: uMask,
    residueType: new Int32Array([alaIdx, alaIdx]),
    paddingMask: new Uint8Array([1, 1]),
    insertionLogits: uLogits,
    inputAtoms: uOnlyAtoms,
    inputFrame: true,
  });
  assert.equal(resUOnly.inputFrame, true);
  for (const a of resUOnly.atoms) {
    assert.ok(Math.hypot(a.x - a.inCoord[0], a.y - a.inCoord[1], a.z - a.inCoord[2]) < 1e-3);
  }
  // Without insertionLogits, geometric fallback must not stamp inCoord onto unindexed targets
  const resNoLogits = processSampleToAtoms({
    coors: uCoors,
    atomMask: uMask,
    residueType: new Int32Array([alaIdx, alaIdx]),
    paddingMask: new Uint8Array([1, 1]),
    insertionLogits: null,
    inputAtoms: uOnlyAtoms,
    inputFrame: true,
  });
  assert.equal(resNoLogits.inputFrame, false);
});
