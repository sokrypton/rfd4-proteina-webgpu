/**
 * RFD4-Proteina interactive browser host: AtomWorks Studio's Design card (start screen, job editor, run settings,
 * Generate, designs, Refold, Save all) with this page's engine behind it, 100% client-side: `Rfd4pGeneratorGpu` on
 * WebGPU (WebGL2 or the CPU where there is none) and LocalFold's fold for refolds. The card is Studio's
 * (`window.atomworksStudio.mountDesignCard`, docs/INTEGRATIONS.md "Design card" in atomworks-studio); this file is the
 * provider it is given (`_provider`): `generate`, the engine's live frames, overlays and designs, the refold and the
 * speed hint. Studio's REST API for the card is answered in the browser by web/studio-api.js.
 */

import {
  DEFAULT_HF_CONFIG_URL,
  DEFAULT_HF_WEIGHTS_URL,
  DEFAULT_LOCAL_CONFIG_URL,
  DEFAULT_LOCAL_SHARDED_DIR,
  DEFAULT_LOCAL_WEIGHTS_URL,
  DEFAULT_SAMPLING,
  RFD4P_PRESETS,
  Rfd4pGeneratorGpu,
  UNIFIED_ATOM37_TOKENS,
  buildBatch,
  decodeTextOrZstd,
  fetchRfd4pConfig,
  fetchSafetensors,
  fetchShardedWeights,
  generateRfd4pCpu,
  shardedWeightsCached,
  getRfd4pPresetById,
  requestWebGpuDevice,
  processSampleToAtoms,
  superimposeCoords3D,
  toCifString,
  toPdbString,
} from '../webgpu/index.js';
import {
  StudioDoc,
  buildLiveFrameCifFromCa,
  caSecondaryStructureCif,
  buildTrajectoryCifFromFrames,
  getSharedStudioStore,
  installStudioApiInterceptor,
} from './studio-api.js';
import { createLocalFoldProvider, refoldRecordForZip, refoldTermsStorage } from './refold-provider.js';
import { motifAtomsOf } from './motif-metrics.js';
import { webgl2Available } from '../webgl/rfd4p/gl.js';
import { Rfd4pModelGL } from '../webgl/rfd4p/model.js';
import { openGuide } from './guide-dialog.js';
import { criteriaKind, designedChains, motifSlots, partialTemplates, polymerChains } from './refold-view.js';

installStudioApiInterceptor();

/**
 * Extract per-token metadata (`chainId`, `resId`, `resName`, `atomName`, `element`, `isAtomized`)
 * from `batch` and `batch.__atoms` for building multi-model Cα trajectory CIFs.
 */
function extractTokenMetadata(batch) {
  const pad = batch.padding_mask.data;
  const n = batch.padding_mask.shape ? batch.padding_mask.shape.at(-1) : pad.length;
  const atoms = batch.__atoms ?? null;
  const rtData = batch.residue_type?.data;
  const tokTypeData = batch.token_type?.data;

  const firstAtomByToken = new Map();
  if (atoms?.token_id) {
    for (let i = 0; i < atoms.token_id.length; i += 1) {
      const tid = atoms.token_id[i];
      if (!firstAtomByToken.has(tid)) firstAtomByToken.set(tid, i);
    }
  }

  const tokens = new Array(n);
  for (let tid = 0; tid < n; tid += 1) {
    const a0 = firstAtomByToken.get(tid);
    const isAtomized =
      (a0 !== undefined && Boolean(atoms.atomize?.[a0])) ||
      (tokTypeData ? tokTypeData[tid] === 3 : false);
    const rt = rtData ? rtData[tid] : 1;
    const defaultResName = rt > 0 && rt < UNIFIED_ATOM37_TOKENS.length ? UNIFIED_ATOM37_TOKENS[rt] : 'ALA';
    const rawResName = a0 !== undefined ? atoms.res_name?.[a0] : defaultResName;
    const resName = !rawResName || rawResName === '<M>' || rawResName === 'UNK' ? defaultResName : rawResName;
    tokens[tid] = {
      tokenId: tid,
      chainId: (a0 !== undefined && atoms.chain_id?.[a0]) ? String(atoms.chain_id[a0]) : 'A',
      resId: (a0 !== undefined && Number.isFinite(Number(atoms.res_id?.[a0]))) ? Number(atoms.res_id[a0]) : tid + 1,
      resName,
      atomName: (a0 !== undefined && atoms.atom_name?.[a0] && atoms.atom_name[a0] !== 'UNK') ? atoms.atom_name[a0] : 'CA',
      element: (a0 !== undefined && atoms.element?.[a0] && atoms.element[a0] !== 'X') ? atoms.element[a0] : 'C',
      isAtomized,
    };
  }
  return tokens;
}

function extractFixedTokenAnchorsNm(batch, n) {
  const atoms = batch.__atoms ?? null;
  const crdArr = atoms?.condition_coordinate_1_atom;
  const hasIdxArr = atoms?.condition_index_1_residue;
  if (!atoms?.token_id || !Array.isArray(crdArr)) return null;

  const anchorByToken = new Map();
  for (let i = 0; i < atoms.token_id.length; i += 1) {
    if (hasIdxArr && !hasIdxArr[i]) continue;
    const c = crdArr[i];
    if (!Array.isArray(c) || !Number.isFinite(c[0]) || !Number.isFinite(c[1]) || !Number.isFinite(c[2])) continue;
    const tid = atoms.token_id[i];
    if (tid < 0 || tid >= n) continue;
    const prev = anchorByToken.get(tid);
    const aname = atoms.atom_name?.[i];
    if (!prev || aname === 'CA' || (aname === "C4'" && prev.atomName !== 'CA')) {
      anchorByToken.set(tid, { atomName: aname, xNm: c[0] / 10.0, yNm: c[1] / 10.0, zNm: c[2] / 10.0 });
    }
  }
  if (anchorByToken.size < 3) return null;
  // Conditioned coordinates are centred in the batch; frames are shown where the input was loaded.
  const [sx, sy, sz] = (atoms.center_shift ?? [0, 0, 0]).map((v) => v / 10.0);
  for (const info of anchorByToken.values()) { info.xNm += sx; info.yNm += sy; info.zNm += sz; }
  const tokenIndices = new Int32Array(anchorByToken.size);
  const fixedCoordsNm = new Float32Array(anchorByToken.size * 3);
  let k = 0;
  for (const [tid, info] of anchorByToken.entries()) {
    tokenIndices[k] = tid;
    fixedCoordsNm[k * 3 + 0] = info.xNm;
    fixedCoordsNm[k * 3 + 1] = info.yNm;
    fixedCoordsNm[k * 3 + 2] = info.zNm;
    k += 1;
  }
  return { tokenIndices, fixedCoordsNm };
}

/** Max distance (nm) of any Cα from the centroid. */
function caExtentRadiusNm(caNm) {
  const n = caNm.length / 3;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i += 1) {
    cx += caNm[i * 3]; cy += caNm[i * 3 + 1]; cz += caNm[i * 3 + 2];
  }
  cx /= n; cy /= n; cz /= n;
  let r2 = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = caNm[i * 3] - cx, dy = caNm[i * 3 + 1] - cy, dz = caNm[i * 3 + 2] - cz;
    r2 = Math.max(r2, dx * dx + dy * dy + dz * dz);
  }
  return Math.sqrt(r2);
}

/**
 * Rigidly superimpose output `atoms` (Å) onto per-token reference Cα coordinates (nm), in place.
 * Returns false (atoms untouched) when the atoms do not cover exactly `n` tokens.
 */
function alignStructureAtomsToCa(atoms, refCaNm, n) {
  const byToken = new Map();
  for (const a of atoms) {
    const tid = a.tokenId ?? 0;
    const prev = byToken.get(tid);
    if (!prev || a.atomName === 'CA' || (a.atomName === "C4'" && prev.atomName !== 'CA')) {
      byToken.set(tid, a);
    }
  }
  if (byToken.size !== n || n < 3) return false;
  const refCoordsA = Float32Array.from(refCaNm, (v) => v * 10.0);
  const mobileA = new Float32Array(n * 3);
  for (let k = 0; k < n; k += 1) {
    const a = byToken.get(k);
    mobileA[k * 3 + 0] = a?.x ?? 0;
    mobileA[k * 3 + 1] = a?.y ?? 0;
    mobileA[k * 3 + 2] = a?.z ?? 0;
  }
  const { rotation: R, fixedCenter, mobileCenter } = superimposeCoords3D(refCoordsA, mobileA);
  for (const a of atoms) {
    const mx = a.x - mobileCenter[0];
    const my = a.y - mobileCenter[1];
    const mz = a.z - mobileCenter[2];
    a.x = Math.fround(mx * R[0] + my * R[1] + mz * R[2] + fixedCenter[0]);
    a.y = Math.fround(mx * R[3] + my * R[4] + mz * R[5] + fixedCenter[1]);
    a.z = Math.fround(mx * R[6] + my * R[7] + mz * R[8] + fixedCenter[2]);
  }
  return true;
}

/**
 * Tokens with no input coordinates at all (unmodelled residues of a fetched structure). They cannot be held fixed, so
 * they do not make an otherwise fixed target chain count as "partly fixed" (a motif) for the overlays.
 */
function unmodelledTokens(inputAtoms) {
  const modelled = new Set();
  const all = new Set();
  for (let i = 0; i < (inputAtoms?.token_id?.length ?? 0); i += 1) {
    const tid = inputAtoms.token_id[i];
    all.add(tid);
    const c = inputAtoms.coord?.[i];
    if (c && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2])) modelled.add(tid);
  }
  return new Set([...all].filter((t) => !modelled.has(t)));
}

/**
 * The reference motif, as atom records at their conditioning coordinates (Å, input frame), for drawing beside each
 * live frame so a motif-scaffolding run keeps showing what it is scaffolding. A motif is a coordinate-conditioned
 * token that is unindexed, or indexed inside a chain that is otherwise designed; a chain whose residues are all
 * coordinate-conditioned is a target and is drawn by the frame itself. Placed on an unused chain id so it can never
 * share a residue key with the scaffold.
 */
/**
 * Whether a chain's held residues are a motif, worth showing against what the model builds: a few residues it scaffolds
 * (1QJG's three), not a kept domain the job extends (6MRR and 100 new residues), whose shadow would only double it.
 */
const isSmallMotif = (fixed, total) => fixed > 0 && fixed < total && fixed <= total / 4;

function buildMotifOverlayAtoms(inputAtoms) {
  if (!inputAtoms?.token_id || !inputAtoms.condition_coordinate_1_atom) return [];
  // Conditioned coordinates are centred in the batch; designs are shown in the input's frame, so the motif is too.
  const [sx, sy, sz] = inputAtoms.center_shift ?? [0, 0, 0];
  const crdOf = (i) => {
    const c = inputAtoms.condition_coordinate_1_atom[i];
    return Array.isArray(c) && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2])
      ? [c[0] + sx, c[1] + sy, c[2] + sz] : null;
  };
  const tokenChain = new Map();
  const tokenHasCrd = new Map();
  const tokenIndexed = new Map();
  const unmodelled = unmodelledTokens(inputAtoms);
  for (let i = 0; i < inputAtoms.token_id.length; i += 1) {
    const tid = inputAtoms.token_id[i];
    if (unmodelled.has(tid)) continue;
    tokenChain.set(tid, inputAtoms.chain_id[i]);
    if (crdOf(i)) tokenHasCrd.set(tid, true);
    if (!tokenIndexed.has(tid)) {
      tokenIndexed.set(tid, inputAtoms.condition_index_1_residue ? Boolean(inputAtoms.condition_index_1_residue[i]) : true);
    }
  }
  const chainTokens = new Map();
  for (const [tid, chain] of tokenChain) {
    const c = chainTokens.get(chain) ?? { total: 0, fixed: 0 };
    c.total += 1;
    if (tokenHasCrd.get(tid)) c.fixed += 1;
    chainTokens.set(chain, c);
  }
  // Residues placed anywhere always (an enzyme's catalytic tips); residues held in place only as a small motif.
  const isMotifToken = (tid) => tokenHasCrd.get(tid) && (!tokenIndexed.get(tid)
    || isSmallMotif(chainTokens.get(tokenChain.get(tid)).fixed, chainTokens.get(tokenChain.get(tid)).total));
  const used = new Set(tokenChain.values());
  const chainId = [...'MNOPQRSTUVWXYZ'].find((c) => !used.has(c)) ?? 'MOTIF';
  const out = [];
  for (let i = 0; i < inputAtoms.token_id.length; i += 1) {
    const crd = crdOf(i);
    const name = inputAtoms.atom_name[i];
    // Ligands are drawn by the frame itself (atomized tokens are always emitted).
    if (!crd || inputAtoms.hetero?.[i] || !isMotifToken(inputAtoms.token_id[i]) || !name || name === 'UNK') continue;
    out.push({
      groupPdb: 'HETATM', isHetero: true, atomName: name, resName: inputAtoms.res_name[i], chainId,
      resId: Number(inputAtoms.res_id[i]), authSeqId: Number(inputAtoms.res_id[i]), insCode: '',
      element: inputAtoms.element?.[i] || undefined, x: crd[0], y: crd[1], z: crd[2],
      occupancy: 1.0, bFactor: NaN, charge: 0, entityId: null,
    });
  }
  return out;
}

/**
 * Side chains (Cβ onward) of the design's own motif residues — an unindexed motif placed by the insertion mapping
 * (e.g. an enzyme's catalytic residues), or a coordinate-conditioned residue in a chain that is otherwise designed —
 * as copies drawn as sticks on top of the cartoon. Target chains (every residue fixed) are
 * left to the frame itself, as in `buildMotifOverlayAtoms`.
 */
function motifSideChainAtoms(structureAtoms, skipTokens = new Set()) {
  const hasCrd = (a) => Boolean(a.crd && Number.isFinite(a.crd[0]));
  const chainTokens = new Map();
  const tokenFixed = new Map();
  const motifTokens = new Set();
  for (const a of structureAtoms) {
    if (a.tokenId == null || a.isHetero || skipTokens.has(a.tokenId)) continue;
    let tokens = chainTokens.get(a.chainId);
    if (!tokens) chainTokens.set(a.chainId, (tokens = new Set()));
    tokens.add(a.tokenId);
    if (hasCrd(a)) tokenFixed.set(a.tokenId, true);
    if (a.isUnindexedInsertion) motifTokens.add(a.tokenId);
  }
  for (const tokens of chainTokens.values()) {
    const fixed = [...tokens].filter((t) => tokenFixed.get(t));
    if (isSmallMotif(fixed.length, tokens.size)) for (const t of fixed) motifTokens.add(t);
  }
  if (!motifTokens.size) return [];
  return structureAtoms
    // Cβ onward: with a Cα, Mol* draws the lone residue's backbone trace as a cone. Cβ sits against the cartoon tube.
    .filter((a) => motifTokens.has(a.tokenId) && !a.isHetero && !['N', 'CA', 'C', 'O', 'OXT'].includes(a.atomName))
    .map((a) => ({ ...a, groupPdb: 'HETATM', isHetero: true }));
}

/** A standalone `non-polymer` CIF for display-only atoms (null if there are none). */
function overlayCifOf(atoms, name) {
  if (!atoms.length) return null;
  const all = atoms.map((a, k) => ({ ...a, serial: k + 1, entityId: 1 }));
  return toCifString(all, { dataBlockName: name }) + entityCif(all, new Set([1]));
}

/**
 * Display-only overlays drawn beside a design, each as its own Mol* structure so none can be distance-bonded to the
 * design or to each other: the reference motif (drawn translucent) and the design's motif side chains.
 */
function designOverlays(structureAtoms, motifOverlay, skipTokens) {
  return {
    motif: overlayCifOf(motifOverlay, 'motif'),
    sideChains: overlayCifOf(motifSideChainAtoms(structureAtoms, skipTokens), 'motif_side_chains'),
  };
}

/** Shows `overlays` (from `designOverlays`) in the Studio viewer, clearing any that are absent. */
async function showDesignOverlays(overlays) {
  if (typeof window.atomworksStudio?.setOverlay !== 'function') return;
  // One ghost color (not element colors) so the reference never reads as extra bonds on the sampled side chain.
  await window.atomworksStudio?.setOverlay('motif', overlays?.motif ?? null, {
    label: 'Motif (reference)', alpha: 0.35, color: 0xe2e8f0, sizeAspectRatio: 0.6,
  });
  await window.atomworksStudio?.setOverlay('sideChains', overlays?.sideChains ?? null, { label: 'Motif side chains' });
  // A refold's own overlays (a magenta cartoon, its ligands) are Studio's Refold panel's, drawn when a design is shown.
}

/** `_entity` loop for `atoms`: the given entity ids are `non-polymer` (drawn as sticks), the rest as before. */
function entityCif(atoms, nonPolymerIds) {
  const types = new Map();
  for (const a of atoms) {
    const id = a.entityId ?? 0;
    const prev = types.get(id);
    const het = nonPolymerIds.has(id) || a.isHetero;
    types.set(id, prev === undefined ? het : prev && het);
  }
  return ['loop_', '_entity.id', '_entity.type',
    ...[...types].map(([id, het]) => `${id} ${het ? 'non-polymer' : 'polymer'}`), '#', ''].join('\n');
}

/** Whether output atoms carry conditioning coordinates (so processSampleToAtoms put them in the input frame). */
const isAnchored = (atoms) => atoms.some((a) => a.crd && Number.isFinite(a.crd[0]));

function alignCaNmToAnchors(caNm, tokenIndices, fixedCoordsNm) {
  const m = tokenIndices.length;
  const mobile = new Float32Array(m * 3);
  for (let k = 0; k < m; k += 1) {
    const tid = tokenIndices[k];
    mobile[k * 3 + 0] = caNm[tid * 3 + 0];
    mobile[k * 3 + 1] = caNm[tid * 3 + 1];
    mobile[k * 3 + 2] = caNm[tid * 3 + 2];
  }
  const { rotation: R, fixedCenter, mobileCenter } = superimposeCoords3D(fixedCoordsNm, mobile);
  const out = new Float32Array(caNm.length);
  const n = caNm.length / 3;
  for (let i = 0; i < n; i += 1) {
    const mx = caNm[i * 3 + 0] - mobileCenter[0];
    const my = caNm[i * 3 + 1] - mobileCenter[1];
    const mz = caNm[i * 3 + 2] - mobileCenter[2];
    out[i * 3 + 0] = Math.fround(mx * R[0] + my * R[1] + mz * R[2] + fixedCenter[0]);
    out[i * 3 + 1] = Math.fround(mx * R[3] + my * R[4] + mz * R[5] + fixedCenter[1]);
    out[i * 3 + 2] = Math.fround(mx * R[6] + my * R[7] + mz * R[8] + fixedCenter[2]);
  }
  return out;
}

const ONE_LETTER = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
  MSE: 'M', UNK: 'X', DA: 'A', DC: 'C', DG: 'G', DT: 'T', A: 'A', C: 'C', G: 'G', U: 'U',
};

/**
 * The designed sequence of each live frame, drawn in Studio's sequence dock while a design runs. It sits beside
 * Studio's own (interactive) `#sequence-view`, which is only hidden meanwhile, so finishing, stopping or failing
 * just removes the strip and Studio's view is back untouched. Residues that changed since the previous frame are
 * highlighted, which shows the sequence settling.
 */
class LiveSequenceStrip {
  constructor() {
    this.dock = document.getElementById('sequence-dock');
    this.view = document.getElementById('sequence-view');
    this.el = null;
    this.tokens = new Map();
  }

  /** @param {Set<string>} [hotspots] `chain|resId|resName` keys (`hotspotResidueKeys`), marked as in Studio's view */
  start(hotspots = new Set()) {
    this.hotspots = hotspots;
    if (!this.view || this.el) return;
    document.getElementById('rfd4p-live-sequence')?.remove();
    this.el = document.createElement('div');
    this.el.id = 'rfd4p-live-sequence';
    this.el.className = 'sequence-view';
    this.el.setAttribute('aria-live', 'off');
    this.view.after(this.el);
    // On the dock, not the view: Studio's sequence render resets the view's className.
    this.dock?.classList.add('rfd4p-live');
    this.tokens.clear();
  }

  /** @param {Array<{chainId: string, resId: number, resName: string}>} atoms */
  update(atoms) {
    if (!this.el) return;
    const residues = new Map();
    for (const a of atoms) {
      const code = ONE_LETTER[a.resName];
      const key = `${a.chainId}:${a.resId}`;
      if (code && !residues.has(key)) residues.set(key, { chainId: a.chainId, resId: a.resId, resName: a.resName, code });
    }
    if (this.tokens.size !== residues.size || [...residues.keys()].some((k) => !this.tokens.has(k))) {
      this.build(residues);
    } else {
      for (const [key, r] of residues) {
        const tok = this.tokens.get(key);
        const diff = tok.textContent !== r.code;
        if (diff) {
          tok.textContent = r.code;
          tok.title = `${r.chainId}/${r.resName} ${r.resId}`;
        }
        tok.classList.toggle('rfd4p-changed', diff);
      }
    }
  }

  build(residues) {
    this.el.replaceChildren();
    this.tokens.clear();
    const tracks = new Map();
    for (const [key, r] of residues) {
      let track = tracks.get(r.chainId);
      if (!track) {
        const chain = document.createElement('div');
        chain.className = 'sequence-chain';
        const color = window.atomworksStudio?.chainColor?.(r.chainId);
        if (color) chain.style.setProperty('--chain-color', color);
        const label = document.createElement('span');
        label.className = 'sequence-chain-label';
        label.textContent = r.chainId;
        track = document.createElement('div');
        track.className = 'sequence-track';
        chain.append(label, track);
        this.el.appendChild(chain);
        tracks.set(r.chainId, track);
      }
      const tok = document.createElement('span');
      tok.className = 'sequence-token';
      if (this.hotspots?.has(`${r.chainId}|${r.resId}|${r.resName}`)) tok.classList.add('condition-hotspot');
      tok.textContent = r.code;
      tok.title = `${r.chainId}/${r.resName} ${r.resId}`;
      track.appendChild(tok);
      this.tokens.set(key, tok);
    }
  }

  stop() {
    if (!this.el) return;
    this.el.remove();
    this.el = null;
    this.tokens.clear();
    this.dock?.classList.remove('rfd4p-live');
  }
}

/** The hotspot residues of a Studio document, as `chain|resId|resName` keys. */
function hotspotResidueKeys(atoms) {
  const keys = new Set();
  // A Studio document keeps it under `annotations`; a built batch's atoms also carry it flat.
  const hot = atoms?.annotations?.condition_hotspot_1_atom ?? atoms?.condition_hotspot_1_atom;
  if (!hot) return keys;
  const n = atoms.nAtoms ?? atoms.chain_id.length;
  for (let i = 0; i < n; i += 1) {
    if (hot[i]) keys.add(`${atoms.chain_id[i]}|${atoms.res_id[i]}|${atoms.res_name[i]}`);
  }
  return keys;
}

const PRESET_TO_TASK = {
  uncond_short: 'unconditional',
  ppi_pdl1: 'unconditional',
  motif_idx_1prw: 'unconditional',
  motif_unidx_1prw: 'unconditional',
  sm_sam: 'unconditional',
  dna_7m5w: 'unconditional',
  yaml_5o45_binder: 'unconditional',
  partial_diffusion_1prw: 'partial_diffusion',
  inverse_folding_1prw: 'inverse_folding',
  custom_cath_monomer: 'unconditional',
};

const NO_WEBGPU_MESSAGE = 'This browser has no WebGPU, which designs need. Use Chrome or Edge (version 113 or later) '
  + 'or Safari 26 or later. You can still open and edit structures here.';

/** Hooks that report nowhere: a run made without the card (a GPU gate calling `runDesign`). */
const NO_HOOKS = {
  status() {}, progress() {}, loading() {}, async frame() {}, async design() { return {}; },
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** The ghost the reference motif is drawn in: one colour, so it never reads as extra bonds on the sampled side chain. */
const MOTIF_OVERLAY = { label: 'Motif (reference)', alpha: 0.35, color: 0xe2e8f0, sizeAspectRatio: 0.6 };
const SIDE_CHAIN_OVERLAY = { label: 'Motif side chains' };

/** `designOverlays` as the card draws them beside a frame or a design (`FrameOptions.overlays`). */
const cardOverlays = (overlays) => ({
  motif: { cif: overlays?.motif ?? null, options: MOTIF_OVERLAY },
  sideChains: { cif: overlays?.sideChains ?? null, options: SIDE_CHAIN_OVERLAY },
});

/** The residues of a hotspot key set, as the viewer names a live frame's (res_id, past any new residues). */
const hotspotsOf = (keys) => [...keys].map((key) => {
  const [chain, residue] = key.split('|');
  return { chain, residue: Number(residue) };
});

const insCodeOf = (atoms, i) => { const c = String(atoms.ins_code?.[i] ?? ''); return c === '.' || c === '?' ? '' : c; };

/**
 * `keys` (`hotspotResidueKeys` of the input) as the viewer names them on a finished design: its CIF's auth_seq_id and
 * insertion code. A design keeps a kept chain's letter and numbering; residues that do not match (chain, number and
 * residue) are left alone.
 */
function designHotspots(cif, keys) {
  if (!keys?.size) return [];
  const a = new StudioDoc(cif, 'design.cif').atoms;
  const found = new Map();
  for (let i = 0; i < a.nAtoms; i += 1) {
    if (!keys.has(`${a.chain_id[i]}|${a.res_id[i]}|${a.res_name[i]}`)) continue;
    const r = { chain: String(a.chain_id[i]), residue: Number(a.auth_seq_id?.[i] ?? a.res_id[i]), insCode: insCodeOf(a, i) };
    found.set(`${r.chain}|${r.residue}|${r.insCode}`, r);
  }
  return [...found.values()];
}

/** Settings of the design card, as this engine takes them (the card's own defaults, written out). */
const CARD_SETTINGS = {
  samples: { min: 1, max: 8, default: 1 },
  // None by default, for speed (about 3x faster than upstream's medium, which its `inference.yaml` and binder cookbook use).
  thinking: { options: ['none', 'low', 'medium', 'high'], default: 'none' },
  seed: true,
  partialDiffusion: { percent: 40 },
  backboneK: { min: 1, max: 20, step: 0.1, default: 16 },
  latentK: { min: 1, max: 6, step: 0.1, default: 4 },
  earlyStop: { default: true },
};

class Rfd4pStudioApp {
  constructor() {
    this.gpuDevice = null;
    this.gpuAdapterInfo = null;
    this.activeConfig = null;
    this.activeWeights = null;
    this.gpuModel = null;
    this.gpuModelHalf = null;
    this.shardedBundle = null;
    this.lastResult = null;
    this.runCounter = 0;
    this.prevArchChoice = 'int8-sharded';
    this.presetBytesCache = new Map();
    this.currentPreset = null;
    this.currentCifBytes = null;
    this.currentCifText = '';
    this.currentBatchResult = null;
    // Steps of a design: the model's own 60. A gate that wants a short run sets it (the card has no control for it).
    this.nsteps = 60;
    // Whether a run is under way (the card never calls `generate` while one is).
    this.running = false;
    // The Design card (Studio's) once mounted; `ready` resolves then.
    this.card = null;
    this.ready = new Promise((resolve) => { this._markReady = resolve; });
    // Designs of the input a run is on are numbered on while it is unchanged, as the card lists them.
    this.designKey = null;
    this.designCount = 0;
  }

  async init() {
    await this._initWebGpu();
    const params = new URLSearchParams(window.location.search);
    // `?gl=1`: the WebGL2 path even where WebGPU is available (for testing it).
    if (params.get('gl') === '1' && this.gpuDevice) {
      this.gpuDevice.destroy?.();
      this.gpuDevice = null;
      this.gpuModel = null;
    }
    // Without WebGPU (Safari before 26, older browsers) designs run on WebGL2 (`webgl/rfd4p`), several times slower but
    // on the GPU. Without that too, the CPU path would take hours with the page frozen: say so instead; `?cpu=1`
    // keeps the CPU path for development. Refolding (LocalFold) runs on either: its WebGL2 fold without WebGPU.
    const cpuOnly = params.get('cpu') === '1';
    this.glFallback = !this.gpuDevice && !cpuOnly && webgl2Available();
    this.noWebGpu = !this.gpuDevice && !this.glFallback && !cpuOnly;
    this.noRefold = !this.gpuDevice && !this.glFallback;
    // AtomWorks Studio runs in this same document; its page API appears once the editor has started.
    for (let i = 0; i < 1200 && typeof window.atomworksStudio?.mountDesignCard !== 'function'; i += 1) await sleep(50);
    if (typeof window.atomworksStudio?.mountDesignCard !== 'function') throw new Error('AtomWorks Studio did not start');
    this.card = window.atomworksStudio.mountDesignCard(null, this._provider());
    this._markReady();
  }

  get backendLabel() {
    return this.gpuDevice ? 'WebGPU FP16' : this.glFallback ? 'WebGL2' : this.noWebGpu ? 'No WebGPU' : 'CPU Float32';
  }

  /**
   * What this page gives Studio's Design card (docs/INTEGRATIONS.md, "Design card" in atomworks-studio): the engine.
   * Everything a person sees and decides is the card's.
   */
  _provider() {
    return {
      engine: { badge: this.backendLabel, disabled: this.noWebGpu ? NO_WEBGPU_MESSAGE : null },
      settings: CARD_SETTINGS,
      generate: (request, hooks) => this.generate(request, hooks),
      // LocalFold, in the browser (web/refold-provider.js); the card hosts Studio's Refold panel with it.
      refold: createLocalFoldProvider(this),
      refoldPanel: {
        // Each design is named by its seed: the same seed and settings make it again.
        nameHeader: { label: 'Seed', title: 'Each design is named by its seed: the same seed and settings make it again' },
        describe: (design) => `seed ${design.name}`,
        // AlphaFold 3's terms were agreed to under this page's own key before they moved into Studio.
        storage: refoldTermsStorage,
      },
      renderExtras: (container) => this._renderSpeedHint(container),
      guide: () => openGuide(),
      zipTitle: 'RFD4-Proteina (WebGPU) results',
      // The refold's own record (its RMSD of the designed chain fitted on itself, its templates) is what the zip keeps.
      zipRefold: (record) => refoldRecordForZip(record),
      // A job is open: have the model ready before Generate, if its weights are already cached (nothing is downloaded).
      inputChanged: () => this.preloadWeights(),
    };
  }

  /**
   * Desktop Chrome exposes subgroup matrices (about 1.4x faster designs on an M2, `hasSubgroupMatrix` in the engine) only behind
   * chrome://flags/#enable-unsafe-webgpu. Pages cannot open chrome:// URLs, so the hint offers to copy the address.
   * It sits beside Generate (the card's `renderExtras`), where the wait is.
   */
  _renderSpeedHint(container) {
    container.insertAdjacentHTML('beforeend', `
      <div id="rfd4p-speed-hint" class="rfd4p-speed-hint" role="note" hidden>
        <p><strong>Up to 1.5× faster designs:</strong> turn on Chrome's experimental WebGPU. Paste
          <code>chrome://flags/#enable-unsafe-webgpu</code> into the address bar, set it to <em>Enabled</em> and
          relaunch Chrome.</p>
        <div class="rfd4p-speed-hint-actions">
          <button id="rfd4p-speed-copy" type="button" class="secondary compact">Copy address</button>
          <button id="rfd4p-speed-dismiss" type="button" class="secondary compact">Not now</button>
        </div>
      </div>`);
    const hint = container.querySelector('#rfd4p-speed-hint');
    if (!hint || !this.gpuDevice) return undefined;
    const KEY = 'rfd4p.speedHintDismissed';
    let dismissed = false;
    try { dismissed = localStorage.getItem(KEY) === '1'; } catch { /* storage blocked: show it */ }
    const brands = navigator.userAgentData?.brands?.map((b) => b.brand) ?? [];
    const desktopChrome = brands.includes('Google Chrome') && navigator.userAgentData?.mobile === false;
    const hasMatrix = this.gpuDevice.features?.has?.('chromium-experimental-subgroup-matrix') === true;
    if (dismissed || !desktopChrome || hasMatrix) return undefined;
    hint.hidden = false;
    // Said again after each design, until dismissed.
    this.speedHintOn = true;
    const copy = container.querySelector('#rfd4p-speed-copy');
    copy?.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText('chrome://flags/#enable-unsafe-webgpu');
        copy.textContent = 'Copied';
      } catch {
        copy.textContent = 'Copy failed: select the address above';
      }
    });
    container.querySelector('#rfd4p-speed-dismiss')?.addEventListener('click', () => {
      hint.hidden = true;
      this.speedHintOn = false;
      try { localStorage.setItem(KEY, '1'); } catch { /* not remembered; fine */ }
    });
    return undefined;
  }

  async _initWebGpu() {
    try {
      const res = await requestWebGpuDevice();
      // The model runs in float16 (`enable f16;` in its kernels), so a device without shader-f16 takes the WebGL2
      // path, as a browser without WebGPU does. Subgroups are optional: without them (Safari on an iPad) the engine
      // reduces in workgroup memory (`subgroupMode` 'shared', `none`).
      if (res?.device && !res.device.features?.has?.('shader-f16')) {
        console.warn('WebGPU device lacks shader-f16; designs run on WebGL2');
        res.device.destroy?.();
      } else if (res && res.device) {
        this.gpuDevice = res.device;
        this.gpuAdapterInfo = res.adapter?.info ?? null;
        this._watchDevice(res.device);
      }
    } catch (err) {
      console.warn('WebGPU fallback to CPU:', err);
    }
  }

  /**
   * A lost device (GPU reset, driver watchdog, memory pressure) takes every buffer and pipeline with it. Record why
   * and drop the device and model; the next Generate requests a new device and rebuilds the model from the weights
   * already in memory (`ensureWeightsLoaded`), instead of failing until the page is reloaded.
   */
  _watchDevice(device) {
    this._deviceLost = device.lost.then((info) => {
      if (this.gpuDevice !== device) return;
      this.gpuDeviceLost = { reason: info.reason, message: info.message, at: new Date().toISOString() };
      console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
      this.gpuDevice = null;
      this.gpuModel = null;
    });
  }

  /**
   * Load the model ahead of the first Generate, once a job is open and nothing runs, when every shard it needs is
   * already in the browser's cache: the first design then starts ~1.5 s sooner (n = 223, M2), and nothing is
   * downloaded that Generate was not asked for (a metered connection pays nothing). A Generate meanwhile waits for it.
   */
  preloadWeights() {
    if (this.activeWeights || this.preloadPromise || this.running || this.prevArchChoice !== 'int8-sharded') return;
    this.preloadPromise = (async () => {
      if (!(await shardedWeightsCached(DEFAULT_LOCAL_SHARDED_DIR))) return;
      const sharded = await fetchShardedWeights(DEFAULT_LOCAL_SHARDED_DIR, { includeEncoder: false });
      const cfg = sharded.config ?? (await fetchRfd4pConfig(DEFAULT_LOCAL_CONFIG_URL));
      if (this.activeWeights) return;
      this.shardedBundle = sharded;
      this.activeConfig = cfg.config ?? cfg;
      this.activeWeights = sharded.tensors;
      if (this.gpuDevice && !this.gpuModel) {
        // A layer at a time, so the page keeps responding while it is built (a job just opened is being edited).
        const model = await Rfd4pGeneratorGpu.create(this.gpuDevice, this.activeWeights, this.activeConfig, { half: true });
        if (this.gpuModel) model.destroy();
        else { this.gpuModel = model; this.gpuModelHalf = true; }
      }
    })().catch((err) => console.warn('Model preload skipped:', err)).finally(() => { this.preloadPromise = null; });
  }

  /** @param {object} [hooks] the run's `hooks` (loading, progress, status): the weights' progress is the run's */
  async ensureWeightsLoaded(source = this.prevArchChoice, { needEncoder = false, hooks = NO_HOOKS } = {}) {
    // A preload already under way finishes first, rather than loading the same shards twice.
    if (this.preloadPromise) await this.preloadPromise;
    const half = true;
    const makeProgressCb = (label) => ({ bytesLoaded, totalBytes, file }) => {
      const mb = (bytesLoaded / (1024 * 1024)).toFixed(1);
      const tag = file ? `${label} (${file})` : label;
      if (totalBytes > 0) {
        const totMb = (totalBytes / (1024 * 1024)).toFixed(1);
        hooks.loading(`${tag} · ${mb} / ${totMb} MiB`);
        hooks.progress(Math.min(0.95, bytesLoaded / totalBytes));
      } else {
        hooks.loading(`${tag} · ${mb} MiB`);
      }
    };

    if (this.activeWeights && this.activeConfig) {
      if (needEncoder && this.shardedBundle && !this.shardedBundle.encoderLoaded) {
        try {
          hooks.loading('Loading encoder shards');
          hooks.progress(0.01);
          hooks.status('Loading RFD4-Proteina encoder shards (131 MiB)...');
          await this.shardedBundle.loadEncoder(makeProgressCb('weights/int8'));
          hooks.progress(0);
        } finally {
          hooks.loading(null);
        }
      }
      if (this.gpuDevice && (!this.gpuModel || this.gpuModelHalf !== half)) {
        if (this.gpuModel) this.gpuModel.destroy();
        this.gpuModel = new Rfd4pGeneratorGpu(this.gpuDevice, this.activeWeights, this.activeConfig, { half });
        this.gpuModelHalf = half;
      }
      return;
    }

    try {
      hooks.loading('Loading weights');
      hooks.progress(0.01);

      if (source === 'int8-sharded') {
        try {
          hooks.status('Loading sharded INT8 RFD4-Proteina weights (block-64)...');
          const sharded = await fetchShardedWeights(DEFAULT_LOCAL_SHARDED_DIR, {
            includeEncoder: needEncoder,
            onProgress: makeProgressCb('weights/int8'),
          });
          const cfg = sharded.config ?? (await fetchRfd4pConfig(DEFAULT_LOCAL_CONFIG_URL));
          this.shardedBundle = sharded;
          this.activeConfig = cfg.config ?? cfg;
          this.activeWeights = sharded.tensors;
          if (this.gpuDevice) {
            if (this.gpuModel) this.gpuModel.destroy();
            this.gpuModel = new Rfd4pGeneratorGpu(this.gpuDevice, this.activeWeights, this.activeConfig, { half });
            this.gpuModelHalf = half;
          }
          hooks.progress(0);
          return;
        } catch (shardedErr) {
          console.warn('Sharded INT8 weights unavailable, falling back to model.safetensors:', shardedErr);
        }
      }

      const useHf = source === 'hf';
      const configUrl = useHf ? DEFAULT_HF_CONFIG_URL : DEFAULT_LOCAL_CONFIG_URL;
      const weightsUrl = useHf ? DEFAULT_HF_WEIGHTS_URL : DEFAULT_LOCAL_WEIGHTS_URL;

      hooks.status('Loading RFD4-Proteina configuration and 1,257-tensor checkpoint...');

      let cfg;
      try {
        cfg = await fetchRfd4pConfig(configUrl);
      } catch {
        cfg = await fetchRfd4pConfig(DEFAULT_HF_CONFIG_URL);
      }

      let loaded;
      const onProgress = makeProgressCb('model.safetensors');
      try {
        loaded = await fetchSafetensors(weightsUrl, { onProgress });
      } catch (err) {
        if (!useHf) {
          loaded = await fetchSafetensors(DEFAULT_HF_WEIGHTS_URL, { onProgress });
        } else {
          throw err;
        }
      }

      this.shardedBundle = null;
      this.activeConfig = cfg.config ?? cfg;
      this.activeWeights = loaded.tensors ?? loaded;
      if (this.gpuDevice) {
        if (this.gpuModel) this.gpuModel.destroy();
        this.gpuModel = new Rfd4pGeneratorGpu(this.gpuDevice, this.activeWeights, this.activeConfig, { half });
        this.gpuModelHalf = half;
      }
      hooks.progress(0);
    } finally {
      hooks.loading(null);
    }
  }

  /** A bundled benchmark preset as the input of `runDesign` (the GPU gates; the card opens structures through Studio). */
  async _loadPreset(presetId) {
    const preset = getRfd4pPresetById(presetId);
    this.currentPreset = preset;
    this.currentCifBytes = null;
    this.currentCifText = '';
    this.presetPartial = PRESET_TO_TASK[preset.id] === 'partial_diffusion'
      ? (preset.partialTrajectoryFractionSteps ?? 0.4) : null;

    if (preset.cifUrl) {
      try {
        let bytes = this.presetBytesCache.get(preset.cifUrl);
        if (!bytes) {
          const resp = await fetch(preset.cifUrl);
          if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${preset.cifUrl}`);
          bytes = new Uint8Array(await resp.arrayBuffer());
          this.presetBytesCache.set(preset.cifUrl, bytes);
        }
        this.currentCifBytes = bytes;
        this.currentCifText = decodeTextOrZstd(bytes);
      } catch (err) {
        console.warn('Failed to fetch preset CIF:', err);
        this.currentCifText = '';
      }
    }

    this._buildCurrentBatch();
  }

  /** The batch of the input: the card's annotated CIF, or (a gate's run) the loaded preset. */
  _buildCurrentBatch(seed, cif = null) {
    if (cif) {
      const res = buildBatch(cif, { seed });
      this.currentBatchResult = res;
      return res;
    }
    const preset = this.currentPreset;
    if (preset && !preset.specMode && this.currentCifBytes) {
      const res = buildBatch(this.currentCifBytes, { seed });
      this.currentBatchResult = res;
      return res;
    }

    const res = buildBatch(
      { contig: preset?.contig || '100-100P', conditions: preset?.conditions ?? {} },
      { inputCif: this.currentCifText || null, seed },
    );
    this.currentBatchResult = res;
    return res;
  }

  /**
   * A run without the card's Generate, from the loaded preset (`_loadPreset`): what the GPU gates call. `overrides`:
   * `nsteps`, `recycleSchedule`, `seed`, `samples`, `earlyStop`. It reports as the card would: the status line
   * (`#rfd4p-status-message`), live frames in the viewer, each design opened as a session of Studio's.
   */
  async runDesign(overrides = {}) {
    const preset = this.currentPreset;
    const settings = {
      samples: overrides.samples ?? 1,
      thinking: overrides.recycleSchedule ?? preset?.recycleSchedule ?? 'none',
      seed: overrides.seed ?? 42,
      partialDiffusion: this.presetPartial != null ? Math.round(this.presetPartial * 100) : null,
      backboneK: 16,
      latentK: 4,
      earlyStop: overrides.earlyStop ?? true,
    };
    const app = this;
    const studio = window.atomworksStudio;
    const status = (text, o = {}) => this.card?.setStatus(text, { error: Boolean(o.error), visible: o.visible ?? true });
    const hooks = {
      ...NO_HOOKS,
      status,
      async frame(cif, o = {}) { await studio?.renderFrame?.(cif, o.name ?? 'frame.cif', { keepCamera: o.keepCamera }); },
      async design(info) {
        app.lastDesign = info;
        const { sessionId } = getSharedStudioStore().createSession(info.cif, `${info.filenameStem ?? 'design'}.cif`);
        await studio?.openSession?.(sessionId);
        return { id: sessionId };
      },
    };
    const request = {
      sessionId: null, cif: null, name: preset?.id ?? 'preset', revision: 1, spec: null, settings, signal: new AbortController().signal,
    };
    const steps = this.nsteps;
    if (overrides.nsteps != null) this.nsteps = overrides.nsteps;
    studio?.setLiveMode?.(true);
    try {
      const result = await this.generate(request, hooks);
      if (result?.message) status(result.message);
    } catch (err) {
      status(err.userMessage ?? `Generation failed. ${err.message} Your input is unchanged.`, { error: true });
    } finally {
      studio?.setLiveMode?.(false);
      this.nsteps = steps;
    }
  }

  /**
   * The card's `generate`: make `request.settings.samples` designs of `request.cif`, live frames and designs through
   * `hooks`. A failure the person should read in other words carries `userMessage`.
   */
  async generate(request, hooks = NO_HOOKS) {
    const liveSequence = new LiveSequenceStrip();
    this.running = true;
    try {
      return await this._generate(request, hooks, liveSequence);
    } catch (err) {
      if (request.signal?.aborted || String(err?.message || '').includes('aborted')) throw err;
      // A failed readback can surface before device.lost settles; give it a moment so a lost device is
      // recognised (and dropped) now rather than failing the next Generate too.
      if (this.gpuDevice && this._deviceLost) {
        await Promise.race([this._deviceLost, new Promise((r) => setTimeout(r, 300))]);
      }
      if (!this.gpuDevice && this.gpuDeviceLost) {
        throw Object.assign(new Error(err.message), {
          userMessage: `The GPU was reset during generation (${this.gpuDeviceLost.reason}: ${this.gpuDeviceLost.message || 'no details'}). Press Generate to retry; WebGPU restarts automatically. Your input is unchanged.`,
        });
      }
      throw err;
    } finally {
      liveSequence.stop();
      this.running = false;
    }
  }

  async _generate(request, hooks, liveSequence) {
    if (this.noWebGpu) throw Object.assign(new Error(NO_WEBGPU_MESSAGE), { userMessage: NO_WEBGPU_MESSAGE });
    const { signal } = request;
    const s = request.settings ?? {};
    const baseSeed = Number.isFinite(s.seed) && s.seed >= 0 ? s.seed : 42;
    const numSamples = clamp(Math.trunc(s.samples ?? 1) || 1, 1, 8);
    const nsteps = this.nsteps;
    const recycleSchedule = s.thinking ?? 'none';
    // Sampling temperature as upstream's Studio provider exposes it: TSR K per modality (lower = hotter).
    const backboneK = clamp(s.backboneK ?? 16, 1, 20);
    const latentK = clamp(s.latentK ?? 4, 1, 6);
    // What to design comes from the input's annotations; partial diffusion is the one option on top. A backbone kept
    // with its sequence redesigned runs the inverse-folding sampler, as upstream's Space does for that task
    // (`generation=inverse_folding`): the backbone is kept exactly, only sequence and side chains are sampled.
    const partialTrajectoryFractionSteps = s.partialDiffusion != null ? clamp(s.partialDiffusion, 1, 100) / 100 : null;
    const taskKey = request.spec?.task?.key ?? null;
    const inverseFolding = partialTrajectoryFractionSteps == null
      && (request.spec ? taskKey === 'inverse' : Boolean(this.currentPreset?.inverseFolding));

    const firstBatchRes = this._buildCurrentBatch(baseSeed, request.cif);
    const needEncoder = Boolean(inverseFolding || partialTrajectoryFractionSteps != null);
    if (needEncoder) {
      const has3dCoords = Boolean(
        firstBatchRes?.atoms?.coord?.some(
          (c) => Array.isArray(c) && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2]),
        ),
      );
      if (!has3dCoords) {
        throw new Error(
          `${inverseFolding ? 'Inverse folding' : 'Partial diffusion'} requires an input structure with 3D coordinates (open a structure or select a 3D example in Examples).`,
        );
      }
    }

    if (!this.gpuDevice && this.gpuDeviceLost) {
      // The previous device was lost: get a new one; ensureWeightsLoaded then rebuilds the model on it.
      hooks.status('Restarting WebGPU after a GPU reset...');
      await this._initWebGpu();
    }
    await this.ensureWeightsLoaded(this.prevArchChoice, { needEncoder, hooks });

    const useGpu = Boolean(this.gpuModel);
    // The WebGL2 model holds the weights it has used on the GPU: made again only when the weights change.
    if (!useGpu && this.glFallback && this.glModelWeights !== this.activeWeights) {
      this.glModel?.destroy();
      this.glModel = new Rfd4pModelGL(this.activeWeights);
      this.glModelWeights = this.activeWeights;
    }
    const backendLabel = useGpu ? 'WebGPU FP16' : this.glFallback ? 'WebGL2' : 'CPU Float32';

    const customSampling = {
      ...DEFAULT_SAMPLING,
      nsteps,
      bb_ca: {
        ...DEFAULT_SAMPLING.bb_ca,
        step: { ...DEFAULT_SAMPLING.bb_ca.step, tsr_k: backboneK },
      },
      local_latents: {
        ...DEFAULT_SAMPLING.local_latents,
        step: { ...DEFAULT_SAMPLING.local_latents.step, tsr_k: latentK },
      },
    };

    // The card lists designs of an input that has not changed after the earlier ones, numbered on.
    const key = `${request.sessionId}@${request.revision}`;
    if (this.designKey !== key) { this.designKey = key; this.designCount = 0; }

    const earlyStops = [];
    const t0 = performance.now();

    for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx += 1) {
      const sampleSeed = baseSeed + sampleIdx;
      const { batch, n, atoms } = sampleIdx === 0 ? firstBatchRes : this._buildCurrentBatch(sampleSeed, request.cif);
      // Hotspots stay marked while the design runs and on every design it makes. Live frames number residues as the job
      // does (res_id, past any new residues), not as the input is shown, so each sample's keys come from its own batch.
      const hotspotKeys = hotspotResidueKeys(atoms);
      liveSequence.start(hotspotKeys);
      const liveHotspots = hotspotsOf(hotspotKeys);
      const tokens = extractTokenMetadata(batch);
      const fixedAnchors = extractFixedTokenAnchorsNm(batch, n);
      const allTokenIndices = Int32Array.from({ length: n }, (_, idx) => idx);
      const trajectoryFrames = [];
      const sampleTag = numSamples > 1 ? ` [Design ${sampleIdx + 1}/${numSamples}]` : '';

      hooks.progress((sampleIdx / numSamples) + 0.02);
      // The button says how far the run is; the status line is for what is not progress.
      hooks.status(`Generating design${sampleTag} (${n} tokens, ${nsteps} steps, ${backendLabel})...`, { visible: false });

      let referenceCaNm = null;
      let liveFramesRendered = 0;
      let liveCamRadius = 0;

      // Live frames use the decoder's all-atom x1 (real backbone geometry, so Mol* can assign secondary
      // structure, and in-context targets keep their input atoms); CA-only frames are the fallback.
      // Every step is shown: each one is decoded (the last is replaced by the final structure), and a step waits
      // for the previous frame's Mol* update rather than dropping its own frame.
      const wantLiveFrame = (step, total) => step < total - 1;
      const motifOverlay = buildMotifOverlayAtoms(atoms);
      const unmodelled = unmodelledTokens(atoms);
      // Overlays for the frame being built; shown once its structure has been drawn.
      let liveOverlays = null;
      const hasUnindexedMotif = Boolean(atoms.condition_index_1_residue?.some((v, i) =>
        !v && !atoms.hetero?.[i] && atoms.condition_coordinate_1_atom?.[i]?.some?.(Number.isFinite)));
      const liveAllAtomCif = (info, frame = null) => {
        if (!info.decoded) return null;
        try {
          const structure = processSampleToAtoms({
            coors: info.decoded.coors,
            atomMask: info.decoded.atomMask,
            residueType: info.decoded.residueType,
            paddingMask: batch.padding_mask?.data ?? null,
            inputAtoms: atoms,
            batch,
            insertionLogits: info.insertionLogits ?? null,
            inputFrame: true,
          });
          // processSampleToAtoms already aligns coordinate-conditioned designs onto the input (motif) frame, and with
          // `inputFrame` one held only by a rigid target onto that target as loaded.
          if (!fixedAnchors && !structure.inputFrame && !isAnchored(structure.atoms) && referenceCaNm) {
            alignStructureAtomsToCa(structure.atoms, referenceCaNm, n);
          }
          liveSequence.update(structure.atoms);
          // Kept for the saved trajectory: each frame as drawn live.
          if (frame) frame.atoms = structure.atoms;
          liveOverlays = designOverlays(structure.atoms, motifOverlay, unmodelled);
          return toCifString(structure) + caSecondaryStructureCif(structure.atoms);
        } catch (err) {
          console.warn('Live all-atom frame failed; showing Cα frame:', err);
          return null;
        }
      };

      const onStep = async (info) => {
        let alignedCaNm = Float32Array.from(info.x1.bb_ca);
        if (fixedAnchors) {
          alignedCaNm = alignCaNmToAnchors(alignedCaNm, fixedAnchors.tokenIndices, fixedAnchors.fixedCoordsNm);
        } else if (n >= 3) {
          if (!referenceCaNm) {
            referenceCaNm = Float32Array.from(alignedCaNm);
          } else {
            alignedCaNm = alignCaNmToAnchors(alignedCaNm, allTokenIndices, referenceCaNm);
          }
        }

        trajectoryFrames.push({
          step: info.step,
          tBb: info.t.bb_ca,
          tLl: info.t.local_latents,
          depth: info.depth,
          caCoordsNm: alignedCaNm,
          // The sequence the model had at this step, so the saved trajectory shows it settling.
          residueType: info.decoded?.residueType ?? null,
        });

        const isStopping = info.step > 0 && (info.step === info.nsteps - 1 || Boolean(info.convergence?.stop));
        // On the GPU path only decoded frames are shown; the CA-only frame is for the CPU backend (no live decode).
        const frameReady = Boolean(info.decoded) || !useGpu;
        if (frameReady && !isStopping) {
          liveOverlays = { motif: overlayCifOf(motifOverlay, 'motif'), sideChains: null };
          const liveCif = liveAllAtomCif(info, trajectoryFrames.at(-1)) ?? buildLiveFrameCifFromCa(alignedCaNm, tokens, null, UNIFIED_ATOM37_TOKENS);
          if (liveCif) {
            // Early x1 estimates are a collapsed blob that later expands several-fold, so a camera
            // fit once to step 1 ends up inside the protein; refit whenever the extent drifts.
            const radius = caExtentRadiusNm(alignedCaNm);
            const keepCamera = liveFramesRendered > 0 && radius < liveCamRadius * 1.3 && radius > liveCamRadius * 0.6;
            if (!keepCamera) liveCamRadius = radius;
            liveFramesRendered += 1;
            // The card waits for the frame before this one to be drawn, so no step's frame is dropped.
            await hooks.frame(liveCif, {
              name: `step_${info.step + 1}.cif`, keepCamera, hotspots: liveHotspots, overlays: cardOverlays(liveOverlays),
            });
          }
        }

        const stepFrac = (info.step + 1) / (info.nsteps + 1);
        hooks.progress((sampleIdx + stepFrac) / numSamples,
          `${numSamples > 1 ? `design ${sampleIdx + 1} of ${numSamples} · ` : ''}step ${info.step + 1}/${info.nsteps}`);
        hooks.status(
          `Generating design${sampleTag} · step ${info.step + 1}/${info.nsteps} (${info.depth}x recycle)`, { visible: false },
        );
      };

      const genOpts = {
        nsteps,
        recycleSchedule,
        seed: sampleSeed,
        sampling: customSampling,
        inverseFolding,
        partialTrajectoryFractionSteps,
        inputAtoms: atoms,
        // A design with nothing held in place but a rigid target is laid on the target as loaded.
        inputFrame: true,
        onStep,
        decodeStep: wantLiveFrame,
        liveInsertion: hasUnindexedMotif,
        earlyStop: s.earlyStop ?? true,
        signal,
      };

      const result = useGpu
        ? await this.gpuModel.generate(batch, genOpts)
        : await generateRfd4pCpu(batch, this.activeWeights, this.activeConfig,
          this.glFallback ? { ...genOpts, kernels: this.glModel.kernels() } : genOpts);

      if (!fixedAnchors && referenceCaNm && result.structure?.atoms && !result.structure.inputFrame && !isAnchored(result.structure.atoms) &&
          alignStructureAtomsToCa(result.structure.atoms, referenceCaNm, n)) {
        result.cif = toCifString(result.structure);
        result.pdb = toPdbString(result.structure);
      }

      this.lastResult = result;
      if (result.stepsRun != null && result.stepsRun < result.nsteps) earlyStops.push(`${result.stepsRun}/${result.nsteps}`);

      const structureAtoms = result.structure?.atoms ?? [];
      const overlays = designOverlays(structureAtoms, motifOverlay, unmodelled);
      const trajectoryCif = buildTrajectoryCifFromFrames(
        trajectoryFrames,
        tokens,
        result.residueType,
        UNIFIED_ATOM37_TOKENS,
        result.structure?.atoms ?? null,
        result.structure ?? {},
      );
      const designed = designedChains(atoms, structureAtoms);
      const slots = motifSlots(structureAtoms, motifSideChainAtoms(structureAtoms, unmodelled));
      const designIndex = (this.designCount += 1);
      await hooks.design({
        name: String(sampleSeed),
        cif: result.cif,
        trajectoryCif,
        pdb: result.pdb,
        filenameStem: `design-${designIndex}-seed${sampleSeed}`,
        // Which of upstream's pass criteria apply: the task, and whether everything the refold folds was designed.
        kind: criteriaKind(taskKey, {
          designedAll: polymerChains(result.pdb ?? '').every((c) => designed.includes(c.letter)),
          hasLigand: /^HETATM/m.test(result.pdb ?? ''),
        }),
        seed: sampleSeed,
        stepsRun: result.stepsRun ?? null,
        nsteps: result.nsteps ?? nsteps,
        designedChains: designed,
        hotspots: designHotspots(result.cif, hotspotKeys),
        overlays: cardOverlays(overlays),
        // What a refold needs (web/refold-provider.js), handed back with the design.
        data: {
          pdb: result.pdb,
          cif: result.cif,
          designedChains: designed,
          // Designed chains mostly held to the input (an antibody framework): templated on those residues in a refold.
          partialTemplates: partialTemplates(structureAtoms, designed),
          // ...and which of the motif residues' atoms the run held, for the motif RMSDs a refold reports.
          motifAtoms: motifAtomsOf(structureAtoms, slots),
        },
      });
    }

    const elapsedS = ((performance.now() - t0) / 1000).toFixed(2);
    this.runCounter += 1;
    const stopNote = earlyStops.length ? ` Converged early at step ${earlyStops.join(', ')}.` : '';
    const seeds = numSamples > 1 ? `seeds ${baseSeed}–${baseSeed + numSamples - 1}` : `seed ${baseSeed}`;
    const faster = this.speedHintOn ? ' Up to 1.5× faster with Chrome\'s experimental WebGPU (see above).' : '';
    return { message: `Generated from input revision ${request.revision} with ${seeds} (${elapsedS} s on ${backendLabel}).${stopNote}${faster}` };
  }
}

const boot = () => {
  const app = new Rfd4pStudioApp();
  window.rfd4pApp = app;
  app.init().catch((err) => console.error('RFD4-Proteina page failed to start:', err));
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
