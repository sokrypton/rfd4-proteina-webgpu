/**
 * RFD4-Proteina interactive browser host — exact match to `nvidia/rfd4-proteina-zerogpu`
 * (`RFD4GenerationProvider` top controls + embedded `AtomWorks Studio` editor with
 * Mol* WebGL, Patinae WebGPU, Sequence dock, Annotations popover, and trajectory playback)
 * powered 100% client-side by `Rfd4pGeneratorGpu` on WebGPU.
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
  getRfd4pPresetById,
  requestWebGpuDevice,
  processSampleToAtoms,
  superimposeCoords3D,
  toCifString,
  toPdbString,
} from '../webgpu/index.js';
import {
  buildLiveFrameCifFromCa,
  caSecondaryStructureCif,
  buildTrajectoryCifFromFrames,
  getSharedStudioStore,
  installStudioApiInterceptor,
} from './studio-api.js';
import { DesignPanel } from './design-panel.js';

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
function buildMotifOverlayAtoms(inputAtoms) {
  if (!inputAtoms?.token_id || !inputAtoms.condition_coordinate_1_atom) return [];
  const crdOf = (i) => {
    const c = inputAtoms.condition_coordinate_1_atom[i];
    return Array.isArray(c) && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2]) ? c : null;
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
  const isMotifToken = (tid) => tokenHasCrd.get(tid) &&
    (!tokenIndexed.get(tid) || chainTokens.get(tokenChain.get(tid)).fixed < chainTokens.get(tokenChain.get(tid)).total);
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
    if (fixed.length < tokens.size) for (const t of fixed) motifTokens.add(t);
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
  if (typeof window.__rfd4pSetOverlay !== 'function') return;
  // One ghost color (not element colors) so the reference never reads as extra bonds on the sampled side chain.
  await window.__rfd4pSetOverlay('motif', overlays?.motif ?? null, {
    label: 'Motif (reference)', alpha: 0.35, color: 0xe2e8f0, sizeAspectRatio: 0.6,
  });
  await window.__rfd4pSetOverlay('sideChains', overlays?.sideChains ?? null, { label: 'Motif side chains' });
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
    this.help = document.querySelector('#sequence-dock .sequence-help');
    this.el = null;
    this.tokens = new Map();
  }

  start() {
    if (!this.view || this.el) return;
    document.getElementById('rfd4p-live-sequence')?.remove();
    this.el = document.createElement('div');
    this.el.id = 'rfd4p-live-sequence';
    this.el.className = 'sequence-view';
    this.el.setAttribute('aria-live', 'off');
    this.view.after(this.el);
    // On the dock, not the view: Studio's sequence render resets the view's className.
    this.dock?.classList.add('rfd4p-live');
    this.helpText = this.help?.textContent ?? null;
    this.tokens.clear();
  }

  /** @param {Array<{chainId: string, resId: number, resName: string}>} atoms */
  update(atoms, step, nsteps, convergence = null) {
    if (!this.el) return;
    const residues = new Map();
    for (const a of atoms) {
      const code = ONE_LETTER[a.resName];
      const key = `${a.chainId}:${a.resId}`;
      if (code && !residues.has(key)) residues.set(key, { chainId: a.chainId, resId: a.resId, resName: a.resName, code });
    }
    let changed = 0;
    if (this.tokens.size !== residues.size || [...residues.keys()].some((k) => !this.tokens.has(k))) {
      this.build(residues);
    } else {
      for (const [key, r] of residues) {
        const tok = this.tokens.get(key);
        const diff = tok.textContent !== r.code;
        if (diff) {
          tok.textContent = r.code;
          tok.title = `${r.chainId}/${r.resName} ${r.resId}`;
          changed += 1;
        }
        tok.classList.toggle('rfd4p-changed', diff);
      }
    }
    if (this.help) {
      const conv = convergence?.stop ? ' · converged'
        : convergence?.streak ? ` · still for ${convergence.streak} step${convergence.streak > 1 ? 's' : ''}` : '';
      this.help.textContent = `Live · step ${step + 1}/${nsteps}` + (step > 0 ? ` · ${changed} changed` : '') + conv;
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
        const color = window.__rfd4pChainColor?.(r.chainId);
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
    if (this.help && this.helpText != null) this.help.textContent = this.helpText;
  }
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

    // AtomWorks Studio integration state
    this.studioSessionId = null;
    this.inputSessionId = null;
    this.useStudioSession = false;
    this.activeAbortController = null;
    this.generatedViews = new Map(); // sessionId -> { label, cif, pdb, hasTrajectory, isInput }

    this.els = {
      cardStaging: document.getElementById('rfd4p-card-staging'),
      studioCard: document.getElementById('rfd4p-studio-card'),
      predictBtn: document.getElementById('predict'),
      thinkingRadioGroup: document.getElementById('thinking-radio-group'),
      samplesInput: document.getElementById('samples-input'),
      seedInput: document.getElementById('seed-input'),
      backboneK: document.getElementById('backbone-k'),
      backboneKNum: document.getElementById('backbone-k-num'),
      latentK: document.getElementById('latent-k'),
      latentKNum: document.getElementById('latent-k-num'),
      partialDiffusion: document.getElementById('partial-diffusion'),
      partialPercent: document.getElementById('partial-diffusion-percent'),
      partialPercentNum: document.getElementById('partial-diffusion-percent-num'),
      earlyStop: document.getElementById('early-stop'),
      statusBox: document.getElementById('generation-status'),
      statusMessage: document.getElementById('rfd4p-status-message'),
      modelLoad: document.getElementById('rfd4p-model-load'),
      modelLoadText: document.getElementById('rfd4p-model-load-text'),
      progress: document.getElementById('rfd4p-progress'),
      designResultsRow: document.getElementById('design-results'),
      designViewRadios: document.getElementById('design-view-radios'),
      downloadsBox: document.getElementById('downloads'),
      downloadCifBtn: document.getElementById('download-cif-btn'),
      downloadAllBtn: document.getElementById('download-all-btn'),
      webgpuBadge: document.getElementById('webgpu-badge'),
    };
  }

  _mountCardIntoStudio() {
    const card = this.els.studioCard;
    const popover = document.getElementById('atomworks-studio-popover');
    if (card && popover && card.parentElement !== popover) {
      popover.prepend(card);
      this._bindControls();
    }
    const closeBtn = document.getElementById('close-structure');
    if (closeBtn && !closeBtn.__rfd4pCloseBound) {
      closeBtn.__rfd4pCloseBound = true;
      closeBtn.addEventListener('click', () => {
        this.useStudioSession = false;
        this.inputSessionId = null;
        this.studioSessionId = null;
        if (this.els.designResultsRow) this.els.designResultsRow.hidden = true;
      });
    }
  }

  setStatus(msg, isError = false, { visible = true } = {}) {
    if (!this.els.statusMessage) return;
    this.els.statusMessage.textContent = msg;
    this.els.statusMessage.classList.toggle('error', Boolean(isError));
    if (this.els.statusBox) {
      this.els.statusBox.hidden = !visible;
    }
    const studioFooter = visible ? document.getElementById('status-message') : null;
    if (studioFooter) {
      studioFooter.textContent = msg;
      studioFooter.classList.toggle('error-text', Boolean(isError));
    }
  }

  setProgress(fraction, state = 'running') {
    if (!this.els.progress) return;
    this.els.progress.dataset.state = state;
    this.els.progress.value = Math.max(0, Math.min(1, fraction));
    this.els.progress.hidden = state === 'idle';
  }

  /** Tick or clear Partial diffusion (optionally setting its %); the % controls are live only when ticked. */
  _setPartialDiffusion(enabled, percent = null) {
    const { partialDiffusion, partialPercent, partialPercentNum } = this.els;
    if (partialDiffusion) partialDiffusion.checked = Boolean(enabled);
    if (percent != null && partialPercent) {
      partialPercent.value = String(Math.round(percent));
      if (partialPercentNum) partialPercentNum.value = partialPercent.value;
    }
    if (partialPercent) partialPercent.disabled = !enabled;
    if (partialPercentNum) partialPercentNum.disabled = !enabled;
  }

  _getSelectedThinking() {
    const checked = this.els.thinkingRadioGroup?.querySelector('input[name="thinking-radio"]:checked');
    return checked?.value || 'none';
  }

  _syncThinkingRadios(val) {
    if (!this.els.thinkingRadioGroup) return;
    const radios = this.els.thinkingRadioGroup.querySelectorAll('input[name="thinking-radio"]');
    for (const r of radios) {
      r.checked = r.value === val;
    }
  }

  async init() {
    this._bindControls();
    await this._loadPreset(RFD4P_PRESETS[0].id, { syncStudio: false });
    this._initStudioBridge();
    this.designPanel = new DesignPanel(this);
    await this._initWebGpu();
    this._syncThinkingRadios('none');
    this.els.predictBtn.disabled = false;
    this.designPanel.render();
    const backendLabel = this.gpuDevice ? 'WebGPU FP16' : 'CPU Float32';
    if (this.els.webgpuBadge) {
      this.els.webgpuBadge.textContent = backendLabel;
    }
    this.setStatus('Ready to generate.', false, { visible: false });

    const params = new URLSearchParams(window.location.search);
    if (params.get('autorun') === '1') {
      await this.runDesign();
    }
  }

  _initStudioBridge() {
    this._mountCardIntoStudio();
    // AtomWorks Studio runs in this same document and announces session changes as window events.
    window.addEventListener('atomworks-studio-message', (ev) => {
      const event = { data: ev.detail };
      if (['ready', 'changed', 'example'].includes(event.data?.type)) {
        this._mountCardIntoStudio();
        const sid = event.data.session_id;
        if (sid) {
          this.studioSessionId = sid;
          const isExplicitChange = event.data.type === 'changed' || event.data.type === 'example';
          if (isExplicitChange || !this.generatedViews.has(sid) || this.generatedViews.get(sid)?.isInput) {
            const previousInput = this.inputSessionId;
            this.inputSessionId = sid;
            this.useStudioSession = true;
            if (isExplicitChange && this.generatedViews.has(sid)) {
              const view = this.generatedViews.get(sid);
              if (view) view.isInput = true;
            }
            if (previousInput && previousInput !== sid && this.els.designResultsRow && event.data.type === 'example') {
              this.els.designResultsRow.hidden = true;
            }
            if (event.data.type === 'example') {
              if (this.activeAbortController) {
                this.activeAbortController.abort();
              }
              this._setPartialDiffusion(event.data.task === 'partial_diffusion');
              this.setStatus('Ready to generate.', false, { visible: false });
            }
          }
        }
      }
    });
  }

  async _navigateStudioToSession(sessionId, { withTrajectory = false } = {}) {
    if (!sessionId) return;
    const trajUrl = withTrajectory
      ? `/api/v1/sessions/${encodeURIComponent(sessionId)}/trajectory.cif`
      : null;
    if (typeof window.__rfd4pOpenSession !== 'function') {
      console.warn('AtomWorks Studio is not ready; cannot open session', sessionId);
      return;
    }
    try {
      await window.__rfd4pOpenSession(sessionId, trajUrl);
      // Display-only reference motif and motif side chains beside a finished design (not part of the saved document).
      await showDesignOverlays(this.generatedViews.get(sessionId)?.overlays);
    } catch (err) {
      console.warn('Motif overlay failed:', err);
    }
  }

  async _initWebGpu() {
    try {
      const res = await requestWebGpuDevice();
      if (res && res.device) {
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

  async ensureWeightsLoaded(source = this.prevArchChoice, { needEncoder = false } = {}) {
    const half = true;
    const makeProgressCb = (label) => ({ bytesLoaded, totalBytes, file }) => {
      const mb = (bytesLoaded / (1024 * 1024)).toFixed(1);
      const tag = file ? `${label} (${file})` : label;
      if (totalBytes > 0) {
        const totMb = (totalBytes / (1024 * 1024)).toFixed(1);
        this.els.modelLoadText.textContent = `${tag} · ${mb} / ${totMb} MiB`;
        this.setProgress(Math.min(0.95, bytesLoaded / totalBytes), 'running');
      } else {
        this.els.modelLoadText.textContent = `${tag} · ${mb} MiB`;
      }
    };

    if (this.activeWeights && this.activeConfig) {
      if (needEncoder && this.shardedBundle && !this.shardedBundle.encoderLoaded) {
        try {
          this.els.modelLoad.hidden = false;
          this.setProgress(0.01, 'running');
          this.setStatus('Loading RFD4-Proteina encoder shards (131 MiB)...');
          await this.shardedBundle.loadEncoder(makeProgressCb('weights/int8'));
          this.setProgress(0, 'idle');
        } finally {
          this.els.modelLoad.hidden = true;
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
      this.els.modelLoad.hidden = false;
      this.setProgress(0.01, 'running');

      if (source === 'int8-sharded') {
        try {
          this.setStatus('Loading sharded INT8 RFD4-Proteina weights (block-64)...');
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
          this.setProgress(0, 'idle');
          return;
        } catch (shardedErr) {
          console.warn('Sharded INT8 weights unavailable, falling back to model.safetensors:', shardedErr);
        }
      }

      const useHf = source === 'hf';
      const configUrl = useHf ? DEFAULT_HF_CONFIG_URL : DEFAULT_LOCAL_CONFIG_URL;
      const weightsUrl = useHf ? DEFAULT_HF_WEIGHTS_URL : DEFAULT_LOCAL_WEIGHTS_URL;

      this.setStatus('Loading RFD4-Proteina configuration and 1,257-tensor checkpoint...');

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
      this.setProgress(0, 'idle');
    } finally {
      this.els.modelLoad.hidden = true;
    }
  }

  async _loadPreset(presetId, { syncStudio = false } = {}) {
    const preset = getRfd4pPresetById(presetId);
    this.currentPreset = preset;
    this.currentCifBytes = null;
    this.currentCifText = '';
    this.useStudioSession = false;

    this._setPartialDiffusion(
      PRESET_TO_TASK[preset.id] === 'partial_diffusion',
      preset.partialTrajectoryFractionSteps != null ? preset.partialTrajectoryFractionSteps * 100 : null,
    );
    if (preset.recycleSchedule) {
      this._syncThinkingRadios(preset.recycleSchedule);
    }

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
        if (syncStudio) {
          const fileName = preset.cifUrl.split('/').pop().replace(/\.zst$/, '');
          const store = getSharedStudioStore();
          const { sessionId } = store.createSession(this.currentCifText, fileName);
          this.studioSessionId = sessionId;
          this.inputSessionId = sessionId;
          this._navigateStudioToSession(sessionId);
        }
      } catch (err) {
        console.warn('Failed to fetch preset CIF:', err);
        this.currentCifText = '';
      }
    }

    this._buildCurrentBatch();
  }

  _buildCurrentBatch(seedOverride = null) {
    const seed = seedOverride ?? (Number.parseInt(this.els.seedInput.value, 10) || 42);

    // Build directly from the active AtomWorks Studio session when the user is working in Studio
    if (this.useStudioSession && this.inputSessionId) {
      const store = getSharedStudioStore();
      const doc = store.getSession(this.inputSessionId);
      if (doc) {
        const res = buildBatch(doc.toCif(), { seed });
        this.currentBatchResult = res;
        return res;
      }
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

  _bindControls() {
    const bindSliderPair = (slider, numInput) => {
      if (!slider || !numInput) return;
      slider.oninput = () => { numInput.value = slider.value; };
      numInput.oninput = () => { slider.value = numInput.value; };
    };
    bindSliderPair(this.els.backboneK, this.els.backboneKNum);
    bindSliderPair(this.els.latentK, this.els.latentKNum);
    bindSliderPair(this.els.partialPercent, this.els.partialPercentNum);
    if (this.els.partialDiffusion) {
      this.els.partialDiffusion.onchange = () => this._setPartialDiffusion(this.els.partialDiffusion.checked);
    }

    if (this.els.predictBtn) {
      this.els.predictBtn.onclick = () => {
        if (this.activeAbortController) {
          this.activeAbortController.abort();
          return;
        }
        this.runDesign();
      };
    }

    if (this.els.downloadCifBtn) {
      this.els.downloadCifBtn.onclick = () => {
        const activeView = this.generatedViews.get(this.studioSessionId);
        const cif = activeView?.cif || this.lastResult?.cif;
        if (!cif) return;
        this._downloadText(cif, `${activeView?.filenameStem || 'design-1'}.cif`, 'chemical/x-mmcif');
      };
    }

    if (this.els.downloadAllBtn) {
      this.els.downloadAllBtn.onclick = async () => {
        await this._downloadAllDesigns();
      };
    }

    if (this.els.designViewRadios) {
      this.els.designViewRadios.onchange = (e) => {
        const input = e.target;
        if (input && input.name === 'design-view-radio' && input.checked) {
          const sid = input.value;
          this.studioSessionId = sid;
          const viewInfo = this.generatedViews.get(sid);
          this._navigateStudioToSession(sid, {
            withTrajectory: Boolean(viewInfo?.hasTrajectory),
          });
        }
      };
    }
  }

  async _downloadAllDesigns() {
    const designs = [...this.generatedViews.values()].filter((v) => !v.isInput && v.cif);
    if (designs.length === 0) {
      if (this.lastResult?.cif) {
        this._downloadText(this.lastResult.cif, 'design-1.cif', 'chemical/x-mmcif');
      }
      return;
    }
    if (designs.length === 1 || typeof window.JSZip !== 'function') {
      this._downloadText(designs[0].cif, `${designs[0].filenameStem || 'design-1'}.cif`, 'chemical/x-mmcif');
      return;
    }
    const zip = new window.JSZip();
    designs.forEach((d, idx) => {
      zip.file(`design-${idx + 1}.cif`, d.cif);
    });
    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'designs.zip';
    a.click();
    URL.revokeObjectURL(url);
  }

  _renderDesignResultsRow(choices, selectedSessionId) {
    if (!this.els.designResultsRow || !this.els.designViewRadios) return;
    this.els.designViewRadios.innerHTML = '';
    for (const choice of choices) {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = 'design-view-radio';
      input.value = choice.sessionId;
      input.checked = choice.sessionId === selectedSessionId;
      const span = document.createElement('span');
      span.textContent = choice.label;
      label.append(input, span);
      this.els.designViewRadios.appendChild(label);
    }
    const designCount = choices.filter((c) => c.label !== 'Input').length;
    if (this.els.downloadsBox) {
      this.els.downloadsBox.hidden = designCount <= 1;
    }
    this.els.designResultsRow.hidden = false;
  }

  _downloadText(text, filename, mime) {
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  async runDesign(overrides = {}) {
    if (this.activeAbortController) {
      this.activeAbortController.abort();
    }
    const abortController = new AbortController();
    this.activeAbortController = abortController;
    const liveSequence = new LiveSequenceStrip();
    try {
      this.els.predictBtn.textContent = 'Stop generation';
      this.els.predictBtn.classList.add('secondary');
      this.els.predictBtn.disabled = false;

      const baseSeed = Number.parseInt(this.els.seedInput.value, 10) || 42;
      const numSamples = Math.max(1, Math.min(8, Number.parseInt(this.els.samplesInput?.value ?? '1', 10) || 1));
      const nsteps = overrides.nsteps ?? 60;
      const recycleSchedule = overrides.recycleSchedule ?? this._getSelectedThinking();
      // Sampling temperature as upstream's Studio provider exposes it: TSR K per modality (lower = hotter).
      const backboneK = Math.max(1, Math.min(20, Number.parseFloat(this.els.backboneK?.value ?? '16') || 16));
      const latentK = Math.max(1, Math.min(6, Number.parseFloat(this.els.latentK?.value ?? '4') || 4));

      const firstBatchRes = this._buildCurrentBatch(baseSeed);

      const preset = this.currentPreset;
      // What to design comes from the input's annotations (as in upstream's Space); partial diffusion is the one
      // option on top. The resampled inverse-folding sampler is kept for the benchmark preset only.
      const inverseFolding = !this.useStudioSession && Boolean(preset?.inverseFolding);
      const partialPercent = Math.max(1, Math.min(100, Number.parseFloat(this.els.partialPercent?.value ?? '70') || 70));
      const partialTrajectoryFractionSteps = this.els.partialDiffusion?.checked ? partialPercent / 100 : null;
      const needEncoder = Boolean(inverseFolding || partialTrajectoryFractionSteps != null);
      if (needEncoder) {
        const has3dCoords = Boolean(
          firstBatchRes?.atoms?.coord?.some(
            (c) => Array.isArray(c) && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2])
          )
        );
        if (!has3dCoords) {
          throw new Error(
            `${inverseFolding ? 'Inverse folding' : 'Partial diffusion'} requires an input structure with 3D coordinates (open a structure or select a 3D example in Examples).`
          );
        }
      }

      if (!this.gpuDevice && this.gpuDeviceLost) {
        // The previous device was lost: get a new one; ensureWeightsLoaded then rebuilds the model on it.
        this.setStatus('Restarting WebGPU after a GPU reset...');
        await this._initWebGpu();
      }
      await this.ensureWeightsLoaded(this.prevArchChoice, { needEncoder });

      const useGpu = Boolean(this.gpuModel);
      const backendLabel = useGpu ? 'WebGPU FP16' : 'CPU Float32';

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

      const store = getSharedStudioStore();
      this.generatedViews.clear();
      const viewChoices = [];
      const earlyStops = [];
      liveSequence.start();
      let inputRevision = 1;
      if (!this.inputSessionId && this.currentCifText) {
        const { sessionId } = store.createSession(this.currentCifText, 'uncond_short.cif');
        this.inputSessionId = sessionId;
      }
      if (this.inputSessionId && store.getSession(this.inputSessionId)) {
        const inputDoc = store.getSession(this.inputSessionId);
        inputRevision = inputDoc.revision ?? 1;
        this.generatedViews.set(this.inputSessionId, {
          label: 'Input',
          cif: inputDoc.toCif(),
          pdb: null,
          isInput: true,
          hasTrajectory: false,
          filenameStem: 'input',
        });
        viewChoices.push({ label: 'Input', sessionId: this.inputSessionId });
      }

      const t0 = performance.now();

      for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx += 1) {
        const sampleSeed = baseSeed + sampleIdx;
        const { batch, n, atoms } = sampleIdx === 0 ? firstBatchRes : this._buildCurrentBatch(sampleSeed);
        const tokens = extractTokenMetadata(batch);
        const fixedAnchors = extractFixedTokenAnchorsNm(batch, n);
        const allTokenIndices = Int32Array.from({ length: n }, (_, idx) => idx);
        const trajectoryFrames = [];
        const sampleTag = numSamples > 1 ? ` [Design ${sampleIdx + 1}/${numSamples}]` : '';

        this.setProgress((sampleIdx / numSamples) + 0.02, 'running');
        this.setStatus(`Generating design${sampleTag} (${n} tokens, ${nsteps} steps, ${backendLabel})...`);

        let referenceCaNm = null;
        let liveFramesRendered = 0;
        let liveCamRadius = 0;
        let lastLiveRenderTime = 0;
        let liveRenderBusy = false;
        let liveRenderPromise = null;

        // Live frames use the decoder's all-atom x1 (real backbone geometry, so Mol* can assign secondary
        // structure, and in-context targets keep their input atoms); CA-only frames are the fallback.
        // Asked before the step's trunk pass, so predict whether a render will be due once it finishes.
        let lastStepEnd = performance.now();
        let lastStepMs = 0;
        const LIVE_FRAME_INTERVAL_MS = 320;
        // Not gated on liveRenderBusy: the previous frame's Mol* update almost always finishes during this trunk pass,
        // and a skipped decode used to fall back to a CA-only frame (frozen sequence, sheets twisted by invented O).
        const wantLiveFrame = (step, total) =>
          step === 0 || (step < total - 1 && performance.now() + lastStepMs - lastLiveRenderTime >= LIVE_FRAME_INTERVAL_MS);
        const motifOverlay = buildMotifOverlayAtoms(atoms);
        const unmodelled = unmodelledTokens(atoms);
        // Overlays for the frame being built; shown once its structure has been drawn.
        let liveOverlays = null;
        const hasUnindexedMotif = Boolean(atoms.condition_index_1_residue?.some((v, i) =>
          !v && !atoms.hetero?.[i] && atoms.condition_coordinate_1_atom?.[i]?.some?.(Number.isFinite)));
        const liveAllAtomCif = (info) => {
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
            });
            // processSampleToAtoms already aligns coordinate-conditioned designs onto the input (motif) frame.
            if (!fixedAnchors && !isAnchored(structure.atoms) && referenceCaNm) {
              alignStructureAtomsToCa(structure.atoms, referenceCaNm, n);
            }
            liveSequence.update(structure.atoms, info.step, info.nsteps, info.convergence);
            liveOverlays = designOverlays(structure.atoms, motifOverlay, unmodelled);
            return toCifString(structure) + caSecondaryStructureCif(structure.atoms);
          } catch (err) {
            console.warn('Live all-atom frame failed; showing Cα frame:', err);
            return null;
          }
        };

        const onStep = (info) => {
          const stepEnd = performance.now();
          lastStepMs = stepEnd - lastStepEnd;
          lastStepEnd = stepEnd;
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
          });

          const now = performance.now();
          const isStopping = info.step > 0 && (info.step === info.nsteps - 1 || Boolean(info.convergence?.stop));
          // On the GPU path only decoded frames are shown; the CA-only frame is for the CPU backend (no live decode).
          const frameReady = Boolean(info.decoded) || !useGpu;
          if (frameReady && !liveRenderBusy && !isStopping && (info.step === 0 || now - lastLiveRenderTime >= LIVE_FRAME_INTERVAL_MS)) {
            if (typeof window.__rfd4pRenderLiveFrame === 'function') {
              liveOverlays = { motif: overlayCifOf(motifOverlay, 'motif'), sideChains: null };
              const liveCif = liveAllAtomCif(info) ?? buildLiveFrameCifFromCa(alignedCaNm, tokens, null, UNIFIED_ATOM37_TOKENS);
              const frameOverlays = liveOverlays;
              if (liveCif) {
                liveRenderBusy = true;
                lastLiveRenderTime = now;
                // Early x1 estimates are a collapsed blob that later expands several-fold, so a camera
                // fit once to step 1 ends up inside the protein; refit whenever the extent drifts.
                const radius = caExtentRadiusNm(alignedCaNm);
                const keepCam = liveFramesRendered > 0 && radius < liveCamRadius * 1.3 && radius > liveCamRadius * 0.6;
                if (!keepCam) liveCamRadius = radius;
                liveFramesRendered += 1;
                liveRenderPromise = Promise.resolve(
                  window.__rfd4pRenderLiveFrame(liveCif, `step_${info.step + 1}.cif`, keepCam),
                )
                  .then(() => showDesignOverlays(frameOverlays))
                  .catch(() => {})
                  .finally(() => {
                    liveRenderBusy = false;
                  });
              }
            }
          }

          const stepFrac = (info.step + 1) / (info.nsteps + 1);
          const overallFrac = (sampleIdx + stepFrac) / numSamples;
          this.setProgress(overallFrac, 'running');
          this.setStatus(
            `Generating design${sampleTag} · step ${info.step + 1}/${info.nsteps} (${info.depth}x recycle)`
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
          onStep,
          decodeStep: wantLiveFrame,
          liveInsertion: hasUnindexedMotif,
          earlyStop: this.els.earlyStop?.checked ?? true,
          signal: abortController.signal,
        };

        let result;
        try {
          result = useGpu
            ? await this.gpuModel.generate(batch, genOpts)
            : await generateRfd4pCpu(batch, this.activeWeights, this.activeConfig, genOpts);
        } finally {
          if (liveRenderPromise) {
            await liveRenderPromise;
          }
        }

        if (!fixedAnchors && referenceCaNm && result.structure?.atoms && !isAnchored(result.structure.atoms) &&
            alignStructureAtomsToCa(result.structure.atoms, referenceCaNm, n)) {
          result.cif = toCifString(result.structure);
          result.pdb = toPdbString(result.structure);
        }

        this.lastResult = result;
        if (result.stepsRun != null && result.stepsRun < result.nsteps) earlyStops.push(`${result.stepsRun}/${result.nsteps}`);

        const overlays = designOverlays(result.structure?.atoms ?? [], motifOverlay, unmodelled);

        // Register generated design in AtomWorks Studio session store with trajectory playback
        const trajectoryCif = buildTrajectoryCifFromFrames(
          trajectoryFrames,
          tokens,
          result.residueType,
          UNIFIED_ATOM37_TOKENS,
        );
        const designName = `design-${sampleIdx + 1}.cif`;
        const { sessionId: designSessionId } = store.createSession(result.cif, designName, {
          trajectoryCif,
        });
        this.generatedViews.set(designSessionId, {
          label: `Design ${sampleIdx + 1}`,
          cif: result.cif,
          pdb: result.pdb,
          isInput: false,
          hasTrajectory: Boolean(trajectoryCif),
          filenameStem: `design-${sampleIdx + 1}`,
          overlays,
        });
        viewChoices.push({ label: `Design ${sampleIdx + 1}`, sessionId: designSessionId });
      }

      const elapsedS = ((performance.now() - t0) / 1000).toFixed(2);
      this.runCounter += 1;

      const firstDesignChoice = viewChoices.find((c) => c.label === 'Design 1');
      if (firstDesignChoice) {
        this._renderDesignResultsRow(viewChoices, firstDesignChoice.sessionId);
        this.studioSessionId = firstDesignChoice.sessionId;
        const viewInfo = this.generatedViews.get(firstDesignChoice.sessionId);
        await this._navigateStudioToSession(firstDesignChoice.sessionId, {
          withTrajectory: Boolean(viewInfo?.hasTrajectory),
        });
      }

      this.setProgress(1, 'idle');
      const stopNote = earlyStops.length ? ` Converged early at step ${earlyStops.join(', ')}.` : '';
      this.setStatus(`Generated from input revision ${inputRevision} (${elapsedS} s on ${backendLabel}).${stopNote}`);
    } catch (err) {
      this.setProgress(0, 'idle');
      try {
        await window.__rfd4pRestoreInputView?.();
        await showDesignOverlays(null);
      } catch {
        // Ignore
      }
      if (abortController.signal.aborted || String(err?.message || '').includes('aborted')) {
        this.setStatus('Generation stopped. Your input is unchanged.', false);
      } else {
        console.error('RFD4-Proteina run failed:', err);
        // A failed readback can surface before device.lost settles; give it a moment so a lost device is
        // recognised (and dropped) now rather than failing the next Generate too.
        if (this.gpuDevice && this._deviceLost) {
          await Promise.race([this._deviceLost, new Promise((r) => setTimeout(r, 300))]);
        }
        const lost = !this.gpuDevice && this.gpuDeviceLost;
        this.setStatus(lost
          ? `The GPU was reset during generation (${this.gpuDeviceLost.reason}: ${this.gpuDeviceLost.message || 'no details'}). Press Generate to retry; WebGPU restarts automatically. Your input is unchanged.`
          : `Generation failed. ${err.message} Your input is unchanged.`, true);
      }
    } finally {
      // A newer run (Generate pressed again) owns the strip once it has started.
      if (this.activeAbortController === abortController) liveSequence.stop();
      if (this.activeAbortController === abortController) {
        this.activeAbortController = null;
      }
      this.els.predictBtn.textContent = 'Generate design';
      this.els.predictBtn.classList.remove('secondary');
      this.els.predictBtn.disabled = false;
      this.designPanel?.render();
    }
  }
}

const boot = () => {
  const app = new Rfd4pStudioApp();
  window.rfd4pApp = app;
  app.init();
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
