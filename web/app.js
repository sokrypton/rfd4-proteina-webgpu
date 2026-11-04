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
import { openGuide } from './guide-dialog.js';
import { resultsFiles } from './results-zip.js';
import {
  REFOLD_MODELS, criteriaKind, describeRefold, designedChains, motifSlots, partialTemplates, passes, pdbToCif,
  polymerChains, rankKey, refoldLigandCif, refoldMetrics, refoldOverlayCif,
} from './refold-view.js';

/** An element: attributes (`text`, `class`, `style`, `on…` handlers; null or false left out), then its children. */
const h = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'text') node.textContent = v;
    else if (k === 'class') node.className = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
};

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
  // A refold's designed chains as a magenta cartoon over the design (web/refold-view.js); magenta is in no chain
  // colour, so it reads against any binder. `representation` needs Studio with RosettaCommons/atomworks-studio#29
  // (studio.json `override`); a Studio without it draws ball-and-stick instead.
  await window.atomworksStudio?.setOverlay('refold', overlays?.refold?.cif ?? null, {
    label: overlays?.refold?.label ?? 'Refold', color: 0xe040fb, alpha: 0.75, representation: 'cartoon',
  });
  // Its ligands, where the refold put them, in the same magenta: a cartoon draws polymers only.
  await window.atomworksStudio?.setOverlay('refoldLigands', overlays?.refold?.ligandCif ?? null, {
    label: `${overlays?.refold?.label ?? 'Refold'} ligands`, color: 0xe040fb, sizeAspectRatio: 0.6,
  });
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

/**
 * Mark `keys` (`hotspotResidueKeys` of the input) as hotspots on a generated design's document, so its sequence bar
 * and viewer show them as the input's do. A design keeps a kept chain's letter and numbering; residues that do not
 * match (chain, number and residue) are left alone.
 */
function markHotspots(doc, keys) {
  if (!doc?.atoms || !keys?.size) return;
  const a = doc.atoms;
  const hits = [];
  for (let i = 0; i < a.nAtoms; i += 1) if (keys.has(`${a.chain_id[i]}|${a.res_id[i]}|${a.res_name[i]}`)) hits.push(i);
  if (hits.length) doc.setCondition('hotspot', hits, { enabled: true, snapshot: false });
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
      seedRandom: document.getElementById('seed-random'),
      seedMode: document.getElementById('seed-mode'),
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
      downloadAllBtn: document.getElementById('download-all-btn'),
      refoldBox: document.getElementById('rfd4p-refold'),
      refoldModel: document.getElementById('rfd4p-refold-model'),
      refoldSeed: document.getElementById('rfd4p-refold-seed'),
      refoldBtn: document.getElementById('rfd4p-refold-btn'),
      refoldResult: document.getElementById('rfd4p-refold-result'),
      refoldTerms: document.getElementById('rfd4p-refold-terms'),
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

  /** How far a run is, drawn as the Generate (now Stop) button filling up. */
  setProgress(fraction, state = 'running') {
    const btn = this.els.predictBtn;
    if (!btn) return;
    btn.style.setProperty('--rfd4p-progress', `${(Math.max(0, Math.min(1, fraction)) * 100).toFixed(1)}%`);
    btn.classList.toggle('rfd4p-running', state === 'running');
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
    // It lives under More settings: shown whenever it is turned on, so it is never on out of sight.
    if (enabled) document.getElementById('sampling-accordion')?.setAttribute('open', '');
  }

  _getSelectedThinking() {
    const checked = this.els.thinkingRadioGroup?.querySelector('input[name="thinking-radio"]:checked');
    // None by default, for speed (about 3x faster than upstream's medium, which its `inference.yaml` and binder cookbook use).
    return checked?.value || 'none';
  }

  _syncThinkingRadios(val) {
    if (!this.els.thinkingRadioGroup) return;
    const radios = this.els.thinkingRadioGroup.querySelectorAll('input[name="thinking-radio"]');
    for (const r of radios) {
      r.checked = r.value === val;
    }
  }

  /**
   * Seeds start random so every visitor gets their own designs. A new one is drawn after each run, unless the user
   * typed a seed (then it stays, to reproduce a design); the dice button draws one and returns to random.
   */
  _initSeed() {
    const { seedInput, seedRandom } = this.els;
    if (!seedInput) return;
    this.seedPinned = false;
    this._rollSeed();
    seedInput.addEventListener('input', () => {
      this.seedPinned = seedInput.value.trim() !== '';
      if (!this.seedPinned) this._rollSeed();
      this._renderSeedMode();
    });
    seedRandom?.addEventListener('click', () => {
      this.seedPinned = false;
      this._rollSeed();
    });
  }

  _rollSeed() {
    if (!this.els.seedInput) return;
    this.els.seedInput.value = String(Math.floor(Math.random() * 1e6));
    this._renderSeedMode();
  }

  _renderSeedMode() {
    if (this.els.seedMode) this.els.seedMode.textContent = this.seedPinned ? 'fixed' : 'new each run';
  }

  async init() {
    // The guide opens as a popup, from the card and from the footer's Guide link (Studio's own guide is linked in it).
    document.getElementById('rfd4p-guide-btn')?.addEventListener('click', () => void openGuide());
    document.getElementById('guide-link')?.addEventListener('click', (ev) => { ev.preventDefault(); void openGuide(); });
    this._initSeed();
    this._bindControls();
    await this._loadPreset(RFD4P_PRESETS[0].id, { syncStudio: false });
    this._initStudioBridge();
    this.designPanel = new DesignPanel(this);
    await this._initWebGpu();
    this._syncThinkingRadios('none');
    this.els.predictBtn.disabled = false;
    this.designPanel.render();
    const params = new URLSearchParams(window.location.search);
    // Without WebGPU a design runs on the CPU path on the main thread: hours for 60 steps, with the page frozen on
    // its last status. Say so instead; `?cpu=1` keeps the CPU path for development.
    this.noWebGpu = !this.gpuDevice && params.get('cpu') !== '1';
    const backendLabel = this.gpuDevice ? 'WebGPU FP16' : this.noWebGpu ? 'No WebGPU' : 'CPU Float32';
    if (this.els.webgpuBadge) {
      this.els.webgpuBadge.textContent = backendLabel;
    }
    this._initSpeedHint();
    if (this.noWebGpu) {
      this.els.predictBtn.disabled = true;
      this.setStatus(NO_WEBGPU_MESSAGE, true);
      return;
    }
    this.setStatus('Ready to generate.', false, { visible: false });

    if (params.get('autorun') === '1') {
      await this.runDesign();
    }
  }

  _initStudioBridge() {
    this._mountCardIntoStudio();
    // AtomWorks Studio runs in this same document and announces session changes as window events.
    window.addEventListener('atomworks-studio', (ev) => {
      const event = { data: ev.detail };
      if (['ready', 'changed', 'example'].includes(event.data?.type)) {
        this._mountCardIntoStudio();
        const sid = event.data.session_id;
        if (sid) {
          this.studioSessionId = sid;
          this._syncRefoldUi();
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
              if (this.noWebGpu) this.setStatus(NO_WEBGPU_MESSAGE, true);
              else this.setStatus('Ready to generate.', false, { visible: false });
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
    if (typeof window.atomworksStudio?.openSession !== 'function') {
      console.warn('AtomWorks Studio is not ready; cannot open session', sessionId);
      return;
    }
    try {
      await window.atomworksStudio?.openSession(sessionId, trajUrl);
      // Display-only reference motif and motif side chains beside a finished design (not part of the saved document).
      await showDesignOverlays(this.generatedViews.get(sessionId)?.overlays);
      this.designPanel?.showHotspots();
    } catch (err) {
      console.warn('Motif overlay failed:', err);
    }
  }

  /**
   * Desktop Chrome exposes subgroup matrices (about 2x faster designs, `hasSubgroupMatrix` in the engine) only behind
   * chrome://flags/#enable-unsafe-webgpu. Pages cannot open chrome:// URLs, so the hint offers to copy the address.
   */
  _initSpeedHint() {
    const hint = document.getElementById('rfd4p-speed-hint');
    if (!hint || !this.gpuDevice) return;
    const KEY = 'rfd4p.speedHintDismissed';
    let dismissed = false;
    try { dismissed = localStorage.getItem(KEY) === '1'; } catch { /* storage blocked: show it */ }
    const brands = navigator.userAgentData?.brands?.map((b) => b.brand) ?? [];
    const desktopChrome = brands.includes('Google Chrome') && navigator.userAgentData?.mobile === false;
    const hasMatrix = this.gpuDevice.features?.has?.('chromium-experimental-subgroup-matrix') === true;
    if (dismissed || !desktopChrome || hasMatrix) return;
    hint.hidden = false;
    const copy = document.getElementById('rfd4p-speed-copy');
    copy?.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText('chrome://flags/#enable-unsafe-webgpu');
        copy.textContent = 'Copied';
      } catch {
        copy.textContent = 'Copy failed: select the address above';
      }
    });
    document.getElementById('rfd4p-speed-dismiss')?.addEventListener('click', () => {
      hint.hidden = true;
      try { localStorage.setItem(KEY, '1'); } catch { /* not remembered; fine */ }
    });
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
    const typed = Number.parseInt(this.els.seedInput.value, 10);
    const seed = seedOverride ?? (Number.isFinite(typed) && typed >= 0 ? typed : 42);

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

    if (this.els.refoldBtn) {
      // Refold all, or Stop while refolds run.
      this.els.refoldBtn.onclick = () => { void this._refoldDesigns(null); };
    }

    if (this.els.downloadAllBtn) {
      this.els.downloadAllBtn.onclick = async () => {
        await this._saveResults();
      };
    }

    if (this.els.designViewRadios) {
      this.els.designViewRadios.onchange = (e) => {
        const input = e.target;
        if (input && input.name === 'design-view-radio' && input.checked) {
          const sid = input.value;
          this.studioSessionId = sid;
          this._syncRefoldUi();
          const viewInfo = this.generatedViews.get(sid);
          this._navigateStudioToSession(sid, {
            withTrajectory: Boolean(viewInfo?.hasTrajectory),
          });
        }
      };
    }
  }

  /**
   * "Save results (.zip)": the annotated input the run designed from, every design (final structure and trajectory),
   * every refold made of them, and a summary (web/results-zip.js).
   */
  async _saveResults() {
    const views = [...this.generatedViews.entries()];
    const input = views.find(([, v]) => v.isInput)?.[1];
    const store = getSharedStudioStore();
    const designs = views.filter(([, v]) => !v.isInput && !v.isRefold && !v.isSnapshot && v.cif)
      .map(([sid, v]) => ({
        index: v.designIndex ?? 1, label: v.label, seed: v.seed, stepsRun: v.stepsRun, nsteps: v.nsteps,
        cif: v.cif, pdb: v.pdb, trajectoryCif: store.getSession(sid)?.trajectoryCif ?? null,
        designedChains: v.designedChains, refolds: v.refolds ?? [],
      }))
      .sort((a, b) => a.index - b.index);
    if (!designs.length) return;
    const files = resultsFiles({
      input: input ? { cif: input.cif, name: input.name, revision: input.revision } : null,
      designs, page: window.location.href.split('?')[0],
    });
    const stamp = new Date().toISOString().slice(0, 16).replace(/[-:]/g, '').replace('T', '-');
    if (typeof window.JSZip !== 'function') {
      // No zip library (offline, or its CDN blocked): the summary at least, rather than nothing.
      const summary = files.find((f) => f.name === 'summary.json');
      this._downloadText(summary.text, `rfd4p-results-${stamp}.json`, 'application/json');
      return;
    }
    const zip = new window.JSZip();
    const folder = zip.folder(`rfd4p-results-${stamp}`);
    for (const f of files) folder.file(f.name, f.text);
    const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rfd4p-results-${stamp}.zip`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  /**
   * The Designs table: Input, then every design of the runs on this input (earlier runs folded away), each row its
   * refold's numbers, ✓ where upstream's pass criteria hold and greyed where they fail. A column head sorts by it;
   * by default rows rank by the job's first criterion, passing designs first, once there are numbers.
   */
  _renderDesigns() {
    const box = this.els.designViewRadios;
    if (!this.els.designResultsRow || !box) return;
    const views = this.generatedViews;
    // Runs belong to the input they were made from; another input opened shows none until it has its own.
    const runs = this.runsInput === this.inputSessionId ? (this.runs ?? []).filter((r) => r.designs.length) : [];
    const designs = runs.flatMap((r) => r.designs).map((sid) => views.get(sid)).filter(Boolean);
    const kind = designs.find((v) => v.kind != null)?.kind ?? null;
    const placed = kind === 'binder' || kind === 'antibody';
    const iface = placed || designs.some((v) => Number.isFinite(v.metrics?.iptm));
    const cols = [
      { key: placed ? 'placedRmsd' : 'rmsd', label: 'RMSD', dir: 1, fmt: (x) => x.toFixed(1),
        title: placed ? 'Å: the designed chain against its refold, fitted on the rest, so where it sits counts'
          : 'Å: the designed chains against their refold' },
      { key: 'plddt', label: 'pLDDT', dir: -1, fmt: (x) => x.toFixed(0), title: "The refold model's confidence, 0–100" },
      ...(iface ? [
        { key: 'iptm', label: 'ipTM', dir: -1, fmt: (x) => x.toFixed(2), title: "The refold model's confidence in the interface, 0–1" },
        { key: 'minIpae', label: 'iPAE', dir: 1, fmt: (x) => x.toFixed(1),
          title: 'Å: the lowest predicted error between a designed residue and the rest (min iPAE)' },
      ] : []),
    ];
    const value = (v, key) => (Number.isFinite(v.metrics?.[key]) ? v.metrics[key] : null);
    const sort = this.designSort ?? (designs.some((v) => v.metrics) ? { ...rankKey(kind), passFirst: true } : null);
    const verdict = (v) => (v.pass === true ? 0 : v.pass === false ? 2 : 1);
    const order = (sids) => {
      const list = sids.map((sid) => [sid, views.get(sid)]).filter(([, v]) => v);
      if (!sort) return list;
      return list.sort(([, a], [, b]) => {
        if (sort.passFirst && verdict(a) !== verdict(b)) return verdict(a) - verdict(b);
        const x = value(a, sort.key);
        const y = value(b, sort.key);
        if (x === null || y === null) return (x === null) - (y === null) || a.designIndex - b.designIndex;
        return (x - y) * sort.dir || a.designIndex - b.designIndex;
      });
    };
    // The name never shrinks below itself ("Design 12", not "Design…"); the numbers take what they need.
    const grid = `minmax(max-content, 1fr) 12px repeat(${cols.length}, 34px) 48px`;
    const busy = Boolean(this.refoldAbort);
    const head = h('div', { class: 'rfd4p-design-head', style: `grid-template-columns: ${grid}` },
      h('span'), h('span'),
      ...cols.map((c) => h('button', {
        type: 'button', class: `rfd4p-design-sort${sort?.key === c.key ? ' is-sorted' : ''}`, title: c.title,
        text: `${c.label}${sort?.key === c.key ? (sort.dir === c.dir ? ' ▾' : ' ▴') : ''}`,
        onclick: () => {
          this.designSort = sort?.key === c.key && !sort.passFirst ? { key: c.key, dir: -sort.dir } : { key: c.key, dir: c.dir };
          this._renderDesigns();
        },
      })),
      h('span'));
    const row = (sid, v, inputTitle = null) => {
      const cls = ['rfd4p-design-row', v.isInput ? 'rfd4p-design-input' : '',
        v.pass === true ? 'rfd4p-design-pass' : '', v.pass === false ? 'rfd4p-design-fail' : ''].filter(Boolean).join(' ');
      const isInputRow = v.isInput || v.isSnapshot;
      const label = h('label', { class: cls, style: `grid-template-columns: ${grid}`, title: isInputRow ? inputTitle : `seed ${v.seed}` });
      const radio = h('input', { type: 'radio', name: 'design-view-radio', value: sid });
      radio.checked = sid === this.studioSessionId;
      label.append(radio, h('span', { class: 'rfd4p-design-name', text: isInputRow ? 'Input' : `Design ${v.designIndex}` }),
        h('span', { class: 'rfd4p-design-mark', text: v.pass === true ? '✓' : '' }));
      const state = isInputRow ? '' : this.refoldSid === sid ? 'refolding…' : this.refoldQueue?.includes(sid) ? 'queued'
        : v.refoldError ? 'refold failed' : null;
      if (state !== null) label.append(h('span', { class: 'rfd4p-design-state', text: state, style: `grid-column: span ${cols.length}` }));
      else for (const c of cols) { const x = value(v, c.key); label.append(h('span', { class: 'rfd4p-design-num', text: x === null ? '' : c.fmt(x) })); }
      const stop = (fn) => (ev) => { ev.preventDefault(); ev.stopPropagation(); fn(); };
      label.append(h('span', { class: 'rfd4p-design-actions' },
        isInputRow || this.noWebGpu ? null : h('button', {
          type: 'button', class: 'rfd4p-design-act rfd4p-design-refold', text: '↻', disabled: busy,
          title: v.metrics ? 'Refold this design again' : 'Refold this design', 'aria-label': `Refold Design ${v.designIndex}`,
          onclick: stop(() => { void this._refoldDesigns([sid]); }),
        }),
        h('button', {
          type: 'button', class: 'rfd4p-design-act', text: '⤓', title: 'Download this structure (CIF)',
          'aria-label': `Download ${isInputRow ? 'the input' : `Design ${v.designIndex}`}`,
          onclick: stop(() => this._downloadText(v.cif, `${v.filenameStem || 'design'}.cif`, 'chemical/x-mmcif')),
        })));
      return label;
    };
    // Each run's own input first in its group: the input itself while it is unchanged since the newest run, else the
    // run's read-only copy. The input as it is now heads the table only once it has changed since that run.
    const input = views.get(this.inputSessionId);
    const revisionNow = getSharedStudioStore().getSession(this.inputSessionId)?.revision ?? null;
    const parts = [head];
    const newest = runs.at(-1);
    const unchanged = (r) => r === newest && r.inputRevision === revisionNow;
    if (input?.isInput && (!newest || !unchanged(newest))) parts.push(row(this.inputSessionId, input, 'The input as it is now (editable)'));
    const seedText = (r) => (r.seeds[0] === r.seeds[1] ? `seed ${r.seeds[0]}` : `seeds ${r.seeds[0]}–${r.seeds[1]}`);
    for (const [k, r] of runs.entries()) {
      const own = unchanged(r) && input?.isInput ? [this.inputSessionId, input, 'The input these designs were made from (editable)']
        : r.inputSnapshot && views.get(r.inputSnapshot) ? [r.inputSnapshot, views.get(r.inputSnapshot), `The input run ${r.index} was made from (read-only)`] : null;
      const rows = [own ? row(...own) : null, ...order(r.designs).map(([sid, v]) => row(sid, v))].filter(Boolean);
      if (runs.length === 1) { parts.push(...rows); continue; }
      const group = h('details', { class: 'rfd4p-run' },
        h('summary', { text: `Run ${r.index} · ${seedText(r)} · ${r.designs.length} design${r.designs.length === 1 ? '' : 's'}` }), ...rows);
      // Open or closed as the reader left it, across redraws (a refold redraws the table); until then, the newest run
      // and the one holding what is shown are open.
      this.runOpen ??= new Map();
      group.open = this.runOpen.has(r.index) ? this.runOpen.get(r.index)
        : k === runs.length - 1 || r.designs.includes(this.studioSessionId) || r.inputSnapshot === this.studioSessionId;
      group.addEventListener('toggle', () => this.runOpen.set(r.index, group.open));
      parts.push(group);
    }
    box.replaceChildren(...(designs.length ? parts : []));
    box.classList.toggle('rfd4p-designs-busy', Boolean(this.activeAbortController));
    if (this.els.downloadsBox) this.els.downloadsBox.hidden = designs.length < 1;
    this.els.designResultsRow.hidden = designs.length < 1;
    this._syncRefoldUi();
  }

  /** The refold controls: Refold all (Stop while refolds run), and what the refolds are doing. */
  _syncRefoldUi() {
    const any = (this.runs ?? []).some((r) => r.designs.length);
    if (this.els.refoldBox) this.els.refoldBox.hidden = !any || this.noWebGpu;
    if (this.els.refoldBtn) this.els.refoldBtn.textContent = this.refoldAbort ? 'Stop' : 'Refold all';
    if (this.els.refoldResult) {
      // The refolds' own news wins; otherwise say if the parameters are still coming down, because that wait is
      // minutes long on a slow link and used to show nothing at all.
      const text = this.refoldStatus ?? this._prefetchNote();
      this.els.refoldResult.textContent = text ?? '';
      this.els.refoldResult.classList.toggle('error', Boolean(this.refoldStatus && this.refoldStatusError));
      this.els.refoldResult.hidden = !text;
    }
  }

  _setRefoldStatus(text, isError = false) {
    this.refoldStatus = text;
    this.refoldStatusError = isError;
    this._syncRefoldUi();
  }

  /** Whether AlphaFold 3's weights terms have already been agreed to in this browser. */
  _af3TermsAccepted() {
    try { return localStorage.getItem('rfd4p.refoldTerms.af3') === 'accepted'; } catch { return false; }
  }

  /**
   * Pull the selected folding model's parameters ahead of the click, and say so.
   *
   * Two problems, one fix. The parameters are a few hundred MB and `refoldDesign` only starts fetching them when
   * Refold is pressed, so the whole transfer lands on someone sitting still — and it passes a no-op progress
   * callback to `loadAf3Weights`, so nothing is reported while it happens. `loadAf3Weights` memoises per family
   * and honours the progress callback of whoever calls it FIRST, so calling it here both starts the download
   * early and makes it the thing that reports. The later `refoldDesign` awaits the same promise.
   *
   * 🔴 ALPHAFOLD 3 IS NOT PREFETCHED UNTIL ITS TERMS ARE ACCEPTED. Its parameters carry Google DeepMind's
   * terms of use and prohibited-use policy, and `_agreeRefoldTerms` is where someone agrees to them. Fetching
   * them in the background first would take that decision away. OpenBind-0 is Apache 2.0 and has no such gate.
   */
  _prefetchFoldWeights() {
    const model = this.els.refoldModel?.value ?? 'openbind0';
    if (this.noWebGpu || !model || this.refoldAbort) return;
    if (model === 'af3' && !this._af3TermsAccepted()) return;
    if (this.weightPrefetch?.model === model) return;
    const name = REFOLD_MODELS[model] ?? model;
    const state = { model, name, loadedBytes: 0, totalBytes: 0, done: false };
    this.weightPrefetch = state;
    void (async () => {
      try {
        const { loadAf3Weights, FOLDERS } = await import('./localfold.js');
        await loadAf3Weights((progress) => {
          state.loadedBytes = progress.loadedBytes ?? 0;
          state.totalBytes = progress.totalBytes ?? 0;
          this._syncRefoldUi();
        }, FOLDERS[model] ?? model);
      } catch {
        // A failed prefetch costs nothing: the refold itself will try again and report properly.
      } finally {
        state.done = true;
        this._syncRefoldUi();
      }
    })();
  }

  /** What the Refold row says while parameters are still coming down, or nothing once they are here. */
  _prefetchNote() {
    const state = this.weightPrefetch;
    if (!state || state.done || !state.totalBytes) return null;
    const mb = (bytes) => (bytes / 1e6).toFixed(0);
    return `Downloading ${state.name} parameters — ${mb(state.loadedBytes)} of ${mb(state.totalBytes)} MB.`
      + ' First time only; it is kept for next time.';
  }

  /**
   * AlphaFold 3's parameters are Google DeepMind's, for non-commercial use under a prohibited-use policy: ask once per
   * browser, and offer OpenBind-0 (same architecture, Apache 2.0) as the way past rather than only through.
   * @returns {Promise<string|null>} the model to refold with, or null to cancel
   */
  async _agreeRefoldTerms(model) {
    const KEY = 'rfd4p.refoldTerms.af3';
    if (model !== 'af3') return model;
    try { if (localStorage.getItem(KEY) === 'accepted') return model; } catch { /* storage blocked: ask */ }
    const dialog = this.els.refoldTerms;
    if (!dialog?.showModal) return null;
    dialog.returnValue = '';
    dialog.showModal();
    await new Promise((resolve) => dialog.addEventListener('close', resolve, { once: true }));
    if (dialog.returnValue === 'accept') {
      try { localStorage.setItem(KEY, 'accepted'); } catch { /* asked again next time */ }
      return 'af3';
    }
    if (dialog.returnValue === 'switch') {
      if (this.els.refoldModel) this.els.refoldModel.value = 'openbind0';
      return 'openbind0';
    }
    return null;
  }

  /**
   * Refold designs one after another with the chosen model, scoring each (`refoldMetrics`, `passes`) and filling its
   * row as it finishes: `sids`, or every design not yet refolded with that model. Pressed again while refolds run,
   * it stops them.
   */
  async _refoldDesigns(sids) {
    if (this.refoldAbort) {
      this.refoldAbort.abort();
      return;
    }
    const model = await this._agreeRefoldTerms(this.els.refoldModel?.value ?? 'af3');
    if (!model) return;
    const name = REFOLD_MODELS[model] ?? model;
    const seed = Math.max(0, Number.parseInt(this.els.refoldSeed?.value ?? '0', 10) || 0);
    const all = (this.runs ?? []).flatMap((r) => r.designs);
    // Refold all skips designs already refolded with this model and this seed: a new seed samples them again.
    const todo = (sids ?? all.filter((sid) => !this.generatedViews.get(sid)?.refolds?.some((r) => r.model === model && (r.seed ?? 0) === seed)))
      .filter((sid) => this.generatedViews.get(sid)?.pdb);
    if (!todo.length) {
      this._setRefoldStatus(`Every design is refolded with ${name}, seed ${seed}.`);
      return;
    }
    const abort = new AbortController();
    this.refoldAbort = abort;
    this.refoldQueue = [...todo];
    let done = 0;
    let failed = 0;
    this._renderDesigns();
    try {
      if (!this.gpuDevice) await this._initWebGpu();
      if (!this.gpuDevice) throw new Error('WebGPU is not available');
      // ~6 MB of LocalFold code (tools/sync-localfold.mjs), loaded on the first refold only.
      const lf = await import('./localfold.js');
      // 🔴 ASK FOR THE PARAMETERS HERE, NOT INSIDE THE REFOLD, SO THE WAIT IS VISIBLE. `refoldDesign` calls
      // `loadAf3Weights` with a no-op progress callback, so a few hundred MB used to come down behind a status
      // line that never moved. The loader memoises per family and honours the FIRST caller's callback, so asking
      // here is what makes the transfer report; the call inside `refoldDesign` then awaits this same promise.
      await lf.loadAf3Weights((progress) => {
        const loaded = progress.loadedBytes ?? 0;
        const total = progress.totalBytes ?? 0;
        if (!total) return;
        this._setRefoldStatus(`${name}: downloading parameters — ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB.`
          + ' First time only; it is kept for next time.');
      }, lf.FOLDERS[model] ?? model);
      for (let k = 0; k < todo.length && !abort.signal.aborted; k += 1) {
        const sid = todo[k];
        const view = this.generatedViews.get(sid);
        this.refoldQueue = todo.slice(k + 1);
        this.refoldSid = sid;
        this._renderDesigns();
        const tag = todo.length > 1 ? ` (${k + 1} of ${todo.length})` : '';
        this._setRefoldStatus(`Refolding Design ${view.designIndex} with ${name}${tag}…`);
        const ok = await this._refoldOne(sid, model, seed, abort, lf,
          (text) => this._setRefoldStatus(`Design ${view.designIndex}${tag}: ${text}`));
        if (ok) done += 1;
        else if (!abort.signal.aborted) failed += 1;
      }
      const stopped = abort.signal.aborted ? ' Stopped.' : '';
      this._setRefoldStatus(`Refolded ${done} design${done === 1 ? '' : 's'} with ${name}.${failed ? ` ${failed} failed.` : ''}${stopped}`,
        failed > 0);
    } catch (err) {
      if (!abort.signal.aborted) console.error('Refold failed:', err);
      this._setRefoldStatus(abort.signal.aborted ? 'Refold stopped.' : `Refold failed: ${err?.message ?? err}`, !abort.signal.aborted);
    } finally {
      if (this.refoldAbort === abort) {
        this.refoldAbort = null;
        this.refoldSid = null;
        this.refoldQueue = [];
      }
      this._renderDesigns();
    }
  }

  /** One design's refold: scored, kept for "Save all", and drawn over the design. False if it failed. */
  async _refoldOne(sid, model, seed, abort, lf, status) {
    const view = this.generatedViews.get(sid);
    const name = REFOLD_MODELS[model] ?? model;
    try {
      const result = await lf.refoldDesign({
        pdb: view.pdb, chain: view.designedChains, model, device: this.gpuDevice, seed, signal: abort.signal,
        partial: view.partialTemplates ?? [],
        status: (text) => status(text.replace(/^refolding with \S+( · )?/, '') || `refolding with ${name}`),
      });
      view.metrics = refoldMetrics(result, view.designedChains, view.pdb);
      view.pass = passes(view.metrics, view.kind);
      view.refoldText = describeRefold(model, result, view.designedChains, view.partialTemplates ?? []);
      (view.refolds ??= []).push({
        model, seed, pdb: result.refold, cif: pdbToCif(result.refold), rmsd: result.rmsd, confidence: result.confidence,
        templated: result.templated, partial: result.partial, text: view.refoldText, metrics: view.metrics, pass: view.pass,
      });
      view.refoldError = false;
      const cif = refoldOverlayCif(result.refold);
      const ligandCif = refoldLigandCif(result.refold);
      view.overlays = { ...view.overlays, refold: cif ? { cif, ligandCif, label: `${name} refold` } : null };
      if (this.studioSessionId === sid) await showDesignOverlays(view.overlays);
      return true;
    } catch (err) {
      if (abort.signal.aborted) return false;
      view.refoldError = true;
      console.error(`Refold of Design ${view.designIndex} failed:`, err);
      return false;
    }
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
    if (this.noWebGpu) {
      this.setStatus(NO_WEBGPU_MESSAGE, true);
      return;
    }
    // A refold shares the GPU and shows designs this run is about to replace.
    this.refoldAbort?.abort();
    if (this.activeAbortController) {
      this.activeAbortController.abort();
    }
    const abortController = new AbortController();
    this.activeAbortController = abortController;
    const liveSequence = new LiveSequenceStrip();
    try {
      this.els.predictBtn.textContent = 'Stop';
      this.els.predictBtn.classList.add('secondary');
      this.els.predictBtn.disabled = false;

      const typedSeed = Number.parseInt(this.els.seedInput.value, 10);
      const baseSeed = Number.isFinite(typedSeed) && typedSeed >= 0 ? typedSeed : 42;
      const numSamples = Math.max(1, Math.min(8, Number.parseInt(this.els.samplesInput?.value ?? '1', 10) || 1));
      const nsteps = overrides.nsteps ?? 60;
      const recycleSchedule = overrides.recycleSchedule ?? this._getSelectedThinking();
      // Sampling temperature as upstream's Studio provider exposes it: TSR K per modality (lower = hotter).
      const backboneK = Math.max(1, Math.min(20, Number.parseFloat(this.els.backboneK?.value ?? '16') || 16));
      const latentK = Math.max(1, Math.min(6, Number.parseFloat(this.els.latentK?.value ?? '4') || 4));

      const firstBatchRes = this._buildCurrentBatch(baseSeed);

      const preset = this.currentPreset;
      // What to design comes from the input's annotations; partial diffusion is the one option on top. A backbone kept
      // with its sequence redesigned runs the inverse-folding sampler, as upstream's Space does for that task
      // (`generation=inverse_folding`): the backbone is kept exactly, only sequence and side chains are sampled.
      const partialPercent = Math.max(1, Math.min(100, Number.parseFloat(this.els.partialPercent?.value ?? '40') || 40));
      const partialTrajectoryFractionSteps = this.els.partialDiffusion?.checked ? partialPercent / 100 : null;
      const inverseFolding = partialTrajectoryFractionSteps == null && (this.useStudioSession
        ? this.designPanel?._spec?.()?.task?.key === 'inverse'
        : Boolean(preset?.inverseFolding));
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
      const earlyStops = [];
      // What the designs are judged by (web/refold-view.js criteriaKind), read before the run.
      const taskKey = this.useStudioSession ? this.designPanel?._spec?.()?.task?.key ?? null : null;
      // Hotspots stay marked while the design runs and on every design it makes. Each sample's keys come from its
      // own batch (below): new residues inside a chain renumber what follows them there, as in the design.
      liveSequence.start(hotspotResidueKeys(store.getSession(this.inputSessionId)?.atoms));
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
          name: inputDoc.name ?? null,
          revision: inputRevision,
        });
      }
      // Earlier runs on the same input stay, folded away under the new one; a different input starts over.
      if (this.runsInput !== this.inputSessionId || !this.runs) {
        for (const sid of this.generatedViews.keys()) if (sid !== this.inputSessionId) this.generatedViews.delete(sid);
        this.runs = [];
        this.designCounter = 0;
        this.designSort = null;
        this.runOpen = new Map();
        this.runsInput = this.inputSessionId;
      }
      const run = { index: (this.runs.at(-1)?.index ?? 0) + 1, seeds: [baseSeed, baseSeed + numSamples - 1], designs: [], inputRevision };
      // The input as this run had it, kept read-only beside its designs: edits after it do not change what it was.
      const inputNow = store.getSession(this.inputSessionId);
      if (inputNow) {
        const cif = inputNow.toCif();
        const { sessionId: snapshotSid } = store.createSession(cif, `input-run-${run.index}.cif`);
        this.generatedViews.set(snapshotSid, {
          label: 'Input', cif, pdb: null, isInput: false, isSnapshot: true, hasTrajectory: false,
          filenameStem: `input-run-${run.index}`, runIndex: run.index,
        });
        run.inputSnapshot = snapshotSid;
      }
      this.runs.push(run);
      this._renderDesigns();

      const t0 = performance.now();

      for (let sampleIdx = 0; sampleIdx < numSamples; sampleIdx += 1) {
        const sampleSeed = baseSeed + sampleIdx;
        const { batch, n, atoms } = sampleIdx === 0 ? firstBatchRes : this._buildCurrentBatch(sampleSeed);
        liveSequence.hotspots = hotspotResidueKeys(atoms);
        // Live frames number residues as the job does (res_id, past any new residues), not as the input is shown.
        this.liveHotspotKeys = liveSequence.hotspots;
        this.designPanel?.showHotspots();
        const tokens = extractTokenMetadata(batch);
        const fixedAnchors = extractFixedTokenAnchorsNm(batch, n);
        const allTokenIndices = Int32Array.from({ length: n }, (_, idx) => idx);
        const trajectoryFrames = [];
        const sampleTag = numSamples > 1 ? ` [Design ${sampleIdx + 1}/${numSamples}]` : '';

        this.setProgress((sampleIdx / numSamples) + 0.02, 'running');
        // The button says how far the run is; the status line is for what is not progress.
        this.setStatus(`Generating design${sampleTag} (${n} tokens, ${nsteps} steps, ${backendLabel})...`, false, { visible: false });

        let referenceCaNm = null;
        let liveFramesRendered = 0;
        let liveCamRadius = 0;
        let liveRenderPromise = null;

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
            liveSequence.update(structure.atoms);
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
          });

          const isStopping = info.step > 0 && (info.step === info.nsteps - 1 || Boolean(info.convergence?.stop));
          // On the GPU path only decoded frames are shown; the CA-only frame is for the CPU backend (no live decode).
          const frameReady = Boolean(info.decoded) || !useGpu;
          if (frameReady && !isStopping) {
            if (typeof window.atomworksStudio?.renderFrame === 'function') {
              if (liveRenderPromise) await liveRenderPromise;
              liveOverlays = { motif: overlayCifOf(motifOverlay, 'motif'), sideChains: null };
              const liveCif = liveAllAtomCif(info) ?? buildLiveFrameCifFromCa(alignedCaNm, tokens, null, UNIFIED_ATOM37_TOKENS);
              const frameOverlays = liveOverlays;
              if (liveCif) {
                // Early x1 estimates are a collapsed blob that later expands several-fold, so a camera
                // fit once to step 1 ends up inside the protein; refit whenever the extent drifts.
                const radius = caExtentRadiusNm(alignedCaNm);
                const keepCam = liveFramesRendered > 0 && radius < liveCamRadius * 1.3 && radius > liveCamRadius * 0.6;
                if (!keepCam) liveCamRadius = radius;
                liveFramesRendered += 1;
                liveRenderPromise = Promise.resolve(
                  window.atomworksStudio?.renderFrame(liveCif, `step_${info.step + 1}.cif`, keepCam),
                )
                  .then(() => showDesignOverlays(frameOverlays))
                  .catch(() => {});
              }
            }
          }

          const stepFrac = (info.step + 1) / (info.nsteps + 1);
          const overallFrac = (sampleIdx + stepFrac) / numSamples;
          this.setProgress(overallFrac, 'running');
          this.els.predictBtn.textContent = `Stop · ${numSamples > 1 ? `design ${sampleIdx + 1} of ${numSamples} · ` : ''}step ${info.step + 1}/${info.nsteps}`;
          this.setStatus(
            `Generating design${sampleTag} · step ${info.step + 1}/${info.nsteps} (${info.depth}x recycle)`, false, { visible: false },
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
        // Mol* draws cheaply while frames stream in, so it competes less with WebGPU for the GPU.
        window.atomworksStudio?.setLiveMode?.(true);
        try {
          result = useGpu
            ? await this.gpuModel.generate(batch, genOpts)
            : await generateRfd4pCpu(batch, this.activeWeights, this.activeConfig, genOpts);
        } finally {
          if (liveRenderPromise) {
            await liveRenderPromise;
          }
          window.atomworksStudio?.setLiveMode?.(false);
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
        const designIndex = (this.designCounter = (this.designCounter ?? 0) + 1);
        const designName = `design-${designIndex}.cif`;
        const { sessionId: designSessionId } = store.createSession(result.cif, designName, {
          trajectoryCif,
        });
        markHotspots(store.getSession(designSessionId), hotspotResidueKeys(atoms));
        const designed = designedChains(atoms, result.structure?.atoms ?? []);
        this.generatedViews.set(designSessionId, {
          label: `Design ${designIndex} · seed ${sampleSeed}`,
          cif: result.cif,
          pdb: result.pdb,
          isInput: false,
          hasTrajectory: Boolean(trajectoryCif),
          filenameStem: `design-${designIndex}-seed${sampleSeed}`,
          overlays,
          designedChains: designed,
          // Designed chains mostly held to the input (an antibody framework): templated on those residues in a refold.
          partialTemplates: partialTemplates(result.structure?.atoms ?? [], designed),
          // Where the motif residues are, so a refold's view can show its side chains there too.
          motifSlots: motifSlots(result.structure?.atoms ?? [], motifSideChainAtoms(result.structure?.atoms ?? [], unmodelled)),
          refoldText: null,
          // For "Save results": which design, how it ran, and every refold made of it.
          designIndex,
          runIndex: run.index,
          // Which of upstream's pass criteria apply: the task, and whether everything the refold folds was designed.
          kind: criteriaKind(taskKey, {
            designedAll: polymerChains(result.pdb ?? '').every((c) => designed.includes(c.letter)),
            hasLigand: /^HETATM/m.test(result.pdb ?? ''),
          }),
          seed: sampleSeed,
          stepsRun: result.stepsRun ?? null,
          nsteps: result.nsteps ?? nsteps,
          refolds: [],
        });
        run.designs.push(designSessionId);
        this._renderDesigns();
      }

      const elapsedS = ((performance.now() - t0) / 1000).toFixed(2);
      this.runCounter += 1;

      const firstDesign = run.designs[0];
      if (firstDesign) {
        this.studioSessionId = firstDesign;
        this._renderDesigns();
        await this._navigateStudioToSession(firstDesign, {
          withTrajectory: Boolean(this.generatedViews.get(firstDesign)?.hasTrajectory),
        });
      }

      this.setProgress(1, 'idle');
      // There is now something to refold, and the Refold button has just appeared. Start pulling the folding
      // model's parameters while the user reads their design, rather than when they click.
      this._prefetchFoldWeights();
      const stopNote = earlyStops.length ? ` Converged early at step ${earlyStops.join(', ')}.` : '';
      const seeds = numSamples > 1 ? `seeds ${baseSeed}–${baseSeed + numSamples - 1}` : `seed ${baseSeed}`;
      this.setStatus(`Generated from input revision ${inputRevision} with ${seeds} (${elapsedS} s on ${backendLabel}).${stopNote}`);
      if (!this.seedPinned) this._rollSeed();
    } catch (err) {
      this.setProgress(0, 'idle');
      try {
        await window.atomworksStudio?.showDocument?.();
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
        this.liveHotspotKeys = null;
        this.designPanel?.showHotspots();
      }
      this.els.predictBtn.textContent = 'Generate design';
      this.els.predictBtn.classList.remove('secondary');
      this.els.predictBtn.disabled = false;
      this.setProgress(0, 'idle');
      // A run stopped before its first design leaves nothing to list.
      if (this.runs) this.runs = this.runs.filter((r) => r.designs.length);
      this._renderDesigns();
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
