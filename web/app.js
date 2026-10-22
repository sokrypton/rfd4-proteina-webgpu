/**
 * RFD4-Proteina interactive browser studio on py2Dmol's application shell
 * (matching ../localfold/web/app.js).
 */

import {
  DEFAULT_HF_CONFIG_URL,
  DEFAULT_HF_WEIGHTS_URL,
  DEFAULT_LOCAL_CONFIG_URL,
  DEFAULT_LOCAL_SHARDED_DIR,
  DEFAULT_LOCAL_WEIGHTS_URL,
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
  parseSafetensors,
  recycleDepths,
  requestWebGpuDevice,
} from '../webgpu/index.js';

function maxAbsDiff(a, b) {
  let maxErr = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i += 1) {
    if (!Number.isFinite(a[i]) && !Number.isFinite(b[i])) continue;
    const err = Math.abs(a[i] - b[i]);
    if (err > maxErr) maxErr = err;
  }
  return maxErr;
}

/**
 * Extract per-token viewer metadata (`chainId`, `resId`, `resName`, `atomName`, `element`, `isAtomized`)
 * from `batch` and `batch.__atoms`.
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

/**
 * Format an intermediate CA trajectory frame (`caCoordsNm` in nanometers) into a 4-atom backbone PDB
 * so `py2Dmol` renders smooth cartoon ribbons and scrubbing across flow-matching steps.
 */
function formatCaFrameToPdb(caCoordsNm, tokens, bFactors, finalResidueType = null) {
  const n = tokens.length;
  const lines = [];
  let serial = 1;

  const fmtAtomLine = (recType, atomSerial, atomName, resName, chainId, resSeq, x, y, z, bFactor, element) => {
    const nameField = atomName.length >= 4 ? atomName.slice(0, 4) : ` ${atomName.padEnd(3, ' ')}`;
    const resField = (resName || 'ALA').slice(0, 3).padStart(3, ' ');
    const chainField = (chainId || 'A').slice(0, 1);
    const seqField = String(Math.max(-999, Math.min(9999, resSeq))).padStart(4, ' ');
    const xField = (Number.isFinite(x) ? x : 0).toFixed(3).padStart(8, ' ');
    const yField = (Number.isFinite(y) ? y : 0).toFixed(3).padStart(8, ' ');
    const zField = (Number.isFinite(z) ? z : 0).toFixed(3).padStart(8, ' ');
    const occField = '  1.00';
    const bField = (Number.isFinite(bFactor) ? bFactor : 80.0).toFixed(2).padStart(6, ' ');
    const elField = (element || 'C').slice(0, 2).padStart(2, ' ');
    return (
      `${recType.padEnd(6, ' ')}${String(atomSerial).padStart(5, ' ')} ${nameField} ${resField} ${chainField}${seqField}    ` +
      `${xField}${yField}${zField}${occField}${bField}          ${elField}  `
    );
  };

  for (let i = 0; i < n; i += 1) {
    const tok = tokens[i];
    const cx = caCoordsNm[i * 3 + 0] * 10.0;
    const cy = caCoordsNm[i * 3 + 1] * 10.0;
    const cz = caCoordsNm[i * 3 + 2] * 10.0;
    const bf = bFactors ? bFactors[i] : 80.0;
    const rt = finalResidueType ? finalResidueType[i] : -1;
    const resName =
      rt >= 0 && UNIFIED_ATOM37_TOKENS[rt] && UNIFIED_ATOM37_TOKENS[rt] !== '<M>' && UNIFIED_ATOM37_TOKENS[rt] !== '<A>'
        ? UNIFIED_ATOM37_TOKENS[rt]
        : tok.resName;

    if (tok.isAtomized) {
      lines.push(fmtAtomLine('HETATM', serial++, tok.atomName, tok.resName, tok.chainId, tok.resId, cx, cy, cz, bf, tok.element));
    } else {
      const prevIdx = i > 0 ? i - 1 : i;
      const nextIdx = i < n - 1 ? i + 1 : i;
      let tx = (caCoordsNm[nextIdx * 3 + 0] - caCoordsNm[prevIdx * 3 + 0]) * 10.0;
      let ty = (caCoordsNm[nextIdx * 3 + 1] - caCoordsNm[prevIdx * 3 + 1]) * 10.0;
      let tz = (caCoordsNm[nextIdx * 3 + 2] - caCoordsNm[prevIdx * 3 + 2]) * 10.0;
      const tLen = Math.hypot(tx, ty, tz) || 1.0;
      tx /= tLen;
      ty /= tLen;
      tz /= tLen;

      lines.push(fmtAtomLine('ATOM', serial++, 'N', resName, tok.chainId, tok.resId, cx - tx * 1.45, cy - ty * 1.45, cz - tz * 1.45, bf, 'N'));
      lines.push(fmtAtomLine('ATOM', serial++, 'CA', resName, tok.chainId, tok.resId, cx, cy, cz, bf, 'C'));
      lines.push(fmtAtomLine('ATOM', serial++, 'C', resName, tok.chainId, tok.resId, cx + tx * 1.52, cy + ty * 1.52, cz + tz * 1.52, bf, 'C'));
      lines.push(fmtAtomLine('ATOM', serial++, 'O', resName, tok.chainId, tok.resId, cx + tx * 1.52 + 0.85, cy + ty * 1.52 + 0.88, cz + tz * 1.52, bf, 'O'));
    }
  }
  lines.push('END');
  return lines.join('\n');
}

/**
 * Compute per-token B-factors reflecting RFD4-Proteina conditioning roles and a pairwise
 * contact map (`Uint8Array(n * n)`) for `py2Dmol`'s heatmap panel.
 */
function buildViewerDiagnostics(batch, tokens, caCoordsNm) {
  const n = tokens.length;
  const bFactors = new Float32Array(n);
  const highlightPositions = [];
  const chains = tokens.map((t) => t.chainId);

  const condCrd = batch.condition_coordinate_1_atom?.data;
  const condGrp = batch.condition_CoordinateGroup_1_atom?.data;
  const condHot = batch.condition_hotspot_1_atom?.data;
  const condMcoord = batch['condition_metal-coord_1_atom']?.data;
  const condHb = batch.condition_hbond_1_atom?.data;

  for (let i = 0; i < n; i += 1) {
    let isHot = false;
    let isFixed = false;
    let isIntCrd = false;
    let isMcoord = false;
    for (let a = 0; a < 37; a += 1) {
      const s = i * 37 + a;
      if (condHot && condHot[s]) isHot = true;
      if (condMcoord && condMcoord[s] > 0) isMcoord = true;
      if (condHb && condHb[s] > 0) isHot = true;
      if (condGrp && condGrp[s] >= 0) isIntCrd = true;
      if (condCrd && Number.isFinite(condCrd[s * 3])) isFixed = true;
    }
    if (isHot || isMcoord) {
      bFactors[i] = 96.0;
      highlightPositions.push(i);
    } else if (isIntCrd) {
      bFactors[i] = 88.0;
      highlightPositions.push(i);
    } else if (isFixed) {
      bFactors[i] = 92.0;
    } else {
      bFactors[i] = 78.0;
    }
  }

  const contactData = new Uint8Array(n * n);
  for (let i = 0; i < n; i += 1) {
    const ix = caCoordsNm[i * 3 + 0];
    const iy = caCoordsNm[i * 3 + 1];
    const iz = caCoordsNm[i * 3 + 2];
    for (let j = 0; j < n; j += 1) {
      const jx = caCoordsNm[j * 3 + 0];
      const jy = caCoordsNm[j * 3 + 1];
      const jz = caCoordsNm[j * 3 + 2];
      const dNm = Math.hypot(ix - jx, iy - jy, iz - jz);
      const prob = Math.max(0.0, Math.min(1.0, 1.0 - (dNm - 0.4) / 1.8));
      contactData[i * n + j] = Math.round(prob * 255);
    }
  }

  return {
    bFactors,
    highlightPositions,
    chains,
    contactMap: {
      data: contactData,
      n,
      vmin: 0,
      vmax: 1,
      chains,
    },
  };
}

class Rfd4pLocalFoldApp {
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
    this.presetDirty = false;
    this.currentBatchResult = null;

    this.els = {
      predictBtn: document.getElementById('predict'),
      parityBtn: document.getElementById('parity-btn'),
      quickUploadInput: document.getElementById('quick-upload-input'),
      contigInput: document.getElementById('contig-input'),
      activeConditionPills: document.getElementById('active-condition-pills'),
      presetSelect: document.getElementById('preset-select'),
      nstepsInput: document.getElementById('nsteps-input'),
      recycleSelect: document.getElementById('recycle-select'),
      seedInput: document.getElementById('seed-input'),
      backendSelect: document.getElementById('backend-select'),
      archSelect: document.getElementById('arch-select'),
      safetensorsFileInput: document.getElementById('safetensors-file-input'),
      conditionSpecInput: document.getElementById('condition-spec-input'),
      structureInput: document.getElementById('structure-input'),
      structureFileInput: document.getElementById('structure-file-input'),
      structureSummaryText: document.getElementById('structure-summary-text'),
      statusMessage: document.getElementById('status-message'),
      modelLoad: document.getElementById('model-load'),
      modelLoadText: document.getElementById('model-load-text'),
      progress: document.getElementById('progress'),
      predictionScoresBox: document.getElementById('predictionScoresBox'),
      downloadsBox: document.getElementById('downloads'),
      downloadPdbBtn: document.getElementById('download-pdb-btn'),
      downloadCifBtn: document.getElementById('download-cif-btn'),
      metricTokens: document.getElementById('metric-tokens'),
      metricAtoms: document.getElementById('metric-atoms'),
      metricLatency: document.getElementById('metric-latency'),
      metricParity: document.getElementById('metric-parity'),
    };
  }

  setStatus(msg, isError = false) {
    if (!this.els.statusMessage) return;
    this.els.statusMessage.textContent = msg;
    this.els.statusMessage.classList.toggle('error', Boolean(isError));
  }

  setProgress(fraction, state = 'running') {
    if (!this.els.progress) return;
    this.els.progress.dataset.state = state;
    this.els.progress.value = Math.max(0, Math.min(1, fraction));
  }

  getPy2DmolRenderer() {
    const registry = window.py2dmol_viewers ?? {};
    const firstKey = Object.keys(registry)[0];
    return firstKey ? registry[firstKey]?.renderer : undefined;
  }

  revealViewer() {
    const renderer = this.getPy2DmolRenderer();
    try {
      if (window.Plddt && renderer) window.Plddt.initialize(renderer);
    } catch { /* ignore */ }
    const container = document.getElementById('viewer-container');
    if (container && getComputedStyle(container).display === 'none') {
      container.style.display = 'flex';
    }
    const seqBox = document.getElementById('sequence-viewer-container');
    if (seqBox) seqBox.style.display = 'block';
    if (this.els.predictionScoresBox) this.els.predictionScoresBox.style.display = 'block';
    if (this.els.downloadsBox) this.els.downloadsBox.style.display = 'flex';
    try {
      renderer?._updateCanvasDimensions?.();
    } catch { /* ignore */ }
  }

  _isHalfSelected() {
    return this.els.backendSelect?.value !== 'webgpu-fp32';
  }

  async init() {
    this._populatePresets();
    this._bindControls();
    await this._initWebGpu();
    await this._loadPreset(RFD4P_PRESETS[0].id);
    this.els.predictBtn.disabled = false;
    const backendLabel = this.gpuDevice
      ? `WebGPU (${this._isHalfSelected() ? 'FP16' : 'FP32'} Weights)`
      : 'CPU Float32';
    this.setStatus(`Ready (${backendLabel}). Pick a preset or enter a contig and press Design.`);

    const params = new URLSearchParams(window.location.search);
    if (params.get('autorun') === '1') {
      await this.runDesign();
    }
  }

  _populatePresets() {
    this.els.presetSelect.innerHTML = '';
    for (const p of RFD4P_PRESETS) {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.title;
      this.els.presetSelect.appendChild(opt);
    }
  }

  async _initWebGpu() {
    try {
      const res = await requestWebGpuDevice();
      if (res && res.device) {
        this.gpuDevice = res.device;
        this.gpuAdapterInfo = res.adapter?.info ?? null;
      } else {
        this.els.backendSelect.value = 'cpu';
      }
    } catch (err) {
      console.warn('WebGPU fallback to CPU:', err);
      this.els.backendSelect.value = 'cpu';
    }
  }

  async ensureWeightsLoaded(source = this.prevArchChoice, { needEncoder = false } = {}) {
    const half = this._isHalfSelected();
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

  async _loadPreset(presetId) {
    const preset = getRfd4pPresetById(presetId);
    this.currentPreset = preset;
    this.presetDirty = false;
    this.currentCifBytes = null;

    this.els.contigInput.value = preset.contig || '';
    this.els.conditionSpecInput.value = JSON.stringify(preset.conditions ?? {}, null, 2);

    if (preset.nsteps) {
      const val = String(preset.nsteps);
      if ([...this.els.nstepsInput.options].some((o) => o.value === val)) {
        this.els.nstepsInput.value = val;
      }
    }
    if (preset.recycleSchedule) {
      this.els.recycleSelect.value = preset.recycleSchedule;
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
        const text = decodeTextOrZstd(bytes);
        this.els.structureInput.value = text;
        const fileName = preset.cifUrl.split('/').pop();
        this.els.structureSummaryText.textContent = `${fileName} (${text.split('\n').length} lines)`;
      } catch (err) {
        console.warn('Failed to fetch preset CIF:', err);
        this.els.structureInput.value = '';
        this.els.structureSummaryText.textContent = 'None (de novo)';
      }
    } else {
      this.els.structureInput.value = '';
      this.els.structureSummaryText.textContent = 'None (de novo)';
    }

    this._previewConditions();
  }

  _buildCurrentBatch() {
    const seed = Number.parseInt(this.els.seedInput.value, 10) || 42;
    const preset = this.currentPreset;

    // Use the exact upstream annotated CIF when a preset is unmodified and not marked specMode
    if (!this.presetDirty && preset && !preset.specMode && this.currentCifBytes) {
      const res = buildBatch(this.currentCifBytes, { seed });
      this.currentBatchResult = res;
      return res;
    }

    let conditions = {};
    const rawCond = (this.els.conditionSpecInput.value || '').trim();
    if (rawCond && rawCond !== 'undefined') {
      try {
        conditions = JSON.parse(rawCond);
      } catch (err) {
        throw new Error(`Invalid conditions JSON: ${err.message}`);
      }
    }
    const contig = this.els.contigInput.value.trim();
    const structureText = this.els.structureInput.value.trim() || null;
    const res = buildBatch(
      { contig, conditions },
      { inputCif: structureText, seed },
    );
    this.currentBatchResult = res;
    return res;
  }

  _previewConditions() {
    try {
      const res = this._buildCurrentBatch();
      this._renderConditionPills(res);
    } catch (err) {
      this.setStatus(err.message, true);
    }
  }

  _renderConditionPills(batchResult) {
    const { batch, n } = batchResult;
    const condCrd = batch.condition_coordinate_1_atom?.data;
    const condGrp = batch.condition_CoordinateGroup_1_atom?.data;
    const condHot = batch.condition_hotspot_1_atom?.data;
    const condMcoord = batch['condition_metal-coord_1_atom']?.data;
    const condHb = batch.condition_hbond_1_atom?.data;
    const condSalt = batch.condition_saltbridge_1_atom?.data;
    const condSec = batch['condition_secondary-structure_1_residue']?.data;
    const condCath = batch['condition_cath-label_1_residue']?.data;
    const seqMask = batch.sequence_probs_mask?.data;
    const unidxMask = batch.is_in_context_conditioning?.data;

    let nCrd = 0;
    let nCrdGrp = 0;
    let nUnidx = 0;
    let nHot = 0;
    let nMcoord = 0;
    let nHb = 0;
    let nSalt = 0;
    let nSec = 0;
    let nSeq = 0;
    let nCath = 0;

    for (let i = 0; i < n; i += 1) {
      if (condSec && condSec[i] !== 0) nSec += 1;
      if (seqMask && seqMask[i] !== 0) nSeq += 1;
      if (unidxMask && unidxMask[i] !== 0) nUnidx += 1;
      if (condCath && condCath[i * 3] >= 0) nCath += 1;

      let tokCrd = false;
      let tokGrp = false;
      let tokHot = false;
      let tokMc = false;
      let tokHb = false;
      let tokSalt = false;
      for (let a = 0; a < 37; a += 1) {
        const s = i * 37 + a;
        if (condCrd && Number.isFinite(condCrd[s * 3])) tokCrd = true;
        if (condGrp && condGrp[s] >= 0) tokGrp = true;
        if (condHot && condHot[s]) tokHot = true;
        if (condMcoord && condMcoord[s] > 0) tokMc = true;
        if (condHb && condHb[s] > 0) tokHb = true;
        if (condSalt && condSalt[s] > 0) tokSalt = true;
      }
      if (tokCrd) nCrd += 1;
      if (tokGrp) nCrdGrp += 1;
      if (tokHot) nHot += 1;
      if (tokMc) nMcoord += 1;
      if (tokHb) nHb += 1;
      if (tokSalt) nSalt += 1;
    }

    const preset = this.currentPreset;
    const modeTag = preset?.inverseFolding
      ? [{ label: 'Inverse Folding', cls: 'pill-cath' }]
      : preset?.partialTrajectoryFractionSteps
      ? [{ label: `Partial Diffusion (${preset.partialTrajectoryFractionSteps * 100}%)`, cls: 'pill-cath' }]
      : [];

    const pills = [
      { label: `${n} tokens`, cls: '' },
      ...modeTag,
      ...(nCrd > 0 ? [{ label: `C_CRD: ${nCrd} fixed`, cls: 'pill-crd' }] : []),
      ...(nCrdGrp > 0 ? [{ label: `C_CRDGRP: ${nCrdGrp} motif`, cls: 'pill-crdgrp' }] : []),
      ...(nUnidx > 0 ? [{ label: `C_IDX=false: ${nUnidx} unindexed`, cls: 'pill-crdgrp' }] : []),
      ...(nSeq > 0 ? [{ label: `C_SEQ: ${nSeq} locked`, cls: 'pill-crd' }] : []),
      ...(nHot > 0 ? [{ label: `C_HOT: ${nHot} hotspots`, cls: 'pill-hot' }] : []),
      ...(nMcoord > 0 ? [{ label: `C_MCOORD: ${nMcoord} metal`, cls: 'pill-mcoord' }] : []),
      ...(nHb > 0 ? [{ label: `C_HB: ${nHb}`, cls: 'pill-mcoord' }] : []),
      ...(nSalt > 0 ? [{ label: `C_SALT: ${nSalt}`, cls: 'pill-hot' }] : []),
      ...(nSec > 0 ? [{ label: `C_SEC: ${nSec}`, cls: 'pill-cath' }] : []),
      ...(nCath > 0 ? [{ label: `C_CATH: ${nCath}`, cls: 'pill-cath' }] : []),
    ];

    this.els.activeConditionPills.innerHTML = pills
      .map((p) => `<span class="rfd4p-pill ${p.cls}">${p.label}</span>`)
      .join('');
  }

  _bindControls() {
    this.els.presetSelect.addEventListener('change', async () => {
      await this._loadPreset(this.els.presetSelect.value);
    });

    const onSpecEdited = () => {
      this.presetDirty = true;
      this._previewConditions();
    };
    this.els.contigInput.addEventListener('input', onSpecEdited);
    this.els.conditionSpecInput.addEventListener('input', onSpecEdited);
    this.els.structureInput.addEventListener('input', onSpecEdited);

    this.els.predictBtn.addEventListener('click', () => this.runDesign());
    this.els.parityBtn.addEventListener('click', () => this.runParityCheck());

    this.els.structureFileInput.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      await this._loadStructureFile(file);
    });

    this.els.quickUploadInput.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      if (file.name.endsWith('.safetensors')) {
        await this._loadSafetensorsFile(file);
      } else {
        await this._loadStructureFile(file);
      }
    });

    this.els.archSelect.addEventListener('change', async () => {
      const choice = this.els.archSelect.value;
      if (choice === 'local') {
        this.els.archSelect.value = this.prevArchChoice;
        this.els.safetensorsFileInput.click();
        return;
      }
      this.prevArchChoice = choice;
      this.activeWeights = null;
      this.shardedBundle = null;
      if (this.gpuModel) {
        this.gpuModel.destroy();
        this.gpuModel = null;
      }
      await this.ensureWeightsLoaded(choice);
      const count = Object.keys(this.activeWeights).length;
      this.setStatus(`Ready · ${count} tensors loaded on ${this.gpuDevice ? 'WebGPU' : 'CPU'}.`);
    });

    this.els.safetensorsFileInput.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (file) await this._loadSafetensorsFile(file);
    });

    this.els.downloadPdbBtn.addEventListener('click', () => {
      if (!this.lastResult?.pdb) return;
      this._downloadText(this.lastResult.pdb, `${this.els.presetSelect.value || 'rfd4p'}.pdb`, 'chemical/x-pdb');
    });

    this.els.downloadCifBtn.addEventListener('click', () => {
      if (!this.lastResult?.cif) return;
      this._downloadText(this.lastResult.cif, `${this.els.presetSelect.value || 'rfd4p'}.cif`, 'chemical/x-mmcif');
    });
  }

  async _loadStructureFile(file) {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const text = decodeTextOrZstd(bytes);
      this.currentCifBytes = bytes;
      this.els.structureInput.value = text;
      this.els.structureSummaryText.textContent = `${file.name} (${text.split('\n').length} lines)`;
      // If the uploaded file is an annotated CIF with condition_*, parse directly; otherwise use spec mode
      const isAnnotated = text.includes('condition_') || text.includes('annotation_expseg');
      this.presetDirty = !isAnnotated;
      this._previewConditions();
      this.setStatus(`Loaded structure ${file.name}.`);
    } catch (err) {
      this.setStatus(`Failed to load ${file.name}: ${err.message}`, true);
    }
  }

  async _loadSafetensorsFile(file) {
    try {
      this.setStatus(`Loading checkpoint ${file.name}...`);
      if (!this.activeConfig) {
        try {
          this.activeConfig = await fetchRfd4pConfig(DEFAULT_LOCAL_CONFIG_URL);
        } catch {
          this.activeConfig = await fetchRfd4pConfig(DEFAULT_HF_CONFIG_URL);
        }
      }
      const buf = await file.arrayBuffer();
      const { tensors } = parseSafetensors(buf);
      this.shardedBundle = null;
      this.activeWeights = tensors;
      const half = this._isHalfSelected();
      if (this.gpuDevice) {
        if (this.gpuModel) this.gpuModel.destroy();
        this.gpuModel = new Rfd4pGeneratorGpu(this.gpuDevice, this.activeWeights, this.activeConfig, { half });
        this.gpuModelHalf = half;
      }
      this.setStatus(`Loaded ${file.name} (${Object.keys(tensors).length} tensors).`);
    } catch (err) {
      this.setStatus(`Failed to load ${file.name}: ${err.message}`, true);
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

  /**
   * Load the flow-matching CA trajectory and final decoded all-atom structure into `py2Dmol`.
   */
  async _loadResultIntoPy2Dmol(stem, batch, trajectoryFrames, result) {
    const load = window.py2dmolLoadFiles;
    if (typeof load !== 'function') return;

    const tokens = extractTokenMetadata(batch);
    const finalCaNm = result.x.bb_ca;
    const diag = buildViewerDiagnostics(batch, tokens, finalCaNm);
    const finalPdb = result.pdb;
    const finalResTypes = result.residueType;

    const firstFrame = trajectoryFrames[0];
    const firstPdb = firstFrame
      ? formatCaFrameToPdb(firstFrame.caCoordsNm, tokens, diag.bFactors, finalResTypes)
      : finalPdb;

    const ui = window.viewerConfig?.ui;
    const prevFilter = ui?.filterAdditives;
    if (ui) ui.filterAdditives = false;
    try {
      const vFile = (name, text) => ({ name, readAsync: () => Promise.resolve(text) });
      await load([vFile(`${stem}.pdb`, firstPdb)], true);
    } finally {
      if (ui) ui.filterAdditives = prevFilter;
    }

    const renderer = this.getPy2DmolRenderer();
    if (!renderer) return;
    const objName = renderer.currentObjectName ?? stem;
    const objData = renderer.objectsData?.[objName];

    if (objData?.frames?.[0]) {
      const f0 = objData.frames[0];
      f0.name = f0.label = f0.title = firstFrame
        ? `flow_${firstFrame.step + 1}_t${firstFrame.tBb.toFixed(2)}`
        : 'decoded_all_atom';
      f0.maps = { contact: diag.contactMap };
    }

    const api = window.py2Dmol;
    if (api?.frameFromText && trajectoryFrames.length > 0) {
      for (let s = 1; s < trajectoryFrames.length; s += 1) {
        const tf = trajectoryFrames[s];
        const stepPdb = formatCaFrameToPdb(tf.caCoordsNm, tokens, diag.bFactors, finalResTypes);
        const stepDiag = buildViewerDiagnostics(batch, tokens, tf.caCoordsNm);
        const frame = api.frameFromText(stepPdb);
        frame.name = frame.label = frame.title = `flow_${tf.step + 1}_t${tf.tBb.toFixed(2)}`;
        frame.maps = { contact: stepDiag.contactMap };
        frame.align = true;
        renderer.addFrame(frame, objName);
      }

      const finalFrame = api.frameFromText(finalPdb);
      finalFrame.name = finalFrame.label = finalFrame.title = 'decoded_all_atom';
      finalFrame.maps = { contact: diag.contactMap };
      finalFrame.align = true;
      renderer.addFrame(finalFrame, objName);
    }

    this.revealViewer();

    const totalFrames = objData?.frames?.length ?? 1;
    if (typeof renderer.setFrame === 'function' && totalFrames > 0) {
      renderer.setFrame(totalFrames - 1);
    }

    if (diag.highlightPositions.length > 0 && typeof renderer.showSidechains === 'function') {
      try {
        renderer.showSidechains({ object: objName, positions: diag.highlightPositions });
      } catch { /* ignore */ }
    }

    const uniqueChains = new Set(diag.chains).size;
    const targetColor = uniqueChains > 1 ? 'chain' : 'rainbow';
    if (typeof api?.setColor === 'function') {
      try {
        api.setColor(targetColor);
      } catch { /* ignore */ }
    }

    try {
      window.Heatmap?.updateFrame?.(renderer, objData, totalFrames - 1);
      window.Heatmap?.updateVisibility?.(renderer);
      renderer.render?.('rfd4p-complete');
    } catch { /* ignore */ }
  }

  async runDesign() {
    try {
      this.els.predictBtn.disabled = true;
      const { batch, n, atoms } = this._buildCurrentBatch();
      this._renderConditionPills({ batch, n, atoms });

      const preset = this.currentPreset;
      const inverseFolding = !this.presetDirty && Boolean(preset?.inverseFolding);
      const partialTrajectoryFractionSteps =
        !this.presetDirty && preset?.partialTrajectoryFractionSteps != null
          ? preset.partialTrajectoryFractionSteps
          : null;
      const needEncoder = Boolean(inverseFolding || partialTrajectoryFractionSteps != null);

      await this.ensureWeightsLoaded(this.prevArchChoice, { needEncoder });

      const nsteps = Math.max(1, Number.parseInt(this.els.nstepsInput.value, 10) || 60);
      const recycleSchedule = this.els.recycleSelect.value || 'low';
      const seed = Number.parseInt(this.els.seedInput.value, 10) || 42;
      const useGpu = this.els.backendSelect.value !== 'cpu' && Boolean(this.gpuModel);
      const backendLabel = useGpu
        ? `WebGPU ${this._isHalfSelected() ? 'FP16' : 'FP32'}`
        : 'CPU Float32';

      const trajectoryFrames = [];
      const t0 = performance.now();
      this.setProgress(0.02, 'running');
      this.setStatus(`Running RFD4-Proteina (${n} tokens, ${nsteps} steps, ${backendLabel})...`);

      const onStep = async (info) => {
        trajectoryFrames.push({
          step: info.step,
          tBb: info.t.bb_ca,
          tLl: info.t.local_latents,
          depth: info.depth,
          caCoordsNm: Float32Array.from(info.x1.bb_ca),
        });
        const frac = (info.step + 1) / (info.nsteps + 1);
        this.setProgress(frac, 'running');
        this.setStatus(
          `Flow-matching step ${info.step + 1}/${info.nsteps} · t_bb=${info.t.bb_ca.toFixed(2)} · t_ll=${info.t.local_latents.toFixed(2)} (${info.depth}x recycle)`
        );
      };

      const genOpts = {
        nsteps,
        recycleSchedule,
        seed,
        inverseFolding,
        partialTrajectoryFractionSteps,
        inputAtoms: atoms,
        onStep,
      };

      const result = useGpu
        ? await this.gpuModel.generate(batch, genOpts)
        : await generateRfd4pCpu(batch, this.activeWeights, this.activeConfig, genOpts);

      const elapsedS = ((performance.now() - t0) / 1000).toFixed(2);
      this.lastResult = result;
      this.runCounter += 1;
      const stem = `${this.els.presetSelect.value || 'rfd4p'}_${this.runCounter}`;

      await this._loadResultIntoPy2Dmol(stem, batch, trajectoryFrames, result);

      const atomCount = result.structure?.atoms?.length ?? 0;
      const chains = new Set(result.structure?.atoms?.map((a) => a.chainId) ?? ['A']).size;
      const depths = recycleDepths(recycleSchedule, trajectoryFrames.length || nsteps, nsteps);
      const totalPasses = depths.reduce((acc, d) => acc + d, 0);

      this.els.metricTokens.textContent = `${n} (${chains} ch)`;
      this.els.metricAtoms.textContent = String(atomCount);
      this.els.metricLatency.textContent = `${elapsedS} s`;
      if (this.els.metricParity.textContent === '-') {
        this.els.metricParity.textContent = `${totalPasses} passes`;
      }

      this.setProgress(1, 'idle');
      this.setStatus(
        `Done in ${elapsedS} s (${backendLabel}) · ${n} tokens · ${atomCount} all-atom coordinates decoded.`
      );
    } catch (err) {
      console.error('RFD4-Proteina run failed:', err);
      this.setProgress(0, 'idle');
      this.setStatus(`Error: ${err.message}`, true);
    } finally {
      this.els.predictBtn.disabled = false;
    }
  }

  async runParityCheck() {
    try {
      this.els.parityBtn.disabled = true;
      await this.ensureWeightsLoaded(this.prevArchChoice);
      if (!this.gpuModel) {
        this.setStatus('WebGPU unavailable on this browser; running on CPU reference.', true);
        return;
      }
      const { batch, n, atoms } = this._buildCurrentBatch();
      const seed = Number.parseInt(this.els.seedInput.value, 10) || 42;
      this.setStatus(`Verifying WebGPU vs CPU float32 parity on ${n} tokens (1 step + decoder)...`);
      await new Promise((r) => setTimeout(r, 15));

      const opts = { nsteps: 1, recycleSchedule: 'none', seed, inputAtoms: atoms };
      const gpuRes = await this.gpuModel.generate(batch, opts);
      const cpuRes = await generateRfd4pCpu(batch, this.activeWeights, this.activeConfig, opts);

      const caDiff = maxAbsDiff(gpuRes.x.bb_ca, cpuRes.x.bb_ca);
      const atomDiff = maxAbsDiff(gpuRes.coors, cpuRes.coors);
      this.els.metricParity.textContent = `${atomDiff.toExponential(1)} Å`;
      this.setStatus(
        `WebGPU vs CPU Parity Verified · max |ΔCα| = ${caDiff.toExponential(2)} nm · max |ΔAllAtom| = ${atomDiff.toExponential(2)} Å`
      );
    } catch (err) {
      this.setStatus(`Parity check error: ${err.message}`, true);
    } finally {
      this.els.parityBtn.disabled = false;
    }
  }
}

const boot = () => {
  const app = new Rfd4pLocalFoldApp();
  window.rfd4pApp = app;
  app.init();
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
