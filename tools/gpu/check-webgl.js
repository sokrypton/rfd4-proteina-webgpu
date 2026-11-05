/**
 * The WebGL2 fallback (`webgl/rfd4p/model.js`) vs the PyTorch oracle, with the released weights: the trunk's network
 * outputs at a full recycle depth, the decoder and the encoder. Runs in any browser with WebGL2 (the WebGPU device
 * the runner passes is not used).
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-webgl.js [--case=uncond_short] [--parts=trunk,decoder,encoder]
 */

import { sequenceForDecoder } from '../../cpu/rfd4p/decoder.js';
import { pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures } from '../../shared/rfd4p/features.js';
import { Rfd4pModelGL } from '../../webgl/rfd4p/model.js';
import { loadOracleCase, loadReleased } from './check-trunk.js';

function compare(got, want) {
  let se = 0;
  let sw = 0;
  let maxAbs = 0;
  let bad = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    if (!Number.isFinite(got[i])) bad += 1;
    se += d * d;
    sw += want[i] * want[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return { relRms: Number(Math.sqrt(se / Math.max(sw, 1e-30)).toExponential(2)), maxAbs: Number(maxAbs.toExponential(2)), nonFinite: bad };
}

export async function main(_device, args) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const caseName = opt.case ?? 'uncond_short';
  const parts = (opt.parts ?? 'trunk,decoder,encoder').split(',');
  const { cfg, weights } = await loadReleased();
  const model = new Rfd4pModelGL(weights);
  const results = { renderer: model.ctx.renderer };
  // Weights are float16 on the GPU, as the WebGPU path's half mode holds them: its tolerance.
  const tol = 2e-2;

  if (parts.includes('trunk')) {
    const { meta, t } = await loadOracleCase(caseName);
    const nnCfg = cfg.nn;
    const batch = {};
    for (const [key, v] of Object.entries(t)) if (key.startsWith('batch.')) batch[key.slice(6)] = v;
    const n = batch.padding_mask.shape[1];
    const xt = { bb_ca: t['step.0.x_t.bb_ca'].data, local_latents: t['step.0.x_t.local_latents'].data };
    const tt = { bb_ca: t['step.0.t.bb_ca'].data[0], local_latents: t['step.0.t.local_latents'].data[0] };
    const draws = meta.rng.filter((r) => r.fn === 'randperm' && r.call === 0).map((r) => t[`rng.${r.i}`].data);
    let d = 0;
    const randperm = () => draws[d++];
    const f = {
      seq: trunkSeqFeatures(batch, { n, xt, feats: nnCfg.feats_seq, weights, randperm, noiseAlignChirality: nnCfg.noise_align_chirality }).data,
      cond0: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: null, feats: nnCfg.feats_cond_seq, weights }).data,
      cond1: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: 1, feats: nnCfg.feats_cond_seq, weights }).data,
      pair: pairFeatures(batch, { n, caNm: xt.bb_ca, feats: nnCfg.feats_pair_repr,
        dist: { dim: nnCfg.xt_pair_dist_dim, min: nnCfg.xt_pair_dist_min, max: nnCfg.xt_pair_dist_max }, seqSepDim: nnCfg.seq_sep_dim }).data,
      pairCond: pairCondFeatures(tt, nnCfg.feats_pair_cond),
    };
    const depth = opt.depth ? Number(opt.depth) : meta.steps[0].n_recycle;
    if (opt.profile === '1') { await model.forwardTrunk(n, f, batch.padding_mask.data, nnCfg, 1); model.ctx.profile = new Map(); model.ctx.lastT = performance.now(); }
    const t0 = performance.now();
    const out = await model.forwardTrunk(n, f, batch.padding_mask.data, nnCfg, depth, { insertion: true });
    const ms = Math.round(performance.now() - t0);
    const ctx = batch.is_in_context_conditioning.data;
    const lat = batch.supports_local_latent.data;
    for (let i = 0; i < n; i += 1) {
      if (ctx[i]) out.bb_ca.fill(0, i * 3, i * 3 + 3);
      if (ctx[i] || !lat[i]) out.local_latents.fill(0, i * 32, i * 32 + 32);
    }
    const checks = {
      bb_ca: compare(out.bb_ca, t['step.0.out.bb_ca.v'].data),
      latents: compare(out.local_latents, t['step.0.out.local_latents.v'].data),
      logits: compare(out.logits, t['step.0.out.sequence.logits'].data),
      ...(t['step.0.out.insertion.logits'] && out.insertion ? { insertion: compare(out.insertion, t['step.0.out.insertion.logits'].data) } : {}),
    };
    if (model.ctx.profile) {
      results.profile = [...model.ctx.profile].sort((a, b) => b[1] - a[1]).slice(0, 14).map(([k, v]) => `${Math.round(v)} ms ${k}`);
      model.ctx.profile = null;
    }
    results.trunk = { case: caseName, n, depth, ms, ok: Object.values(checks).every((r) => r.relRms < tol && !r.nonFinite), ...checks };
  }

  if (parts.includes('decoder')) {
    const { meta, t } = await loadOracleCase(caseName);
    const batch = {};
    for (const [k, v] of Object.entries(t)) if (k.startsWith('batch.')) batch[k.slice(6)] = v;
    const n = batch.padding_mask.shape[1];
    const last = meta.steps.length - 1;
    const seq = sequenceForDecoder(batch, n, t[`step.${last}.out.sequence.logits`].data);
    const t0 = performance.now();
    const out = await model.decode(batch, {
      n, caNm: t['decode.in.ca_coors_nm'].data, z: t['decode.in.z_latent'].data,
      seq: Int32Array.from(t['decode.in.C_SEQ'].data), cfg: cfg.autoencoder_config.nn_ae.decoder,
    });
    const ms = Math.round(performance.now() - t0);
    const wantRt = t['decode.out.residue_type'].data;
    let rtDiff = 0;
    for (let i = 0; i < n; i += 1) if (out.residueType[i] !== wantRt[i]) rtDiff += 1;
    const r = compare(out.coorsNm, t['decode.out.coors_nm'].data);
    results.decoder = { case: caseName, n, ms, ok: rtDiff === 0 && r.maxAbs < 5e-2 && !r.nonFinite, rtDiff, seqOk: seq.length === n, ...r };
  }

  if (parts.includes('encoder')) {
    const { meta, t } = await loadOracleCase('encoder_oracle');
    const caseE = meta.cases[0];
    const batch = {};
    const prefix = `${caseE}.batch.`;
    for (const [k, v] of Object.entries(t)) if (k.startsWith(prefix)) batch[k.slice(prefix.length)] = v;
    const n = batch.padding_mask.shape[1];
    const eps = t[`${caseE}.enc.eps`].data;
    const t0 = performance.now();
    const out = await model.encode(batch, { n, cfg: cfg.autoencoder_config.nn_ae.encoder, rng: { randn: () => eps } });
    const ms = Math.round(performance.now() - t0);
    const rMean = compare(out.mean, t[`${caseE}.enc.mean`].data);
    const rZ = compare(out.zLatent, t[`${caseE}.enc.z_latent`].data);
    results.encoder = { case: caseE, n, ms, ok: rMean.relRms < tol && rZ.relRms < tol, mean: rMean, z: rZ };
  }

  results.gpuMB = Math.round(model.ctx.bytes / 2 ** 20);
  model.destroy();
  return results;
}
