/**
 * The WebGL2 fallback (`webgl/rfd4p/model.js`) vs the PyTorch oracle, with the released weights: the trunk's network
 * outputs at a full recycle depth, the decoder and the encoder. Runs in any browser with WebGL2 (the WebGPU device
 * the runner passes is not used).
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-webgl.js [--case=uncond_short] [--parts=trunk,decoder,encoder]
 */

import { sequenceForDecoder } from '../../cpu/rfd4p/decoder.js';
import { generateRfd4pCpu } from '../../cpu/rfd4p/reference.js';
import { pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures } from '../../shared/rfd4p/features.js';
import { Rfd4pModelGL } from '../../webgl/rfd4p/model.js';
import { recordedOracleRng } from './check-generate.js';
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
  if (opt.nopacked === '1') model.noPacked = true;
  if (opt.skipdraw === '1') model.ctx.skipDraw = true;
  // `--skip=<regex>`: those kernels' draws not issued (a timing probe: what they cost in a real pass; results wrong).
  if (opt.skip) model.ctx.skipKeys = new RegExp(opt.skip);
  for (const k of (opt.flags ?? '').split(',').filter(Boolean)) model[k] = true;
  if (opt.only) model.noPackedSites = new Set(['zn', 'xn', 'zt', 'h', 'np', 'zc'].filter((x) => x !== opt.only));
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
      pairCond: pairCondFeatures(tt, nnCfg.feats_pair_cond),
    };
    // As the sampler gives them (`pairDistsOnGpu`): the pair features without the Cα distances, and the coordinates;
    // `--pairgpu=0`: all of them from the CPU.
    const dist = { dim: nnCfg.xt_pair_dist_dim, min: nnCfg.xt_pair_dist_min, max: nnCfg.xt_pair_dist_max };
    const pf = pairFeatures(batch, { n, caNm: xt.bb_ca, feats: nnCfg.feats_pair_repr, dist, seqSepDim: nnCfg.seq_sep_dim, skipPairDists: opt.pairgpu !== '0' });
    f.pair = pf.data;
    if (opt.pairgpu !== '0') f.pairDists = { caNm: xt.bb_ca, offset: pf.distOffset, ...dist };
    const depth = opt.depth ? Number(opt.depth) : meta.steps[0].n_recycle;
    if (opt.warm === '1' && opt.profile !== '1') await model.forwardTrunk(n, f, batch.padding_mask.data, nnCfg, 1);
    if (opt.profile === '1') { await model.forwardTrunk(n, f, batch.padding_mask.data, nnCfg, 1); model.ctx.profileSizes = opt.sizes === '1'; model.ctx.profile = new Map(); model.ctx.lastT = performance.now(); }
    if (opt.counts === '1') model.ctx.counts = new Map();
    // `--reps=N`: N more calls after this one, the fastest reported as `minMs` (a single pass is noisy).
    if (opt.reps) {
      let best = Infinity;
      for (let r = 0; r < Number(opt.reps); r += 1) {
        const tr = performance.now();
        await model.forwardTrunk(n, f.pairDists ? { ...f, pair: null } : f, batch.padding_mask.data, nnCfg, depth);
        best = Math.min(best, performance.now() - tr);
      }
      results.minMs = Math.round(best);
    }
    const t0 = performance.now();
    const out = await model.forwardTrunk(n, f, batch.padding_mask.data, nnCfg, depth, { insertion: true });
    if (model.ctx.counts) { const all = [...model.ctx.counts]; results.draws = all.reduce((a, [, v]) => a + v, 0); results.drawsTop = all.sort((a, b) => b[1] - a[1]).slice(0, 12).map(([k, v]) => `${v} ${k}`); model.ctx.counts = null; }
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
      results.profile = [...model.ctx.profile].sort((a, b) => b[1] - a[1]).slice(0, Number(opt.top ?? 14)).map(([k, v]) => `${Math.round(v)} ms ${k}`);
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

  if (parts.includes('generate')) {
    // The whole sampler (`generateRfd4pCpu` on these kernels, as the app runs it) with the oracle's recorded noise.
    const { meta, t } = await loadOracleCase(caseName);
    const batch = {};
    for (const [k, v] of Object.entries(t)) if (k.startsWith('batch.')) batch[k.slice(6)] = v;
    const rng = recordedOracleRng(t, meta.rng);
    const kernels = model.kernels();
    if (opt.pairgpu === '0') kernels.pairDistsOnGpu = false;
    const t0 = performance.now();
    const res = await generateRfd4pCpu(batch, weights, cfg, { rng, nsteps: meta.steps.length, recycleSchedule: meta.recycle_depths, kernels });
    const ms = Math.round(performance.now() - t0);
    const ca = compare(res.x.bb_ca, t['final.x.bb_ca'].data);
    const wantRt = t['final.residue_type'].data;
    let rtDiff = 0;
    for (let i = 0; i < wantRt.length; i += 1) if (res.residueType[i] !== wantRt[i]) rtDiff += 1;
    const wantMask = t['final.atom_mask'].data;
    const want = t['final.coors'].data;
    let ss = 0;
    let cnt = 0;
    for (let a = 0; a < wantMask.length; a += 1) {
      if (!wantMask[a]) continue;
      for (let k = 0; k < 3; k += 1) ss += (res.coors[a * 3 + k] - want[a * 3 + k]) ** 2;
      cnt += 1;
    }
    const rmsdA = Math.sqrt(ss / Math.max(cnt, 1));
    results.generate = { case: caseName, nsteps: meta.steps.length, ms, randn: `${rng.randnUsed}/${rng.randnTotal}`, ca, rtDiff,
      rmsdA: Number(rmsdA.toFixed(3)), ok: rng.randnUsed === rng.randnTotal && rtDiff === 0 && rmsdA < 0.5 && !ca.nonFinite };
  }

  results.gpuMB = Math.round(model.ctx.bytes / 2 ** 20);
  model.destroy();
  return results;
}
