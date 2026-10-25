/**
 * Wall-clock trunk forward (one cycle) at chosen lengths: one warm-up, then the median of --reps.
 *
 *   LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/bench-trunk.js --lengths=400,600 --reps=3
 *   ... --half=0 for the f32 arm. Run old and new trees from separate worktrees, one at a time.
 */
import {
  DEFAULT_LOCAL_SHARDED_DIR, Rfd4pTrunkGpu, buildBatch, fetchShardedWeights,
  pairCondFeatures, pairFeatures, seededRng, trunkCondFeatures, trunkSeqFeatures,
} from '../../webgpu/index.js';

function feats(batch, n, config, weights) {
  const rng = seededRng(42);
  const c = config.nn;
  const xt = { bb_ca: new Float32Array(n * 3), local_latents: new Float32Array(n * 32) };
  for (let i = 0; i < n * 3; i += 1) xt.bb_ca[i] = Math.sin(i * 0.7) * 1.5;
  for (let i = 0; i < n * 32; i += 1) xt.local_latents[i] = Math.cos(i * 0.31);
  const tt = { bb_ca: 0.5, local_latents: 0.5 };
  const cache = {};
  return {
    seq: trunkSeqFeatures(batch, { n, xt, feats: c.feats_seq, weights, randperm: (l) => rng.randperm(l, 0), noiseAlignChirality: c.noise_align_chirality, cache }).data,
    cond0: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: null, feats: c.feats_cond_seq, weights, cache }).data,
    cond1: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: 1, feats: c.feats_cond_seq, weights, cache }).data,
    pair: pairFeatures(batch, { n, caNm: xt.bb_ca, feats: c.feats_pair_repr, dist: { dim: c.xt_pair_dist_dim, min: c.xt_pair_dist_min, max: c.xt_pair_dist_max }, seqSepDim: c.seq_sep_dim, cache }).data,
    pairCond: pairCondFeatures(tt, c.feats_pair_cond),
  };
}

export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const lengths = (opt.lengths ?? '200,400').split(',').map(Number);
  const half = opt.half !== '0';
  const reps = Number(opt.reps ?? 3);
  const { tensors: weights, config } = await fetchShardedWeights(DEFAULT_LOCAL_SHARDED_DIR);
  const rows = [];
  for (const n of lengths) {
    const { batch } = buildBatch({ contig: `${n}-${n}` }, { seed: 42 });
    const f = feats(batch, n, config, weights);
    const trunk = new Rfd4pTrunkGpu(device, weights, config.nn, { half });
    try {
      await trunk.forward(n, f, 1, {});
      const t = [];
      for (let r = 0; r < reps; r += 1) {
        const t0 = performance.now();
        await trunk.forward(n, f, 1, {});
        t.push(performance.now() - t0);
      }
      t.sort((a, b) => a - b);
      rows.push({ n, chunks: trunk.chunks.length, triF16: trunk.triF16, medianMs: Math.round(t[Math.floor(t.length / 2)]), allMs: t.map(Math.round) });
    } catch (err) {
      rows.push({ n, error: String(err.message ?? err).slice(0, 200) });
    } finally {
      trunk.engine.destroy();
    }
  }
  return { half, rows };
}
