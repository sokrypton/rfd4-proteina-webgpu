/**
 * Does the CHUNKED pair track compute what the single-chunk one computes?
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-rfd4p-chunked.js
 *   ... tools/gpu/check-rfd4p-chunked.js --n=200 --arms=f32,f16 --chunks=2,3,5
 *
 * 🔴 NOTHING HAD EVER CHECKED IT. Every oracle case is 54 to 191 tokens and `chunkPlan`
 * keeps one chunk to n = 362 in f32 and 512 in f16, so `check-rfd4p.js`'s whole suite -
 * trunk, decoder, encoder, end-to-end CPU parity - exercises the single-chunk arm and only
 * that arm. The chunked arm runs a DIFFERENT set of kernels: it norms without transposing,
 * projects into a per-chunk `proj` and scatters that into aT/bT through a `tri-gate` map.
 * None of them had a reference.
 *
 * 🔴 AND IT HAS FOUND FIVE REAL FAULTS, each invisible to every other gate here:
 *   1. a chunk starting off a four-row boundary (`transposeLnOut` reads `x` four pair rows
 *      a word): relRMS ~1.0 on the pair track, and the shipped bound's R was 4-aligned by
 *      luck rather than by design;
 *   2. the recycle's own norm reading an f16 pair track as f32 - `inF16` missing where
 *      `outF16` was given, which was RIGHT while f16 implied one chunk;
 *   3. the chunked triangle contracting over k = n rather than the 16-padded nPad the
 *      subgroup-matrix GEMM wants, which the single-chunk arm has always padded for;
 *   4. the stock GEMMs declaring elementwise B as `array<f32>` while it holds halves, so
 *      the incoming triangle - which reads `bT` by stride - read f16 bytes as floats:
 *      relRMS 0.43 on the alpha carbons and 1.02 on the latents;
 *   5. `inNorm: 'ln'` DROPPED IN SILENCE when the selected kernel cannot fuse it, which a
 *      ragged tail chunk provokes by sending the pair transition's `l1Fused` linear to the
 *      stock small kernel.
 * Each produced a plausible structure rather than an error.
 *
 * 🔴 AND (4) AND (5) ONLY APPEAR UNDER `LOCALFOLD_STOCK_FLAGS=1`, WHICH IS WHAT A VISITOR
 * RUNS. Both live in `gemmSource`/`gemmSmallSource`; the subgroup-matrix kernel handles
 * either case, so on developer flags both arms pass and the three commits that built this
 * never saw them - they were found by running THIS gate with the flag, on an M2. Run both
 * flag sets, and say which one a number came from.
 *
 * 🔴 AND THE TELL THAT SEPARATED (2) FROM AN ACCUMULATION was that its residual was
 * IDENTICAL for two equal chunks and for four ragged ones, to four decimal places. An error
 * that grows with the number of chunks is an accumulation; one that does not is a code path
 * taken on the `chunks.length > 1` branch, and that narrowed six candidate kernels to the
 * two that live there.
 *
 * 🔴 `chunkBytes` IS WHAT MAKES THE QUESTION ASKABLE AT A LENGTH THAT FITS. It is a
 * constructor option, so the same n can be forced down either arm: one chunk at the default
 * bound, several at a small one. The two then differ ONLY in the slicing, so the
 * single-chunk arm is the reference and the bar is float non-associativity rather than a
 * tolerance anybody picked. Measured, the f32 arm is EXACTLY ZERO where the chunks divide
 * evenly, so it is a bit-for-bit comparison and not an approximate one.
 *
 * 🔴 THE f16 ARM IS NOT, AND THE RESIDUAL IS STATED RATHER THAN EXPLAINED AWAY. Measured at
 * n = 200 on an M5, over every plan the default sweep runs:
 *
 *   stock flags       1.33e-4 even, 1.67e-4 with a 4-row tail, 2.87e-4 with a 64-row tail
 *   developer flags   6.59e-4, the SAME to three figures on all four plans
 *
 * where the f32 arm is 1.9e-7 to 5.1e-7 throughout. An M2 reads 1.5e-4 / 2.6e-4 / 2.9e-4 and
 * 6.9e-4 on the same arms, so neither column is one machine's.
 *
 * 🔴 AND THE SPLIT BETWEEN THE TWO COLUMNS IS THE CLUE, which is why both are here: the
 * subgroup-matrix path disagrees 4x WORSE than the stock one and is perfectly flat in both
 * the chunk count and the tail, while the stock path moves with the tail. A residual that
 * does not move with the slicing is not the slicing - most of this belongs to the submat
 * kernels' own selection (the incoming GEMM falls off `vecIO`, where the single-chunk arm
 * takes the fused `triProjG`) rather than to chunking at all. That is a far narrower search
 * than "the two formulations round at different points somewhere in the pass", which is what
 * this comment said while only the developer column existed. It survives `--recycles=1`, so
 * it is not the recycle seam either. Still unaccounted for; this is not a substitute for an
 * account, but it now says where to look.
 *
 * It compares the trunk's OUTPUT, not a trajectory, so it costs seconds rather than a
 * generation: one `forward` an arm at one set of features.
 */

import {
  DEFAULT_LOCAL_SHARDED_DIR,
  Rfd4pTrunkGpu,
  buildBatch,
  chunkPlan,
  fetchShardedWeights,
  pairCondFeatures,
  pairFeatures,
  seededRng,
  trunkCondFeatures,
  trunkSeqFeatures,
} from '../../webgpu/index.js';

/**
 * One relative RMS, GUARDED.
 *
 * 🔴 THE UNGUARDED VERSION CANNOT FAIL, WHICH IS WORSE THAN NOT CHECKING: hand it something
 * that is not an array and `a.length` is undefined, the loop never runs, and it returns
 * `Math.sqrt(0 / 1e-30)` - exactly zero, a perfect score from comparing nothing.
 */
function relativeRms(actual, expected) {
  if (!actual?.length || !expected?.length) throw new Error('relativeRms: not an array');
  if (actual.length !== expected.length) throw new Error(`relativeRms: ${actual.length} vs ${expected.length}`);
  let num = 0;
  let den = 0;
  for (let i = 0; i < actual.length; i += 1) {
    const d = actual[i] - expected[i];
    num += d * d;
    den += expected[i] * expected[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-30));
}

/**
 * The byte bound that makes `chunkPlan` choose about `rows` rows a chunk.
 *
 * 🔴 ASKED OF `chunkPlan` RATHER THAN DERIVED, because a checker that restates the thing it
 * is checking goes quiet exactly when that thing moves: an earlier draft had the row cost
 * written out by hand, the cost changed underneath it, and the bound that used to force
 * three chunks silently stopped forcing any - so the f16 arm compared a single-chunk run
 * against itself and reported a perfect zero. R is monotone in the bound, so bisect.
 */
function boundForRows(nn, wantF16, rows) {
  let lo = 16;
  let hi = 8 * 1024 * 1024 * 1024;
  for (let i = 0; i < 64; i += 1) {
    const mid = Math.floor((lo + hi) / 2);
    const plan = chunkPlan(nn, { wantF16, chunkBytes: mid });
    if (plan.chunks === 1 || plan.R > rows) hi = mid; else lo = mid;
  }
  return lo;
}

/**
 * An R whose LAST chunk is about `tail` pair rows, four-aligned, in three chunks and a bit.
 *
 * 🔴 ASKED FOR IN ROWS RATHER THAN A CHUNK COUNT, because a count says nothing about the tail:
 * one that divides `nn` leaves none at all, and which counts divide is a property of the
 * length. Three full chunks is enough to be a chunked run and cheap enough to be a default.
 * `chunkPlan` rounds R down to a multiple of four, so the tail that comes back is this one
 * plus whatever that rounding adds - the caller reads `tailRows` off the row rather than
 * trusting this, and `plan.tail` makes an even result a refusal.
 */
function tailRows(nn, tail) {
  return Math.max(4, Math.floor((nn - tail) / 12) * 4);
}

/** The features one `forward` needs, built once and shared by every arm. */
function buildFeatures(batch, n, config, weights) {
  const rng = seededRng(42);
  const nnCfg = config.nn;
  const distCfg = { dim: nnCfg.xt_pair_dist_dim, min: nnCfg.xt_pair_dist_min, max: nnCfg.xt_pair_dist_max };
  // A fixed, non-zero x_t: an all-zero pair distance makes the pair track degenerate, and a
  // degenerate reference agrees with anything.
  const xt = { bb_ca: new Float32Array(n * 3), local_latents: new Float32Array(n * 32) };
  for (let i = 0; i < n * 3; i += 1) xt.bb_ca[i] = Math.sin(i * 0.7) * 1.5;
  for (let i = 0; i < n * 32; i += 1) xt.local_latents[i] = Math.cos(i * 0.31);
  const tt = { bb_ca: 0.5, local_latents: 0.5 };
  const cache = {};
  return {
    seq: trunkSeqFeatures(batch, {
      n, xt, feats: nnCfg.feats_seq, weights, randperm: (len) => rng.randperm(len, 0),
      noiseAlignChirality: nnCfg.noise_align_chirality, cache,
    }).data,
    cond0: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: null, feats: nnCfg.feats_cond_seq, weights, cache }).data,
    cond1: trunkCondFeatures(batch, { n, t: tt, recycleRemaining: 1, feats: nnCfg.feats_cond_seq, weights, cache }).data,
    pair: pairFeatures(batch, {
      n, caNm: xt.bb_ca, feats: nnCfg.feats_pair_repr, dist: distCfg, seqSepDim: nnCfg.seq_sep_dim, cache,
    }).data,
    pairCond: pairCondFeatures(tt, nnCfg.feats_pair_cond),
  };
}

export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const n = Number(opt.n ?? 200);
  const arms = (opt.arms ?? 'f32,f16').split(',');
  const chunkCounts = (opt.chunks ?? '3,5').split(',').map(Number);
  // `--rows=` sets R directly, which is how "the tail is short" was separated from "R is not
  // a multiple of four": every chunk count that divides nn evenly also happened to give a
  // four-aligned R, so both fit the same measurements until R was named directly. When set,
  // it is ONE plan and the `--chunks`/`--tails` lists are not swept.
  const explicitRows = opt.rows === undefined ? null : Number(opt.rows);
  // 🔴 AND A SHORT TAIL IS IN THE DEFAULT SWEEP, BECAUSE AN EVEN SPLIT CANNOT REACH THE STOCK
  // SMALL KERNEL. A full chunk is thousands of pair rows and that kernel is chosen below ~640,
  // so the ONLY way a chunked run sends a GEMM there is a ragged last chunk - which is exactly
  // where fault (5) lived: on stock Chrome every even split passed and every ragged one failed,
  // whatever the tail length. The old `--chunks=3,5` default reached it only through a 4-row
  // tail, short enough that nothing downstream of it mattered. A tail of 33-640 rows has teeth.
  const tails = (opt.tails ?? '64').split(',').map(Number).filter((t) => t > 0);
  // 🔴 RECYCLES, BECAUSE THE SECOND CYCLE TAKES A DIFFERENT ROUTE IN EACH ARM. With one
  // chunk the last layer FUSES the next cycle's norm (`outZc`); sliced, `recycleIn` does it
  // separately. The two are equivalent and round differently, so `--recycles=1` is the arm
  // that excludes them and says whether a residual belongs to the pair track or that seam.
  const recycles = Number(opt.recycles ?? 2);
  // 🔴 THE BAR FOLLOWS THE ARITHMETIC. The contraction is one global GEMM either way, so what
  // chunking changes in f32 is the per-chunk epilogue's rounding and, at a small R, the tile
  // geometry some row-wise GEMMs resolve to - a reordering. 1e-5 is the band for that and
  // nothing looser; the fault it was built for reads 0.72 to 0.98 on the latents, five
  // orders away.
  const bound = { f32: 1e-5, f16: 6e-3 };

  const { tensors: weights, config } = await fetchShardedWeights(DEFAULT_LOCAL_SHARDED_DIR);
  const { batch } = buildBatch({ contig: `${n}-${n}` }, { seed: 42 });
  const f = buildFeatures(batch, n, config, weights);
  const nn = n * n;

  async function run(half, chunkBytes) {
    const trunk = new Rfd4pTrunkGpu(device, weights, config.nn, { half, chunkBytes });
    try {
      const out = await trunk.forward(n, f, recycles, {});
      return {
        chunks: trunk.chunks.length,
        chunkRows: trunk.R,
        triF16: trunk.triF16,
        bb_ca: out.bb_ca.slice(),
        local_latents: out.local_latents.slice(),
        logits: out.logits.slice(),
      };
    } finally {
      trunk.engine.destroy();
    }
  }

  // 🔴 `--rows=` IS ONE PLAN, NOT ONE PER `--chunks=` ENTRY. It used to be swept inside that
  // loop, so naming R ran the identical plan once per chunk count and printed the same row
  // three times - three identical numbers that read as three agreeing cases.
  const plans = explicitRows !== null
    ? [{ rows: explicitRows, asked: `R=${explicitRows}`, tail: 0 }]
    : [
        ...chunkCounts.map((want) => ({ rows: Math.max(1, Math.ceil(nn / want)), asked: `${want} chunks`, tail: 0 })),
        ...tails.map((tail) => ({ rows: tailRows(nn, tail), asked: `~${tail}-row tail`, tail })),
      ];

  const rows = [];
  let failures = 0;
  for (const arm of arms) {
    const half = arm === 'f16';
    const ref = await run(half, undefined);
    if (ref.chunks !== 1) throw new Error(`the ${arm} reference is ${ref.chunks} chunks at n=${n} - lower n`);
    for (const plan of plans) {
      const got = await run(half, boundForRows(nn, half, plan.rows));
      // 🔴 A TAIL CASE THAT CAME BACK EVEN IS AN INERT CASE: it tests the arm the even splits
      // already test, while still printing a green row. Refuse rather than report it.
      if (plan.tail && nn % got.chunkRows === 0) {
        throw new Error(`the ~${plan.tail}-row tail case divided evenly at R=${got.chunkRows} (n=${n}) - it tests nothing the even splits do not`);
      }
      // A row is only a slicing comparison when both arms hold the same arithmetic.
      if (got.triF16 !== ref.triF16) {
        rows.push({
          arm, asked: plan.asked, chunks: got.chunks, chunkRows: got.chunkRows, comparable: false,
          why: `the chunked arm is ${got.triF16 ? 'f16' : 'f32'} where its reference is ${ref.triF16 ? 'f16' : 'f32'} - not a slicing comparison`,
        });
        continue;
      }
      const dev = {
        bb_ca: relativeRms(got.bb_ca, ref.bb_ca),
        local_latents: relativeRms(got.local_latents, ref.local_latents),
        logits: relativeRms(got.logits, ref.logits),
      };
      const worst = Math.max(dev.bb_ca, dev.local_latents, dev.logits);
      const ok = Number.isFinite(worst) && worst <= bound[arm];
      if (!ok) failures += 1;
      rows.push({
        arm,
        asked: plan.asked,
        chunks: got.chunks,
        chunkRows: got.chunkRows,
        // Whether the last chunk is short - reported because it is the first thing anybody
        // suspects here, and it is NOT the discriminator.
        ragged: nn % got.chunkRows !== 0,
        tailRows: nn - got.chunkRows * (got.chunks - 1),
        triF16: got.triF16,
        bb_ca: Number(dev.bb_ca.toExponential(2)),
        local_latents: Number(dev.local_latents.toExponential(2)),
        logits: Number(dev.logits.toExponential(2)),
        comparable: true,
        bound: bound[arm],
        ok,
      });
    }
  }

  const compared = rows.filter((r) => r.comparable).length;
  const report = { n, recycles, bound, compared, skipped: rows.length - compared, rows, ok: failures === 0 && compared > 0 };
  if (failures) throw new Error(`rfd4p chunked check: ${failures} arm(s) out of band: ${JSON.stringify(report)}`);
  // A gate that compared nothing is not a passing gate, which is how a skip-based arm goes
  // quiet.
  if (!compared) throw new Error(`rfd4p chunked check: every row was skipped: ${JSON.stringify(report)}`);
  return report;
}
