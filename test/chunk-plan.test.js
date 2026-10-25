/**
 * `chunkPlan` - how the pair track is sliced, and in which arithmetic.
 *
 * It is a pure function of the pair count and a byte bound, so the whole policy that used
 * to be inline arithmetic inside `record()` is checkable without a GPU. These are the
 * invariants the kernels depend on; one of them (the four-row start) is a correctness
 * requirement rather than a preference, and the shipped bound satisfied it only by luck.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_CHUNK_BYTES, chunkPlan } from '../webgpu/rfd4p/trunk.js';

test('keeps one chunk to n = 362 in f32, which is where it always fell', () => {
  // The f32 arm is unchanged on purpose: restating the bound in bytes was meant to let f16
  // reach further, not to move the strict path, so 362/363 is the assertion that the
  // restatement was faithful rather than a retune wearing its clothes.
  assert.equal(chunkPlan(362 * 362, { wantF16: false }).chunks, 1);
  assert.equal(chunkPlan(363 * 363, { wantF16: false }).chunks, 2);
});

test('keeps one chunk to n = 512 in f16, because half the bytes is twice the rows', () => {
  assert.equal(chunkPlan(512 * 512, { wantF16: true }).chunks, 1);
  assert.equal(chunkPlan(512 * 512, { wantF16: true }).triF16, true);
  assert.ok(chunkPlan(513 * 513, { wantF16: true }).chunks > 1);
});

test('keeps f16 when it gives up the single chunk', () => {
  // The chunked kernels read and write halves now, so the arithmetic is a property of the
  // DEVICE and the request while the slicing is a property of the length - two independent
  // questions that used to be one. While they were coupled, crossing n = 512 to 513 widened
  // the pair track back to f32 at the same moment the slicing changed: memory rose 46%
  // (2056 -> 3001 MiB) and the pass took 2.02x as long (2914 -> 5882 ms). A cliff that size
  // in one residue is what a coupling looks like from the outside.
  for (const n of [100, 200, 362, 400, 512, 513, 700, 1000]) {
    assert.equal(chunkPlan(n * n, { wantF16: true }).triF16, true, `n=${n}`);
  }
});

test('never asks f16 of a device that cannot do it', () => {
  for (const n of [100, 400, 1000]) {
    assert.equal(chunkPlan(n * n, { wantF16: false }).triF16, false);
  }
});

test('honours the byte bound in BOTH shapes', () => {
  // A bug that was written and caught: sizing R from f16's row cost and then allocating f32
  // buffers against it put 1.88 GB behind a 896 MB bound, and it ran only because the
  // machine it was written on has 4 GB bindings. The bound has to hold whichever shape the
  // plan actually chooses, so both are swept.
  for (const wantF16 of [false, true]) {
    for (const n of [54, 100, 200, 362, 363, 400, 512, 513, 520, 700, 1000]) {
      const plan = chunkPlan(n * n, { wantF16 });
      assert.ok(plan.scratchBytes <= DEFAULT_CHUNK_BYTES,
        `n=${n} f16=${wantF16}: ${plan.scratchBytes} > ${DEFAULT_CHUNK_BYTES}`);
    }
  }
});

test('covers every pair row, with chunks that tile the tensor', () => {
  for (const n of [54, 100, 363, 400, 513, 1000]) {
    const nn = n * n;
    const plan = chunkPlan(nn, { wantF16: true });
    assert.ok(plan.R * plan.chunks >= nn, `n=${n} does not cover`);
    // One chunk short of the count must NOT cover it, or the count is padded.
    assert.ok(plan.R * (plan.chunks - 1) < nn, `n=${n} count is padded`);
  }
});

test('starts every chunk on a multiple of four pair rows', () => {
  // 🔴 THE INVARIANT `transposeLnOut` DEPENDS ON. It reads the contraction output as
  // `array<vec4<f32>>` along the ROW axis - `x[c * nnStride4 + (r0 + rLocal) / 4]` - so four
  // consecutive pair rows share a word and a chunk beginning off that boundary reads every
  // row of every channel from the wrong place. `rLocal` is already a multiple of four, so
  // the whole requirement lands on r0 = k * R.
  //
  // Measured against the single-chunk arm at n = 60: R = 1800, 1208, 1200, 900 and 512 all
  // agree at 1.3e-6 on the pair track, and R = 515, 450 and 225 come back at relRMS 0.89 to
  // 1.04. Raggedness is NOT the discriminator - 450 and 225 divide 3600 exactly and still
  // fail, while 1208 and 512 leave a short tail and pass - so a tail of any length is fine
  // and a misaligned start is not.
  //
  // Swept over bounds rather than a chosen few, because the default bound happens to give a
  // four-aligned R and has hidden this since the chunking was written.
  for (let mb = 1; mb <= 2048; mb += 1) {
    for (const nn of [400 * 400, 513 * 513, 1000 * 1000]) {
      const { R, chunks } = chunkPlan(nn, { wantF16: true, chunkBytes: mb * 1024 * 1024 });
      if (chunks === 1) continue;
      assert.equal(R % 4, 0, `${mb} MiB, nn=${nn}: R=${R}`);
    }
  }
});

test('still returns a usable plan when the bound is tiny', () => {
  // A bound smaller than one row is still a plan - and the floor is FOUR rows, not one,
  // for the reason above. Zero rows is not a plan and one row is a wrong one.
  const plan = chunkPlan(400 * 400, { wantF16: true, chunkBytes: 16 });
  assert.equal(plan.R, 4);
  assert.equal(plan.triF16, true);
  assert.equal(plan.chunks, 40000);
});

test('gives a bigger bound more rows, monotonically', () => {
  let previous = 0;
  for (const mb of [16, 64, 256, 896, 2048]) {
    const { R } = chunkPlan(1000 * 1000, { wantF16: true, chunkBytes: mb * 1024 * 1024 });
    assert.ok(R >= previous, `${mb} MiB gave ${R} after ${previous}`);
    previous = R;
  }
});
