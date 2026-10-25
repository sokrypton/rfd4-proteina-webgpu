# M2 → M5 Handoff: validating the chunked pair track (PR #3)

Written 2026-10-25 on an Apple M2 for the agent working on the M5. PR #3 (`perf/chunked-pair-track`) was built
and measured on the M5. This file records what happened when the M2 re-checked it on **stock Chrome** before the
merge (9fa19f9): two bugs found and fixed, speedups confirmed, and what the M5 should do next. Read
[`SPEED-HANDOFF.md`](./SPEED-HANDOFF.md) section 1 (rules) first.

---

## 1. Result in one paragraph

PR #3 had a real correctness bug, and it was visible only without developer flags. On stock Chrome the f16
chunked arm disagreed with the single-chunk arm at **relRMS 0.43** on the alpha carbons. Every design past
n ≈ 512 in f16 would have come out garbled. With developer flags it passed, which is why the M5 never saw it.
Two stock-only GEMM gaps caused it; they are fixed in 2399e1b. After the fix every gate passes on both flag
sets, and the PR is **1.9× faster on M2 at every length above the old f16 cliff** (n > 362).

---

## 2. Machine and how it was run

| | |
|---|---|
| Host | Apple M2 (10-core GPU), 16 GB, macOS 13.2.1, Chrome 156 |
| Stock features (`LOCALFOLD_STOCK_FLAGS=1`) | `shader-f16`, `subgroups`, `timestamp-query`; **no subgroup-matrix** |
| Developer flags | adds `chromium-experimental-subgroup-matrix` (8x8x8) |
| Weights | `weights/int8` shards; oracle parity also uses f32 `weights/model.safetensors` + `oracle-dumps/` |

The M5 does not have the oracle dumps or the f32 model. The PR says so, and the M2 ran those gates for it.

---

## 3. The two bugs (both fixed in 2399e1b)

Both were older kernel gaps that stayed unreachable while `triF16` required one chunk. The PR made f16
chunked, which exposed them. Both hit only the **stock** kernels (`gemmSource`, `gemmSmallSource`). The
subgroup-matrix kernel handles both cases, so developer-flag runs pass.

### 3.1 Elementwise B read as f32 when it holds halves

- `gemmSubmatSource` types B as f16 whenever `aF16`.
- The stock kernels only read halves through the packed "word B" path, which needs `B.k === 1`. Otherwise they
  declared `b: array<f32>`.
- The PR's incoming triangle reads `bT` by stride (`{n: 1, k: st}`), so it read f16 bytes as floats.
- **Fix:** elementwise B now shares A's type (`EBelem`) in both stock kernels.

### 3.2 The fused input LayerNorm was silently dropped

- `gemm()` computes `inLn` as a conjunction of kernel capabilities. Whenever the selected kernel couldn't fuse
  the norm, `inNorm: 'ln'` was ignored without an error.
- `gemmSmallSource` has no `inLn` at all.
- On stock Chrome the small kernel is chosen for `tiles64 < 32`, which is **any m up to ~640 rows** at n = 384,
  not just m ≤ 32.
- A ragged tail chunk therefore sent the f16 pair transition's `l1Fused` linear (`inNorm: 'ln'`) to the small
  kernel, and that kernel skipped the LayerNorm.
- **Fix:** `'ln'` GEMMs stay off the stock small kernel. An `'ln'` that the selected kernel can't honour now
  **throws** instead of being dropped.

### 3.3 How it was located (reusable recipe)

1. Run `check-rfd4p-chunked.js` with stock flags. The f16 arm failed; the same run with developer flags
   passed. That points at kernel selection.
2. After fix 1, even splits passed (R=8000, R=10000) and every ragged split failed, whatever the tail length
   (4, 16, 40 and 64 rows). Check with `--rows=`.
3. Disabling the small kernel for `cF16` GEMMs fixed it. Disabling it only for f16-gated GEMMs did not. That
   narrowed it to the ungated `l1Fused` linear, the only `inNorm` caller in the chunk loops.

---

## 4. Gates at the merged head (stock flags unless noted)

| gate | result |
|---|---|
| `npm test` | 35/35 |
| `check-rfd4p.js` (oracle parity) | 25/25 |
| `check-rfd4p.js --half=1` | 25/25 |
| `check-rfd4p-chunked.js` f32 | 1.9e-7 even, 4.1e-7 with a 4-row tail |
| `check-rfd4p-chunked.js` f16 | 1.5e-4 even, 2.6e-4 with a 4-row tail, 2.9e-4 with a 64-row tail (`--rows=13312`) |
| `check-rfd4p-chunked.js` f16, developer flags | 6.9e-4 |
| `check-generate.js` | final RMSD 0.036 Å |
| `check-webapp.js` | ok |

On the M2's stock path the f16 chunked arm agrees **better** than on the subgroup-matrix path
(1.5e-4 vs 6.9e-4). That is a clue for the unexplained f16 residual (section 6.4).

---

## 5. Speed on M2

`tools/gpu/bench-trunk.js` measures one trunk forward (one cycle, 14 layers, f16 requested as in the app):
one warm-up, then the median of 3. Spread was under 1.5%. The old and new trees ran from separate worktrees,
one at a time.

### 5.1 Stock Chrome (what visitors run)

| n | main | chunks / arithmetic | PR #3 | chunks / arithmetic | speedup |
|---:|---:|---|---:|---|---:|
| 200 | 1095 ms | 1 / f16 | 1111 ms | 1 / f16 | — (same path) |
| 362 | 3516 | 1 / f16 | 3475 | 1 / f16 | 1.01× |
| 400 | 8576 | 2 / **f32** | 4520 | 1 / f16 | **1.90×** |
| 513 | 14881 | 3 / f32 | 7746 | 2 / f16 | **1.92×** |
| 600 | 20056 | 3 / f32 | 10388 | 2 / f16 | **1.93×** |
| 800 | 37104 | 5 / f32 | 19340 | 2 / f16 | **1.92×** |

- At n ≤ 362 the code path is unchanged, and so is the time. The gain starts exactly where main used to fall
  back to f32.
- On M5 (developer flags) the PR reported 1.80× at n=513 and 1.84× at n=600. The M2 stock gain is about the
  same, slightly larger.

### 5.2 Developer flags (subgroup-matrix)

| n | main | PR #3 |
|---:|---:|---:|
| 400 | **fails to compile** | 2080 ms |
| 600 | **fails to compile** | 4668 ms |

Main's f32 chunked arm needs `gemm-sm8-f-ve`, the ragged-k submatrix variant with the `(k + 1u)` typo that the
PR's first commit fixes. Main could not run n ≥ 363 with developer flags at all.

Stock vs subgroup-matrix on M2 after the PR: 2.2× at n = 400 and 600. The P0 lever in SPEED-HANDOFF.md
(getting subgroup-matrix to visitors) is unchanged.

---

## 6. What the M5 should consider next

### 6.1 Run every gate with `LOCALFOLD_STOCK_FLAGS=1`

This is now the third bug that shipped or nearly shipped because only developer flags were tested (see
SPEED-HANDOFF.md section 1). The M5 numbers in the PR body are all developer-flag numbers. Quote both.

### 6.2 Put a ragged tail in the default chunked sweep — done in #4 (`--tails=`, default 64)

- The default `--chunks=3,5` at n = 200 gives one ragged plan (4-row tail) and one even plan.
- An even split never reaches the small kernel, because full chunks are thousands of rows.
- Add an explicit tail case of 33–640 rows, for example `--rows=13312` (64-row tail), to the defaults. A
  short tail is the only way the stock small kernel gets exercised by the chunked arm.

### 6.3 Audit for other silently dropped GEMM options — done in #4 (gemv `inNorm` set, non-`'ln'` on tiled kernels, f16 operands without `shader-f16` all throw)

`inLn` was a capability that the selected kernel could quietly fail to provide. Look for the same pattern:

- Any `spec` field that `gemmSmallSource`, `gemmSource` or `gemvSource` ignores. Small has no `pair`, no
  `inLn` and no vector epilogue.
- Any flag computed as `cond && spec.x` where the false branch changes **results**, not just speed.

Make those throw like `inLn` does now. Performance-only fallbacks (`vecIO`, `vecB`) are fine.

### 6.4 Command-buffer length past n ≈ 400 on slower GPUs

`LAYERS_PER_SUBMIT = 2`, so a cycle is 7 command buffers. Average per command buffer on M2 stock after the
PR:

| n | 400 | 513 | 600 | 800 |
|---|---:|---:|---:|---:|
| ms per command buffer | ~650 | ~1100 | ~1500 | ~2800 |

- SPEED-HANDOFF.md rule 4 says to keep each one well under ~1 s, because visible macOS Chrome kills long
  ones as a GPU hang.
- Main was about twice as long, so the PR helps, but n ≥ ~500 on an M2 is still over the line.
- Headless runs don't show this. It needs a visible-Chrome test at n = 600.
- Candidates:
  - scale layers per submit with n (1 layer per submit past ~400);
  - split the per-chunk loops of `triangle` across submits.

### 6.5 The remaining f16 chunked-vs-single residual

The PR body leaves this unexplained: 6.6e-4 after 68daebb on M5, 6.9e-4 on M2 with developer flags.
On the **stock** path it is 1.5e-4–2.9e-4, so most of the residual is specific to the subgroup-matrix
kernels. Compare the two arms' kernel selection under developer flags:

- the incoming GEMM falls off `vecIO`;
- the single-chunk arm uses the fused `triProjG` with `pad0`.

That is a narrower search than "reordering".

### 6.6 The f16 cliff moved to 512 but still exists

With `DEFAULT_CHUNK_BYTES` = 896 MB, f16 runs one chunk up to n = 512. Past that the 2-chunk plan costs little
on M2 (n = 513 is 1.71× n = 400 for 1.64× the pairs). That's not a priority, but a second bound for
16 GB-class devices would remove the transition.

---

## 7. Files touched in this round

- `webgpu/rfd4p/engine.js`: the stock GEMM fixes (2399e1b) on top of PR #3.
- `tools/gpu/bench-trunk.js`: new. Length sweep wall-clock bench; used for section 5.
- `M2-TO-M5.md`: this file.
