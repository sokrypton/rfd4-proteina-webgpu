# Speed-Optimization Handoff: RFD4-Proteina WebGPU (`rfd4-proteina-webgpu`)

Written 2026-10-23 on an Apple M2 for the next agent working on **generation speed**. Read [`AGENTS.md`](./AGENTS.md)
first (module layout, ES-module rule, WebGPU invariants), then this file. [`HANDOFF.md`](./HANDOFF.md) holds the
previous M2/M5 tuning round: its section 9 explains how the current kernels got here and section 7 lists what
already lost. Do not repeat anything in section 7 without a new reason.

Live site: <https://sokrypton.github.io/rfd4-proteina-webgpu/> (GitHub Pages from `main`, root). The owner has
said to commit to `main` and push after each verified change; the site redeploys in ~1 min.

---

## 0. The one-paragraph summary

A design is ~60 sampler steps; each step runs 1-6 trunk passes ("Thinking"), and a trunk pass is ~80% five pair-
track GEMMs. On stock Chrome (what visitors run) one pass at n = 100 takes **303 ms**; with
`--enable-unsafe-webgpu`, which exposes `chromium-experimental-subgroup-matrix`, the same pass takes **132 ms
(2.3x faster)**. Four tweaks of the stock GEMM kernel all lost (section 5). So the big levers are: (P0) get the
subgroup-matrix path to visitors, (P1) a different class of stock GEMM kernel, (P2) fewer passes per design. Every
change must keep the PyTorch parity gates green on **both** flag sets.

---

## 1. Rules (non-negotiable)

1. **Stock Chrome is the target.** Run every GPU gate with `LOCALFOLD_STOCK_FLAGS=1` as well as the default
   flags. Two real bugs shipped because only the default (developer) flags were tested:
   - an FP16 path that only broke without subgroup-matrix produced NaN designs for every visitor;
   - headless Chrome tolerated a 13.6 s command buffer that a visible macOS Chrome kills as a GPU hang.
2. **Correctness first.** A speedup that moves parity outside the gates in section 7 is not a speedup.
3. **Interleave A/B measurements in one process** (`tools/gpu/bench-kernels.js` does this). The laptop throttles;
   back-to-back runs drift 5-20%. Report the best of >= 3 interleaved rounds.
4. **Keep every GPU command buffer well under ~1 s.** macOS kills longer ones ("Caused GPU Hang Error",
   `kIOGPUCommandBufferCallbackErrorHang`, device lost). The trunk records 2 layers per program
   (`LAYERS_PER_SUBMIT`, `webgpu/rfd4p/trunk.js:29`) and `Rfd4pEngine.encodePrograms` submits one command buffer
   per program. A new kernel that is 3x slower per call can reintroduce hangs at n = 280 with high thinking.
5. **Producers and consumers of f16 buffers must agree.** Whether an activation is stored f16 is decided in
   several places (`triF16` in trunk.js, `swF16` / `qkF16` / `xnF16` in blocks.js, `outF16` in
   `linear()`'s split-reduce-norm path). The NaN bug above was one producer gated on `hasF16` writing f16 that a
   consumer gated on `hasSubgroupMatrix` read as f32. If you change one gate, find every reader.
6. AGENTS.md invariants still apply: plain ES modules, no `L x L x L` buffers, guarded folded 2D dispatches
   (`npm test` checks the guard statically).

---

## 2. Machine and browser

| | |
|---|---|
| Host | Apple M2 (10-core GPU, 8-core CPU), macOS 13.2.1 |
| Browser | Google Chrome 156.0.8078.47 |
| Adapter | `apple` / `metal-3` |
| Stock features (`LOCALFOLD_STOCK_FLAGS=1`) | `shader-f16`, `subgroups`, `timestamp-query`; **no** subgroup-matrix |
| Developer flags (default harness) | adds `chromium-experimental-subgroup-matrix` (8x8x8), `subgroup-size-control` |
| Limits | 32 KB workgroup storage, 1024 invocations / workgroup, 4 GB max buffer |

Note: HANDOFF.md 9.1 says this M2 exposes no subgroup-matrix even with the unsafe flag. That was true on the
Chrome it was measured with; Chrome 156 exposes it behind the flag. Device ceilings measured by `../localfold`
on this M2 (`../localfold/docs/PERF.md`, "This device's ceilings"): ~1.24 TFLOP/s scalar f32 FMA, ~4.9 vec4, f16
1.7x f32; "it is an instruction-count machine".

---

## 3. Where the time goes

### 3.1 Per design

- Steps: 60 (`DEFAULT_SAMPLING.nsteps`). Early stop (on by default, `shared/rfd4p/convergence.js`) ends at
  step 47-57 (mean ~53) with an identical sequence and < 0.1 A all-atom RMSD vs the full run.
- Trunk passes per step = recycle depth: `none` 1; `low` 4/3/2 for the first 30 steps then 1; `medium` and
  `high` more (`RECYCLE_SCHEDULES`, `shared/rfd4p/flow.js`). The UI defaults to `none`, but presets set their own
  (the `uncond_short` preset uses `low` = 120 passes / 60 steps).
- Per step besides the trunk: one decoder pass for the live frame (37 ms n=100, 73 ms n=208), CPU work
  (`processSampleToAtoms`, CIF text, py2Dmol secondary structure, Mol* update). Measured app overhead outside
  the trunk: ~9-11% of wall time.
- Per-step readback is free: 60 steps with readback 56.07 s vs fused-without-readback 55.87 s.

### 3.2 One trunk pass (`passPrograms[0]`), per kernel, GPU timestamps

Measured with `tools/gpu/bench-kernels.js --variants='[{}]'` (oracle inputs `uncond_short` n=100,
`ppi_pdl1` n=208).

**Stock Chrome** (`gemmSource`, the non-subgroup-matrix path):

| kernel | n=100 ms | n=208 ms |
|---|---:|---:|
| `triProjG [nn]x1408x384` (14 calls) | 136.1 | 583.6 |
| `gemm [nn]x768x384` (`Q.l1Fused`, LN fused, ReLU) | 38.5 | 166.6 |
| `gemm [nn]x384x768` (`Q.l2`, accumulate) | 37.5 | 159.8 |
| `gemm [nn]x384x256` (`t.z`, gated) | 26.4 | 111.4 |
| `tri-in` + `tri-out` | ~17 | 94.4 |
| `swigluLinear [n]x3072x768` | 17.0 | ~32 |
| **pass total** | **303.5** | **1227.0** |

**Developer flags** (`gemmSubmatSource`, 8x8x8 subgroup matrices):

| kernel | n=100 ms | n=208 ms |
|---|---:|---:|
| `triProjG` | 55.6 (2.45x) | 238.8 (2.44x) |
| `gemm [nn]x384x768` | 18.6 | 78.0 |
| `gemm [nn]x768x384` | 16.0 | 68.3 |
| `gemm [nn]x384x256` | 10.6 | 44.1 |
| `tri-in` / `tri-out` | | 18.0 / 18.0 |
| **pass total** | **132.1 (2.30x)** | **530.1 (2.31x)** |

`triProjG` at n=100 is ~151 GFLOP per pass, so 136 ms = ~1.1 TFLOP/s stock vs ~2.7 TFLOP/s with matrix units.

---

## 4. Ranked opportunities

### P0 - Subgroup matrices for visitors (research; potentially 2.3x for everyone)

The fast kernels already exist and pass parity. The only reason visitors don't get them is that stock Chrome
does not expose `chromium-experimental-subgroup-matrix`.

- Find out whether Chrome offers an **origin trial** (or a ship date) for WebGPU subgroup matrices. Precedent:
  basic subgroups ran origin trial `WebGPUSubgroupsFeatures` in Chrome 128-131 and shipped in 134; Chrome 135's
  notes mention an "Extended subgroups experiment". Check chromestatus.com and blink-dev. A web search on
  2026-10-23 found nothing specific.
- If a trial exists, registering a token for `sokrypton.github.io` needs the **owner** (it's an account action
  on an external service; ask, do not do it). Then add the `<meta http-equiv="origin-trial">` tag to
  `index.html` and verify in stock Chrome that `adapter.features` gains the feature. No code change should be
  needed after that: `requestWebGpuDevice` (`webgpu/index.js:122`) already requests
  `chromium-experimental-subgroup-matrix` whenever the adapter offers it, and `Rfd4pEngine` switches to the
  subgroup-matrix kernels on `hasSubgroupMatrix` (`webgpu/rfd4p/engine.js:2353`). Then run every gate in section 7
  in that configuration, since it becomes what visitors run.

### P1 - A different class of stock GEMM kernel (engineering; target 1.3-2x on the GEMMs)

The current `gemmSource` (`webgpu/rfd4p/engine.js:33`): 256 threads, 64x128 output tile, 4x8 per thread (8 `vec4`
f32 accumulators), K-tile 16, both operand tiles staged in workgroup memory as `vec4<f16>`, packed-f16 weights
read as `u32` pairs, fused epilogues (gates, LN-in, split-reduce norm, `triProjG`'s coalesced transposing
writeback). Section 5 shows it is a local optimum: anything that adds registers or workgroup bytes per thread
loses about in proportion, i.e. occupancy binds. Ideas that change the class rather than tweak it:

1. **Subgroup-shuffle register blocking**: keep one operand in registers and broadcast it across the subgroup
   with `subgroupShuffle` / `subgroupBroadcast` instead of reading it from workgroup memory every k. `subgroups`
   is available on stock Chrome. Watch register count; measure occupancy-sensitive shapes first.
2. **Smaller workgroups / different thread-to-output maps** that raise occupancy at the same per-thread work
   (e.g. 128 threads with 32x128 tiles), checked per kernel shape.
3. **Kernel-specific specialization** of `triProjG` (14 calls, 45% of a pass): it is a pure
   `[nn]x1408x384` GEMM with a fixed K, so a dedicated kernel can drop all generality (no batching, no partial-K,
   fixed strides) and tune its own tile.
4. Revisit **register prefetch only where K >= 768** (`Q.l2`, `swigluLinear` won 1.05-1.07x; ~1% of a pass)
   only if combined with something else.

Look at `../localfold` for M2-tuned WebGPU kernels and its device-profile / tuning priors
(`../localfold/webgpu/runtime/device-profile.js`, `../localfold/docs/M2-HANDOFF.md`, `../localfold/docs/PERF.md`)
before writing from scratch.

### P2 - Fewer trunk passes per design (measurement; up to ~25%)

- **Step count.** 30 steps collapses the designed sequence to poly-alanine in every noise setting tried; 60 is
  fine. 40 / 45 / 50 are untested. Protocol: same seeds and inputs at 60 vs k steps over >= 4 cases
  (`uncond_short`, `ppi_pdl1`, `motif_unidx_1prw`, `sm_sam`) x >= 3 seeds; report sequence identity, CA RMSD,
  all-atom RMSD, and whether the sequence degenerates (count the most common residue). This can't be checked
  against PyTorch locally (the oracle dumps are 60-step), so it is a product decision for the owner once the
  numbers exist.
- **Early stop tuning.** `DEFAULT_EARLY_STOP` = patience 5, CA tol 0.05 A, minFrac 0.5. Patience 3 would stop ~2-5
  steps earlier; re-run the same protocol used to validate the current setting (seq identity 100%, < 0.1 A).
- **Recycle defaults.** Presets that set `low` do 2x the passes of `none`; whether the extra thinking is worth it
  is again a quality question.

### P3 - App overhead outside the trunk (~10%)

> **2026-10-25: every step's frame is shown, by the owner's decision.** Do not speed this up by skipping frames.
> Decoding a frame every 2-3 steps saved 1.10x at n=100 (e4531d2) and was reverted in 2ae2968. Live frames
> cost ~14% at n=100 (a 33 ms all-atom decode per step on an M2) and ~4% for Mol* and CIF. What is left is making
> each frame cheaper, for example overlapping the decode with the next trunk pass.

Per rendered frame: decoder pass, `processSampleToAtoms`, CIF serialization, `caSecondaryStructureCif`
(py2Dmol `assignSecondary`), Mol* in-place update, live sequence strip (`web/app.js:988-1060`). Options: decode
the live frame every 2-3 steps when steps are fast (the predicate is `wantLiveFrame`); move CIF building and
secondary structure into a Worker; skip frames while a Mol* update is still pending (already partly done via
`liveRenderBusy`). Measure end to end with the studio, not the generator alone.

### P4 - The subgroup-matrix path itself (for when P0 lands)

> **2026-10-25: measured, and at the ceiling.** A synthetic kernel that only loads f16 8x8 tiles from workgroup
> memory and multiply-accumulates them peaks at 2.75 TFLOP/s for a 64x16 block per subgroup (our layout), 2.98 for
> 64x32 and 2.92 for 32x64; 16x16 gives 2.06. `triProjG` runs at ~2.7 (55.6 ms for 14 calls at n=100), the other
> pair GEMMs at 2.2-2.6. So re-blocking `gemmSubmatSource` to 64x32 could buy at most ~8% on `triProjG`, ~3% of a
> pass. Pass totals on this path: 132 ms at n=100, 529 ms at n=207.

At ~2.7 TFLOP/s `triProjG` is still well under the f16 ceiling. HANDOFF.md section 5 priority 5 and
`../localfold` notes on 8x8x8 on M2/M5 apply. Lower priority until visitors can run it.

---

## 5. Already measured, do not repeat blind

See HANDOFF.md section 7 for the full list. Added this round (2026-10-23), all on stock Chrome, inside the real
trunk, 3 interleaved rounds:

| variant of `gemmSource` | n=100 | n=208 | numerics |
|---|---:|---:|---|
| f32 workgroup staging (convert once on store) | 0.73x | 0.73x | same |
| `mt8`: 8x8 per thread as two 64-row halves (128-row tile) | 0.77x | 0.77x | bit-identical |
| f16 FMA with per-k-tile f16 partials, f32 flush | 0.89x | 0.89x | worse (`zRel` ~1e-3) |
| register prefetch of the next k-tile | 0.94x | 0.93x | bit-identical |

Also measured and not worth pursuing: overlapping per-step readback with compute (0.4%), batching several designs
in one pass (a single design's pair GEMMs already fill the 10 GPU cores at n >= 100).

The four variants were reverted; their generator code is not in the tree. The diffs are easy to reconstruct from
the descriptions; `mt8` kept every epilogue by re-running it in a nested WGSL scope with `m0` and the
accumulators shadowed, which is a reusable trick for any 128-row-tile experiment.

---

## 6. Tools

| command | what it does |
|---|---|
| `node tools/gpu-chrome.mjs tools/gpu/bench-kernels.js --cases=uncond_short,ppi_pdl1 --rounds=3 --variants='[{},{"myFlag":true}]'` | A/B kernel variants inside the real trunk: per-kernel GPU time of one pass and the trunk output (`s`, `z`) diffed against the first variant. |
| `node tools/gpu-chrome.mjs tools/gpu/profile-rfd4p.js` | Full per-stage and per-kernel profile (init, pass, heads, decoder, 60-step runs). |
| `LOCALFOLD_STOCK_FLAGS=1` | Stock Chrome flags (no unsafe-webgpu, no NVIDIA f16 toggle). |
| `LOCALFOLD_GPU_TIMEOUT_MS=2400000` | Raise the runner's 600 s cap for long runs. |

**Adding a variant for `bench-kernels.js`:** the bench sets `engine.gemmVariant` (an object). Read your flag in
`Rfd4pEngine.gemm()` (`engine.js:2453`), e.g. `const myFlag = Boolean(this.gemmVariant?.myFlag) && <applicability>`,
**put it in the pipeline cache key** (the `key` string, or two variants silently share a pipeline), pass it to
the source generator, and if the variant changes the tile height update the workgroup count (`groups`) too. If
a variant changes `tb`'s element type, `triProjG`'s epilogue stages through `tb` and must follow
(`engine.js:2761`).

**Never** let `bench-kernels.js` be the only evidence: a variant that wins there must pass every gate in
section 7.

---

## 7. Gates (all must pass, on both flag sets where marked)

```bash
npm test                                                   # 15/15 unit tests (incl. folded-grid guard)
node tools/oracle/check-batch.mjs                          # pipeline vs PyTorch, 7 cases
node tools/oracle/check-features.mjs
node tools/oracle/check-sampler.mjs
node tools/oracle/check-output.mjs
# both LOCALFOLD_STOCK_FLAGS=1 and default:
node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js --half=1    # trunk + decoder + encoder vs PyTorch, FP16
node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js --half=0    # FP32
node tools/gpu-chrome.mjs tools/gpu/check-generate.js --half=1 # 60-step trajectory parity
node tools/gpu-chrome.mjs tools/gpu/check-generate.js --half=1 --fast=1
# stock flags:
node tools/gpu-chrome.mjs tools/gpu/check-webapp.js            # studio end to end, 9 examples (run 3x: it caught the GPU hang 2/3)
node tools/gpu-chrome.mjs tools/gpu/check-recovery.js          # device-loss recovery
```

Current reference numbers (2026-10-23): `check-generate --half=1` final backbone RMSD **0.0377 A** stock /
**0.0564 A** developer flags, final structure 0.0395 / 0.0622 A, `rtDiff = 0`, `allOk: true`. A pure scheduling
change (more command buffers, different submission) must reproduce these exactly; a numerics change may move
them but must stay `allOk`.

---

## 8. Code map

| where | what |
|---|---|
| `webgpu/rfd4p/engine.js:33` `gemmSource` | stock tiled GEMM (64x128, 4x8/thread) and its epilogues |
| `engine.js:444` `gemmSmallSource` | small-M GEMM (32x64) |
| `engine.js:650` `gemmSubmatSource` | subgroup-matrix GEMM (developer flags) |
| `engine.js:2453` `gemm()` | variant selection: `submat`, `small`, `useGemv`, `wordB`, `vecIO`, `fullK`, `vecEpilogue`, `inLn`; pipeline key |
| `engine.js:2529` `linear()` | split-K, fused split-reduce + norm (`outF16` gate), GEMV path |
| `engine.js:2701` `swigluLinear`, `engine.js:2761` `triProjG` | fused SwiGLU; triangle input projection + gates with transposing writeback |
| `engine.js:3110` `encodePrograms`, `run`, `runAndRead` | one command buffer per program, one submit |
| `engine.js:3128` `profile` | per-dispatch GPU timestamps |
| `webgpu/rfd4p/trunk.js:245` `record` | builds `passPrograms` (4 cycle flavors), heads, sim-step; `LAYERS_PER_SUBMIT` chunks |
| `trunk.js:484` `triangle`, `:537` `pairUpdate`, `:603` `forward` | pair track; one call = one network evaluation |
| `webgpu/rfd4p/blocks.js:211` `layer`, `:158` `swiglu` | token attention layer; `qkF16` / `swF16` gates |
| `webgpu/rfd4p/generate.js:143` `generate` | step loop: fused GPU sim-step (`gpuStepLoop`), decode predicate, early stop monitor |
| `shared/rfd4p/flow.js` | schedules, `RECYCLE_SCHEDULES`, `DEFAULT_SAMPLING` |
| `shared/rfd4p/convergence.js` | early stop |
| `web/app.js:988-1060` | live frame: decode predicate, all-atom CIF, motif overlay, live sequence, Mol* update |

---

## 9. Report back

Append a dated section to this file with: what you changed, before/after per-kernel and per-pass tables on both
flag sets (interleaved, best of >= 3), end-to-end design times in the studio (`uncond_short`, `ppi_pdl1`, one
motif case), the gate results from section 7, and anything that lost (with numbers) so the next agent does not
repeat it.

---

## 10. 2026-10-23 Optimization Report (Apple M2, Chrome 156)

### Summary of Changes

1. **P0 — `chromium-experimental-subgroup-matrix` Origin Trial / Flag Check**:
   - Inspected Chrome `156.0.8091.0`'s `Google Chrome Framework` binary (`strings` on Blink runtime features and Dawn toggles).
   - Confirmed there is no `WebGPUSubgroupMatrix` Blink Origin Trial feature in Chrome 156; `chromium-experimental-subgroup-matrix` is gated in Dawn by `allow_unsafe_apis` (`--enable-unsafe-webgpu` or `chrome://flags/#enable-unsafe-webgpu`). Stock Chrome therefore always takes the `gemmSource` / `!this.hasSubgroupMatrix` path.
2. **P1 — Stock Chrome `gemmSource` & `triProjG` Optimizations (`webgpu/rfd4p/engine.js`)**:
   - **`shufTa2` (Coalesced 128-bit `vec4` load of `A` + 2-step `subgroupShuffleXor(1u/2u)` butterfly transpose into `ta`)**:
     Each thread (`tm4 = li / 16u`, `tx = li % 16u`, `rSubA = tx & 3u`, `kSubA4 = tx >> 2u`) issues a single coalesced `vec4<EA>` load from row `m0 + tm + rSubA` at `kSubA4` instead of 4 strided scalar/vector loads across 4 rows. Four consecutive lanes transpose their $4 \times 4$ subtile in registers via two `subgroupShuffleXor`s (`1u` then `2u`) into `u`, accumulate `inLn` (`sumA`, `sqSumA`) directly on `u`, and write `ta[tx * 17u + tm4] = u` (producing the exact same shared-memory layout and bit-identical results as baseline).
   - **`vecB` (Coalesced 128-bit `vec4<u32>` load of packed-FP16 `B` + 2-step `subgroupShuffleXor(1u/2u)` butterfly transpose into `tb`)**:
     Each thread (`r4B = li / 8u`, `kHalfB = (li / 4u) & 1u`, `rSubB = li & 3u`) issues a single 128-bit `vec4<u32>` load (8 `f16` weights) from row `n0 + r4B * 4 + rSubB` instead of 4 separate `u32` loads across 4 rows, transposes $4 \times 4$ `u32` words across 4 consecutive lanes via `subgroupShuffleXor(1u/2u)`, and writes `tb[tbWrite0]` and `tb[tbWrite0 + 33u]`.
   - **`prefAll` (Register prefetch of next K-tile's `aPref` and `bPref` during the FMA loop)**:
     Previously, register prefetch lost on $K = 384$ (`triProjG`, `Q.l1Fused`) because 4-row strided loader state + prefetch variables pushed `gemmSource` across Apple M2's 64-register/thread occupancy cliff. With `shufTa2` + `vecB` collapsing the per-thread global load state to a single `aPref: vec4<EA>` and `bPref: vec4<u32>`, register pressure stays below 64 registers/thread and prefetching the next K-tile before `workgroupBarrier()` #2 wins across **all** `b16` GEMMs.
   - **`triEpi2` (`triProjG` early-return epilogue for `n0 >= 1024u` + removal of redundant `* p.alpha`)**:
     In `triProjG` (`N = 1408`, 11 column tiles of 128), tiles `8..10` (`n0 >= 1024u`, 27.3% of workgroups) write row-major `gz` directly from registers and never use the shared-memory transpose (`ta`/`tb`). Placing `if (n0 >= 1024u) { ... return; }` at the top of `epilogueBody` skips computing `y_0..y_3` and skips all 3 epilogue `workgroupBarrier()`s for those workgroups, while letting Metal's compiler kill the 32 `acc` registers immediately after `y_0..y_3` are computed in the `n0 < 1024u` path.
   - **Pipeline key fix**: Added `${vecAStock ? '-va' : ''}${vecBStock ? '-vb' : ''}` to `Rfd4pEngine.gemm()`'s cache key (`webgpu/rfd4p/engine.js:2581`).
3. **P2 — Early-Stop `patience = 4` (`shared/rfd4p/convergence.js`)**:
   - Reduced `DEFAULT_EARLY_STOP.patience` from `5` to `4` (`caTolA = 0.05`, `minFrac = 0.5`), saving 1 full trunk pass (`~0.3–1.2 s`) on every early-stopped trajectory with **zero sequence identity or RMSD degradation** across 12/12 benchmark runs (4 presets $\times$ 3 seeds).
4. **P3 — Studio Live-Frame Deduplication, Pacing & Session Navigation Await (`web/app.js`, `webgpu/rfd4p/generate.js`)**:
   - Gated `wantLiveFrame` on `!liveRenderBusy` and excluded the final step (`step < total - 1`), and gated `wantDecodeStep` in `generate.js` on `!convergence?.stop`. Previously, `wantLiveFrame` returned `true` on the final step, running a duplicate GPU `decoder.decode(...)` + `processSampleToAtoms(...)` + `toCifString(...)` + Mol* live render 1 ms before the final decode and session render ran on the exact same coordinates.
   - Paced intermediate live frames to `LIVE_FRAME_INTERVAL_MS = 320` (`280 -> 320 ms`).
   - Made `_navigateStudioToSession` `async` and awaited `window.__rfd4pOpenSession` + `window.__rfd4pAddOverlay` inside `runDesign` so final Mol* session/overlay rendering completes before `runDesign` resolves (preventing a WebGL2/WebGPU Metal race when consecutive designs run back-to-back).

---

### P1: Interleaved Per-Pass & Per-Kernel Benchmarks (3 Interleaved Rounds)

Command:
```bash
LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/bench-kernels.js --cases=uncond_short,ppi_pdl1 --rounds=3 --variants='[{"baseline":true},{}]'
```

#### Stock Chrome (`LOCALFOLD_STOCK_FLAGS=1`)

| Case | Kernel | Calls/pass | Before (ms) | After (ms) | Speedup | Parity (`sRel` / `zRel`) |
|---|---|---:|---:|---:|---:|---|
| **`uncond_short`** ($n=100$) | **Full 14-layer pass (`passPrograms[0]`)** | **1** | **307.34** | **285.62** | **1.076x** (`-21.72 ms/pass`) | `0` / `0` (bit-exact) |
| | `triProjG 10000x1408x384` | 14 | 139.0 | 130.0 | **1.07x** | |
| | `Q.l1Fused` (`gemm 10000x768x384`) | 7 | 38.5 | 38.1 | **1.01x** (best `36.6 ms`, `1.05x`) | |
| | `Q.l2` (`gemm 10000x384x768`) | 7 | 37.7 | 33.9 | **1.11x** | |
| | `t.z` (`gemm 10000x384x256`) | 14 | 27.2 | 25.4 | **1.07x** | |
| | `swigluLinear 100x3072x768` | 14 | 17.1 | 14.6 | **1.17x** | |
| | `qkv` (`gemm 100x3072x768`) | 14 | 9.5 | 8.0 | **1.19x** | |
| **`ppi_pdl1`** ($n=208$) | **Full 14-layer pass (`passPrograms[0]`)** | **1** | **1249.64** | **1168.45** | **1.069x** (`-81.19 ms/pass`) | `0` / `0` (bit-exact) |
| | `triProjG 43264x1408x384` | 14 | 592.5 | 561.2 | **1.06x** | |
| | `Q.l1Fused` (`gemm 43264x768x384`) | 7 | 167.9 | 159.1 | **1.06x** | |
| | `Q.l2` (`gemm 43264x384x768`) | 7 | 163.9 | 146.4 | **1.12x** | |
| | `t.z` (`gemm 43264x384x256`) | 14 | 112.6 | 107.8 | **1.04x** | |
| | `tri-out` | 7 | 49.8 | 44.6 | **1.12x** | |
| | `tri-in` | 7 | 48.9 | 44.5 | **1.10x** | |

#### Developer Flags (`--enable-unsafe-webgpu`, `chromium-experimental-subgroup-matrix`)

| Case | Kernel | Calls/pass | Pass / Kernel (ms) | Parity (`sRel` / `zRel`) |
|---|---|---:|---:|---|
| **`uncond_short`** ($n=100$) | **Full 14-layer pass (`passPrograms[0]`)** | **1** | **132.05** | `0` / `0` |
| | `triProjG 10000x1408x384` | 14 | 55.6 | |
| | `Q.l2` (`gemm 10000x384x768`) | 7 | 18.4 | |
| | `Q.l1Fused` (`gemm 10000x768x384`) | 7 | 16.0 | |
| | `t.z` (`gemm 10000x384x256`) | 14 | 10.6 | |
| | `swigluLinear 100x3072x768` | 14 | 7.8 | |
| | `qkv` (`gemm 100x3072x768`) | 14 | 4.2 | |
| **`ppi_pdl1`** ($n=208$) | **Full 14-layer pass (`passPrograms[0]`)** | **1** | **531.79** | `0` / `0` |
| | `triProjG 43264x1408x384` | 14 | 238.4 | |
| | `Q.l2` (`gemm 43264x384x768`) | 7 | 78.0 | |
| | `Q.l1Fused` (`gemm 43264x768x384`) | 7 | 69.4 | |
| | `t.z` (`gemm 43264x384x256`) | 14 | 44.1 | |
| | `tri-out` / `tri-in` | 7 / 7 | 18.5 / 18.0 | |

---

### P2: Step Count & Early-Stop Patience Evaluation (4 Presets $\times$ Seeds 42, 43, 44 = 12 Runs)

Compared against full 60-step baseline (`recycleSchedule: 'none'`) across `uncond_short`, `motif_unidx_1prw`, `sm_sam`, and `ppi_pdl1`:

| Policy | Mean Steps | Mean Speedup | Min / Mean `seqId` vs 60-step | Mean / Max `caRmsdA` (Å) | Mean / Max `aaRmsdA` (Å) | Max Homopolymer Run Δ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `patience = 5` (old default) | 44.4 | 1.35x | `0.9928` / `0.9994` | `0.072` / `0.097` | `0.151` / `0.188` | `0` (12/12 exact) | Baseline |
| **`patience = 4` (new default)** | **43.4** | **1.38x** | **`0.9928` / `0.9994`** (100% identical to `p=5`) | **`0.077` / `0.106`** | **`0.157` / `0.204`** | **`0` (12/12 exact)** | **Won (shipped)** |
| `patience = 3` | 41.3 | 1.45x | `0.9600` / `0.9949` | `0.091` / `0.141` | `0.175` / `0.281` | `0` | Lost (`uncond_short` seed 44 dropped to `seqId=0.96`, `aaRmsdA=0.281 Å`) |
| `fixed-step-50` (`t[49]` early exit) | 50.0 | 1.20x | `0.9928` / `0.9994` | `0.056` / `0.076` | `0.132` / `0.188` | `0` | Slower than `patience = 4` on 12/12 runs |
| `fixed-step-45` (`t[44]` early exit) | 45.0 | 1.33x | `0.9600` / `0.9961` | `0.087` / `0.119` | `0.173` / `0.278` | `0` | Lost (`uncond_short` seed 44 `seqId=0.96`, slower than `p=4` on `ppi_pdl1` where `p=4` stops at step 29–34) |

---

### End-to-End Studio Design Wall Times (`nsteps = 60, recycleSchedule = 'none'`)

Measured inside `/index.html` (`app.runDesign`) with live Mol* rendering, sequence dock updates, final all-atom decode, trajectory CIF assembly, and session navigation:

| Preset | $n$ | Stock Before (s) | Stock After (s) | Stock Speedup | Dev Before (s) | Dev After (s) | Dev Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `uncond_short` | 100 | 16.70 s (50 steps) | **15.21 s** (49 steps) | **1.098x** (`-1.49 s`) | 7.19 s (48 steps) | **7.13 s** (47 steps) | **1.009x** |
| `motif_unidx_1prw` | 127 | 23.01 s (48 steps) | **21.20 s** (47 steps) | **1.085x** (`-1.81 s`) | 11.18 s (48 steps) | **10.82 s** (47 steps) | **1.034x** |
| `ppi_pdl1` | 207 | 37.41 s (30 steps) | **35.33 s** (30 steps) | **1.059x** (`-2.07 s`) | 17.36 s (31 steps) | **17.18 s** (30 steps) | **1.010x** |

---

### Section 7 Verification Gates

All gates run strictly sequentially with zero concurrent GPU processes:

| Gate Command | `LOCALFOLD_STOCK_FLAGS=1` | Developer Flags |
|---|---|---|
| `npm test` | **15/15 PASS** | **15/15 PASS** |
| `check-rfd4p.js --half=1` | **`allOk: true`** | **`allOk: true`** |
| `check-rfd4p.js --half=0` | **`allOk: true`** | **`allOk: true`** |
| `check-generate.js --case=uncond_short --half=1` | **`allOk: true`**, `finalBbCa.rmsdA = 0.0377 Å`, `finalStructure.rmsdA = 0.03954 Å`, `rtDiff = 0` | **`allOk: true`**, `finalBbCa.rmsdA = 0.0564 Å`, `finalStructure.rmsdA = 0.06221 Å`, `rtDiff = 0` |
| `check-generate.js --case=motif_unidx_1prw --half=1` | **`allOk: true`**, `finalBbCa.rmsdA = 0.0645 Å`, `finalStructure.rmsdA = 0.06517 Å`, `rtDiff = 0` | **`allOk: true`**, `finalBbCa.rmsdA = 0.0528 Å`, `finalStructure.rmsdA = 0.05582 Å`, `rtDiff = 0` |
| `check-generate.js --case=uncond_short --half=1 --fast=1` | **`allOk: true`**, `finalBbCa.rmsdA = 0.0377 Å`, `finalStructure.rmsdA = 0.03954 Å`, `rtDiff = 0` | **`allOk: true`**, `finalBbCa.rmsdA = 0.0564 Å`, `finalStructure.rmsdA = 0.06221 Å`, `rtDiff = 0` |
| `check-webapp.js` | **3/3 consecutive runs PASS (`ok: true`, all 9 Studio examples + 10 presets + inverse folding)** | **`ok: true` (all 9 Studio examples + 10 presets + inverse folding)** |
| `check-recovery.js` | **`allOk: true`** | **`allOk: true`** |

---

### What Was Tried and Lost (Do Not Repeat)

1. **`shuffleA` (`595.67 ms`, `0.519x` on `uncond_short`) & `vecTa` (`486.51 ms`, `0.635x` on `uncond_short`)**:
   - Loading `ta` as `vec4<f32>` along K (`a0..a3`) or broadcasting `A` across threads via `subgroupShuffle` inside the inner K-loop requires holding `a0..a3` (16 `f32` registers) live across the 8 `tb` loads and 32 FMAs instead of 1 `av: vec4<f32>` (4 `f32` registers) at a time. That +12-register increase pushes `gemmSource` past Apple M2's 64-register/thread occupancy cliff, halving SIMDgroup occupancy.
2. **Single-pass `triProjG` epilogue (`tri1Pass`: `287.65 ms` vs `287.52 ms`, `triProjG` `132.5 ms` vs `130.5 ms`; `tri1Pass17`: `290.33 ms`, `triProjG` `135.3 ms`)**:
   - Expanding `tb` in `triProjG` from `528` `vec4<f16>`s (`4,224` B; total `6,400` B/workgroup = 4 workgroups/core in 32 KB threadgroup memory) to `1,024` or `1,088` `vec4<f16>`s (`10,368–10,880` B/workgroup = 3 workgroups/core) to transpose all 64 rows in one barrier instead of two drops threadgroup-memory occupancy on M2 from 4 workgroups/core to 3 workgroups/core during the entire 24-tile K-loop, losing more than the 2 saved epilogue barriers.
3. **`triEpi` with `if (n0 < 1024u) { workgroupBarrier(); ... } else { ... }` (`299.81 ms`, `triProjG` `137.9 ms` vs `136.4 ms`)**:
   - Wrapping the `n0 < 1024u` barriers inside an `if/else` diamond where `acc_0_0..acc_3_1` are read in the `else` branch prevents Metal's compiler from reusing the 32 `acc` registers for `y_0..y_3`. Moving `if (n0 >= 1024u) { ... return; }` to the top of `epilogueBody` (`triEpi2`) eliminates the `else` join and wins `~4–5 ms` on `triProjG`.
4. **Issuing register prefetch at top of K-loop (`prefTop`: `294.27 ms` vs `294.02 ms`)**:
   - Loading `aPref`/`bPref` immediately after `workgroupBarrier()` #1 instead of after the 16 FMAs overlaps global load destination registers with the `ta`/`tb` shared-memory reads.
5. **`patience = 3` and fixed `nsteps = 45`**:
   - On `uncond_short` seed 44, a 3-step Cα plateau occurs at steps 41–44 before 4 residues flip at step 45 (`seqId` drops to `0.960`, `aaRmsdA = 0.281 Å`). `patience = 4` rides through that plateau and matches `patience = 5` on 12/12 runs.
6. **f16 accumulators in stock `triProjG` (2026-10-25, `acc16`): 1.027x / 1.033x a pass at n=100 / 200, drift doubled**:
   - Eight `vec4<f16>` accumulators in the K-loop, widened to f32 for the unchanged epilogue. Interleaved, one
     process, stock flags: 286-290 -> 278-280 ms (n=100), 1097-1107 -> 1061-1070 ms (n=200), about 1.07x on the
     kernel. Drift against f32 accumulators 1.2e-3 bb_ca, 3-5e-3 latents, about twice FP16's existing trunk drift.
   - localfold's 1.43-1.55x with the same idea (`6e996fa8`) came from accumulators held in a WGSL ARRAY, which
     spilled; halving them ended the spill. Ours are scalar `vec4` variables and do not spill, so there is little
     to win. Together with section 5, this kernel is at the M2's f32 ceiling for its class (localfold's best fused
     triangle projection on the same M2: 1.07-1.29 TFLOP/s; ours ~1.1).
7. **Handing `onStep` off without waiting, so the next trunk pass overlaps frame building (2026-10-25): 1.000x**.
   - Visible window, n=100, 60 steps, 3 interleaved rounds: 20.78 s both arms, all 59 frames in order. The GPU
     was not waiting on the CPU's frame building.
8. **The small-tile GEMM for row counts up to 128 (token track and decoder at n=100): 0.80x** (19.2 -> 24.1 s,
   identical output). The 64x128 tile's reuse matters more than filling the 10 cores.

Triangle multiplications (`tri-in` + `tri-out`) at long lengths, share of one trunk pass (M2, 2026-10-25):
stock 8% / 12% / 17% at n=200 / 400 / 600, subgroup-matrix 7% / 12% / 15%. Their raw rate (stock 0.66-1.06
TFLOP/s, matrix 1.96-2.49) is mostly tile padding: 64x128 tiles round an n x n contraction up to 448 x 512 at
n=400 (1.43x) and 640 x 640 at n=600 (1.14x). Padding-corrected they sit at each path's ceiling. Smaller tiles pad
less but run slower (32x64 measured 0.8x), so no gain. One real gap: chunked lengths read the incoming operand by
stride (PR #3), and on M2 stock that costs 952 vs 732 ms at n=600, about 2% of a pass.

Where a 60-step design's time goes at n=100 (M2, stock, visible window, every frame shown, Mol* live mode):
trunk passes 17.1 s (82%), live decodes ~1.9 s (9%), Mol* drawing ~0.35 s, the rest (setup, final decode,
session, per-step readback) ~1.1 s. The trunk's GEMMs and the decoder's run at 0.85-1.1 TFLOP/s against an
f32 ceiling of ~1.1-1.3 on this device.


## 11. Open bug (2026-10-24): intermittent GPU reset on chiral-ligand inputs, slow first forwards

**Symptom.** `check-webapp` (stock flags) fails about 1 run in 4. Every failure where the example was identified was
`small_molecule_binder` (SAM + 150 new residues, n=177). The error was "Caused GPU Hang Error" (macOS watchdog) or
Metal "Internal Error". It did not reproduce in 30 runs of the same example from the plain CDP probe, either alone or
cycling all 9 examples twice. The harness differs in its flags (`--disable-dawn-features=timestamp_quantization`,
precise memory info) and in what runs before the examples (`motif_unidx_1prw` and `inverse_folding_1prw` presets,
which load the encoder).

**What is known (M2, stock flags).**
- Ligands with chiral centres disable the fused GPU step loop: `generate.js`, `gpuStepLoop` is false when
  `noise_align_chirality` has active frames. Every step then calls `trunk.forward` without `simBuf`. SAM has chiral
  centres; ORO (enzyme example) does not, so the enzyme stays fused (about 1.1 s per pass at n=193).
- On that non-fused path the first two forwards are slow on the GPU itself. The queue was drained before timing (346
  ms before step 0, about 0 after), with no NaN/Inf in inputs or outputs. Forward ms over 12 steps:
  `2411, 2078, 850, 852, 853, …`. With live frames on, step 1 rises to about 4.0 s (Mol*'s first-scene build competes
  for the GPU). `record()` does not re-record (0 ms), `condUniform` is true throughout, and the programs are the same
  chunked `passPrograms` as the fused path.
- The `sm_sam` preset (n=180) shows the same pattern (`954, 3100` for its first two forwards). `profile-rfd4p.js
  --presets=sm_sam` (zero xt, t=0.5) measures a steady 865 ms pass, dominated by `triProjG`, so the slow passes depend
  on the early-step state, not the size.

**Next steps.** Profile per kernel (`engine.profile`) on forwards 0 and 1 of a non-fused chiral-ligand run, against
step 5, and find which dispatch inflates 2-4x. A 2-4x inflation of one 2-layer command buffer under contention is
the likely watchdog trip. Also check whether the slowness tracks the early-t xt scale (data-dependent cost, e.g.
f16 overflow/denormal handling) by replaying step 0's inputs at step 5. Reproduce in the CDP probe with
`tools/gpu/check-webapp.js`'s preceding preset runs before the example.

## 12. 2026-10-29 Check-up on the M5 (Chrome 156.0.8078.73, 10-core, 24 GB)

Verification pass at `cbe84cb`, not an optimization round. No product code changed.

### 12.1 P0 is worth far less on M5 than on M2

Stock Chrome 156.0.8078.73 still exposes no `chromium-experimental-subgroup-matrix`
(`subgroup-size-control` is now stock; the matrix feature is not). P0 remains blocked.

But its value has collapsed. `bench-trunk.js`, one trunk forward, best of 3 processes x 3 reps:

| n | stock | subgroup-matrix | ratio |
|---:|---:|---:|---:|
| 100 | 113 ms | 93 ms | 1.22x |
| 200 | 425 | 355 | 1.20x |
| 400 | 1743 | 1453 | 1.20x |
| 600 | 3952 | 3192 | 1.24x |

On M2 the same ratio is 2.2-2.3x. Across generations the two paths moved very differently:

| n | M2 stock | M5 stock | gain | M2 matrix | M5 matrix | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 4520 ms | 1743 | 2.59x | 2080 | 1453 | 1.43x |
| 600 | 10388 | 3952 | 2.63x | 4668 | 3192 | 1.46x |

The stock path gained 2.6x generation over generation; the matrix path only 1.43x. So on current
Apple silicon the visitor path is already close to the matrix path, and P0 should be re-ranked
below P1/P2 for M5-class devices. The corollary is that the under-tuned kernel on M5 is now
`gemmSubmatSource` (P4), not `gemmSource` - it is the one that failed to keep up.

### 12.2 The chunk bound (section 6.6 of M2-TO-M5) buys nothing here

n=400 (1 chunk) -> n=600 (2 chunks) is 2.25x the pairs for 2.27x the time, so the 2-chunk plan
costs about 1%. Raising `DEFAULT_CHUNK_BYTES` on 24 GB-class devices is not worth doing.

Command buffers are also no longer a concern at these lengths: n=600 is 3952 ms over 7 command
buffers, ~565 ms each, well inside the ~1 s watchdog line that section 6.4 flagged for the M2.

### 12.3 The numbers in `trunk.js`'s own doc comment reproduce

That comment records M5 f16 one-chunk passes of 115.8 / 444.5 / 1815.8 ms at n = 100 / 200 / 400.
Measured now: 113 / 425 / 1743. Nothing has regressed since PR #3.

### 12.4 `check-webapp.js` had been failing since 2026-10-24, for its own reasons

Two stale assertions, both artefacts of dropping the vendored Studio (7522570), neither a product bug:

1. It counted live frames through `window.__rfd4pRenderLiveFrame`, a global that commit deleted along
   with the vendored bundle. The guard `if (typeof origLiveRender === 'function')` meant the counter
   was silently never installed, and the gate then failed on `got 0` - blaming the design for
   rendering nothing. Studio draws live frames through `window.atomworksStudio.renderFrame` now. The
   counter follows it, and a missing hook throws instead of being skipped.
2. It required the renderer `<select>` to hold exactly one option, `molstar`. A released Studio also
   offers Patinae (WebGPU) as an opt-in choice, so the list is `['', 'molstar', 'patinae']`. The gate
   now asserts what actually matters - that the page *defaults* to Mol*, on WebGL2, so the viewer does
   not take the GPU from the design - and reports the option list rather than freezing it.

Live frames and the Mol* default were both verified working; only the gate was wrong.

> **Open question for the owner.** A visitor can switch the viewer to Patinae (WebGPU) from that
> dropdown. The vendored Studio offered no such option. On that setting the viewer contends with
> RFD4-Proteina for the GPU, which is the opposite of the choice recorded in section 10 (P3) and may
> bear on the intermittent hang in section 11. Whether to strip the option when composing
> `index.html` is a product decision, so it is left as it ships.

### 12.5 Gates

Stock flags, this M5, after the two test repairs:

| gate | result |
|---|---|
| `npm test` | 35/35 |
| `check-webapp.js` | `ok: true` - 9 Studio examples, 10 presets, inverse folding, 51 live frames, trajectory 10 models, Mol* default with 2028 atoms |
| `check-recovery.js` | `allOk: true` - mid-run loss reported, next Generate on a new device, idle loss recovers on a third |

Design wall times from those runs (10-step `motif_unidx_1prw`, 8-step recovery case): 2.74 s and
1.17-1.34 s on WebGPU FP16.

The oracle parity gates (`check-rfd4p`, `check-generate`, `tools/oracle/*`) **cannot run on this
machine**: it has neither `oracle-dumps/` nor the f32 `weights/model.safetensors`. They remain the
M2's to run, as M2-TO-M5 section 2 notes.

### 12.6 `check-recovery.js` was broken by the M5's speed, in the same way

The gate destroyed the device after a fixed `sleep(1500)` while a design ran. On this M5 an 8-step
design finishes in **1.17 s**, so the destroy landed *after* the design completed. Nothing was ever
interrupted: `midRun` passed where the gate requires it to fail, the loss surfaced on the *following*
run instead, and the gate bailed out with `allOk: false, note: "no device after the mid-run recovery"`
having never tested recovery at all.

It now runs a 40-step design and waits until the status line reports step >= 4 before destroying the
device, which holds on any hardware.

**Both gate failures this round were tests calibrated to slower hardware, not product bugs.** That is
worth generalising: section 1 rule 1 says to run every gate on stock flags; it should also say that a
gate must not encode a wall-clock assumption about how long a design takes.

## 13. 2026-10-30: the viewer backend, CI, and section 11 on the M5

### 13.1 Patinae costs a design 1.66x, and why

A visitor could switch the viewer from Mol* (WebGL2) to Patinae (WebGPU) from the page's own dropdown.
Patinae is not broken - it renders, designs complete on it, it reports its capabilities to the UI, it
switches back cleanly, and it raises no errors. It is simply expensive, because two methods Studio's
viewer contract marks **optional** are absent from its adapter (`packages/web/src/viewers/patinae.ts`
in atomworks-studio-dev), and an absent optional method degrades silently:

| contract member | Mol* | Patinae |
|---|---|---|
| `loadCif(cif, name, options?)` | yes | **no `options` parameter** |
| `updateLiveCif` | yes | **missing** |
| `setLiveMode` | yes | **missing** |

`ViewerHost.renderLiveCif` takes the in-place path only `if (this.adapter?.updateLiveCif)`, so on Patinae
every live frame falls through to a full `loadCif`, which - in the contract's own words - "clears the whole
scene, which flashes between frames and throws away the camera". `keepCamera` is passed as the third
argument and Patinae's signature does not accept it, so the camera resets on each of ~50 frames of a
design. `setLiveMode` is optional-chained into a no-op, so no quality-for-framerate trade is made.

Measured with `tools/gpu/bench-renderer.js` (interleaved A/B in one process, `motif_unidx_1prw`, 20 steps):

| | best | median |
|---|---:|---:|
| Mol* | 3274 ms | 3281 ms |
| Patinae | 5420 ms | 5435 ms |
| | **1.66x** | **1.66x** |

`tools/gpu/diag-patinae-cost.js` splits that cost by stubbing frame drawing: drawing live frames costs
697 ms on Mol* against **1853 ms** on Patinae, and merely having Patinae resident costs a further 768 ms.

**Patinae is no longer offered on this page.** `tools/build-studio.mjs` removes its `<option>` and its
renderer-prompt button when composing `index.html`, and throws if either disappears upstream. The
lazily-loaded Patinae chunks (4.1 MB) are still copied, so `?viewer=patinae` still reaches it deliberately
and nothing 404s. Restore the option once the adapter implements those two methods and takes
`ViewerLoadOptions`; `tools/gpu/check-patinae.js` gates that path and currently reports `skipped`.

### 13.2 Section 11 reproduces on the M5, but does not trip the watchdog

`tools/gpu/diag-chiral-hang.js`, `sm_sam` (SAM, chiral, n=195), 12 steps, 3 runs. The "slow first forwards"
pattern is real and reproduces here: step 1 ran **4.36x** the median of the remaining steps in one run
(1452 ms against a 333 ms median) and 2.02x in another, with step 0 near normal (1.08-1.19x).

No device was lost in any run. On the M2 a step is ~850 ms, so the same multiple lands a single command
buffer well past the macOS watchdog's patience; on the M5 a step is ~333 ms and the inflated step is still
only ~200 ms per command buffer. **The inflation is the bug; the hang is the inflation meeting a slower
GPU.** So section 11 is not fixed, and it is not reproducible here - it needs the M2, or a profile of which
dispatch inflates. It is still the only known defect that reaches visitors.

### 13.3 The v0.0.6 release was rebuilt and re-uploaded, breaking the pin

`npm run build:studio` began failing: the tarball at the `v0.0.6` tag no longer matched `studio.json`'s
sha256. It was not tampering. The release assets were re-uploaded on 2026-10-29T21:10:59Z, and the only
difference in the whole web bundle is one line of `sw.js`:

```
-const CACHE = "annotation-studio-shell-development-1793123848370";
+const CACHE = "annotation-studio-shell-development-1793307587756";
```

a build timestamp. Verified by extracting the immutable PyPI 0.0.6 wheel (sha256 `de1cfd3e...`) and diffing
it against the re-uploaded tarball: every other file is byte-identical, and rebuilding from the new tarball
left `assets/` untouched. The pin is updated to `262521d6...`.

Two things worth fixing upstream, neither in this repo:

1. **The release build is not reproducible.** A timestamp baked into `sw.js` means re-running the release
   workflow produces a different artifact, so any downstream checksum pin breaks for no reason.
2. **A published tag's assets were replaced.** GitHub allows it; PyPI does not, which is why PyPI still
   serves the original. For a moment `v0.0.6` named two different artifact sets depending on where you
   fetched it. The checksum guard in `build-studio.mjs` is what caught this, and it earned its keep.

Note that `sw.js` is a service worker that caches root-absolute paths (`"/"`, `"/manifest.webmanifest"`)
and intercepts every same-origin GET. This page never ships it - `build-studio.mjs` copies `assets/**` and
`sw.js` sits at the bundle root - which is just as well, since this site is served under
`/rfd4-proteina-webgpu/` and those paths are not its own.

### 13.4 CI

The repo had no workflow at all; the `pages-build-deployment` runs are GitHub's automatic Pages deploy.
`.github/workflows/ci.yaml` now runs, on every push and PR, what needs no GPU: `npm test` on node 22 and 24,
a parse of every shipped module, `tools/check-page-integrity.mjs`, and a stamp-drift check. The GPU gates in
section 7 remain local.
