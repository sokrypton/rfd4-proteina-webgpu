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

Per rendered frame: decoder pass, `processSampleToAtoms`, CIF serialization, `caSecondaryStructureCif`
(py2Dmol `assignSecondary`), Mol* in-place update, live sequence strip (`web/app.js:988-1060`). Options: decode
the live frame every 2-3 steps when steps are fast (the predicate is `wantLiveFrame`); move CIF building and
secondary structure into a Worker; skip frames while a Mol* update is still pending (already partly done via
`liveRenderBusy`). Measure end to end with the studio, not the generator alone.

### P4 - The subgroup-matrix path itself (for when P0 lands)

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
