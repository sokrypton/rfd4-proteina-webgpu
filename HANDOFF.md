# HANDOFF — RFD4-Proteina WebGPU Port (`localdesign`) for Apple Silicon (M2 & M5)

Updated 2026-10-22 for the Apple Silicon (M2 / M5) verification and tuning pass. Read this first, then [`AGENTS.md`](file:///home/ubuntu/localdesign/AGENTS.md).

---

## 1. Current Status — 100% Complete & Differential-Verified

Every subsystem from [`HANDOFF.md`](./HANDOFF.md)'s original roadmap is implemented, verified against the PyTorch oracle dumps, and optimized (**5.01x–5.36x** end-to-end speedup on NVIDIA Ampere):

| Subsystem | Files | Verification & Status |
|---|---|---|
| Input pipeline (`.cif.zst`, mmCIF/PDB, YAML/JSON spec, `sel(...)`, 3D chirality, bond BFS) | `shared/rfd4p/pipeline/*.js` | `node tools/oracle/check-batch.mjs` (6/6 oracle cases exact) + `npm test` |
| Feature factories & static pair cache | `shared/rfd4p/features.js` | `node tools/oracle/check-features.mjs` (6/6 cases exact to `~1e-7`) |
| Flow-matching schedules, `vf_tsr`, partial diffusion, inverse folding | `shared/rfd4p/flow.js`, `shared/rfd4p/partial.js` | `node tools/oracle/check-sampler.mjs` (6/6 cases) |
| Structure output (Hungarian unindexed motif assignment, internal-coord grafting, Kabsch, CIF/PDB) | `shared/rfd4p/output.js` | `node tools/oracle/check-output.mjs` (6/6 cases, `< 1e-4 Å` RMSD) |
| CPU reference (Trunk, Decoder, Encoder, Generator) | `cpu/rfd4p/{trunk,decoder,encoder,reference}.js` | `node tools/oracle/check-{cpu-trunk,decoder,encoder}.mjs` |
| WebGPU Engine, Trunk, Decoder, Encoder, Generator | `webgpu/rfd4p/{engine,blocks,trunk,decoder,encoder,generate}.js` | `npm run test:gpu` (`check-rfd4p.js`, FP32 & `--half=1`, with & without `--no-submat=1`) + `check-generate.js` |
| Interactive Browser Studio & Block-64 INT8 Sharded Weights | `index.html`, `web/app.js`, `shared/rfd4p/safetensors.js` | `node tools/gpu-chrome.mjs tools/gpu/check-webapp.js` (10/10 presets, 224 MB INT8 shards) |

---

## 2. How to Run Tests & Profiles on macOS (M2 / M5)

`tools/gpu-chrome.mjs` automatically launches `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` on `darwin` (or `$LOCALFOLD_CHROME` if set). No `DISPLAY=:99` is needed on macOS.

```bash
# 1. Node unit tests (8/8 tests: WGSL bounds guards, encoding, pipeline, features, flow, output, INT8 shards)
npm test

# 2. Full WebGPU differential oracle suite (Trunk + Decoder + Encoder) in FP32 and FP16
node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js
node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js --half=1

# 3. Full 60-step recorded-RNG trajectory parity vs PyTorch oracle (with and without step readback)
node tools/gpu-chrome.mjs tools/gpu/check-generate.js --half=1
node tools/gpu-chrome.mjs tools/gpu/check-generate.js --half=1 --fast=1

# 4. Interactive browser studio end-to-end verification (10 presets + inverse folding + py2Dmol + PDB/CIF)
node tools/gpu-chrome.mjs tools/gpu/check-webapp.js

# 5. Detailed per-kernel GPU timestamp profiler + 60-step end-to-end benchmarks
node tools/gpu-chrome.mjs tools/gpu/profile-rfd4p.js
LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/profile-rfd4p.js
```

> **Note on Weights & Oracle Dumps**:
> - `weights/int8/` (224 MB across 6 block-64 INT8 shards + `manifest.json` + `config.json`) is committed/present in the workspace and used by `profile-rfd4p.js` and `web/app.js`.
> - `weights/model.safetensors` (1.74 GB FP32) and `oracle-dumps/*.safetensors` are used by `check-rfd4p.js` and `check-generate.js`. Copy or symlink `weights/model.safetensors` and `oracle-dumps/` from the sibling checkout if not already present.

---

## 3. What Was Done to Prevent Overfitting to NVIDIA Ampere

1. **Proper `16x16x16` Subgroup-Matrix Capability Check ([`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L2135-L2158), [`webgpu/index.js`](file:///home/ubuntu/localdesign/webgpu/index.js#L149-L151))**:
   - On NVIDIA Ampere, `adapter.info.subgroupMatrixConfigs` exposes `16x16x16` (`f16 -> f32`).
   - On Apple Silicon M2 and M5 (when Chrome is launched with `--enable-unsafe-webgpu`), `adapter.info.subgroupMatrixConfigs` exposes **`8x8x8` ONLY** (`f32 -> f32` and `f16 -> f16` where `f16` accumulates in `f16`). In stock Chrome (`LOCALFOLD_STOCK_FLAGS=1`), `chromium-experimental-subgroup-matrix` is off on both M2 and M5.
   - `Rfd4pEngine` now inspects `device.adapterInfo.subgroupMatrixConfigs` (populated from `adapter.info` in `requestWebGpuDevice()`) and only enables `this.hasSubgroupMatrix` when a `16x16x16` (`f16 -> f32`) configuration is supported (and `device.noSubmat !== true`).
   - This prevents Metal shader compilation failures (`subgroup_matrix_left<f16, 16, 16>`) on M2/M5 when running under `tools/gpu-chrome.mjs` (which passes `--enable-unsafe-webgpu`).

2. **Decoupled `shader-f16` + `subgroups` Kernels from `hasSubgroupMatrix`**:
   - Both M2 and M5 natively support `shader-f16` and `subgroups` (`simdgroup` size = 32, matching our 32-lane subgroup reductions) in **stock Chrome**.
   - All subgroup + `f16` kernels that do **not** use `chromium_experimental_subgroup_matrix` are gated on `this.hasF16` (not `this.hasSubgroupMatrix`), so M2 and M5 get them automatically:
     - `gemvSource` (`m in {1, 2}` zero-barrier subgroup GEMV with fused `rmsCond` / `lnCond` / `rmsPre`)
     - `swigluGemvSource` (`m in {1, 2}` zero-barrier fused SwiGLU GEMV)
     - `precomputeCond` (single-dispatch conditioning across all 14 trunk / 12 AE layers)
     - `pairAdaln` (`rmsPre` + linear fused into 1 GEMV dispatch)
     - `qkvgQkNorm` (2-way Split-K + fused `splitReduceQkNormSource`)
     - `splitReduceNormSource`, `normSource`, `pairToBiasSource`, `transposeLnOutSource`, `pairWriteLnInSource`, `headFastSource`, `softmaxSource`

3. **Verified `--no-submat=1` Fallback Path**:
   - Passing `--no-submat=1` to `check-rfd4p.js`, `check-trunk.js`, `check-generate.js`, or `profile-rfd4p.js` forces `this.hasSubgroupMatrix = false` (`gemmSource` `64x128` and `gemmSmallSource` `32x64` bank-conflict-free `vec4` register-FMA kernels + `pairToBiasSource`).
   - Verified 100% passing in both FP32 and FP16 (`check-rfd4p.js --half=1 --no-submat=1`: `allOk: true`, `v bb_ca relRms = 0.000400`, `rtDiff = 0`, `maskDiff = 0`).

4. **GPU-Resident Step Loop (`simStepProgram`) Active With & Without `onStep` ([`webgpu/rfd4p/trunk.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/trunk.js#L393-L436), [`webgpu/rfd4p/generate.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/generate.js#L225-L332))**:
   - `simStepProgram` updates `seqF` in place on the GPU and writes both `xt` (`B.out[0 .. 35*n]`) and clean estimate `x1` (`B.out[182*n .. 217*n]`).
   - Both headless batch runs (`!opts.onStep`) and interactive browser runs (`opts.onStep` in `web/app.js`) use `headFastProgram` + `simStepProgram` on the GPU, avoiding per-step CPU `trunkSeqFeatures`, `writeBuffer(B.seqF)`, and `sequence_readout` `[n, 768] x [147, 768]^T` GEMM dispatches.

---

## 4. Reference Numbers on NVIDIA Ampere (A100)

| Benchmark / Metric | Initial Baseline | Current (Ampere `submat`) | Speedup |
|---|---:|---:|---:|
| `uncond_short_low_60steps` (`n = 100`, 120 passes) | `5,378.7 ms` | **`1,002.7 ms`** | **5.36x** |
| `uncond_short_medium_60steps` (`n = 100`, 180 passes) | `7,620.4 ms` | **`1,475.9 ms`** | **5.16x** |
| `ppi_pdl1_low_60steps` (`n = 207`, 120 passes) | `15,196.6 ms` | **`3,034.0 ms`** | **5.01x** |
| `fwd3WallMs` (`n = 100` / `n = 207`, 3 passes) | `71.02 ms` / `120.75 ms` | **`34.96 ms`** / **`77.66 ms`** | **2.03x** / **1.55x** |
| `initGpu` + conditioning (`n = 100` / `n = 207`) | `1.240 ms` / `1.421 ms` | **`0.284 ms`** / **`0.342 ms`** | **4.37x** / **4.15x** |
| `passGpu` single trunk pass (`n = 100` / `n = 207`) | `15.468 ms` / `111.737 ms` | **`10.901 ms`** / **`27.468 ms`** | **1.42x** / **4.07x** |
| `decoderGpu` (`n = 100` / `n = 207`) | `6.143 ms` / `7.717 ms` | **`4.113 ms`** / **`4.790 ms`** | **1.49x** / **1.61x** |

---

## 5. Checklist & Tuning Guide for the M2 / M5 Agent

Lessons from `../localfold/docs/M5-HANDOFF.md` and `../localfold/webgpu/runtime/device-profile.js` that apply directly here:

1. **Step 1 — Run Correctness Gates First (No Code Changes)**:
   - Run `npm test`, `node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js`, `node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js --half=1`, and `node tools/gpu-chrome.mjs tools/gpu/check-generate.js --half=1`.
   - Confirm `allOk: true` on both M2 and M5.

2. **Step 2 — Profile Baseline on M2 and M5 (`profile-rfd4p.js`)**:
   - Run `LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/profile-rfd4p.js` (and without `LOCALFOLD_STOCK_FLAGS=1`) on both M2 and M5.
   - Note that Apple laptops throttle under sustained load and drift 5–20% between back-to-back runs; **always interleave A/B arms in the same process** before trusting a <10% difference.

3. **Step 3 — Apple Silicon Specific Tuning Levers to Investigate**:
   - **Inner-Tile `f16` vs `f32` FMA Accumulation in `gemmSource` & `gemmSmallSource` ([`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L198-L226))**:
     - When `b16 = true` (`half = true && hasF16`), `gemmSource` and `gemmSmallSource` stage `ta` and `tb` as `vec4<f16>` in workgroup memory (12.8 KB total, well within Apple's 32 KB threadgroup memory limit) and accumulate each 16-step K-tile in `vec4<f16>` (`p_0_0 = fma(...)`) before adding `vec4<f32>(p_0_0)` to `acc_0_0`.
     - In `../localfold`, M2 saw 1.7x higher throughput from `f16` ALU whereas M5 saw 1.3x, and on some kernels M2 preferred `f32` accumulation over `f16`. Race `b16` inner-loop `f16` vs `f32` accumulation on M2 and M5.
   - **Workgroup Tile Occupancy (`gemmSmallSource` Crossover & Split-K Thresholds in [`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L2261-L2277) and [`linear()`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L2310))**:
     - The current `small` (`32x64` vs `64x128`) crossover (`m * B0 * B1 <= 1024 || n <= 64`) and Split-K threshold (`rows <= 512 && inC >= 768`) were tuned on an 80-SM A100.
     - An M2 (10 GPU cores) and M5 (8–10 GPU cores) need ~8–10x fewer workgroups to saturate all cores than an A100! At `n = 100..207`, 4-way/8-way Split-K on token projections (`rows = 100..207`) may create unnecessary `splitReduce` traffic on M2/M5 where `64x128` or `32x64` without Split-K already fills every Apple GPU core. Sweep the Split-K `rows` crossover on M2 and M5!
   - **FP16 Pair Activations (`triF16` / `xnF16`) on Non-`submat` Path**:
     - Currently `triF16` (`trunk.js`) and `xnF16` (`blocks.js`) are gated on `this.hasSubgroupMatrix` because `gemmSource` and `gemmSmallSource` bind `a: array<f32>` and `c: array<f32>`.
     - Adding `aF16` / `cF16` support to `gemmSource` and `gemmSmallSource` would halve pair-track global memory traffic (`z`, `zc`, `aT`, `bT`, `x`, `xc`, `gz`, `phid`) on M2 and M5 as well!
   - **Avoid Workgroup `vec4` Scalar Component Writes on Metal**:
     - As documented in `../localfold/docs/M5-HANDOFF.md` (`probe-register-gemm.js`), writing individual scalar components (`ta[i].x = ...`) of a shared `var<workgroup> ta: array<vec4<...>>` from different threads races on Metal (which compiles it as a read-modify-write of the whole `vec4`). Keep full `vec4` writes (`ta[taWrite] = vec4(...)`) or scalar `array<f16>` / `array<f32>` workgroup arrays.
