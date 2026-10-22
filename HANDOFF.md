# Handoff to Apple Silicon (M2 & M5): RFD4-Proteina WebGPU Port (`localdesign`)

Written on the Linux / NVIDIA Ampere A100 box on 2026-10-22, for an agent working on an **Apple M2** and **Apple M5**.
Read [`AGENTS.md`](./AGENTS.md) first (module layout, ES-module rules, and WebGPU numerical invariants), then this file.
Also consult `../localfold/docs/M5-HANDOFF.md` and `../localfold/webgpu/runtime/device-profile.js` if `../localfold` is checked out alongside this repo.

---

## 1. Rules

1. **Plain ES Modules Only**: Every `.js` file must load directly in the browser without a bundler and use explicit `.js` extensions on relative imports.
2. **Every change goes through the differential gates** (Step 1 below) before it is committed, with before/after numbers recorded.
3. **Measure arms interleaved in one process** (or alternating back-to-back in a loop): Apple laptops throttle under sustained load and share the GPU with the window server. On M2 and M5, back-to-back runs can drift 5–20% with temperature; never trust a single un-alternated run for a `<10%` change.
4. **Guarded 2D Workgroup Dispatches**: Every shader in [`webgpu/rfd4p/engine.js`](./webgpu/rfd4p/engine.js) that folds `wid.x + wid.y * 65535u` must guard bounds before reading/writing storage buffers (`npm test` enforces this via static AST/regex analysis in [`test/folded-grid-guard.test.js`](./test/folded-grid-guard.test.js)).
5. **No `O(L^3)` Materialization**: Triangle multiplicative updates contract over `k` in registers/tiles and never allocate `L x L x L` buffers.

---

## 2. Fresh Clone Setup on macOS (M2 / M5)

The repository at `https://github.com/sokrypton/localdesign` tracks all source code, unit tests, oracle verifiers, GPU harnesses, interactive presets (`web/examples/*.cif.zst`), `py2Dmol` vendor assets (`web/vendor/`), and weight manifests (`weights/config.json`, `weights/int8/config.json`, `weights/int8/manifest.json`).

Large binary tensors (`weights/model.safetensors`, `weights/int8/*.bin`, and `oracle-dumps/*.safetensors`) are gitignored. On a fresh M2/M5 clone, set them up as follows:

### 2.1. Bootstrap `weights/model.safetensors` (1.74 GB FP32) and `weights/int8/*.bin` (448 MB INT8 Shards)

1. Download or symlink the HuggingFace snapshot `nvidia/RFD4-Proteina` (`model.safetensors`, revision `a06e2509c7033b010fc43614e9fd840303588502`):
   ```bash
   # If already in ~/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/*/:
   ln -sf ~/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/*/model.safetensors weights/model.safetensors

   # Or download via huggingface-cli:
   huggingface-cli download nvidia/RFD4-Proteina model.safetensors --local-dir weights
   ```
2. Generate the 16 block-64 INT8 browser shards (`weights/int8/shard_000..010.bin` and `weights/int8/encoder_shard_000..004.bin`) from `weights/model.safetensors` (~3 seconds on CPU, requires only `numpy`):
   ```bash
   python3 tools/export_rfd4p_model.py \
     --input ./weights/model.safetensors \
     --config ./weights/config.json \
     --output ./weights/int8 \
     --format int8 \
     --shard-mb 32
   ```

### 2.2. Copy `oracle-dumps/` (~455 MB) for Differential Verification

Copy `oracle-dumps/` from the Linux machine (or sibling checkout) into `./oracle-dumps/`:
- `oracle-dumps/{uncond_short,ppi_pdl1,sm_sam,dna_7m5w,motif_idx_1prw,motif_unidx_1prw}.{json,safetensors,cif}`
- `oracle-dumps/encoder_oracle.{json,safetensors}`
- `oracle-dumps/cpu/{uncond_short,ppi_pdl1,sm_sam,dna_7m5w,motif_idx_1prw,motif_unidx_1prw,binder_spec}.{json,safetensors,atoms.json,parsed.json}`

---

## 3. Step 1 — Correctness Gates on M2 & M5 (No Code Changes)

On macOS (`darwin`), [`tools/gpu-chrome.mjs`](./tools/gpu-chrome.mjs) automatically launches `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` (or `$LOCALFOLD_CHROME` if set) in headless mode (`--headless=new`). No `DISPLAY=:99` is needed on macOS.

Run all gates before changing any code:

```bash
# 1. Deterministic Node unit tests (8/8 tests, runs without GPU or oracle-dumps)
npm test

# 2. CPU pipeline + feature + sampler + output oracle checks (requires oracle-dumps/ + weights/model.safetensors)
node tools/oracle/check-batch.mjs
node tools/oracle/check-features.mjs
node tools/oracle/check-sampler.mjs
node tools/oracle/check-output.mjs

# 3. Full WebGPU differential oracle suite (Trunk + Decoder + Encoder) in FP32 and FP16
node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js
node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js --half=1

# 4. Full 60-step recorded-RNG trajectory parity vs PyTorch oracle (with and without step readback)
node tools/gpu-chrome.mjs tools/gpu/check-generate.js --half=1
node tools/gpu-chrome.mjs tools/gpu/check-generate.js --half=1 --fast=1

# 5. Interactive browser studio end-to-end verification (10 presets + inverse folding + py2Dmol + PDB/CIF)
node tools/gpu-chrome.mjs tools/gpu/check-webapp.js
```

### Expected Reference Gate Metrics (Measured on Ampere A100)

| Gate | Ampere (`submat`, `--half=1`) | Ampere (`--no-submat=1`, `--half=1`) | Pass Threshold |
|---|---|---|---|
| `check-trunk.js` (`uncond_short` `v bb_ca x6`) | `relRms = 0.000352` | `relRms = 0.000400` | `relRms < 0.01` |
| `check-trunk.js` (`uncond_short` `v latents x6`) | `relRms = 0.000265` | `relRms = 0.000383` | `relRms < 0.01` |
| `check-trunk.js` (`motif_unidx_1prw` `insertion x6`) | `relRms = 0.00112` | `relRms = 0.00125` | `relRms < 0.015` |
| `check-decoder.js` (6/6 cases) | `rtDiff = 0`, `maskDiff = 0`, `relRms <= 9.3e-5` | `rtDiff = 0`, `maskDiff = 0`, `relRms <= 1.04e-4` | `rtDiff = 0`, `relRms < 0.005` |
| `check-encoder.js` (3/3 cases) | `emitDiff = 0`, `zRelRms <= 0.0015` | `emitDiff = 0`, `zRelRms <= 0.0031` | `emitDiff = 0`, `zRelRms < 0.02` |
| `check-generate.js --half=1` (`uncond_short`, 60 steps) | `finalBbCa.rmsdA = 0.0396 Å`, `finalStructure.rmsdA = 0.04571 Å`, `rtDiff = 0` | `allOk: true`, `rtDiff = 0` | `rmsdA < 0.25 Å`, `rtDiff = 0` |

---

## 4. Step 2 — Re-Measure Baselines on M2 & M5 (No Code Changes)

Run the profiler under both stock Chrome flags (`LOCALFOLD_STOCK_FLAGS=1`, what real browser visitors get) and default harness flags (`--enable-unsafe-webgpu`):

```bash
LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/profile-rfd4p.js
node tools/gpu-chrome.mjs tools/gpu/profile-rfd4p.js
```

### 4.1. End-to-End & Stage Wall-Clock Table (Fill in M2 & M5 Columns)

| Benchmark / Metric | A100 Initial | A100 (`submat`) | A100 (`--no-submat=1`) | M2 (stock) | M5 (stock) |
|---|---:|---:|---:|---:|---:|
| `uncond_short_low_60steps` (`n = 100`, 120 passes) | `5,378.7 ms` | **`1,002.7 ms`** | `2,534.6 ms` | *measure* | *measure* |
| `uncond_short_medium_60steps` (`n = 100`, 180 passes) | `7,620.4 ms` | **`1,475.9 ms`** | `3,698.8 ms` | *measure* | *measure* |
| `ppi_pdl1_low_60steps` (`n = 207`, 120 passes) | `15,196.6 ms` | **`3,034.0 ms`** | `8,358.4 ms` | *measure* | *measure* |
| `fwd3WallMs` (`n = 100`, 3 passes) | `71.02 ms` | **`34.96 ms`** | `90.10 ms` | *measure* | *measure* |
| `fwd3WallMs` (`n = 207`, 3 passes) | `120.75 ms` | **`77.66 ms`** | `223.40 ms` | *measure* | *measure* |
| `initGpu` (`n = 100` / `n = 207`) | `1.240 / 1.421 ms` | **`0.284 / 0.342 ms`** | `0.306 / 0.348 ms` | *measure* | *measure* |
| `passGpu` 1 trunk pass (`n = 100` / `n = 207`) | `15.47 / 111.74 ms` | **`10.90 / 27.47 ms`** | `23.31 / 71.37 ms` | *measure* | *measure* |
| `headGpu` (`n = 100` / `n = 207`) | `0.194 / 0.205 ms` | **`0.130 / 0.143 ms`** | `0.132 / 0.142 ms` | *measure* | *measure* |
| `decoderGpu` (`n = 100` / `n = 207`) | `6.143 / 7.717 ms` | **`4.113 / 4.790 ms`** | `5.860 / 7.090 ms` | *measure* | *measure* |

### 4.2. Per-Kernel Breakdown of 1 Trunk Pass (`passGpu`) at `n = 207` (`ppi_pdl1`, `nn = 42,849`)

Look closely at the contrast between A100 `submat` (`27.47 ms/pass`) and A100 `--no-submat=1` (`71.37 ms/pass`), because **M2 and M5 currently run the `--no-submat=1` path**:

| Kernel Label (`n = 207`, `nn = 42,849`) | Calls | A100 `submat` (ms) | A100 `--no-submat=1` (ms) | Why `--no-submat=1` is Slower (and How to Fix on M2/M5) |
|---|---:|---:|---:|---|
| `triProjG 42849x1408x384` | 14 | `6.535` | **`28.571`** | **40% of pass!** In `triProjG` (`engine.js:2543`), `submat=false` writes `c[(cBase + k) * sC0 + mPad]` directly to global memory with stride `sC0 = 42,849` floats (`171 KB` stride = uncoalesced scatter writes!) and `f32` outputs, whereas `submat` transposes via shared memory and fuses `t.lnIn`. |
| `gemm 42849x384x768` (`Q.l2`) | 7 | `3.671` | **`8.559`** | Pair transition output GEMM; reads/writes `f32` `phid` and `z` instead of `f16` (`triF16 = false`). |
| `gemm 42849x768x384` (`Q.l1`) | 7 | `3.575` | **`8.457`** | Pair transition input GEMM; reads/writes `f32` `zc` and `phid` instead of `f16`, and does not fuse `Q.trLnW` (`inNorm: 'ln'`). |
| `gemm 42849x384x256` (`t.z`) | 14 | `3.600` | **`7.017`** | Triangle output linear `t.z`; reads `f32` `xc`, `gz`, `z` instead of `f16`. |
| `tri-in` + `tri-out` (`207x207x207[256]`) | 14 | `2.659` | **`5.514`** | Batched `256 x [n, n] x [n, n]^T` contraction; reads/writes `f32` `aT`, `bT`, `x` instead of `f16`. |
| `ln 42849x384` (`t.in.lnIn` + `Q.trLnW`) | 14 vs 7 | `0.606` (7 calls) | **`1.469`** (14 calls) | Under `--no-submat=1`, `z` and `zc` are `f32` (2x memory traffic) AND `Q.trLnW` is not fused into `Q.l1` (7 extra calls). |
| `pairWriteLnIn 42849x384` | 7 | `0.858` | **`1.190`** | Reads/writes `f32` `z` and `zc` instead of `f16` (`outF16 = false`). |
| `triLnOut 42849x256` | 14 | `0.785` | **`1.054`** | Reads/writes `f32` `x` and `xc` instead of `f16` (`inF16 = outF16 = false`). |
| `pairToBias` | 7 | `0.761` | **`0.948`** | `pairToBiasSource` (32-lane subgroup dot product) is already very fast (`0.948 ms` vs `0.761 ms`)! |
| `swigluLinear 207x3072x768` | 14 | `0.944` | **`2.114`** | Token transition SwiGLU (`gemmSource` `64x128` register FMA). |
| `gemm 207x3072x768` (`qkvg` Split-K=2) | 14 | `0.781` | **`1.712`** | Token `qkvg` projection (`gemmSmallSource` / `gemmSource`). |
| `gemm 207x768x3072` (`swOut` Split-K=8) | 14 | `0.713` | **`1.364`** | Token transition output projection (8-way Split-K). |
| `gemm 207x768x768` (`L.out` + `Q.linX` Split-K=4) | 21 | `0.631` | **`0.943`** | Token attention output + pair `linX` (4-way Split-K). |
| `splitReduce+norm 207x768` | 28 | `0.648` | **`0.665`** | Identical subgroup kernel on both paths (`hasF16`). |
| `pv` + `qk` + `softmax` + `rms2` | 56 | `1.391` | **`1.776`** | Token self-attention (`n = 207`). |

---

## 5. Step 3 — High-ROI Optimization Roadmap for M2 & M5

Based on the per-kernel profile above and the findings from `../localfold/docs/M5-HANDOFF.md`, here are the highest-leverage optimizations for Apple Silicon, in priority order:

### Priority 1: Fix `triProjG` Uncoalesced Scatter Writes in `gemmSource` (`28.57 ms` -> `~7 ms` at `n = 207`)
- **Where**: [`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L2543-L2569) (`triProjG` `pairOut` when `submat === false`).
- **The Problem**:
  In `gemmSource` (`64x128` tile, 256 threads: `tm4 = li / 16` in `0..15`, `tx = li % 16` in `0..15`), each thread holds 4 rows (`m = m0 + tm4*4 + 0..3`) and 4 columns (`tn0 = tx*4 + 0..3`) of `r0` (`ap`/`bp`) and `r1` (`ag`/`bg`).
  When `pairOut` executes:
  ```wgsl
  c[cBase * p.sC0 + mPad] = y.x;
  c[(cBase + 1u) * p.sC0 + mPad] = y.y;
  c[(cBase + 2u) * p.sC0 + mPad] = y.z;
  c[(cBase + 3u) * p.sC0 + mPad] = y.w;
  ```
  Adjacent lanes `tx = 0..15` have the **same `mPad`** and **different `cBase = tx * 4`**, so 16 threads in a half-simdgroup write to addresses separated by `4 * sC0 = 171,396` floats (`685 KB` apart)! And for `incoming = true`, `mPad = (m % n) * n + (m / n)` jumps by `n` between consecutive `m` rows!
- **The Fix**:
  In `gemmSource`, after the `k0` loop completes, `ta` (`272` `vec4`s) and `tb` (`528` `vec4`s) in workgroup memory are **completely free** (or allocate a small `var<workgroup> shOut: array<f32, 4160>` = `64 * 65` floats = `16.6 KB`, well under Apple's 32 KB limit)!
  If each thread writes its `4 rows x 4 cols` of `y = r0 * sigm(r1)` into `shOut[col * 65 + rowLocal]`, followed by `workgroupBarrier()`, then `li` can read `rowLocal = li % 64` (contiguous `m`!) and `col = li / 64` and write `c[(c0 + col) * sC0 + mPad]` with **coalesced memory transactions**!
  Even better, for `n0 >= 1024` (`gz`), write `vec4<f32>` (or `vec2<u32>` `f16`).

### Priority 2: Enable `triF16` (FP16 Pair Activations) on the Non-`submat` Path (`gemmSource` & `gemmSmallSource`)
- **Where**:
  - [`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L33-L269) (`gemmSource` and `gemmSmallSource`)
  - [`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L2263-L2265) (`aF16`, `cF16`, `gF16`)
  - [`webgpu/rfd4p/trunk.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/trunk.js#L270) (`const triF16 = this.half && e.hasSubgroupMatrix && R === nn;` -> `e.hasF16`)
- **Why**:
  Every norm/transpose kernel on the pair track (`normSource`, `pairWriteLnInSource`, `transposeLnOutSource`, `pairToBiasSource`) **already supports `inF16` and `outF16` using only `shader-f16` (`this.hasF16`)**!
  The ONLY reason `triF16` in `trunk.js` (lines 270, 451, 479, 537, 568) and `outF16` in `splitReduceNormSource` (`engine.js:2338`) and `xnF16` in `blocks.js` (line 221) are currently gated on `hasSubgroupMatrix` is that `gemmSource` and `gemmSmallSource` declare `a: array<f32>`, `c: array<f32>`, and `gsc: array<f32>`.
  Adding `aF16`, `cF16`, and `gF16` (`array<f16>` storage bindings) to `gemmSource` and `gemmSmallSource` will immediately cut global memory traffic in half across the entire pair track (`zInitStatic`, `zInit`, `z`, `zc`, `aT`, `bT`, `x`, `xc`, `gz`, `phid`) on M2 and M5!
  > **Note from `../localfold/docs/M5-HANDOFF.md`**: Keep the inner-loop accumulator `acc_i_j` in `vec4<f32>` even when `a`, `b`, and `c` are `f16` so triangle contractions (`K = 207`) never overflow `f16` (`65504`).

### Priority 3: Sweep Split-K and `gemmSmallSource` Thresholds for Apple Silicon (8–10 GPU Cores vs 80 SMs)
- **Where**:
  - [`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L2261-L2262) (`small` tile selector: `m * B0 * B1 <= 1024 || n <= 64`)
  - [`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L2310-L2311) (`linear()` Split-K selector: `if (rows <= 512 && inC >= 768 && outC >= 256 && outC <= 1024 && inC % 64 === 0 && !cF16)`)
- **Why**:
  On an 80-SM NVIDIA A100, a token GEMM at `rows = 100, outC = 768, inC = 768` with a `64x128` tile launches only `2 * 6 = 12` workgroups (15% of the GPU), which is why 4-way Split-K (`32x64` tile -> `4 * 4 * 12 = 192` workgroups) was a huge win on A100.
  On an **M2 (10 GPU cores)** or **M5 (8–10 GPU cores)**, 24–48 workgroups already saturate the entire GPU! Splitting K into 4 or 8 slices at `rows = 207` forces an extra `splitBuf` global write + `splitReduce` readback.
  - Note: When `nextNorm` is attached to `linear()` (`L.out` and `swOut` in `blocks.js`), `splitReduceNormSource` fuses the gated residual + LayerNorm/RMSNorm + AdaLN into the Split-K reduction! So test whether `splits = 2` (or `splits = 4` vs no Split-K) is faster on M2 and M5 at `n = 100` and `n = 207`.

### Priority 4: Race `b16` Inner-Loop `f16` vs `f32` Accumulation in `gemmSource` & `gemmSmallSource`
- **Where**: [`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L198-L226) (`gemmSource`) and [`webgpu/rfd4p/engine.js`](file:///home/ubuntu/localdesign/webgpu/rfd4p/engine.js#L587-L604) (`gemmSmallSource`).
- **Why**:
  Currently, when `b16 = true` (`half && hasF16`), `ta` and `tb` are staged as `vec4<f16>` in workgroup memory, and the 16-step inner K-tile accumulates into `p_i_j: vec4<f16>` via `fma(vec4<f16>(av.x), bv0, p_0_0)` before adding `vec4<f32>(p_0_0)` to `acc_0_0`.
  In `../localfold/docs/M5-HANDOFF.md`, `probe-alu-lanes.js` measured M5's `f16` ALU at only `1.3x` `f32` (vs `1.7x` on M2), and on M2 `grid.project` ran faster accumulating in `f32` than `f16`. Race `vec4<f16>` inner accumulation vs `vec4<f32>` accumulation (while keeping `ta` and `tb` staged as `vec4<f16>` in shared memory) on both M2 and M5.

### Priority 5: Check `8x8x8` `simdgroup_matrix` (`chromium-experimental-subgroup-matrix`) on M2 & M5
- **Context from `../localfold`**:
  - Stock Chrome on macOS does **not** expose `chromium-experimental-subgroup-matrix` (`LOCALFOLD_STOCK_FLAGS=1` -> `subgroupMatrixConfigs: []`).
  - With `--enable-unsafe-webgpu`, Chrome 154 on both M2 and M5 exposes `8x8x8` (`f32 -> f32` and `f16 -> f16`; note these are Metal's `simdgroup_matrix` instructions, not M5's Metal 4 MPP `matmul2d` cooperative tensors, which WebGPU/Dawn does not expose yet).
  - In `../localfold`, `8x8x8` WebGPU subgroup matrices lost on M2 (`-6%` to `-15%`) and were within `1.9%` on M5 for linear projections, **except** on deep-K outer-product contractions where `8x8x8` won by `1.12x–1.32x`.
  - If you experiment with an `8x8x8` variant for `gemmSubmatSource` under `--enable-unsafe-webgpu`, remember:
    1. Keep the `32x32` per-subgroup accumulator budget (16 `8x8` tiles = 32 floats/lane); larger accumulator sets spill catastrophically on M2 (`122 GFLOP/s` vs `1,466 GFLOP/s`).
    2. Use `f32` accumulation (`resultComponentType === 'f32'`) or test `f16` accumulation carefully against `check-rfd4p.js --half=1`.
    3. Never regress `LOCALFOLD_STOCK_FLAGS=1` (stock Chrome), which is what browser visitors run.

---

## 6. Summary of All 39 Optimizations Already Landed (Do Not Regress)

To avoid re-investigating solved problems or breaking subtle invariants, here is what is already built into [`webgpu/rfd4p/`](./webgpu/rfd4p/) and [`shared/rfd4p/`](./shared/rfd4p/):

1. **Recorded Command Lists (`Rfd4pEngine.program`)**:
   - Every pipeline, bind group, and uniform buffer is recorded once in `record()` and replayed inside a single `beginComputePass()` per step (`engine.js:2817`). Zero per-step buffer/bind-group allocations.
2. **Static Pair Feature Cache (`features.js` + `trunk.js`)**:
   - The 70 non-distance-bin channels of `pairFeatures` (`seq_sep`, `token_bonds`, `relative_coords`, `crd_mask`) depend only on the static batch and are computed and projected (`pairStaticProgram` -> `B.zInitStatic`) **once on step 0**.
   - On steps `0..59`, `pairToBias` (`zInitAda` mode in `pairToBiasSource`) computes the 30 `xt_pair_dists` bins on the GPU directly from `B.seqF[0..2]` (`caNm`), looks up `wLnBins[1 + bin]`, adds `B.zInitStatic`, and fuses `RMSNorm + AdaLN + Layer 0 toBias` in a single pass!
3. **Uniform Conditioning Broadcast (`condRows = 1`)**:
   - When conditioning features (`cond0`, `cond1`) are uniform across all tokens (true for all standard unconditional, PPI, motif, and small-molecule designs), `trunk.js` projects and runs `tc[0]`, `tc[1]`, and `precomputeCond` on `batchCondRows = 2` rows instead of `2 * n` rows (`100x` less work).
   - `precomputeCond` (`blocks.js:172`) folds `norm_cond.weight` and `norm_cond.bias` into `allCondGbW` / `allCondGbB` at weight upload time and computes all 14 layers' `[mhaGb, trGb, mhaSc, trSc]` in **1 zero-barrier subgroup GEMV dispatch** (`gemvSource`).
4. **Dual-Layer `pairToBias` (`toBiasDual`, `trunk.js:221`, `engine.js:1645,1765`)**:
   - Since `z` only updates on even layers (`0, 2, 4, 6, 8, 10, 12`), odd layers `i = 1, 3, 5, 7, 9, 11` and the following even layers `i + 1` share the exact same `z`!
   - `pairToBias` computes both `B.bias` (layer `i`) and `B.bias2` (layer `i + 1`) in a single read of `z`, eliminating 6 full-pair `pairToBias` dispatches per pass.
   - `pair_norm.weight * sqrt(384)` is folded into `toBias.weight` at upload time (`w_eff = w_bias * w_norm * sqrt(384)`), since $\text{RMS}(z) \cdot (w_{\text{norm}} \odot w_{\text{bias}}) = (z / \|z\|_{\text{rms}}) \cdot w_{\text{eff}}$.
   - On layer 13 (`i === nlayers - 1`), when `hasNextCycle` is true, `pairToBiasSource` also writes `B.zc = RMSNorm(z) * zCycNorm` (`outZc`), eliminating a separate `zCycNorm` dispatch on recycle passes.
5. **Fused Triangle Projections & Transposes (`triProjG`, `pairWriteLnIn`, `transposeLnOut`)**:
   - `pairWriteLnInSource` fuses `z[a, j] += pp[j, :384] * pp[a, 384:]` (reading `pp` directly from `splitBuf` when `splitPp` is active) with `zc = LayerNorm(z)` for `tri_mult_out`.
   - `triProjG` fuses all 5 triangle projections (`a_p, a_g, b_p, b_g, g` -> `[1408, 384]`), applies `a = a_p * sigm(a_g)` and `b = b_p * sigm(b_g)` in the epilogue, and writes `aT` and `bT` directly in transposed `[256, n, nPad]` layout plus `gz = sigm(g)`.
   - `transposeLnOutSource` fuses the `[256, nn] -> [nn, 256]` transpose of `x` with `LayerNorm(256)` (`t.lnOut`) using a coalesced `16 x 258` shared-memory tile.
6. **Algebraic LayerNorm + `Q.l1` GEMM Fusion (`Q.l1Fused`, `trunk.js:189`, `engine.js:723,911`)**:
   - Uses $\text{LN}(z) W_1^T + b_1 = \sigma^{-1}(z \tilde{W}_1^T - \mu \tilde{w}_{\text{row}}) + \tilde{b}_1$ where $\tilde{W}_1 = W_1 \text{diag}(w_{\text{ln}})$, $\tilde{w}_{\text{row}} = \tilde{W}_1 \mathbf{1}$ (summed from the exact `f16`-quantized $\tilde{W}_1$), and $\tilde{b}_1 = b_1 + b_{\text{ln}} W_1^T$.
   - Accumulates $\mu$ and $\sigma^{-1}$ on the fly during `A` tile loads in `gemmSubmatSource` (`inNorm: 'ln'`) and stores them in the unused padding columns `128..135` of `scratch`, eliminating 7 full-pair `norm` dispatches per pass.
7. **Token-Track Split-K + Fused `splitReduceNorm` / `splitReduceQkNorm`**:
   - `qkvgQkNorm` runs a 2-way Split-K GEMM followed by `splitReduceQkNormSource`, which sums the 2 splits, adds `qkvgB`, applies `RMSNorm`/`LayerNorm` to `Q` and `K`, pre-applies `sigmoid(G)` when `outF16` is true, and writes packed `B.qk` and `B.qkv`.
   - `linear()` with `nextNorm` (`L.out` and `swOut`) runs Split-K followed by `splitReduceNormSource`, fusing the Split-K sum, gated residual add (`y += proj * sigm(gate)`), next layer's `RMSNorm + AdaLN`, and (on even layers) the pair update's `LayerNorm(s)` (`nextNorm2`) into a single dispatch!
8. **GPU-Resident Flow-Matching Step (`headFastSource` + `simStepProgram`)**:
   - `headFastSource` fuses the shared `RMSNorm(s)` with both `ca_linear` (`[n, 3]`) and `local_latents_linear` (`[n, 32]`) in 1 dispatch, skipping the `[n, 147]` `sequence_readout` head on steps `0..58`.
   - `simStepProgram` performs the `vf_tsr` Euler-Maruyama update on the GPU, updating `B.seqF` in place and writing both `xt` and `x1` into `B.out`.

---

## 7. Known Traps & What Lost (Do Not Repeat Blind)

1. **Do NOT use `Q.l1Fused` algebraic LN fusion on `triProjG` (`t.projG`)**:
   - We tested fusing `t.in.lnIn` into `triProjG` (`42849 x 1408 x 384`) via $\tilde{W} = W \text{diag}(w_{\text{ln}})$. Because `t.projG` has `11` column tiles (`N = 1408`), each workgroup re-reduced `sumA` and `sqSumA` across `K = 384` in all 11 N-tiles, adding more register pressure and ALU than the standalone `norm` kernel (`0.086 ms`) cost (`+0.55 ms/pass` slower). Only use `inNorm: 'ln'` where `N / 128` is small (like `Q.l1`, `N = 768`, 6 tiles).
2. **In `pairToBiasSubmatSource`, `right1` (`bias2`) is at `staged_b` offset `192u` (`12 * 16`), NOT `256u`**:
   - `wBias` has 12 heads per layer (`hB = 0..11` for `bias`, `hB = 12..23` for `bias2`), with stride `16` in `staged_b`. Offset `256u` was head 16 and caused a `0.108` relative error on `v bb_ca` before it was fixed to `192u`.
3. **WebIDL `GPUSubgroupMatrixConfig` objects cannot be spread (`{ ...c }`) or `JSON.stringify`'d**:
   - Spreading or `JSON.stringify`ing `adapter.info.subgroupMatrixConfigs[i]` yields `{}`. Always extract `{ componentType: c.componentType, resultComponentType: c.resultComponentType, M: c.M, N: c.N, K: c.K }` explicitly, as [`Rfd4pEngine`](./webgpu/rfd4p/engine.js) now does.
4. **Workgroup `vec4` Scalar Component Writes Race on Metal**:
   - Writing scalar components (`ta[i].x = ...`) of a `var<workgroup> ta: array<vec4<...>>` from different threads compiles to a non-atomic read-modify-write of the whole `vec4` on Metal. Always write whole `vec4`s (`ta[taWrite] = vec4(...)`) or declare scalar workgroup arrays (`array<f16, ...>` / `array<f32, ...>`).

---

## 8. Report Back

Append a section **"## 9. Apple Silicon (M2 & M5) Reply"** at the end of this file with:
1. Step 1 gate results on M2 and M5 (`check-rfd4p.js`, `check-generate.js`, `check-webapp.js`).
2. Step 2 baseline table (§4.1 and §4.2) with the M2 and M5 columns filled in.
3. Each Step 3 optimization measured (interleaved before/after numbers on M2 and M5, plus gate verification).
