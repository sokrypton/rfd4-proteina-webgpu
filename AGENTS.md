# AGENTS.md

Operational guide for coding agents working in `rfd4p` (WebGPU implementation of `RosettaCommons/RFD4-Proteina`, inspired by `../localfold`).

## Architecture & Module Layout

1. **Plain ES Modules Only**:
   - All JavaScript files must be standard ES modules loadable directly by the browser without a bundler or compiler.
   - Always include explicit `.js` file extensions in relative `import` specifiers.
2. **Directory Structure**:
   - `shared/rfd4p/`:
     - `encoding.js` & `conditions.js`: 147-token `UNIFIED_ATOM37_ENCODING` vocabulary, atom37 layouts, and condition / CATH enums.
     - `pipeline/`: Pure-JS input transform pipeline (`buildBatch` in `pipeline/index.js`) with Zstandard `.cif.zst` frame decompressor (`zstd.js`), mmCIF/PDB/annotated-CIF parser + WL entity hashes (`cif.js`), YAML/JSON condition-spec & `sel(...)` query evaluator (`spec.js`), 3D/topological stereochemistry (`chirality.js`), token bond graph & BFS path lengths (`bonds.js`), and validation transforms (`transforms.js`).
     - `features.js`: Upstream feature factories (`trunkSeqFeatures`, `trunkCondFeatures`, `pairFeatures` with static cache, `pairCondFeatures`).
     - `flow.js` & `partial.js`: Flow-matching schedules (`scheduleTsrSafe`, `noiseSchedule`, `recycleDepths`), Euler/SDE `simulationStep`, partial diffusion (`corruptBatch`, `buildPartialSchedule`), and inverse folding (`RESAMPLED_FORWARD_MARGINAL`).
     - `output.js`: Rectangular Hungarian linear sum assignment (`linearSumAssignmentMaximize`), unindexed motif insertion (`buildPredictedInsertionMapping`), internal-coordinate group grafting (`insertInternalCoordConditionedGroups`), input-frame Kabsch alignment (`alignAtomsToInputFrame`), `processSampleToAtoms`, and `toCifString` / `toPdbString`.
     - `safetensors.js` & `presets.js`: Zero-copy `parseSafetensors`, streaming + Cache API `fetchSafetensors`, and 10 interactive benchmark presets (`RFD4P_PRESETS`).
   - `cpu/rfd4p/`: Deterministic float32 CPU reference implementations (`trunk.js`, `decoder.js`, `encoder.js`, `reference.js`).
   - `webgpu/rfd4p/`: Recorded-command-list WebGPU engine (`engine.js` with `64x128` and `32x64` bank-conflict-free `vec4` GEMMs, `16x16` `triGate`, 64M-float triangle chunking, and FP16 weight packing), shared token blocks (`blocks.js`), `Rfd4pTrunkGpu` (`trunk.js`), `Rfd4pDecoderGpu` (`decoder.js`), `Rfd4pEncoderGpu` (`encoder.js`), and `Rfd4pGeneratorGpu` (`generate.js`).
   - `web/`: Interactive browser studio (`index.html`, `web/app.js`, `web/examples/*.cif.zst`) with `py2Dmol` 3D molecular viewer, live trajectory playback, conditioning drawer, and PDB/mmCIF export.
   - `tools/`: Headless Chrome WebGPU runner (`tools/gpu-chrome.mjs`), GPU differential test suites (`tools/gpu/check-{rfd4p,trunk,decoder,encoder,generate,webapp}.js`), and PyTorch oracle verifiers (`tools/oracle/check-{batch,features,sampler,cpu-trunk,decoder,encoder,output}.mjs`).
   - `test/`: Deterministic Node unit tests (`npm test`).

## WebGPU & Numerical Invariants

1. **Differential Verification**:
   - Every WebGPU kernel and network layer in `webgpu/rfd4p/` must be verified against the PyTorch oracle dumps and `cpu/rfd4p/` via `npm run test:gpu` (`DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js`).
2. **No `O(L^3)` Materialization**:
   - Triangle multiplicative updates (`tri_mult_out`, `tri_mult_in`) contract over `k` inside registers in the WGSL shader and never allocate `L x L x L` buffers.
3. **Guarded 2D Workgroup Dispatches**:
   - Any folded 1D workgroup index (`wid.x + wid.y * 65535u`) must check `< totalWorkgroups` before reading or writing storage buffers.
