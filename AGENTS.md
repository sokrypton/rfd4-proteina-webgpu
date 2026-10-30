# AGENTS.md

> Working on speed? Read [`SPEED-HANDOFF.md`](./SPEED-HANDOFF.md) first: current per-kernel profiles on stock and
> developer Chrome, ranked opportunities, what already lost, and the gates.

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
     - `safetensors.js` & `presets.js`: Zero-copy `parseSafetensors`, streaming + Cache API `fetchSafetensors` and `fetchShardedWeights` (keeps the int8 shard bytes; each tensor's `.data` decodes on read, so read large weights once, at GPU upload), and 10 interactive benchmark presets (`RFD4P_PRESETS`).
   - `cpu/rfd4p/`: Deterministic float32 CPU reference implementations (`trunk.js`, `decoder.js`, `encoder.js`, `reference.js`).
   - `webgpu/rfd4p/`: Recorded-command-list WebGPU engine (`engine.js` with `64x128` and `32x64` bank-conflict-free `vec4` GEMMs, `16x16` `triGate`, 64M-float triangle chunking, and FP16 weight packing), shared token blocks (`blocks.js`), `Rfd4pTrunkGpu` (`trunk.js`), `Rfd4pDecoderGpu` (`decoder.js`), `Rfd4pEncoderGpu` (`encoder.js`), and `Rfd4pGeneratorGpu` (`generate.js`).
   - `web/`: Interactive browser studio (`index.html`, `web/app.js`, `web/studio-api.js`, `web/studio/`, `web/examples/*.cif.zst`) embedding the `nvidia/rfd4-proteina-zerogpu` / `AtomWorks Studio` interface (Mol* WebGL 3D viewer, sequence dock, condition popovers, live trajectory playback, and PDB/mmCIF/ZIP export) backed 100% client-side by WebGPU.
   - `web/design-spec.js` (pure: task, chain roles, lengths and contig read from a document's conditions; tested in `test/design-spec.test.js`) and `web/design-panel.js` (start screen, Design panel, guided set-ups, simple/advanced switch). The panel edits documents only through `window.__rfd4pMutate`, so Undo and Studio refreshes work as for Studio's own edits.
   - AtomWorks Studio is **not vendored**. `studio.json` pins the release; `npm run build:studio` downloads that bundle, verifies its sha256, copies it into `assets/` and composes `index.html` from Studio's own page plus the two fragments this repo owns (`web/page/head.html`, `web/page/card.html`). Never hand-edit `assets/` or `index.html` — both are build output. Live frames, overlays, sessions, edits, the selection, chain colours and live mode go through Studio's documented embedding API, `window.atomworksStudio`. To try an unreleased Studio: `npm run build:studio -- --from <path-to-bundle>`. **Right now the site runs an unreleased Studio**: `studio.json` `override` names the commit (sokrypton/atomworks-studio `live-frame-cartoon`: cartoon overlays, RosettaCommons/atomworks-studio#29, and live frames drawn as cartoon plus ligands), and `build:studio` refuses a plain release rebuild until `override` is removed after a release contains it.
   - Refold (`web/refold-view.js`, the Refold row under a finished design): LocalFold's `refoldDesign` refolds the design's sequence with OpenBind-0 or AlphaFold 3 (kept chains as templates), and the refold's designed chains are overlaid as a magenta backbone trace. That code is **vendored, not written here**: `localfold.json` pins a `sokrypton/localfold` commit, and `npm run sync:localfold [<commit>]` copies the import closure of its entries into `vendor/localfold-<commit>/` (the commit in the path is the cache-busting) and rewrites `web/localfold.js`. Never edit `vendor/`; fix LocalFold and re-sync. AlphaFold 3's parameters are Google DeepMind's (non-commercial, prohibited-use policy), so the page asks once per browser before using them and offers OpenBind-0 (Apache 2.0) instead.
   - Cache-busting: `index.html` carries an import map pinning every module under `web/`, `shared/`, `webgpu/`, `cpu/` to `?v=<content hash>` (plus the `app.js` / `styles.css` tags). Run `npm run stamp` after editing any of them (`build:studio` runs it too), or browsers can mix new and cached modules for up to 10 minutes after a deploy.
   - Chain colours come from Studio's shared palette, keyed by chain id; the design panel reads it through `window.atomworksStudio.chainColor`.
   - `tools/`: Headless Chrome WebGPU runner (`tools/gpu-chrome.mjs`), GPU differential test suites (`tools/gpu/check-{rfd4p,trunk,decoder,encoder,generate,webapp}.js`), and PyTorch oracle verifiers (`tools/oracle/check-{batch,features,sampler,cpu-trunk,decoder,encoder,output}.mjs`).
   - `test/`: Deterministic Node unit tests (`npm test`).

## WebGPU & Numerical Invariants

1. **Differential Verification**:
   - Every WebGPU kernel and network layer in `webgpu/rfd4p/` must be verified against the PyTorch oracle dumps and `cpu/rfd4p/` via `npm run test:gpu` (`DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js`).
2. **No `O(L^3)` Materialization**:
   - Triangle multiplicative updates (`tri_mult_out`, `tri_mult_in`) contract over `k` inside registers in the WGSL shader and never allocate `L x L x L` buffers.
3. **Guarded 2D Workgroup Dispatches**:
   - Any folded 1D workgroup index (`wid.x + wid.y * 65535u`) must check `< totalWorkgroups` before reading or writing storage buffers.
