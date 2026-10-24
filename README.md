# RFD4-Proteina on WebGPU

A browser port of [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) (all-atom flow-matching protein
design) that runs entirely on your GPU through WebGPU. There is no server: inputs are prepared, the model runs, and
designs are written out in the page. The editor is [AtomWorks Studio](https://github.com/RosettaCommons/atomworks-studio)
with the Mol* viewer.

**Live:** https://sokrypton.github.io/rfd4-proteina-webgpu/ (Chrome or Edge with WebGPU; the first design downloads
~330 MB of INT8 weights, cached for later visits; the 138 MB encoder is fetched only for partial
diffusion and inverse folding, so the full set is 469 MB).

## Run locally

```sh
npm run serve          # http://localhost:4173 (no-cache static server)
```

No build step is needed for the app itself: everything here is plain ES modules served as-is. `npm run build:studio`
refreshes the embedded AtomWorks Studio from its pinned release (`studio.json`).

## Layout

| Path | What |
| --- | --- |
| `index.html`, `web/` | The page: Studio markup plus the RFD4-Proteina card (`web/app.js`), the in-browser Studio API (`web/studio-api.js`), examples |
| `shared/rfd4p/` | Input pipeline (mmCIF parsing, RCSB-style preparation, CCD fetch), features, flow-matching schedules, output |
| `webgpu/rfd4p/` | WebGPU engine (WGSL kernels), trunk, decoder, encoder, generator |
| `cpu/rfd4p/` | Float32 CPU reference used by the parity checks |
| `studio.json`, `web/page/` | The AtomWorks Studio release this site embeds, and the two page fragments this repo owns (its head additions and the RFD4-Proteina card). `npm run build:studio` downloads that release and composes `index.html` and `assets/` from it |
| `weights/int8/` | INT8 weight shards served to the browser |
| `tools/` | Dev server, Studio build, headless-Chrome GPU checks (`tools/gpu/`), PyTorch oracle checks (`tools/oracle/`) |
| `test/` | Node unit tests |

## Tests

```sh
npm test                                                     # Node unit tests
LOCALFOLD_STOCK_FLAGS=1 node tools/gpu-chrome.mjs tools/gpu/check-webapp.js   # end-to-end in headless stock Chrome
npm run test:gpu                                             # WebGPU vs oracle parity (needs oracle-dumps/)
```

Contributor and agent notes: [`AGENTS.md`](AGENTS.md); performance work: [`SPEED-HANDOFF.md`](SPEED-HANDOFF.md).

## Weights

`weights/int8/` is exported from the [`nvidia/RFD4-Proteina`](https://huggingface.co/nvidia/RFD4-Proteina) checkpoint
with `tools/export_rfd4p_model.py`; see the model card for its terms.

## License

Code in this repository: MIT. Third-party components (AtomWorks Studio, Mol*, fzstd, py2Dmol): see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
