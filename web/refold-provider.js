/**
 * What this page contributes to Studio's Refold panel: the fold itself. Everything a person sees and decides (the
 * Model, Seed, Recycles and Steps controls, AlphaFold 3's terms dialog, Refold all and Stop, the progress line, the
 * numbers in each design's row, the refold drawn over the design) is Studio's, mounted by web/app.js
 * (`window.atomworksStudio.mountRefoldPanel`); docs/INTEGRATIONS.md, "Refold panel", has the contract.
 *
 * The fold is LocalFold's, in the browser: WebGPU, or its WebGL2 path where there is no WebGPU.
 */
import { refoldDesign } from './refold-design.js';
import { motifRmsds } from './motif-metrics.js';
import { describeRefold, pdbToCif, refoldLigandCif, refoldMetrics, refoldOverlayCif, REFOLD_MODELS } from './refold-view.js';

/**
 * Where AlphaFold 3's terms were agreed to before they moved into Studio: the same key, so a browser that has
 * agreed already is not asked again. Studio asks the storage for its own key and this maps it.
 */
const STUDIO_AF3_KEY = 'atomworks-studio.refoldTerms.af3';
const PAGE_AF3_KEY = 'rfd4p.refoldTerms.af3';

export const refoldTermsStorage = {
  get(key) { try { return localStorage.getItem(key === STUDIO_AF3_KEY ? PAGE_AF3_KEY : key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key === STUDIO_AF3_KEY ? PAGE_AF3_KEY : key, value); } catch { /* asked again next time */ } },
};

/**
 * @param {object} app the page (web/app.js): `gpuDevice`, `glFallback`, `noRefold` and `_initWebGpu()`
 * @returns {object} a `RefoldProvider` for Studio's panel
 */
export function createLocalFoldProvider(app) {
  /** The LocalFold module, loaded on the first refold only (~6 MB, tools/sync-localfold.mjs). */
  const localFold = () => import('./localfold.js');

  return {
    // Studio names both: AlphaFold 3 is gated behind its terms there, OpenBind-0 (Apache 2.0) is not.
    models: [{ id: 'af3' }, { id: 'openbind0' }],

    // No WebGPU and no WebGL2 fallback: nothing to fold with, so the Refold box stays away.
    available: () => !app.noRefold,

    /**
     * The parameters are a few hundred MB: ask for them here, ahead of the fold, so the transfer reports on the
     * button. LocalFold memoises per family and honours the progress callback of whoever calls it FIRST, so the
     * `loadAf3Weights` inside `refoldDesign` then awaits this same promise.
     */
    async prepare(model, onProgress) {
      if (!app.gpuDevice && !app.glFallback) await app._initWebGpu();
      if (!app.gpuDevice && !app.glFallback) throw new Error('WebGPU is not available');
      const lf = await localFold();
      await lf.loadAf3Weights((progress) => onProgress({
        loadedBytes: progress.loadedBytes ?? 0, totalBytes: progress.totalBytes ?? 0,
      }), lf.FOLDERS[model] ?? model);
    },

    /** One design's refold: scored here (the page knows which chains it designed), handed to the panel to show and keep. */
    async refold(design, { model, seed, recycles, steps }, { signal, onStatus }) {
      const view = design.data;
      const lf = await localFold();
      const name = REFOLD_MODELS[model] ?? model;
      const result = await refoldDesign(lf, {
        pdb: view.pdb, cif: view.cif, chain: view.designedChains, model, device: app.gpuDevice, seed, recycles, steps, signal,
        ...(app.gpuDevice ? {} : { backend: 'webgl2' }),
        partial: view.partialTemplates ?? [],
        // LocalFold says "phase · 40%": the panel reads the percentage and fills the button by it.
        status: (text) => onStatus(text.replace(/^refolding with \S+( · )?/, '') || `refolding with ${name}`),
      });
      const motif = motifRmsds(view.pdb, result.refold, view.motifAtoms, view.designedChains);
      return {
        cif: refoldOverlayCif(result.refold),
        ligandCif: refoldLigandCif(result.refold),
        metrics: {
          ...refoldMetrics(result, view.designedChains, view.pdb),
          motifRmsd: motif?.motifRmsd ?? NaN, motifBackboneRmsd: motif?.motifBackboneRmsd ?? NaN,
        },
        text: describeRefold(model, result, view.designedChains, view.partialTemplates ?? [], motif),
        // What "Save all (.zip)" keeps of the refold (web/results-zip.js).
        data: {
          pdb: result.refold, cif: pdbToCif(result.refold), rmsd: result.rmsd, confidence: result.confidence,
          templated: result.templated, partial: result.partial,
        },
      };
    },
  };
}

/** A panel record as web/results-zip.js reads a refold: the page's own record shape before the panel existed. */
export function refoldRecordForZip(record) {
  return {
    model: record.model, seed: record.seed, recycles: record.recycles, steps: record.steps,
    pdb: record.data?.pdb, cif: record.data?.cif, rmsd: record.data?.rmsd, confidence: record.data?.confidence,
    templated: record.data?.templated, partial: record.data?.partial, text: record.text,
    metrics: record.metrics, pass: record.pass,
  };
}
