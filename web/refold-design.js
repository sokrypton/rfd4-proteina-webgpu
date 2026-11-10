/**
 * Refold a finished design, reading it the way RFD4's own AF3 oracle does when LocalFold's reader cannot.
 *
 * LocalFold's `refoldDesign` reads the design's PDB: it refuses a modified residue ("chain A has KCX ... cannot read
 * yet") and asks the PDB for every ligand by name, which fails for a ligand a design brings with it ("Could not reach
 * the PDB for ligand L:0"). A design with either never reached the fold, with any model. `foldInputFromDesign`
 * (refold-input.js) reads it from the CIF instead; this is `refoldDesign` run on that input. A design with neither
 * goes to LocalFold's own `refoldDesign` unchanged.
 *
 * The steps below are LocalFold's `refoldDesign` (web/validate-design.js) with the input replaced: its templates, its
 * fit onto the design and its score are reused as they are, from the pinned vendor copy.
 */
import { foldInputFromDesign, pdbWithParents, restoreLigandNames } from './refold-input.js';

/** A module of the vendored LocalFold closure, by its path in LocalFold (the pin names the directory). */
async function vendored(path) {
  const { commit } = await (await fetch(new URL('../localfold.json', import.meta.url))).json();
  return import(/* @vite-ignore */ new URL(`../vendor/localfold-${commit}/${path}`, import.meta.url).href);
}

/**
 * @param {object} lf the LocalFold module (web/localfold.js)
 * @param {object} options `lf.refoldDesign`'s, plus `cif`: the design as mmCIF
 */
export async function refoldDesign(lf, options) {
  const { pdb, cif, chain, model, device, seed = 0, signal, recycles = 3 } = options;
  const input = foldInputFromDesign(pdb, cif ?? '');
  if (!input.needed) return lf.refoldDesign(options);
  const steps = options.steps ?? lf.AF3_COUNTS.diffusion.preferred;
  const status = options.status ?? (() => {});
  if (lf.FOLDERS[model] === undefined) throw new Error(`no folding model ${model} (have ${Object.keys(lf.FOLDERS).join(', ')})`);
  const [{ keptTemplates, refoldOverlay }, { describeValidation, selfConsistency }, { ccdUrl }] = await Promise.all([
    vendored('cpu/design/refold.js'), vendored('cpu/design/validate.js'), vendored('shared/af3/featurise/ccd-component.js')]);
  const templates = keptTemplates(pdbWithParents(pdb), chain, options.partial ?? []);
  status(`refolding with ${model}`);
  const weights = await lf.loadAf3Weights(() => {}, lf.FOLDERS[model]);
  const { sequence, chainKinds, ligandCodes, modifications } = input;

  // foldAf3 fetches a component by code from the PDB. A ligand the design brought is served from the design instead,
  // for the length of the fold and for its stand-in codes alone; every other request goes through.
  const served = new Map(Object.entries(input.components).map(([code, text]) => [ccdUrl(code), text]));
  const realFetch = globalThis.fetch;
  globalThis.fetch = (resource, init) => {
    const url = typeof resource === 'string' ? resource : resource?.url ?? String(resource);
    return served.has(url) ? Promise.resolve(new Response(served.get(url))) : realFetch(resource, init);
  };
  let refolded;
  try {
    refolded = await lf.foldAf3({
      sequence, chainKinds, ligandCodes, modifications, templates, weights, device, backend: options.backend, signal, seed,
      calls: steps, recycles,
      onStatus: (text) => status(`refolding with ${model} · ${text}`),
      onProgress: () => {},
    });
  } finally {
    globalThis.fetch = realFetch;
  }
  if ((refolded.templateCoverage ?? []).length !== templates.length) {
    throw new Error(`${templates.length} kept chains were templated and the fold took ${(refolded.templateCoverage ?? []).length}`);
  }
  const fitted = refoldOverlay(pdb, restoreLigandNames(refolded.pdb, input.standIns), chain);
  const { rmsd, comparable, residues } = selfConsistency(pdb, fitted.refold,
    Array.isArray(chain) ? (chain.length === 1 ? chain[0] : undefined) : chain);
  const result = {
    model, confidence: refolded.confidence ?? {}, rmsd, comparable, residues,
    templated: templates.map((template) => template.chainId),
    partial: templates.filter((template) => template.mapping !== undefined)
      .map((template) => ({ chain: template.chainId, residues: template.mapping.length })),
    refold: fitted.refold, deviations: fitted.deviations,
  };
  const templated = result.templated.length === 0 ? ''
    : ` (chain${result.templated.length === 1 ? '' : 's'} ${result.templated.join(', ')} templated)`;
  return { ...result, described: `${describeValidation(result)}${templated}` };
}
