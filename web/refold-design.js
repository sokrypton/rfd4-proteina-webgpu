/**
 * Refold a finished design from the one input RFD4's AF3 oracle gives AF3: a CIF holding the design, its user CCD and its
 * templates (refold-input.js reads it; refold-cif.js writes it in the browser; tools/oracle/af3_refold_input.py writes
 * it by running the oracle itself). LocalFold's own `refoldDesign` reads a PDB and cannot name a modified residue or a
 * ligand the PDB does not have; this is its steps (web/validate-design.js) with the fold's input taken from that CIF,
 * and its fit and score reused as they are from the pinned vendor copy.
 */
import { foldInputFromRefoldCif, restoreLigandNames } from './refold-input.js';
import { dictionaryCodes, parentsOf, pdbWithParents, refoldCifFromDesign } from './refold-cif.js';

/** A module of the vendored LocalFold closure, by its path in LocalFold (the pin names the directory). */
async function vendored(path) {
  const { commit } = await (await fetch(new URL('../localfold.json', import.meta.url))).json();
  return import(/* @vite-ignore */ new URL(`../vendor/localfold-${commit}/${path}`, import.meta.url).href);
}

/** The PDB's entry for each of the design's non-standard components that has one; a component it lacks is left out. */
async function dictionaryFor(cif, ccdUrl, signal) {
  const dictionary = {};
  await Promise.all(dictionaryCodes(cif).map(async (code) => {
    try {
      const response = await fetch(ccdUrl(code), { signal });
      if (response.ok) dictionary[code] = await response.text();
    } catch (error) {
      if (signal?.aborted) throw error;   // (any other failure: the design's own atoms define it)
    }
  }));
  return dictionary;
}

/**
 * @param {object} lf the LocalFold module (web/localfold.js)
 * @param {object} options `lf.refoldDesign`'s, plus `cif`: the design as mmCIF (the refold input is made from it),
 *   or `refoldCif`: a refold input made elsewhere (tools/oracle/af3_refold_input.py), read as it is. `pdb` is still
 *   what the refold is fitted onto and scored against.
 */
export async function refoldDesign(lf, options) {
  const { pdb, chain, model, device, seed = 0, signal, recycles = 3 } = options;
  const steps = options.steps ?? lf.AF3_COUNTS.diffusion.preferred;
  const status = options.status ?? (() => {});
  if (lf.FOLDERS[model] === undefined) throw new Error(`no folding model ${model} (have ${Object.keys(lf.FOLDERS).join(', ')})`);
  const [{ keptTemplates, refoldOverlay }, { describeValidation, selfConsistency }, { ccdUrl }] = await Promise.all([
    vendored('cpu/design/refold.js'), vendored('cpu/design/validate.js'), vendored('shared/af3/featurise/ccd-component.js')]);

  // The refold's one input. For a design the page made, the CIF is written here the way the oracle writes it: which
  // chains to template is LocalFold's rule (kept chains whole, a designed chain in part where the job kept part of it).
  let refoldCif = options.refoldCif;
  if (refoldCif === undefined) {
    const dictionary = await dictionaryFor(options.cif, ccdUrl, signal);
    const kept = keptTemplates(pdbWithParents(pdb, parentsOf(dictionary)), chain, options.partial ?? []);
    refoldCif = refoldCifFromDesign(options.cif, {
      templates: kept.map((t) => ({ chainId: t.chainId, residues: t.mapping?.map(([at]) => at) ?? null })), dictionary });
  }
  const input = foldInputFromRefoldCif(refoldCif);
  status(`refolding with ${model}`);
  const weights = await lf.loadAf3Weights(() => {}, lf.FOLDERS[model]);
  const { sequence, chainKinds, ligandCodes, modifications, bonds, templates } = input;

  // foldAf3 asks the PDB for a component by code. The refold input defines them, so for the length of the fold those
  // requests are answered from it (a ligand's own name is not a code the PDB has); everything else goes through.
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
      ...(bonds.length === 0 ? {} : { bonds }),
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
