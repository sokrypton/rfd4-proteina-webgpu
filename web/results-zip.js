/**
 * What "Save results (.zip)" holds: the annotated input a run designed from, each design (final structure and its
 * diffusion trajectory), every refold made of it, and a summary of all of them, as mmCIF only (a PDB would repeat
 * the same coordinates and lose the input's conditions). Pure, so it is unit-tested without a
 * browser; web/app.js gathers the data and zips the files with JSZip.
 */
import { REFOLD_MODELS, REFOLD_TERMS } from './refold-view.js';

const THREE_TO_ONE = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I', LEU: 'L',
  LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
  DA: 'A', DC: 'C', DG: 'G', DT: 'T', A: 'A', C: 'C', G: 'G', U: 'U',
};

/** Polymer sequence per chain of a PDB, in file order: `{A: 'MKV...'}`; unknown residues read as X. */
export function pdbSequences(pdb) {
  const chains = {};
  const seen = new Set();
  for (const line of (pdb ?? '').split('\n')) {
    if (!line.startsWith('ATOM')) continue;
    const key = `${line[21]}:${line.slice(22, 27)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    chains[line[21]] = (chains[line[21]] ?? '') + (THREE_TO_ONE[line.slice(17, 20).trim()] ?? 'X');
  }
  return chains;
}

const round = (v, digits) => (Number.isFinite(v) ? Number(v.toFixed(digits)) : null);

/** A refold's numbers as the summary records them. */
function refoldSummary(r) {
  const c = r.confidence ?? {};
  return {
    model: r.model,
    model_name: REFOLD_MODELS[r.model] ?? r.model,
    rmsd_designed_chain: round(r.rmsd, 3),
    plddt: round(c.meanPlddt, 2),
    ptm: round(c.ptm, 3),
    iptm: round(c.iptm, 3),
    templated_chains: r.templated ?? [],
    partial_templates: (r.partial ?? []).map((p) => ({ chain: p.chain, residues: p.residues })),
    description: r.text ?? null,
  };
}

const csvCell = (v) => {
  const s = v == null ? '' : Array.isArray(v) ? v.join(' ') : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/**
 * The package's files.
 *
 * @param {object} run
 * @param {{cif: string|null, name?: string, revision?: number}} run.input the annotated input, as designed from
 * @param {Array<{index: number, label: string, seed: number, stepsRun?: number, nsteps?: number, cif: string,
 *   pdb: string|null (read for the summary's sequences), trajectoryCif?: string|null, designedChains?: string[],
 *   refolds?: Array<{model: string, pdb: string, cif: string|null, rmsd?: number, confidence?: object,
 *   templated?: string[], partial?: object[], text?: string}>}>} run.designs
 * @param {string} [run.generatedAt] ISO time
 * @param {string} [run.page] the page's URL
 * @returns {{name: string, text: string}[]}
 */
export function resultsFiles({ input, designs, generatedAt = new Date().toISOString(), page = '' }) {
  const files = [];
  if (input?.cif) files.push({ name: 'input.cif', text: input.cif });
  const rows = [];
  const summary = {
    generated_at: generatedAt,
    page,
    input: input?.cif ? { file: 'input.cif', name: input.name ?? null, revision: input.revision ?? null } : null,
    designs: [],
  };
  let usesAf3 = false;
  for (const d of designs) {
    const stem = `design-${d.index}`;
    files.push({ name: `${stem}.cif`, text: d.cif });
    if (d.trajectoryCif) files.push({ name: `${stem}-trajectory.cif`, text: d.trajectoryCif });
    const refolds = [];
    for (const [k, r] of (d.refolds ?? []).entries()) {
      // a second refold with the same model gets its own name rather than replacing the first
      const sameModelBefore = (d.refolds ?? []).slice(0, k).filter((x) => x.model === r.model).length;
      const rstem = `${stem}-refold-${r.model}${sameModelBefore ? `-${sameModelBefore + 1}` : ''}`;
      if (r.cif) files.push({ name: `${rstem}.cif`, text: r.cif });
      if (REFOLD_TERMS[r.model]) usesAf3 = true;
      refolds.push({ files: r.cif ? [`${rstem}.cif`] : [], ...refoldSummary(r) });
    }
    const sequences = pdbSequences(d.pdb);
    summary.designs.push({
      index: d.index, label: d.label, seed: d.seed ?? null, steps_run: d.stepsRun ?? null, steps: d.nsteps ?? null,
      designed_chains: d.designedChains ?? [], sequences,
      files: [`${stem}.cif`, d.trajectoryCif ? `${stem}-trajectory.cif` : null].filter(Boolean),
      refolds,
    });
    const base = {
      design: d.index, seed: d.seed ?? null, steps_run: d.stepsRun ?? null, designed_chains: d.designedChains ?? [],
      sequences: Object.entries(sequences).map(([c, s]) => `${c}:${s}`),
    };
    if (refolds.length === 0) rows.push(base);
    for (const r of refolds) {
      rows.push({
        ...base, refold_model: r.model, rmsd_designed_chain: r.rmsd_designed_chain, plddt: r.plddt, ptm: r.ptm,
        iptm: r.iptm, templated_chains: r.templated_chains,
        partial_templates: r.partial_templates.map((p) => `${p.chain}:${p.residues}`),
      });
    }
  }
  files.push({ name: 'summary.json', text: `${JSON.stringify(summary, null, 2)}\n` });
  const columns = ['design', 'seed', 'steps_run', 'designed_chains', 'sequences', 'refold_model', 'rmsd_designed_chain',
    'plddt', 'ptm', 'iptm', 'templated_chains', 'partial_templates'];
  files.push({
    name: 'summary.csv',
    text: `${[columns.join(','), ...rows.map((r) => columns.map((c) => csvCell(r[c])).join(','))].join('\n')}\n`,
  });
  files.push({ name: 'README.txt', text: readme({ hasInput: Boolean(input?.cif), usesAf3, page, generatedAt }) });
  return files;
}

function readme({ hasInput, usesAf3, page, generatedAt }) {
  const lines = [
    'RFD4-Proteina (WebGPU) results',
    `Saved ${generatedAt}${page ? ` from ${page}` : ''}`,
    '',
    ...(hasInput ? ['input.cif                     the annotated input the run designed from (conditions included);',
      '                              open it in the page to design again'] : []),
    'design-N.cif                  design N, the final structure',
    'design-N-trajectory.cif       design N, every diffusion step (multi-model)',
    'design-N-refold-<model>.cif   a refold of design N\'s sequence, in the design\'s frame;',
    '                              the B-factor column is the model\'s pLDDT',
    'summary.json / summary.csv    per design: seed, steps, designed chains, sequences; per refold: model,',
    '                              RMSD of the designed chain, pLDDT, pTM, ipTM, templated chains',
    '',
    'The refold RMSD compares the designed chain\'s shape after fitting it on itself; it does not measure where a',
    'binder sits on its target. ipTM is the refold model\'s own confidence in the interface.',
  ];
  if (usesAf3) {
    lines.push('', 'AlphaFold 3 refolds were made with Google DeepMind\'s AlphaFold 3 parameters, which are for',
      'non-commercial use and carry a prohibited-use policy:',
      `  ${REFOLD_TERMS.af3.terms}`, `  ${REFOLD_TERMS.af3.policy}`);
  }
  return `${lines.join('\n')}\n`;
}
