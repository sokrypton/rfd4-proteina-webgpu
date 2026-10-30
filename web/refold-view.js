/**
 * Refolding a finished design with a folding model (LocalFold's `refoldDesign`, vendored by tools/sync-localfold.mjs):
 * which chains count as designed, how the refold is drawn over the design, and how its numbers read. Pure, so it is
 * unit-tested without a GPU; web/app.js loads the refold code itself only when a refold is asked for.
 */
import { toCifString } from '../shared/rfd4p/output.js';

/** Models offered for a refold, in menu order. AlphaFold 3's parameters carry Google DeepMind's terms. */
export const REFOLD_MODELS = {
  openbind0: 'OpenBind-0',
  af3: 'AlphaFold 3',
};

/** Models whose parameters need the reader's agreement first, with where the terms are. */
export const REFOLD_TERMS = {
  af3: {
    terms: 'https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md',
    policy: 'https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md',
  },
};

/** Residue sequences per chain, in file order: `Map<chainId, string[]>` of residue names. */
function chainSequences(records) {
  const chains = new Map();
  const seen = new Set();
  for (const r of records) {
    if (r.hetero) continue;
    const key = `${r.chainId}:${r.resId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (!chains.has(r.chainId)) chains.set(r.chainId, []);
    chains.get(r.chainId).push(r.resName);
  }
  return chains;
}

/**
 * The design's chains whose sequence the run changed or created: those are refolded free, and every other polymer
 * chain is handed to the fold as a template and is what the refold is fitted on. When nothing differs (an input the
 * run could not change), every polymer chain counts as designed, so the refold is fitted on itself.
 *
 * @param {object} inputAtoms the batch's columnar input atoms (`chain_id`, `res_id`, `res_name`, `hetero`)
 * @param {Array<{chainId: string, resId: number, resName: string, isHetero?: boolean}>} resultAtoms
 * @returns {string[]} designed chain ids, in the design's order
 */
export function designedChains(inputAtoms, resultAtoms) {
  const input = [];
  for (let i = 0; i < (inputAtoms?.chain_id?.length ?? 0); i += 1) {
    input.push({ chainId: inputAtoms.chain_id[i], resId: inputAtoms.res_id[i], resName: inputAtoms.res_name[i],
      hetero: Boolean(inputAtoms.hetero?.[i]) });
  }
  const before = chainSequences(input);
  const after = chainSequences(resultAtoms.map((a) => ({ ...a, hetero: Boolean(a.isHetero) })));
  const changed = [...after].filter(([id, seq]) => before.get(id)?.join(' ') !== seq.join(' ')).map(([id]) => id);
  return changed.length ? changed : [...after.keys()];
}

const BACKBONE = new Set(['N', 'CA', 'C', 'O']);

/**
 * A PDB's atoms as CIF, optionally only some chains and only the backbone. The refold's B-factor column (the model's
 * pLDDT) is kept.
 *
 * @param {string} pdb
 * @param {{chains?: string[] | null, backboneOnly?: boolean, dataBlockName?: string}} [options]
 * @returns {string|null} null when nothing is kept
 */
export function pdbToCif(pdb, { chains = null, backboneOnly = false, dataBlockName = 'refold' } = {}) {
  const keep = chains ? new Set(chains) : null;
  const atoms = [];
  for (const line of pdb.split('\n')) {
    const isHetero = line.startsWith('HETATM');
    if (!line.startsWith('ATOM') && !isHetero) continue;
    const atomName = line.slice(12, 16).trim();
    const chainId = line[21];
    if ((keep && !keep.has(chainId)) || (backboneOnly && (isHetero || !BACKBONE.has(atomName)))) continue;
    const bFactor = Number.parseFloat(line.slice(60, 66));
    atoms.push({
      atomName, chainId, isHetero, resName: line.slice(17, 20).trim(), resId: Number.parseInt(line.slice(22, 26), 10),
      element: line.slice(76, 78).trim() || atomName[0],
      x: Number.parseFloat(line.slice(30, 38)), y: Number.parseFloat(line.slice(38, 46)), z: Number.parseFloat(line.slice(46, 54)),
      ...(Number.isFinite(bFactor) ? { bFactor } : {}),
    });
  }
  return atoms.length ? toCifString(atoms, { dataBlockName }) : null;
}

/**
 * The refold's designed chains as a CIF for a cartoon overlay over the design (Studio's `setOverlay` with
 * `representation: 'cartoon'`). Templated chains are left out, since they sit on the design's own by construction.
 *
 * @param {string} pdb the refold, already in the design's frame and chain letters
 * @param {string[]} chains
 */
export function refoldOverlayCif(pdb, chains) {
  return pdbToCif(pdb, { chains });
}

/** One line for a refold result: RMSD to the design, then the model's own confidence. */
export function describeRefold(model, result, chains) {
  const name = REFOLD_MODELS[model] ?? model;
  const c = result.confidence ?? {};
  const parts = [];
  if (Number.isFinite(result.rmsd)) {
    parts.push(`${result.rmsd.toFixed(2)} Å RMSD${chains.length === 1 ? ` (chain ${chains[0]})` : ''}`);
  }
  if (Number.isFinite(c.meanPlddt)) parts.push(`pLDDT ${c.meanPlddt.toFixed(0)}`);
  if (Number.isFinite(c.ptm)) parts.push(`pTM ${c.ptm.toFixed(2)}`);
  if (Number.isFinite(c.iptm)) parts.push(`ipTM ${c.iptm.toFixed(2)}`);
  return `${name} refold: ${parts.join(' · ') || 'done'}`;
}
