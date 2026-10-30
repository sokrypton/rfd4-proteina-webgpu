/**
 * Refolding a finished design with a folding model (LocalFold's `refoldDesign`, vendored by tools/sync-localfold.mjs):
 * which chains count as designed, how the refold is drawn over the design, and how its numbers read. Pure, so it is
 * unit-tested without a GPU; web/app.js loads the refold code itself only when a refold is asked for.
 */
import { toCifString } from '../shared/rfd4p/output.js';

/** Models offered for a refold, in menu order. AlphaFold 3's parameters carry Google DeepMind's terms. */
export const REFOLD_MODELS = {
  af3: 'AlphaFold 3',
  openbind0: 'OpenBind-0',
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
 * @param {{chains?: string[] | null, backboneOnly?: boolean, heteroOnly?: boolean, dataBlockName?: string}} [options]
 * @returns {string|null} null when nothing is kept
 */
export function pdbToCif(pdb, { chains = null, backboneOnly = false, heteroOnly = false, dataBlockName = 'refold' } = {}) {
  const keep = chains ? new Set(chains) : null;
  const atoms = [];
  for (const line of pdb.split('\n')) {
    const isHetero = line.startsWith('HETATM');
    if (!line.startsWith('ATOM') && !isHetero) continue;
    const atomName = line.slice(12, 16).trim();
    const chainId = line[21];
    if ((keep && !keep.has(chainId)) || (backboneOnly && (isHetero || !BACKBONE.has(atomName))) || (heteroOnly && !isHetero)) continue;
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
 * The refold's polymer chains as a CIF for a cartoon overlay over the design (Studio's `setOverlay` with
 * `representation: 'cartoon'`): every chain, the target as well as the binder, so where the refold put each one is
 * on screen. A templated target sits on the design's own, and a refold that moved it shows that too. Ligands are a
 * separate overlay (`refoldLigandCif`), since a cartoon draws polymers only.
 *
 * @param {string} pdb the refold, already in the design's frame and chain letters
 */
export function refoldOverlayCif(pdb) {
  return pdbToCif(pdb);
}

/**
 * The refold's ligands as a CIF for a ball-and-stick overlay beside its cartoon (a cartoon draws polymers only). They
 * are placed by the refold, not templated, so where they land is part of what the refold says.
 *
 * @param {string} pdb the refold, already in the design's frame
 * @returns {string|null} null when the refold has no ligand
 */
export function refoldLigandCif(pdb) {
  return pdbToCif(pdb, { heteroOnly: true, dataBlockName: 'refold_ligands' });
}

/**
 * Where the design's motif residues sit, as positions along their chain's polymer residues. A refold numbers its
 * residues its own way, but keeps each chain's residues in the design's order, which is how its chains are matched
 * to the design's (LocalFold's `refoldOverlay`), so a position finds the same residue in both.
 *
 * @param {Array<{chainId: string, resId: number, isHetero?: boolean}>} designAtoms
 * @param {Array<{chainId: string, resId: number}>} motifAtoms atoms of the motif residues (any of their atoms)
 * @returns {{chain: string, index: number}[]}
 */
export function motifSlots(designAtoms, motifAtoms) {
  const order = new Map();
  for (const a of designAtoms) {
    if (a.isHetero) continue;
    if (!order.has(a.chainId)) order.set(a.chainId, new Map());
    const residues = order.get(a.chainId);
    if (!residues.has(a.resId)) residues.set(a.resId, residues.size);
  }
  const slots = new Map();
  for (const a of motifAtoms) {
    const index = order.get(a.chainId)?.get(a.resId);
    if (index != null) slots.set(`${a.chainId}:${index}`, { chain: a.chainId, index });
  }
  return [...slots.values()];
}

const MAIN_CHAIN = new Set(['N', 'CA', 'C', 'O', 'OXT']);

/**
 * The refold's side chains (Cβ onward) at the motif positions from `motifSlots`, as records for a sticks overlay:
 * how the refold puts the motif back, to compare with the reference motif drawn beside it.
 *
 * @param {string} pdb the refold, in the design's frame and chain letters
 * @param {{chain: string, index: number}[]} slots
 */
export function refoldMotifSideChainAtoms(pdb, slots) {
  if (!slots.length) return [];
  const wanted = new Set(slots.map((s) => `${s.chain}:${s.index}`));
  const order = new Map();
  const atoms = [];
  for (const line of pdb.split('\n')) {
    if (!line.startsWith('ATOM')) continue;
    const chainId = line[21];
    const residueKey = line.slice(22, 27);
    if (!order.has(chainId)) order.set(chainId, new Map());
    const residues = order.get(chainId);
    if (!residues.has(residueKey)) residues.set(residueKey, residues.size);
    const atomName = line.slice(12, 16).trim();
    const element = line.slice(76, 78).trim() || atomName[0];
    if (!wanted.has(`${chainId}:${residues.get(residueKey)}`) || MAIN_CHAIN.has(atomName) || element === 'H') continue;
    atoms.push({
      atomName, chainId, element, resName: line.slice(17, 20).trim(), resId: Number.parseInt(line.slice(22, 26), 10),
      x: Number.parseFloat(line.slice(30, 38)), y: Number.parseFloat(line.slice(38, 46)), z: Number.parseFloat(line.slice(46, 54)),
      isHetero: true, groupPdb: 'HETATM',
    });
  }
  return atoms;
}

/**
 * Designed chains worth a partial template: those whose residues the run mostly held to the input's coordinates
 * (an antibody's framework around redesigned CDR loops, a protein with a few loops redesigned). The template covers
 * the held residues only, so the redesigned ones and the docking are still the refold's to find. A chain with only a
 * small motif held is left free, since templating a motif would hand a motif-scaffolding refold its answer.
 *
 * A residue is held when the run conditioned it on the input's geometry: absolute coordinates (`crd`, a motif or a
 * target) or internal coordinates (an antibody framework, whose shape is fixed and whose placement is not). Both
 * set `crdSrcComponent`.
 *
 * @param {Array<{chainId: string, resId: number, isHetero?: boolean, crd?: number[], crdSrcComponent?: string}>}
 *   designAtoms the design's atoms
 * @param {string[]} chains the designed chains
 * @param {number} [minFraction] the share of a chain that must be held
 * @returns {{chain: string, residues: number[], of: number}[]} held residues as places along the chain
 */
export function partialTemplates(designAtoms, chains, minFraction = 0.5) {
  const wanted = new Set(chains);
  const residues = new Map();
  for (const a of designAtoms) {
    if (a.isHetero || !wanted.has(a.chainId)) continue;
    if (!residues.has(a.chainId)) residues.set(a.chainId, new Map());
    const chain = residues.get(a.chainId);
    if (!chain.has(a.resId)) chain.set(a.resId, { index: chain.size, held: false });
    if ((a.crd && Number.isFinite(a.crd[0])) || (a.crdSrcComponent && a.crdSrcComponent !== '.')) {
      chain.get(a.resId).held = true;
    }
  }
  const out = [];
  for (const [chain, byResidue] of residues) {
    const held = [...byResidue.values()].filter((r) => r.held).map((r) => r.index);
    if (held.length >= minFraction * byResidue.size && held.length < byResidue.size) {
      out.push({ chain, residues: held, of: byResidue.size });
    }
  }
  return out;
}

/** One line for a refold result: RMSD to the design, then the model's own confidence. */
export function describeRefold(model, result, chains, partial = []) {
  const name = REFOLD_MODELS[model] ?? model;
  const c = result.confidence ?? {};
  const parts = [];
  if (Number.isFinite(result.rmsd)) {
    parts.push(`${result.rmsd.toFixed(2)} Å RMSD${chains.length === 1 ? ` (chain ${chains[0]})` : ''}`);
  }
  if (Number.isFinite(c.meanPlddt)) parts.push(`pLDDT ${c.meanPlddt.toFixed(0)}`);
  if (Number.isFinite(c.ptm)) parts.push(`pTM ${c.ptm.toFixed(2)}`);
  if (Number.isFinite(c.iptm)) parts.push(`ipTM ${c.iptm.toFixed(2)}`);
  for (const p of partial) parts.push(`chain ${p.chain} templated on ${p.residues.length}/${p.of} kept residues`);
  return `${name} refold: ${parts.join(' · ') || 'done'}`;
}
