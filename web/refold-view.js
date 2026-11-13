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

/** The RMSD at or below which a de novo design is usually called self-consistent. */
export const SELF_CONSISTENT_RMSD_A = 2;

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
export function describeRefold(model, result, chains, partial = [], motif = null) {
  const name = REFOLD_MODELS[model] ?? model;
  const c = result.confidence ?? {};
  const parts = [];
  if (result.comparable === false) {
    // `selfConsistency` returns NaN rather than a large number when the two folds describe different numbers of
    // residues, because an RMSD between them compares residue 1 of one thing to residue 1 of another. Say that,
    // rather than dropping the RMSD and leaving a line that looks like a result.
    parts.push('RMSD not comparable — the design and the refold have different lengths');
  } else if (Number.isFinite(result.rmsd)) {
    // The number and its verdict together: 2 Å is the threshold the field holds a self-consistent de novo design
    // to, and a bare "9.04 Å" beside a confident pLDDT reads as success to anyone who has not met that literature.
    const verdict = result.rmsd <= SELF_CONSISTENT_RMSD_A ? '' : ` — above the ${SELF_CONSISTENT_RMSD_A} Å a`
      + ' self-consistent design is usually held to';
    parts.push(`${result.rmsd.toFixed(2)} Å RMSD${chains.length === 1 ? ` (chain ${chains[0]})` : ''}${verdict}`);
  }
  if (Number.isFinite(c.meanPlddt)) parts.push(`pLDDT ${c.meanPlddt.toFixed(0)}`);
  if (Number.isFinite(c.ptm)) parts.push(`pTM ${c.ptm.toFixed(2)}`);
  if (Number.isFinite(c.iptm)) parts.push(`ipTM ${c.iptm.toFixed(2)}`);
  // The motif put back: fitted on itself, then fitted on the designed backbone (see motif-metrics.js)
  if (motif && Number.isFinite(motif.motifRmsd)) {
    parts.push(`motif ${motif.motifRmsd.toFixed(2)} Å`
      + (Number.isFinite(motif.motifBackboneRmsd) ? ` (${motif.motifBackboneRmsd.toFixed(2)} Å on the backbone)` : ''));
  }
  for (const p of partial) parts.push(`chain ${p.chain} templated on ${p.residues.length}/${p.of} kept residues`);
  return `${name} refold: ${parts.join(' · ') || 'done'}`;
}

const WATER = new Set(['HOH', 'WAT', 'DOD']);

/**
 * The polymer chains of a PDB in file order, each with its residue count, as LocalFold reads them for a refold
 * (`chainsOf`, `foldInputFromPdb`): a chain is a polymer if it has any ATOM record, water left out. A refold's tokens
 * are these residues first, one each, in this order, then the ligands.
 * @returns {{letter: string, residues: number}[]}
 */
export function polymerChains(pdb) {
  const chains = new Map();
  let previous = null;
  for (const line of String(pdb ?? '').split('\n')) {
    if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) continue;
    if (WATER.has(line.slice(17, 20).trim())) continue;
    const letter = line[21];
    if (!chains.has(letter)) chains.set(letter, { letter, polymer: false, residues: 0 });
    const chain = chains.get(letter);
    if (line.startsWith('ATOM  ')) chain.polymer = true;
    const key = line.slice(21, 27);
    if (key !== previous) { chain.residues += 1; previous = key; }
  }
  return [...chains.values()].filter((c) => c.polymer).map(({ letter, residues }) => ({ letter, residues }));
}

/**
 * A refold's numbers as upstream's pass criteria read them (`configs/metrics/pass_criteria`):
 * - `rmsd`: the designed chains' shape, fitted on themselves (`refolded_backbone_rmsd`)
 * - `placedRmsd`: the designed chains after fitting the refold on everything else, so where a binder sits on its
 *   target counts (`refolded_target_aligned_binder_rmsd`); the refold's per-residue `deviations` are measured so
 * - `chainPtm`: the designed chain's own pTM (`ptm_coi`), the lowest when several were designed
 * - `minIpae`: the lowest predicted aligned error between a designed residue and any other chain's (`min_ipae_coi`)
 * - `plddt`, `ptm`, `iptm`: the refold model's confidence
 * NaN where a number does not apply (no interface, no confidence head).
 */
export function refoldMetrics(result, designed, designPdb) {
  const c = result?.confidence ?? {};
  const mine = new Set(designed ?? []);
  const dev = (result?.deviations ?? []).filter((d) => mine.has(d.chain) && Number.isFinite(d.distance));
  const placedRmsd = dev.length ? Math.sqrt(dev.reduce((s, d) => s + d.distance ** 2, 0) / dev.length) : NaN;
  const chains = polymerChains(designPdb);
  const tokenChain = chains.flatMap((ch) => new Array(ch.residues).fill(ch.letter));
  // chainPtm is keyed by the fold's chain number: polymers first, in this order, from 1.
  const ptms = chains.map((ch, k) => (mine.has(ch.letter) ? Number(c.chainPtm?.[k + 1]) : NaN)).filter(Number.isFinite);
  let minIpae = NaN;
  const pae = c.predictedAlignedError;
  const n = pae ? Math.round(Math.sqrt(pae.length)) : 0;
  if (n && n * n === pae.length && tokenChain.length <= n && chains.some((ch) => !mine.has(ch.letter))) {
    let best = Infinity;
    for (let i = 0; i < tokenChain.length; i += 1) {
      if (!mine.has(tokenChain[i])) continue;
      for (let j = 0; j < tokenChain.length; j += 1) {
        if (mine.has(tokenChain[j])) continue;
        best = Math.min(best, pae[i * n + j], pae[j * n + i]);
      }
    }
    if (Number.isFinite(best)) minIpae = best;
  }
  return {
    rmsd: result?.comparable === false ? NaN : Number(result?.rmsd ?? NaN),
    placedRmsd,
    plddt: Number(c.meanPlddt ?? NaN),
    ptm: Number(c.ptm ?? NaN),
    iptm: Number(c.iptm ?? NaN),
    chainPtm: ptms.length ? Math.min(...ptms) : NaN,
    minIpae,
  };
}

/**
 * Which of upstream's pass criteria judge a job (`kind` on the design handed to Studio's Refold panel, which holds the
 * thresholds): its task, and whether everything the refold folds was designed. Motif, enzyme and small-molecule jobs
 * also need ligand RMSDs and thresholds not applied here, so they show numbers and no verdict.
 */
export function criteriaKind(taskKey, { designedAll = false, hasLigand = false } = {}) {
  if (taskKey === 'binder') return 'binder';
  if (taskKey === 'antibody') return 'antibody';
  if (designedAll && !hasLigand && !['motif', 'enzyme', 'sm_binder'].includes(taskKey)) return 'monomer';
  return null;
}
