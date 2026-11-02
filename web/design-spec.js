/**
 * What a Studio document asks RFD4-Proteina to design, read from its condition annotations: per-chain roles, lengths,
 * an RFdiffusion-style contig string and the task it amounts to. Pure (atom array in, plain object out) so the design
 * panel can recompute it on every edit and tests can pin it on the bundled examples.
 */

const BACKBONE_ATOMS = new Set(['N', 'CA', 'C', 'O']);

/** Display names for roles; `detail` is filled in per chain. */
export const ROLE_LABELS = {
  fixed: 'Fixed',
  backbone: 'Sequence redesigned',
  sequence: 'Fold predicted',
  free: 'Redesigned',
  motif: 'Partly fixed',
  new: 'New chain',
  ligand: 'Ligand',
};

const isSentinel = (atoms, i) => {
  const min = atoms.annotations?.annotation_expsegmin_1_residue?.[i];
  return (Number.isFinite(min) && min > 0) || /^\d+-\d+$/.test(String(atoms.res_name[i]));
};

const hasCrd = (c) => Array.isArray(c) && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2]);

/** Polymer kind of a residue name, or null for ligands / ions. */
function polymerKind(resName, hetero) {
  if (hetero) return null;
  if (/^D[ACGTU]$/.test(resName)) return 'dna';
  if (/^[ACGU]$/.test(resName)) return 'rna';
  return 'protein';
}

const lengthText = (min, max) => (min === max ? String(min) : `${min}–${max}`);

/**
 * @param {object} atoms AtomArray-like (`parseCifToAtomArray` / `StudioDoc.atoms`)
 * @param {{partialDiffusion?: boolean, partialPercent?: number}} [opts]
 */
export function designSpec(atoms, { partialDiffusion = false, partialPercent = 70 } = {}) {
  const n = atoms?.nAtoms ?? atoms?.chain_id?.length ?? 0;
  const ann = atoms?.annotations ?? {};
  const crd = ann.condition_coordinate_1_atom;
  const seqCond = ann.condition_sequence_1_residue;
  const indexCond = ann.condition_index_1_residue;
  const hotspot = ann.condition_hotspot_1_atom;
  const group = ann.condition_CoordinateGroup_1_atom;
  const internal = ann.condition_InternalCoordinate_1_atom;
  // Where a residue came from ("A38"), when a set-up renumbered it.
  const source = ann['annotation_src-component_1_atom'];
  const segMin = ann.annotation_expsegmin_1_residue;
  const segMax = ann.annotation_expsegmax_1_residue;

  // Residues in document order, grouped per chain.
  const chains = new Map();
  for (let i = 0; i < n; i += 1) {
    const chainId = String(atoms.chain_id[i]);
    let chain = chains.get(chainId);
    if (!chain) {
      chain = { id: chainId, residues: [], byKey: new Map(), atomIndices: [] };
      chains.set(chainId, chain);
    }
    chain.atomIndices.push(i);
    const key = `${atoms.res_id[i]}|${atoms.ins_code?.[i] ?? ''}`;
    let res = chain.byKey.get(key);
    if (!res) {
      const sentinel = isSentinel(atoms, i);
      res = {
        resId: Number(atoms.res_id[i]),
        resName: String(atoms.res_name[i]),
        // is_polymer, not HETATM: modified residues (MSE, SEP, ...) are polymer residues stored as HETATM.
        kind: sentinel ? 'segment' : polymerKind(String(atoms.res_name[i]),
          atoms.is_polymer?.length === n ? !atoms.is_polymer[i] : Boolean(atoms.hetero?.[i])),
        segment: sentinel ? { min: Number(segMin?.[i] ?? 0), max: Number(segMax?.[i] ?? 0), atomIndex: i } : null,
        crdAtoms: 0,
        crdBackbone: 0,
        resolved: false,
        seqFixed: false,
        unindexed: false,
        relative: false,
        source: null,
        hotspots: 0,
        atoms: 0,
        atomIndices: [],
      };
      if (res.segment && !(res.segment.min > 0)) {
        const m = /^(\d+)-(\d+)$/.exec(res.resName);
        if (m) res.segment = { min: Number(m[1]), max: Number(m[2]), atomIndex: i };
      }
      chain.byKey.set(key, res);
      chain.residues.push(res);
    }
    res.atoms += 1;
    res.atomIndices.push(i);
    if (hasCrd(atoms.coord?.[i])) res.resolved = true;
    if (hasCrd(crd?.[i])) {
      res.crdAtoms += 1;
      if (BACKBONE_ATOMS.has(String(atoms.atom_name[i]))) res.crdBackbone += 1;
    }
    const s = seqCond?.[i];
    if (s != null && s !== '<M>' && s !== '.' && s !== '' && s !== '?') res.seqFixed = true;
    if (indexCond && indexCond[i] === false) res.unindexed = true;
    // Held by its shape only (a framework): a coordinate group other than the absolute one, or internal coordinates.
    if (Number(group?.[i]) > 0 || hasCrd(internal?.[i])) res.relative = true;
    if (res.source == null && /^[A-Za-z]{1,4}-?\d+$/.test(String(source?.[i] ?? ''))) res.source = String(source[i]);
    if (hotspot?.[i]) res.hotspots += 1;
  }

  const out = [];
  for (const chain of chains.values()) {
    const polymer = chain.residues.filter((r) => r.kind && r.kind !== 'segment');
    // Where each segment sits relative to the chain's existing residues (for labels).
    const ordered = chain.residues.filter((r) => r.kind);
    const segments = [];
    ordered.forEach((r, k) => {
      if (r.kind !== 'segment') return;
      const prev = ordered.slice(0, k).reverse().find((x) => x.kind !== 'segment');
      const next = ordered.slice(k + 1).find((x) => x.kind !== 'segment');
      const where = !prev && !next ? '' : !prev ? 'N-terminal' : !next ? 'C-terminal'
        : `between ${chain.id}${prev.resId} and ${chain.id}${next.resId}`;
      segments.push({ ...r.segment, where });
    });
    const ligands = chain.residues.filter((r) => !r.kind);
    const fixed = polymer.filter((r) => r.crdAtoms > 0);
    // Unmodelled residues (no coordinates) cannot be held in place, so roles are judged on the modelled ones.
    const resolved = polymer.filter((r) => r.resolved || r.crdAtoms > 0);
    const allFixed = fixed.length > 0 && fixed.length === resolved.length;
    const seqFixed = polymer.filter((r) => r.seqFixed);
    const freeRes = polymer.filter((r) => !r.crdAtoms && !r.seqFixed);
    const segMinSum = segments.reduce((s, g) => s + g.min, 0);
    const segMaxSum = segments.reduce((s, g) => s + g.max, 0);

    let role;
    if (!polymer.length && !segments.length) role = 'ligand';
    else if (!polymer.length) role = 'new';
    else if (allFixed && seqFixed.length === polymer.length && !segments.length) role = 'fixed';
    else if (allFixed && !seqFixed.length && !segments.length) role = 'backbone';
    else if (!fixed.length && seqFixed.length === polymer.length && !segments.length) role = 'sequence';
    else if (!fixed.length && !seqFixed.length) role = 'free';
    else role = 'motif';

    // Contig: fixed residue runs as `A12-20`, designed runs and segments as lengths, `/` between pieces.
    const pieces = [];
    let run = null;
    const flush = () => {
      if (!run) return;
      const range = run.start === run.end ? `${chain.id}${run.start}` : `${chain.id}${run.start}-${run.end}`;
      pieces.push(run.type === 'fixed' ? range : run.type === 'unindexed' ? `[${range}]` : String(run.count));
      run = null;
    };
    for (const r of chain.residues) {
      if (!r.kind) continue;
      if (r.kind === 'segment') {
        flush();
        pieces.push(lengthText(r.segment.min, r.segment.max).replace('–', '-'));
        continue;
      }
      // Unindexed motif residues are placed by the model anywhere in the chain: bracketed, and not counted in length.
      const type = r.unindexed ? 'unindexed' : r.crdAtoms > 0 ? 'fixed' : 'designed';
      if (run && run.type === type && (type === 'designed' || r.resId === run.end + 1)) {
        run.end = r.resId;
        run.count += 1;
      } else {
        flush();
        run = { type, start: r.resId, end: r.resId, count: 1 };
      }
    }
    flush();

    const placed = polymer.filter((r) => !r.unindexed).length;
    const segmentWhere = new Map(segments.map((g) => [g.atomIndex, g.where]));
    const lengthMin = placed + segMinSum;
    const lengthMax = placed + segMaxSum;
    out.push({
      id: chain.id,
      role,
      kind: polymer[0]?.kind ?? (segments.length ? 'protein' : 'ligand'),
      residues: polymer.length,
      fixed: fixed.length,
      backboneOnly: fixed.filter((r) => r.crdAtoms === r.crdBackbone).length,
      unresolved: polymer.length - resolved.length,
      seqFixed: seqFixed.length,
      designed: freeRes.length + polymer.filter((r) => r.crdAtoms && !r.seqFixed).length,
      // Sequence kept but structure free: not designed, but still built by the model.
      predicted: polymer.filter((r) => !r.crdAtoms && r.seqFixed).length,
      unindexed: polymer.filter((r) => r.unindexed).length,
      hotspots: chain.residues.filter((r) => r.hotspots > 0).length,
      segments,
      ligands: ligands.map((r) => ({
        name: r.resName,
        fixed: r.crdAtoms > 0,
        pose: r.crdAtoms === 0 ? 'predicted' : r.crdAtoms < r.atoms ? 'partly fixed' : 'fixed',
        atoms: r.atoms,
        fixedAtoms: r.crdAtoms,
      })),
      lengthMin,
      lengthMax,
      contig: pieces.join('/'),
      pieces: chainPieces(chain.residues, segmentWhere),
      hotspotResIds: polymer.filter((r) => r.hotspots > 0).map((r) => r.resId),
      atomIndices: chain.atomIndices,
      polymerAtomIndices: polymer.flatMap((r) => r.atomIndices),
      backboneAtomIndices: polymer.flatMap((r) => r.atomIndices.filter((i) => BACKBONE_ATOMS.has(String(atoms.atom_name[i])))),
      ligandAtomIndices: ligands.flatMap((r) => r.atomIndices),
      firstResId: polymer[0]?.resId ?? null,
      lastResId: polymer[polymer.length - 1]?.resId ?? null,
    });
  }

  const polymers = out.filter((c) => c.role !== 'ligand');
  const ligands = out.filter((c) => c.role === 'ligand');
  const designedMin = polymers.reduce((s, c) => s + c.designed + c.segments.reduce((t, g) => t + g.min, 0), 0);
  const designedMax = polymers.reduce((s, c) => s + c.designed + c.segments.reduce((t, g) => t + g.max, 0), 0);
  const predicted = polymers.reduce((s, c) => s + c.predicted, 0);
  const totalMin = polymers.reduce((s, c) => s + c.lengthMin, 0);
  const totalMax = polymers.reduce((s, c) => s + c.lengthMax, 0);
  const anyCondition = out.some((c) => c.fixed || c.seqFixed || c.hotspots || c.ligands.some((l) => l.fixed));
  const ligandNames = ligands.flatMap((c) => c.ligands.map((l) => l.name));

  const task = inferTask({ polymers, ligands, anyCondition, designedMax, predicted, partialDiffusion, partialPercent });
  const warnings = [];
  if (!n) warnings.push('Open a structure or pick a task to start.');
  else if (task.key === 'redesign_all') {
    warnings.push('Nothing in this structure is fixed, so all of it will be redesigned. Choose what to keep above.');
  } else if (task.key === 'nothing') {
    warnings.push('Everything is fixed, so there is nothing to design. Free a chain or add new residues.');
  }

  return {
    empty: n === 0,
    task,
    chains: out,
    designedMin,
    designedMax,
    predicted,
    totalMin,
    totalMax,
    contig: polymers.filter((c) => c.contig).map((c) => c.contig).join('/0 ') +
      (ligandNames.length ? ` + ${ligandNames.join(', ')}` : ''),
    warnings,
  };
}

/**
 * How a kept piece is held: everything (the default), or only part of it. Each is a modifier the panel shows as a chip.
 */
export const KEEP_MODES = {
  all: { chip: '', label: 'Keep everything', help: 'Structure and sequence kept as they are' },
  sequence: { chip: 'sequence only', label: 'Keep the sequence only', help: 'Same amino acids; the model predicts the shape' },
  backbone: { chip: 'backbone only', label: 'Keep the backbone only', help: 'Same shape; the model writes a new sequence' },
  sidechain: { chip: 'side chains only', label: 'Keep the side chains only', help: 'Catalytic or binding groups kept; backbone free' },
  shape: { chip: 'shape only', label: 'Keep the shape, not the position', help: 'Held rigid as a unit, but free to move (a framework)' },
};

/** A residue's piece kind ('kept', 'designed', 'new') and, for a kept one, its keep mode. */
function pieceOf(r) {
  if (r.kind === 'segment') return { kind: 'new' };
  let keep = null;
  if (r.relative) keep = 'shape';
  else if (r.crdAtoms > 0) keep = r.crdBackbone === 0 ? 'sidechain' : r.seqFixed ? 'all' : 'backbone';
  else if (r.seqFixed) keep = 'sequence';
  return keep ? { kind: 'kept', keep, anywhere: r.unindexed } : { kind: 'designed' };
}

/** "A38" -> ["A", 38] */
const splitSource = (s) => {
  const m = /^([A-Za-z]{1,4})(-?\d+)$/.exec(s ?? '');
  return m ? [m[1], Number(m[2])] : null;
};

/**
 * A chain as the pieces a contig is made of, in order: runs of consecutive residues treated alike (`kept`, with a
 * keep mode and whether it may sit anywhere, or `designed` in place), and each new-residue segment (`new`) on its
 * own. A run renumbered by a set-up carries where it came from (`source`, e.g. "A38" or "A38–40"). Ligands are not
 * pieces (they are listed per chain).
 */
function chainPieces(residues, segmentWhere) {
  const out = [];
  let run = null;
  for (const r of residues) {
    if (!r.kind) continue;
    const p = pieceOf(r);
    if (p.kind === 'new') {
      run = null;
      out.push({ kind: 'new', min: r.segment.min, max: r.segment.max, atomIndex: r.segment.atomIndex,
        where: segmentWhere.get(r.segment.atomIndex) ?? '', count: 0, atomIndices: r.atomIndices, hotspots: [] });
      continue;
    }
    const src = splitSource(r.source);
    const sameRun = run && run.kind === p.kind && run.keep === p.keep && run.anywhere === p.anywhere
      && r.resId === run.end + 1
      && (!run.src && !src || (run.src && src && src[0] === run.src[0] && src[1] === run.srcEnd + 1));
    if (sameRun) {
      run.end = r.resId;
      run.count += 1;
      if (src) run.srcEnd = src[1];
      run.atomIndices.push(...r.atomIndices);
    } else {
      run = { ...p, start: r.resId, end: r.resId, count: 1, atomIndices: [...r.atomIndices], hotspots: [],
        src, srcEnd: src?.[1] ?? null };
      out.push(run);
    }
    if (r.hotspots > 0) run.hotspots.push(r.resId);
  }
  for (const p of out) {
    if (p.src && (p.src[1] !== p.start || p.srcEnd !== p.end)) {
      p.source = p.src[1] === p.srcEnd ? `${p.src[0]}${p.src[1]}` : `${p.src[0]}${p.src[1]}–${p.srcEnd}`;
    }
    delete p.src;
    delete p.srcEnd;
  }
  return out;
}

/**
 * Residues as typed: "A56, A115-123, B7" (chain letter then number or range; a bare number takes `defaultChain`).
 * Returns `{residues: [{chain, resId}], bad: ['tokens not understood']}`.
 */
export function parseResidueList(text, defaultChain = null) {
  const residues = [];
  const bad = [];
  for (const token of String(text ?? '').split(/[\s,;]+/).filter(Boolean)) {
    const m = /^([A-Za-z]{1,4})?:?(-?\d+)(?:[-–](-?\d+))?$/.exec(token.replace(/^([A-Za-z]+)(\d)/, '$1:$2'));
    const chain = m?.[1] ?? defaultChain;
    if (!m || !chain) { bad.push(token); continue; }
    const lo = Number(m[2]);
    const hi = m[3] == null ? lo : Number(m[3]);
    if (hi < lo || hi - lo > 2000) { bad.push(token); continue; }
    for (let id = lo; id <= hi; id += 1) residues.push({ chain, resId: id });
  }
  return { residues, bad };
}

/** The inverse of parseResidueList for one chain: [56, 115, 116, 117] -> "A56, A115–117". */
export function formatResidueList(chain, resIds) {
  const ids = [...new Set(resIds)].sort((a, b) => a - b);
  const parts = [];
  for (let k = 0; k < ids.length;) {
    let j = k;
    while (j + 1 < ids.length && ids[j + 1] === ids[j] + 1) j += 1;
    parts.push(j === k ? `${chain}${ids[k]}` : `${chain}${ids[k]}–${ids[j]}`);
    k = j + 1;
  }
  return parts.join(', ');
}

function inferTask({ polymers, ligands, anyCondition, designedMax, predicted, partialDiffusion, partialPercent }) {
  const T = (key, label, detail) => ({ key, label, detail });
  if (partialDiffusion) {
    return T('partial', 'Diversify (partial diffusion)', `Starts ${partialPercent}% of the way from this structure.`);
  }
  if (!polymers.length && !ligands.length) return T('empty', 'Nothing loaded', '');
  if (designedMax === 0 && predicted === 0) return T('nothing', 'Nothing to design', '');
  if (designedMax === 0) {
    return T('fold', 'Fold from sequence', 'Sequences kept; the model predicts the structure.');
  }
  // A structure as fetched (no conditions, no new residues) is "everything redesigned", whatever it contains.
  if (!anyCondition && !polymers.some((c) => c.segments.length) && !polymers.every((c) => c.role === 'new')) {
    return T('redesign_all', 'Full redesign', 'No constraints: structure and sequence are both redesigned.');
  }
  const roles = new Set(polymers.map((c) => c.role));
  const unindexed = polymers.some((c) => c.unindexed > 0);
  const hasLigand = ligands.length > 0;
  const fixedChains = polymers.filter((c) => c.role === 'fixed');
  const designedChains = polymers.filter((c) => c.role !== 'fixed');
  if (polymers.every((c) => c.role === 'backbone')) {
    return T('inverse', 'Redesign sequence', 'Backbone kept; new sequence and side chains.');
  }
  if (unindexed && hasLigand) return T('enzyme', 'Enzyme design', 'Catalytic residues placed around the ligand.');
  if (fixedChains.length && designedChains.length) {
    const framework = designedChains.some((c) => c.seqFixed > 0);
    return framework
      ? T('antibody', 'Antibody / VHH design', 'Framework sequence kept; loops designed against the target.')
      : T('binder', 'Binder design', 'New protein designed to bind the fixed target.');
  }
  if (hasLigand && !fixedChains.length) {
    return T('sm_binder', 'Small-molecule binder', 'Protein designed around the ligand.');
  }
  if (roles.has('motif') || unindexed) {
    return T('motif', 'Motif scaffolding', 'Fixed motif residues kept; scaffold designed around them.');
  }
  if (!anyCondition) {
    const onlyNew = polymers.every((c) => c.role === 'new');
    return onlyNew
      ? T('unconditional', 'Protein from scratch', 'No constraints: a new fold of the given length.')
      : T('redesign_all', 'Full redesign', 'No constraints: structure and sequence are both redesigned.');
  }
  return T('custom', 'Custom design', 'Mixed constraints; see the chains below.');
}
