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
        kind: sentinel ? 'segment' : polymerKind(String(atoms.res_name[i]), Boolean(atoms.hetero?.[i])),
        segment: sentinel ? { min: Number(segMin?.[i] ?? 0), max: Number(segMax?.[i] ?? 0), atomIndex: i } : null,
        crdAtoms: 0,
        crdBackbone: 0,
        resolved: false,
        seqFixed: false,
        unindexed: false,
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
    if (hotspot?.[i]) res.hotspots += 1;
  }

  const out = [];
  for (const chain of chains.values()) {
    const polymer = chain.residues.filter((r) => r.kind && r.kind !== 'segment');
    const segments = chain.residues.filter((r) => r.kind === 'segment').map((r) => r.segment);
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
    else if (!fixed.length && !seqFixed.length && !segments.length) role = 'free';
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
  const totalMin = polymers.reduce((s, c) => s + c.lengthMin, 0);
  const totalMax = polymers.reduce((s, c) => s + c.lengthMax, 0);
  const anyCondition = out.some((c) => c.fixed || c.seqFixed || c.hotspots || c.ligands.some((l) => l.fixed));
  const ligandNames = ligands.flatMap((c) => c.ligands.map((l) => l.name));

  const task = inferTask({ polymers, ligands, anyCondition, designedMax, partialDiffusion, partialPercent });
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
    totalMin,
    totalMax,
    contig: polymers.filter((c) => c.contig).map((c) => c.contig).join('/0 ') +
      (ligandNames.length ? ` + ${ligandNames.join(', ')}` : ''),
    warnings,
  };
}

function inferTask({ polymers, ligands, anyCondition, designedMax, partialDiffusion, partialPercent }) {
  const T = (key, label, detail) => ({ key, label, detail });
  if (partialDiffusion) {
    return T('partial', 'Diversify (partial diffusion)', `Starts ${partialPercent}% of the way from this structure.`);
  }
  if (!polymers.length && !ligands.length) return T('empty', 'Nothing loaded', '');
  if (designedMax === 0) return T('nothing', 'Nothing to design', '');
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
