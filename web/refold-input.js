/**
 * A finished design read as a fold's input when it holds residues and ligands a refold cannot name from a PDB alone:
 * a modified residue in a chain (the PTE motif's carboxy-lysine, KCX) and a ligand that is not in the PDB's component
 * dictionary (the PTE's transition-state analogue, "L:0", and its "W:1" bridge).
 *
 * 🔴 THE SAME RULES RFD4's AF3 ORACLE FOLLOWS (rfproteina/metrics/oracles/af3_oracle.py). It hands AF3 the design's CIF,
 * so AF3's own reader takes a modified residue as a modification of its parent (a K carrying KCX) and every
 * non-standard residue and ligand is defined for AF3 from the design's own atoms and bonds (`_build_user_ccd_string`:
 * AF3's "user CCD"), never looked up by name. LocalFold's `foldInputFromPdb` reads a PDB: it has no bonds to build a
 * ligand from and refuses any residue outside the twenty ("chain A has KCX ..., which a refold cannot read yet"), so
 * every design carrying either one failed to refold, with every model, before the fold started.
 *
 * Here the two are done the way the oracle does them, in the terms `foldAf3` already takes:
 *   - a modified residue is `modifications: [{chain, position, code}]` on its parent's letter, its component from the
 *     PDB's dictionary like any other (KCX is there);
 *   - a ligand whose name is not a dictionary code (a name like "L:0" is not one) is a component built from the
 *     design's own atoms and `_chem_comp_bond` rows. `foldAf3` fetches components by code, so such a ligand goes in
 *     under a stand-in code (`ZZ0`, `ZZ1`, ...) and `components` says what text to serve for each.
 *
 * Pure: strings in, plain objects out, so test/refold-input.test.js holds it without a browser.
 */
import { catRows, parseCifBlock } from '../shared/rfd4p/pipeline/cif.js';

const AMINO = { ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V' };
const DNA = { DA: 'A', DC: 'C', DG: 'G', DT: 'T' };
const RNA = { A: 'A', C: 'C', G: 'G', U: 'U' };
const WATER = new Set(['HOH', 'WAT', 'DOD']);

/**
 * Modified amino acids and the standard residue each stands for: the letter the fold's sequence carries at that place.
 * The PDB dictionary's own parent is what the fold then uses for the residue (`mon_nstd_parent_comp_id`); this table
 * only has to name the letter before the dictionary has been asked. A residue outside it is refused by name.
 */
export const MODIFIED_PARENT = {
  KCX: 'LYS', MLY: 'LYS', M3L: 'LYS', MLZ: 'LYS', ALY: 'LYS', LLP: 'LYS', HIC: 'HIS', MSE: 'MET', SEP: 'SER',
  TPO: 'THR', PTR: 'TYR', CSO: 'CYS', CME: 'CYS', CSD: 'CYS', OCS: 'CYS', HYP: 'PRO', PCA: 'GLN', SMC: 'CYS',
};

/** A component code the PDB serves (and `ccdUrl` accepts): 1-5 capitals or digits. "L:0" is not one. */
export const isDictionaryCode = (code) => /^[A-Z0-9]{1,5}$/.test(code);

const isRecord = (line) => line.startsWith('ATOM  ') || line.startsWith('HETATM');

/** The PDB's chains in file order, water left out: letter, whether any residue is an ATOM record, residues (name). */
function chainsOf(pdb) {
  const chains = new Map();
  let previous = null;
  for (const line of pdb.split('\n')) {
    if (!isRecord(line)) continue;
    const name = line.slice(17, 20).trim();
    if (WATER.has(name)) continue;
    const letter = line[21];
    if (!chains.has(letter)) chains.set(letter, { letter, polymer: false, residues: [] });
    const chain = chains.get(letter);
    if (line.startsWith('ATOM  ')) chain.polymer = true;
    const key = line.slice(21, 27);
    if (key !== previous) { chain.residues.push({ name }); previous = key; }
  }
  return [...chains.values()];
}

/** A chain's kind and its one-letter sequence, modified residues read as their parents (and listed). */
function polymerOf(chain) {
  for (const [kind, table] of [['protein', AMINO], ['dna', DNA], ['rna', RNA]]) {
    const parentOf = (name) => (kind === 'protein' && MODIFIED_PARENT[name] !== undefined ? AMINO[MODIFIED_PARENT[name]] : undefined);
    if (chain.residues.every((r) => table[r.name] !== undefined || parentOf(r.name) !== undefined)) {
      const modifications = [];
      const sequence = chain.residues.map((r, at) => {
        if (table[r.name] !== undefined) return table[r.name];
        modifications.push({ position: at + 1, code: r.name });
        return parentOf(r.name);
      }).join('');
      return { kind, sequence, modifications };
    }
  }
  const odd = chain.residues.find((r) => AMINO[r.name] === undefined && MODIFIED_PARENT[r.name] === undefined
    && DNA[r.name] === undefined && RNA[r.name] === undefined);
  throw new Error(odd === undefined
    ? `chain ${chain.letter} mixes amino acids and nucleotides, which a refold cannot read`
    : `chain ${chain.letter} has ${odd.name}, which a refold cannot read: it is neither a standard residue nor one of `
      + `the modified residues known here (${Object.keys(MODIFIED_PARENT).join(', ')})`);
}

const quoted = (value) => (/[\s'"]/.test(value) ? `"${value}"` : value);

/**
 * One ligand as the text of a dictionary entry, from the design's own atoms and bond rows: what AF3's user CCD holds
 * for it in the oracle. Heavy atoms only (a fold tokenises those), coordinates as the design has them (a fold
 * centres them), bonds from `_chem_comp_bond` where the design lists the ligand there.
 */
function componentText(code, name, letter, block) {
  const atoms = catRows(block.atom_site);
  const rows = [];
  for (let r = 0; r < atoms.n; r += 1) {
    if (String(atoms.get('label_comp_id', r)) !== name) continue;
    // (one chain's atoms: a ligand drawn twice is two chains of the same name)
    if (![atoms.get('auth_asym_id', r), atoms.get('label_asym_id', r)].some((v) => String(v) === letter)) continue;
    const element = String(atoms.get('type_symbol', r) ?? '').toUpperCase();
    if (element === 'H' || element === 'D') continue;
    const charge = Number.parseInt(String(atoms.get('pdbx_formal_charge', r) ?? ''), 10);
    rows.push({ atom: String(atoms.get('label_atom_id', r)), element, charge: Number.isFinite(charge) ? charge : 0,
      xyz: ['Cartn_x', 'Cartn_y', 'Cartn_z'].map((c) => Number.parseFloat(atoms.get(c, r))) });
  }
  if (rows.length === 0) throw new Error(`ligand ${name}: the design's CIF has no atoms for it`);
  const bonds = [];
  if (block.chem_comp_bond) {
    const table = catRows(block.chem_comp_bond);
    const have = new Set(rows.map((a) => a.atom));
    for (let r = 0; r < table.n; r += 1) {
      if (String(table.get('comp_id', r)) !== name) continue;
      const a = String(table.get('atom_id_1', r));
      const b = String(table.get('atom_id_2', r));
      if (have.has(a) && have.has(b)) {
        bonds.push([a, b, String(table.get('value_order', r) ?? 'SING'), String(table.get('pdbx_aromatic_flag', r) ?? 'N')]);
      }
    }
  }
  return [
    `data_${code}`, `_chem_comp.id ${code}`, '_chem_comp.type NON-POLYMER',
    'loop_', ...['comp_id', 'atom_id', 'type_symbol', 'charge', 'pdbx_leaving_atom_flag', 'model_Cartn_x',
      'model_Cartn_y', 'model_Cartn_z'].map((c) => `_chem_comp_atom.${c}`),
    ...rows.map((a) => [code, quoted(a.atom), a.element, a.charge, 'N', ...a.xyz.map((v) => v.toFixed(3))].join(' ')),
    ...(bonds.length === 0 ? [] : ['loop_', ...['comp_id', 'atom_id_1', 'atom_id_2', 'value_order', 'pdbx_aromatic_flag']
      .map((c) => `_chem_comp_bond.${c}`), ...bonds.map((b) => [code, quoted(b[0]), quoted(b[1]), b[2], b[3]].join(' '))]),
    '',
  ].join('\n');
}

/**
 * What a fold needs to make the design again, as `foldInputFromPdb` returns it, plus the two things it cannot.
 *
 * @param {string} pdb the design (what the refold is fitted onto)
 * @param {string} cif the same design as mmCIF: the source of a custom ligand's atoms and bonds
 * @returns {{sequence: string, chainKinds: string[], ligandCodes: string[], modifications: {chain: number,
 *   position: number, code: string}[], components: Object<string, string>, standIns: Object<string, string>,
 *   needed: boolean}} `components` maps each stand-in code to its dictionary text; `standIns` maps each stand-in code
 *   back to the ligand's own name (a refold names the ligand by the code it was given); `needed` is whether this design has any
 *   modified residue or custom ligand, that is, whether `foldInputFromPdb` could not have read it.
 */
export function foldInputFromDesign(pdb, cif) {
  const chains = chainsOf(pdb);
  const polymers = chains.filter((chain) => chain.polymer).map(polymerOf);
  if (polymers.length === 0) throw new Error('the design has no polymer chain to refold');
  const modifications = polymers.flatMap((polymer, chain) => polymer.modifications.map((m) => ({ chain, ...m })));
  const components = {};
  const standIns = {};
  let block = null;
  let custom = 0;
  const ligandCodes = chains.filter((chain) => !chain.polymer).map((chain) => {
    if (chain.residues.length !== 1) {
      throw new Error(`ligand chain ${chain.letter} has ${chain.residues.length} residues; a refold reads a one-residue`
        + ' ligand chain by its dictionary code');
    }
    const name = chain.residues[0].name;
    if (isDictionaryCode(name)) return name;
    block ??= parseCifBlock(cif);
    const code = `ZZ${custom}`;
    custom += 1;
    components[code] = componentText(code, name, chain.letter, block);
    standIns[code] = name;
    return code;
  });
  return {
    sequence: polymers.map((polymer) => polymer.sequence).join(':'),
    chainKinds: polymers.map((polymer) => polymer.kind),
    ligandCodes, modifications, components, standIns,
    needed: modifications.length > 0 || custom > 0,
  };
}

/**
 * The design's PDB with each modified residue named as its parent: what LocalFold's `keptTemplates` (which reads the
 * polymer chains by name) can take, with the residue still where it was. Only used for templates and nothing else.
 */
export function pdbWithParents(pdb) {
  return pdb.split('\n').map((line) => {
    if (!isRecord(line)) return line;
    const parent = MODIFIED_PARENT[line.slice(17, 20).trim()];
    return parent === undefined || !line.startsWith('ATOM  ') ? line : `${line.slice(0, 17)}${parent}${line.slice(20)}`;
  }).join('\n');
}

/** A refold's PDB with each stand-in ligand code written back as the design's own name for it (`standIns`: code -> name). */
export function restoreLigandNames(pdb, standIns) {
  return pdb.split('\n').map((line) => {
    if (!isRecord(line)) return line;
    const name = standIns[line.slice(17, 20).trim()];
    return name === undefined ? line : `${line.slice(0, 17)}${name.padStart(3).slice(-3)}${line.slice(20)}`;
  }).join('\n');
}
