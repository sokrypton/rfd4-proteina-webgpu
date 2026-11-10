/**
 * The refold's one input: a CIF made the way RFD4's AF3 oracle makes the input it folds (see
 * tools/oracle/af3_refold_input.py, which runs the oracle's own `_build_manifest` and writes the result as this file;
 * web/refold-cif.js writes the same thing in the browser for a design the page has just made).
 *
 * A multi-block mmCIF:
 *
 *   data_<name>                the manifest's `cif`: the design's chains and atoms, `_chem_comp` types, `_struct_conn`
 *   data_<code> ...            its `user_ccd`: a dictionary block for every non-standard residue and ligand, built
 *                              from the design's own atoms and bonds (a modified residue's block also names its
 *                              parent as the canonical letter of its `_entity_poly`)
 *   data_template_chain_<X>    a `template` of the manifest: the chain's atoms, with `_refold_template` giving its
 *                              `query_indices` and `template_indices`
 *
 * 🔴 NOTHING ELSE GOES INTO A REFOLD. LocalFold's `refoldDesign` was handed a PDB and read a fold's input off it, which
 * has no bonds to build a ligand from and refuses any residue outside the twenty ("chain A has KCX ..., cannot read
 * yet"; "Could not reach the PDB for ligand L:0"), so a design with a modified residue or a ligand of its own never
 * reached the fold with any model. This reads what the oracle writes, the way AF3 reads it: a modified residue stays
 * in its chain, on its parent's letter, as a modification; a ligand is the component its own block defines, whatever
 * its name; templates are the blocks that say so.
 *
 * Pure: a string in, plain objects out, so test/refold-input.test.js holds it without a browser.
 */
import { catRows, parseCifBlock } from '../shared/rfd4p/pipeline/cif.js';

export const AMINO = { ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V' };
export const DNA = { DA: 'A', DC: 'C', DG: 'G', DT: 'T' };
export const RNA = { A: 'A', C: 'C', G: 'G', U: 'U' };

/** A component code the PDB serves (and `ccdUrl` accepts): 1-5 capitals or digits. "L:0" is not one. */
export const isDictionaryCode = (code) => /^[A-Z0-9]{1,5}$/.test(code);

/** The text split at its `data_` lines, each block with its name. */
export function blocksOf(text) {
  return text.split(/^(?=data_)/m).filter((b) => b.trim() !== '').map((block) => ({
    name: block.match(/^data_(\S+)/)?.[1] ?? '', text: block }));
}

const BACKBONE = [new Set(['C', 'N']), new Set(['P', "O3'"])];
const isBackboneLink = (a, b) => BACKBONE.some((pair) => pair.has(a) && pair.has(b));

/**
 * A refold input, read as what a fold takes.
 *
 * @param {string} text the refold-input CIF
 * @returns {{sequence: string, chainKinds: string[], ligandCodes: (string|{codes: string[]})[],
 *   modifications: {chain: number, position: number, code: string}[], bonds: object[],
 *   templates: {text: string, chainId: string, chain: number, mapping?: number[][]}[],
 *   components: Object<string, string>, standIns: Object<string, string>, chains: string[]}}
 *   `components` is the dictionary text to serve for each code the fold asks the PDB for, `standIns` the original name
 *   of every ligand that went in under a code of its own (a refold names it by that code), `chains` the chain ids
 *   in the order the fold takes them (polymers, then ligands).
 */
export function foldInputFromRefoldCif(text) {
  const [first, ...rest] = blocksOf(text);
  if (first === undefined) throw new Error('the refold input is empty');
  const dictionary = new Map();
  const templateBlocks = [];
  for (const block of rest) {
    if (block.text.includes('_refold_template.chain_id')) templateBlocks.push(block);
    else dictionary.set(block.name, block.text);
  }
  const cif = parseCifBlock(first.text);
  const atoms = catRows(cif.atom_site);
  if (atoms.n === 0) throw new Error(`the first block of the refold input (${first.name}) has no atoms`);

  const types = new Map();
  const comps = catRows(cif.chem_comp);
  for (let r = 0; r < comps.n; r += 1) types.set(String(comps.get('id', r)), String(comps.get('type', r)).toUpperCase());

  // Chains in the order they first appear, each with its residues in order (one per auth number and insertion code)
  const chains = new Map();
  for (let r = 0; r < atoms.n; r += 1) {
    const asym = String(atoms.get('label_asym_id', r));
    if (!chains.has(asym)) chains.set(asym, { asym, residues: [], seen: new Map() });
    const chain = chains.get(asym);
    const key = `${atoms.get('auth_seq_id', r)}|${atoms.get('pdbx_PDB_ins_code', r)}|${atoms.get('label_comp_id', r)}`;
    if (!chain.seen.has(key)) {
      chain.seen.set(key, chain.residues.length);
      chain.residues.push({ comp: String(atoms.get('label_comp_id', r)), label: String(atoms.get('label_seq_id', r)) });
    }
  }
  const kindOf = (chain) => {
    const type = types.get(chain.residues[0].comp) ?? '';
    return type.includes('PEPTIDE LINKING') ? 'protein' : type.startsWith('DNA') ? 'dna' : type.startsWith('RNA') ? 'rna' : null;
  };
  const polymers = [];
  const ligands = [];
  for (const chain of chains.values()) {
    const kind = kindOf(chain);
    if (kind === null) ligands.push(chain); else polymers.push({ ...chain, kind });
  }
  if (polymers.length === 0) throw new Error('the refold input has no polymer chain to refold');

  const components = {};
  const standIns = {};
  /** The block that defines a component, served under `as` for the fold's dictionary request. */
  const serve = (comp, as, parent = null) => {
    let block = dictionary.get(comp);
    if (comp !== as) block = block.replaceAll(comp, as);
    if (parent !== null) block = block.replace(/(_chem_comp\.mon_nstd_parent_comp_id\s+)\S+/, `$1${parent}`);
    components[as] = block;
  };

  const modifications = [];
  const sequence = polymers.map((chain, index) => {
    const table = { protein: AMINO, dna: DNA, rna: RNA }[chain.kind];
    return chain.residues.map((residue, at) => {
      if (table[residue.comp] !== undefined) return table[residue.comp];
      // A modified residue: the letter of its parent, which its block names as the canonical letter of its entity or,
      // as the PDB's entries do, as the parent component
      const block = dictionary.get(residue.comp);
      const named = block?.match(/_entity_poly\.pdbx_seq_one_letter_code_can\s+(\S+)/)?.[1]
        ?? block?.match(/_chem_comp\.mon_nstd_parent_comp_id\s+(\S+)/)?.[1]?.split(',')[0];
      const word = named?.replace(/["']/g, '').toUpperCase();
      const parentCode = word !== undefined && table[word] !== undefined ? word      // (a residue's own code: LYS, DC, A)
        : Object.keys(table).find((code) => table[code] === word && (chain.kind === 'protein' || code.length <= 2));  // (its letter)
      const parent = parentCode === undefined ? undefined : table[parentCode];
      if (block === undefined || !parent) {
        throw new Error(`chain ${chain.asym} has ${residue.comp} at ${at + 1}, which is neither a standard residue nor a modified residue `
          + 'the refold input defines (a block naming its parent)');
      }
      modifications.push({ chain: index, position: at + 1, code: residue.comp });
      serve(residue.comp, residue.comp, parentCode);
      return parent;
    }).join('');
  }).join(':');

  let custom = 0;
  const ligandCodes = ligands.map((chain) => {
    const codes = chain.residues.map((residue) => {
      const comp = residue.comp;
      if (!dictionary.has(comp)) {
        if (isDictionaryCode(comp)) return comp;
        throw new Error(`ligand ${comp} (chain ${chain.asym}) has no block in the refold input and is not a dictionary code`);
      }
      if (isDictionaryCode(comp)) { serve(comp, comp); return comp; }
      const code = `ZZ${custom}`;
      custom += 1;
      standIns[code] = comp;
      serve(comp, code);
      return code;
    });
    return codes.length === 1 ? codes[0] : { codes };
  });

  // Declared bonds between residues; the peptide links of a chain are the polymer's own
  const order = [...polymers, ...ligands];
  const position = (asym, label) => {
    const chain = order.find((c) => c.asym === asym);
    const at = chain?.residues.findIndex((residue) => residue.label === String(label)) ?? -1;
    return { asym: order.indexOf(chain), residue: at + 1 };
  };
  const bonds = [];
  const conn = catRows(cif.struct_conn);
  for (let r = 0; r < conn.n; r += 1) {
    const type = String(conn.get('conn_type_id', r));
    if (type !== 'covale' && type !== 'disulf') continue;
    const [a, b] = [1, 2].map((k) => ({ asym: String(conn.get(`ptnr${k}_label_asym_id`, r)),
      label: String(conn.get(`ptnr${k}_label_seq_id`, r)), atom: String(conn.get(`ptnr${k}_label_atom_id`, r)) }));
    const from = position(a.asym, a.label);
    const to = position(b.asym, b.label);
    if (from.asym < 0 || to.asym < 0 || from.residue < 1 || to.residue < 1) continue;
    if (a.asym === b.asym && Math.abs(from.residue - to.residue) === 1 && isBackboneLink(a.atom, b.atom)) continue;
    bonds.push({ from: { ...from, atom: a.atom }, to: { ...to, atom: b.atom } });
  }

  const templates = templateBlocks.map((block) => {
    const value = (item) => block.text.match(new RegExp(`_refold_template\\.${item}\\s+(?:"([^"]*)"|(\\S+))`));
    const chainId = value('chain_id')[2] ?? value('chain_id')[1];
    const list = (item) => (value(item)?.[1] ?? value(item)?.[2] ?? '').split(/\s+/).filter(Boolean).map(Number);
    const query = list('query_indices');
    const through = list('template_indices');
    const chain = polymers.findIndex((c) => c.asym === chainId);
    if (chain < 0) throw new Error(`a template names chain ${chainId}, which is not a polymer chain of the refold input`);
    // (a template of the whole chain, residue for residue, is the chain templated; anything else is part of it)
    const whole = query.length === polymers[chain].residues.length && query.every((q, i) => q === i && through[i] === i);
    return { text: block.text, chainId, chain, ...(whole ? {} : { mapping: query.map((q, i) => [q, through[i]]) }) };
  });

  return { sequence, chainKinds: polymers.map((c) => c.kind), ligandCodes, modifications, bonds, templates, components,
    standIns, chains: order.map((c) => c.asym) };
}

/** A refold's PDB with each stand-in ligand code written back as the design's own name for it (`standIns`: code -> name). */
export function restoreLigandNames(pdb, standIns) {
  return pdb.split('\n').map((line) => {
    if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) return line;
    const name = standIns[line.slice(17, 20).trim()];
    return name === undefined ? line : `${line.slice(0, 17)}${name.padStart(3).slice(-3)}${line.slice(20)}`;
  }).join('\n');
}
