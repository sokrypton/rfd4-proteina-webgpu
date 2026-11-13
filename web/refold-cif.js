/**
 * A design the page has just made, written as the refold input RFD4's AF3 oracle would have made of it: the multi-block
 * CIF web/refold-input.js reads (and tools/oracle/af3_refold_input.py writes, by running the oracle itself).
 *
 * The oracle's `_build_manifest` does, to the AtomWorks array of a design: drop atoms with no coordinates; give
 * every polymer chain its residues as they are, a modified residue among them as itself; write the design as a CIF with
 * `_chem_comp` types and the peptide links a modified residue's neighbours make; define every non-standard residue and
 * ligand in a user CCD from the design's own atoms and bonds (`_build_user_ccd_string`); and template the chains and
 * atoms its `template_query` selects. This does the same from the CIF the page holds, which carries the atoms and
 * its `_chem_comp_bond` rows. test/refold-cif.test.js holds the result to the oracle's on a design with a modified
 * residue, a custom ligand and a metal: both read to the same fold input. What makes a residue a modification of which
 * standard one is read from the PDB's dictionary entry for it (`describeComponent`), as the oracle's AtomWorks reads the CCD.
 *
 * Pure: strings in and out.
 */
import { catRows, parseCifBlock } from '../shared/rfd4p/pipeline/cif.js';
import { AMINO, DNA, RNA } from './refold-input.js';

const TABLE = { protein: AMINO, dna: DNA, rna: RNA };

/**
 * What the PDB's dictionary says of a component, read from its entry: whether it is a link in a polymer (its type), and
 * which standard residue it is a modification of (`mon_nstd_parent_comp_id`). This is where the oracle's AtomWorks gets
 * both from too (the CCD), so any residue the dictionary describes is read, and none is named in this code.
 *
 * @returns {{kind: 'protein'|'dna'|'rna'|null, parent: string|null, type: string}|null} null with no entry
 */
export function describeComponent(entry) {
  if (!entry) return null;
  const type = (entry.match(/_chem_comp\.type\s+(?:"([^"]*)"|'([^']*)'|(\S+))/) ?? []).slice(1).find(Boolean) ?? '';
  const parent = (entry.match(/_chem_comp\.mon_nstd_parent_comp_id\s+(\S+)/)?.[1] ?? '').replace(/["']/g, '').split(',')[0].toUpperCase();
  const upper = type.toUpperCase();
  const kind = upper.includes('PEPTIDE LINKING') ? 'protein' : upper.startsWith('DNA') ? 'dna' : upper.startsWith('RNA') ? 'rna' : null;
  return { kind, parent: parent === '' || parent === '?' || parent === '.' ? null : parent, type };
}

/** Each polymer-linking component of the dictionary mapped to its parent's code: what templates are chosen by. */
export function parentsOf(dictionary) {
  const out = {};
  for (const [code, entry] of Object.entries(dictionary ?? {})) {
    const info = describeComponent(entry);
    if (info?.kind && info.parent) out[code] = info.parent;
  }
  return out;
}

const quoted = (value) => (/[\s'"]/.test(String(value)) || value === '' ? `"${value}"` : String(value));
const peptide = (comp) => (comp === 'GLY' ? 'PEPTIDE LINKING' : 'L-PEPTIDE LINKING');

/**
 * @param {string} designCif the design as mmCIF (what the page holds for it)
 * @param {{name?: string, templates?: {chainId: string, residues?: number[]|null}[], dictionary?: Object<string, string>}}
 *   [options] `templates`: the chains to template, and for a chain templated in part its residues (0-based among the
 *   chain's residues); `dictionary`: the PDB's entry for any non-standard component that has one (see
 *   `dictionaryCodes`), used as its block: the oracle's user CCD takes a known residue's definition from the
 *   dictionary too (its reference conformer is the dictionary's ideal one), and only a component the dictionary
 *   does not have (a design's own ligand) is defined from the design's atoms
 * @returns {string} the refold input
 */
export function refoldCifFromDesign(designCif, { name = 'design', templates = [], dictionary = {} } = {}) {
  const block = parseCifBlock(designCif);
  const table = catRows(block.atom_site);
  const bondTable = catRows(block.chem_comp_bond);

  // The atoms with coordinates, by chain and residue, in file order
  const chains = new Map();
  for (let r = 0; r < table.n; r += 1) {
    const xyz = ['Cartn_x', 'Cartn_y', 'Cartn_z'].map((c) => Number.parseFloat(table.get(c, r)));
    if (!xyz.every(Number.isFinite)) continue;
    const asym = String(table.get('label_asym_id', r));
    if (!chains.has(asym)) chains.set(asym, { asym, residues: [], seen: new Map() });
    const chain = chains.get(asym);
    const comp = String(table.get('label_comp_id', r));
    const key = `${table.get('auth_seq_id', r)}|${table.get('pdbx_PDB_ins_code', r)}|${comp}`;
    if (!chain.seen.has(key)) {
      chain.seen.set(key, chain.residues.length);
      chain.residues.push({ comp, auth: String(table.get('auth_seq_id', r)), ins: String(table.get('pdbx_PDB_ins_code', r) ?? '.'), atoms: [] });
    }
    chain.residues[chain.seen.get(key)].atoms.push({
      name: String(table.get('label_atom_id', r)), element: String(table.get('type_symbol', r)),
      charge: String(table.get('pdbx_formal_charge', r) ?? '?'), group: String(table.get('group_PDB', r)), xyz });
  }
  if (chains.size === 0) throw new Error('the design has no atoms with coordinates');

  // A polymer chain is one whose residues are standard or polymer links the dictionary describes (and gives a parent for)
  const info = (comp) => describeComponent(dictionary[comp]);
  const linkKind = (comp) => {
    if (AMINO[comp] !== undefined) return 'protein';
    if (DNA[comp] !== undefined) return 'dna';
    if (RNA[comp] !== undefined) return 'rna';
    const known = info(comp);
    return known?.kind && known.parent && TABLE[known.kind][known.parent] !== undefined ? known.kind : null;
  };
  const kindOf = (chain) => {
    const comps = chain.residues.map((residue) => residue.comp);
    const standard = comps.filter((c) => AMINO[c] !== undefined || DNA[c] !== undefined || RNA[c] !== undefined);
    if (standard.length === 0) return null;                       // a ligand (a glycan, a lone residue): not a chain to fold
    const kinds = new Set(comps.map(linkKind));
    if (kinds.size === 1 && !kinds.has(null)) return [...kinds][0];
    const odd = comps.find((c) => linkKind(c) === null);
    throw new Error(`chain ${chain.asym} has ${odd}, which a refold cannot read: it is not a standard residue and the `
      + `PDB's dictionary ${dictionary[odd] === undefined ? 'has no entry for it' : 'does not describe it as a modified residue of a standard one'}`
      + (kinds.size > 1 && !kinds.has(null) ? ' (the chain mixes amino acids and nucleotides)' : ''));
  };
  for (const chain of chains.values()) chain.kind = kindOf(chain);

  const comps = new Map();       // every component, its type and its first residue (the user CCD takes that one)
  for (const chain of chains.values()) {
    for (const residue of chain.residues) {
      const standard = chain.kind !== null && TABLE[chain.kind][residue.comp] !== undefined;
      if (comps.has(residue.comp)) continue;
      comps.set(residue.comp, {
        type: chain.kind === 'protein' ? (info(residue.comp)?.type || peptide(residue.comp)) : chain.kind === 'dna' ? 'DNA LINKING'
          : chain.kind === 'rna' ? 'RNA LINKING' : 'NON-POLYMER',
        standard, residue, kind: chain.kind });
    }
  }

  const out = [`data_${name}`, '#', `_entry.id ${name}`, '#'];
  out.push('loop_', ...['id', 'name', 'type', 'formula', 'mon_nstd_parent_comp_id', 'pdbx_synonyms', 'formula_weight']
    .map((c) => `_chem_comp.${c}`), ...[...comps.keys()].sort().map((id) => `${quoted(id)} ${quoted(id)} '${comps.get(id).type}' ? ? ? ?`), '#');

  // A modified residue's peptide links to its neighbours, as the oracle's writer emits them for an atomised residue
  const conn = [];
  for (const chain of chains.values()) {
    if (chain.kind !== 'protein') continue;
    chain.residues.forEach((residue, at) => {
      if (AMINO[residue.comp] !== undefined) return;
      if (at > 0) conn.push([chain.asym, chain.residues[at - 1].comp, at, 'C', residue.comp, at + 1, 'N']);
      if (at < chain.residues.length - 1) conn.push([chain.asym, residue.comp, at + 1, 'C', chain.residues[at + 1].comp, at + 2, 'N']);
    });
  }
  if (conn.length > 0) {
    out.push('loop_', ...['id', 'conn_type_id', 'pdbx_value_order', 'ptnr1_label_asym_id', 'ptnr2_label_asym_id', 'ptnr1_label_comp_id',
      'ptnr2_label_comp_id', 'ptnr1_label_seq_id', 'ptnr2_label_seq_id', 'ptnr1_label_atom_id', 'ptnr2_label_atom_id'].map((c) => `_struct_conn.${c}`),
    ...conn.map(([asym, c1, s1, a1, c2, s2, a2], k) => [k + 1, 'covale', 'sing', asym, asym, quoted(c1), quoted(c2), s1, s2, a1, a2].join(' ')), '#');
  }

  const atomHeader = ['group_PDB', 'type_symbol', 'label_atom_id', 'label_alt_id', 'label_comp_id', 'label_asym_id', 'label_entity_id',
    'label_seq_id', 'pdbx_PDB_ins_code', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id', 'id', 'B_iso_or_equiv', 'occupancy',
    'pdbx_formal_charge', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'pdbx_PDB_model_num'].map((c) => `_atom_site.${c}`);
  const atomRows = (chainList, entityOf, serial = { n: 0 }) => chainList.flatMap((chain) => chain.residues.flatMap((residue, at) => residue.atoms.map((a) => {
    serial.n += 1;
    return [a.group, a.element, quoted(a.name), '.', quoted(residue.comp), chain.asym, entityOf(chain), at + 1, residue.ins, residue.auth,
      quoted(residue.comp), chain.asym, quoted(a.name), serial.n, '0.0', '1.0', a.charge, ...a.xyz.map((v) => v.toFixed(3)), 1].join(' ');
  })));
  const entities = new Map([...chains.keys()].map((asym, k) => [asym, k]));
  out.push('loop_', ...atomHeader, ...atomRows([...chains.values()], (chain) => entities.get(chain.asym)), '#');

  // The user CCD: every non-standard residue and ligand, defined by the design's own atoms and its bond rows
  for (const [code, info] of comps) {
    if (info.standard) continue;
    if (dictionary[code] !== undefined) { out.push(dictionary[code].trimEnd(), '#'); continue; }
    out.push(`data_${code}`, '#', '_entry.id user_ccd_entry', '#');
    if (info.kind !== null) {
      const parent = describeComponent(dictionary[code])?.parent;
      out.push('_entity_poly.entity_id 0', `_entity_poly.type ${info.kind === 'protein' ? 'polypeptide(L)' : info.kind === 'dna' ? 'polydeoxyribonucleotide' : 'polyribonucleotide'}`,
        `_entity_poly.pdbx_seq_one_letter_code_can ${TABLE[info.kind][parent] ?? 'X'}`, '#');
    }
    out.push(`_chem_comp.id ${quoted(code)}`, `_chem_comp.name ${quoted(code)}`, `_chem_comp.type '${info.type}'`,
      '_chem_comp.mon_nstd_parent_comp_id ?', '#');
    const atoms = info.residue.atoms;
    out.push('loop_', ...['comp_id', 'atom_id', 'type_symbol', 'charge', 'pdbx_aromatic_flag', 'pdbx_leaving_atom_flag',
      'pdbx_model_Cartn_x_ideal', 'pdbx_model_Cartn_y_ideal', 'pdbx_model_Cartn_z_ideal'].map((c) => `_chem_comp_atom.${c}`),
    ...atoms.map((a) => [quoted(code), quoted(a.name), a.element, Number.parseInt(a.charge, 10) || 0, 'N', 'N',
      ...a.xyz.map((v) => v.toFixed(3))].join(' ')), '#');
    const have = new Set(atoms.map((a) => a.name));
    const bonds = [];
    for (let r = 0; r < bondTable.n; r += 1) {
      if (String(bondTable.get('comp_id', r)) !== code) continue;
      const [a, b] = [bondTable.get('atom_id_1', r), bondTable.get('atom_id_2', r)].map(String);
      if (have.has(a) && have.has(b)) bonds.push([quoted(code), quoted(a), quoted(b), bondTable.get('value_order', r) ?? 'SING', bondTable.get('pdbx_aromatic_flag', r) ?? 'N'].join(' '));
    }
    if (bonds.length > 0) {
      out.push('loop_', ...['comp_id', 'atom_id_1', 'atom_id_2', 'value_order', 'pdbx_aromatic_flag'].map((c) => `_chem_comp_bond.${c}`), ...bonds, '#');
    }
  }

  // Templates: the chain's atoms, and which of its residues are the template
  for (const template of templates) {
    const chain = chains.get(template.chainId);
    if (chain === undefined) throw new Error(`a template names chain ${template.chainId}, which the design does not have`);
    const residues = template.residues ?? chain.residues.map((_, at) => at);
    out.push(`data_template_chain_${chain.asym}`, '#', `_entry.id template_chain_${chain.asym}`, '#',
      'loop_', ...atomHeader, ...atomRows([chain], () => 0), '#',
      `_refold_template.chain_id ${chain.asym}`, `_refold_template.query_indices "${residues.join(' ')}"`,
      `_refold_template.template_indices "${residues.join(' ')}"`, '#');
  }
  return `${out.join('\n')}\n`;
}

/** The non-standard components of a design that the PDB's dictionary can have an entry for (codes of 1-5 characters). */
export function dictionaryCodes(designCif) {
  const table = catRows(parseCifBlock(designCif).atom_site);
  const standard = (c) => AMINO[c] !== undefined || DNA[c] !== undefined || RNA[c] !== undefined;
  const codes = new Set();
  for (let r = 0; r < table.n; r += 1) {
    const comp = String(table.get('label_comp_id', r));
    if (!standard(comp) && /^[A-Z0-9]{1,5}$/.test(comp)) codes.add(comp);
  }
  return [...codes];
}

/**
 * The design's PDB with each modified residue named as its parent (`parents`: code -> parent code, from `parentsOf`):
 * what LocalFold's `keptTemplates` (which reads the polymer chains by name) can take, with the residue still where it
 * was. Only used to say which chains to template.
 */
export function pdbWithParents(pdb, parents = {}) {
  return pdb.split('\n').map((line) => {
    if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) return line;
    const parent = parents[line.slice(17, 20).trim()];
    return parent === undefined || !line.startsWith('ATOM  ') ? line : `${line.slice(0, 17)}${parent.padStart(3).slice(-3)}${line.slice(20)}`;
  }).join('\n');
}
