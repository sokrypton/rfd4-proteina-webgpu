/**
 * Client-side in-browser implementation of the AtomWorks Studio `/api/v1/*` REST contract.
 *
 * Lets the embedded AtomWorks Studio frontend and the RFD4-Proteina generation host
 * (`web/app.js`), which share `index.html`, run 100% locally in the browser without a Python/FastAPI backend.
 */

import { assignSecondary } from './vendor/py2dmol-ss.js';
import {
  BOND_TYPE,
  CHAIN_TYPE,
  NONSCALAR_1BODY_ANNOTATIONS,
  SCALAR_1BODY_ANNOTATIONS,
  STD_RESIDUE_AROMATIC,
  buildBondsForAtomArray,
  computeIdAndEntityAnnotations,
  getResidueSegments,
  parseCifBlock,
  parseCifToAtomArray,
  pdbTextToCifText,
} from '../shared/rfd4p/pipeline/cif.js';
import { toPdbString } from '../shared/rfd4p/output.js';
import { decodeTextOrZstd } from '../shared/rfd4p/pipeline/zstd.js';
import { prepareRcsbStructure, removeResidues, removeUnresolvedResidues } from '../shared/rfd4p/pipeline/prepare.js';
import { componentsNeedingCcd, fetchCcdComponents, mergeCcdIntoBlock } from '../shared/rfd4p/pipeline/ccd.js';
import { evalPathSelection, evalSelection } from '../shared/rfd4p/pipeline/spec.js';

const RESIDUE_CODES = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C',
  GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P',
  SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
  DA: 'A', DC: 'C', DG: 'G', DT: 'T',
  A: 'A', C: 'C', G: 'G', U: 'U',
};

const BOND_NAME_TO_TYPE = {
  ANY: BOND_TYPE.ANY,
  SINGLE: BOND_TYPE.SINGLE,
  DOUBLE: BOND_TYPE.DOUBLE,
  TRIPLE: BOND_TYPE.TRIPLE,
  QUADRUPLE: BOND_TYPE.QUADRUPLE,
  AROMATIC_SINGLE: BOND_TYPE.AROMATIC_SINGLE,
  AROMATIC_DOUBLE: BOND_TYPE.AROMATIC_DOUBLE,
  AROMATIC_TRIPLE: BOND_TYPE.AROMATIC_TRIPLE,
  COORDINATION: BOND_TYPE.COORDINATION,
  AROMATIC: BOND_TYPE.AROMATIC,
};

const BOND_TYPE_TO_NAME = Object.fromEntries(
  Object.entries(BOND_NAME_TO_TYPE).map(([k, v]) => [v, k])
);

const CONDITION_SPECS = [
  {
    key: 'sequence',
    col: 'condition_sequence_1_residue',
    label: 'Sequence conditioning',
    kind: 'mask',
    level: 'residue',
    unit: 'residues',
    default: false,
    invert_mask: false,
    visible: true,
  },
  {
    key: 'coordinate',
    col: 'condition_coordinate_1_atom',
    label: 'Coordinate conditioning',
    kind: 'mask',
    level: 'atom',
    unit: 'atoms',
    default: false,
    invert_mask: false,
    visible: true,
  },
  {
    key: 'index',
    col: 'condition_index_1_residue',
    label: 'Index conditioning',
    kind: 'mask',
    level: 'residue',
    unit: 'residues',
    default: true,
    invert_mask: true,
    visible: true,
  },
  {
    key: 'hotspot',
    col: 'condition_hotspot_1_atom',
    label: 'Hotspot',
    kind: 'mask',
    level: 'atom',
    unit: 'atoms',
    default: false,
    invert_mask: false,
    visible: true,
  },
  {
    key: 'rasa',
    col: 'condition_rasa-binned_1_atom',
    label: 'RASA',
    kind: 'enum',
    level: 'atom',
    unit: 'atoms',
    default: 0,
    values: { UNSPECIFIED: 0, BURIED: 1, INTERMEDIATE: 2, EXPOSED: 3 },
    visible: true,
  },
  {
    key: 'hbond',
    col: 'condition_hbond_1_atom',
    label: 'Hbond',
    kind: 'enum',
    level: 'atom',
    unit: 'atoms',
    default: 0,
    values: { NONE: 0, DONOR: 1, ACCEPTOR: 2, BOTH: 3 },
    visible: true,
  },
  {
    key: 'pistack',
    col: 'condition_pistack_1_atom',
    label: 'Pistack',
    kind: 'mask',
    level: 'atom',
    unit: 'atoms',
    default: false,
    invert_mask: false,
    visible: true,
  },
  {
    key: 'pication',
    col: 'condition_pication_1_atom',
    label: 'Pication',
    kind: 'enum',
    level: 'atom',
    unit: 'atoms',
    default: 0,
    values: { NONE: 0, AROMATIC: 1, CATION: 2, BOTH: 3 },
    visible: true,
  },
  {
    key: 'saltbridge',
    col: 'condition_saltbridge_1_atom',
    label: 'Salt bridge',
    kind: 'enum',
    level: 'atom',
    unit: 'atoms',
    default: 0,
    values: { NONE: 0, POSITIVE: 1, NEGATIVE: 2 },
    visible: true,
  },
  {
    key: 'hydrophobic',
    col: 'condition_hydrophobic_1_atom',
    label: 'Hydrophobic',
    kind: 'mask',
    level: 'atom',
    unit: 'atoms',
    default: false,
    invert_mask: false,
    visible: true,
  },
  {
    key: 'metal_coord',
    col: 'condition_metal-coord_1_atom',
    label: 'Metal coordination',
    kind: 'enum',
    level: 'atom',
    unit: 'atoms',
    default: 0,
    values: { NONE: 0, METAL: 1, COORDINATING: 2 },
    visible: true,
  },
  {
    key: 'secondary_structure',
    col: 'condition_secondary-structure_1_residue',
    label: 'Secondary Structure',
    kind: 'enum',
    level: 'residue',
    unit: 'residues',
    default: 0,
    values: { NONE: 0, HELIX: 1, SHEET: 2, LOOP: 3 },
    visible: true,
  },
  {
    key: 'index_group',
    col: 'condition_index-group_1_token',
    label: 'Index Group',
    kind: 'integer',
    level: 'token',
    unit: 'groups',
    default: -1,
    group_editor: true,
    reserved_group_ids: [],
    visible: true,
  },
  {
    key: 'cath_label',
    col: 'condition_cath-label_1_residue',
    label: 'CATH fold',
    kind: 'text',
    level: 'residue',
    unit: 'residues',
    default: 'x.x.x.x',
    visible: true,
  },
  {
    key: 'chain_centroid',
    col: 'condition_ChainCentroid_1_atom',
    label: 'Chain centroid',
    kind: 'vector3',
    level: 'chain',
    unit: 'chains',
    default: null,
    visible: true,
  },
  {
    key: 'coordinate_group',
    col: 'condition_CoordinateGroup_1_atom',
    label: 'Coordinate Group',
    kind: 'integer',
    level: 'atom',
    unit: 'groups',
    default: -1,
    group_editor: true,
    reserved_group_ids: [-1, 0],
    visible: true,
  },
  {
    key: 'na_ss_state',
    col: 'condition_na-ss-state_1_residue',
    label: 'Nucleic Acid Secondary Structure',
    kind: 'enum',
    level: 'residue',
    unit: 'residues',
    default: 0,
    values: { UNSPECIFIED: 0, PAIR: 1, LOOP: 2 },
    visible: true,
  },
  {
    key: 'internal_coordinate',
    col: 'condition_InternalCoordinate_1_atom',
    label: 'Internal coordinates',
    kind: 'operation',
    level: 'atom',
    unit: 'atoms',
    default: false,
    visible: false,
  },
  {
    key: 'distance',
    col: 'condition_distance_2_atom',
    label: 'Distance restraints',
    kind: 'pair',
    level: 'atom',
    unit: 'pairs',
    default: null,
    visible: true,
  },
  ...[
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT',
  ].map((tok) => ({
    key: `sequence_probability_${tok.toLowerCase()}`,
    col: `condition_sequence-probability-${tok}_1_residue`,
    label: `Sequence Probability ${tok[0]}${tok.slice(1).toLowerCase()}`,
    kind: 'scalar',
    level: 'residue',
    unit: 'residues',
    default: NaN,
    visible: true,
  })),
];

const CONDITION_BY_KEY = new Map();
for (const spec of CONDITION_SPECS) {
  CONDITION_BY_KEY.set(spec.key, spec);
  CONDITION_BY_KEY.set(spec.col, spec);
}
CONDITION_BY_KEY.set('rasa_binned', CONDITION_BY_KEY.get('rasa'));
CONDITION_BY_KEY.set('rasa-binned', CONDITION_BY_KEY.get('rasa'));

function cloneAtoms(atoms) {
  const copy = {
    nAtoms: atoms.nAtoms,
    hetero: atoms.hetero.slice(),
    element: atoms.element.slice(),
    atom_name: atoms.atom_name.slice(),
    res_name: atoms.res_name.slice(),
    chain_id: atoms.chain_id.slice(),
    label_entity_id: atoms.label_entity_id.slice(),
    res_id: atoms.res_id.slice(),
    ins_code: atoms.ins_code.slice(),
    auth_seq_id: atoms.auth_seq_id.slice(),
    atom_id: atoms.atom_id.slice(),
    b_factor: atoms.b_factor.slice(),
    occupancy: atoms.occupancy.slice(),
    charge: atoms.charge.slice(),
    label_alt_id: atoms.label_alt_id.slice(),
    coord: atoms.coord.map((c) => c.slice()),
    transformation_id: (atoms.transformation_id || []).slice(),
    is_polymer: atoms.is_polymer.slice(),
    chain_type: atoms.chain_type.slice(),
    chem_comp_type: atoms.chem_comp_type.slice(),
    atomic_number: atoms.atomic_number.slice(),
    is_aromatic: atoms.is_aromatic.slice(),
    is_leaving_atom: atoms.is_leaving_atom.slice(),
    pn_unit_id: (atoms.pn_unit_id || []).slice(),
    molecule_id: (atoms.molecule_id || []).slice(),
    chain_entity: (atoms.chain_entity || []).slice(),
    pn_unit_entity: (atoms.pn_unit_entity || []).slice(),
    molecule_entity: (atoms.molecule_entity || []).slice(),
    chain_iid: (atoms.chain_iid || []).slice(),
    pn_unit_iid: (atoms.pn_unit_iid || []).slice(),
    molecule_iid: (atoms.molecule_iid || []).slice(),
    bonds: (atoms.bonds || []).map((b) => b.slice()),
    annotations: {},
    distanceRestraints: (atoms.distanceRestraints || []).map((r) => ({ ...r })),
    cathText: atoms.cathText ? atoms.cathText.slice() : null,
  };
  for (const [k, v] of Object.entries(atoms.annotations || {})) {
    copy.annotations[k] = Array.isArray(v)
      ? v.map((item) => (Array.isArray(item) ? item.slice() : item))
      : v;
  }
  return copy;
}

function fmtCifVal(v) {
  if (v === null || v === undefined || v === '') return '.';
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return 'nan';
    if (!Number.isFinite(v)) return '.';
    return Number.isInteger(v) ? String(v) : v.toFixed(6).replace(/\.?0+$/, (m) => (m === '.000000' ? '.0' : m));
  }
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  const s = String(v);
  if (s === '') return '.';
  // CIF quoting: delimit with the quote character the value does not contain (primed nucleic-acid atom names such as
  // O5' must round-trip), and quote values a parser would read as syntax.
  if (/[\s'"]/.test(s) || /^[_#$;[\]]/.test(s) || /^(data|loop|save|global|stop)_/i.test(s)) {
    if (!s.includes('"')) return `"${s}"`;
    if (!s.includes("'")) return `'${s}'`;
    return `"${s.replace(/"/g, "'")}"`;
  }
  return s;
}

/**
 * `new StudioDoc`, plus the chemistry a plain RCSB file leaves out: multi-atom components with no bond tables
 * (nonstandard residues, ligands) get their CCD entries fetched from RCSB and the document is rebuilt with them,
 * as AtomWorks does from its bundled CCD. Offline, the document is returned as before.
 */
export async function createStudioDocWithCcd(text, name, opts = {}, fetchFn = globalThis.fetch?.bind(globalThis)) {
  const doc = new StudioDoc(text, name, opts);
  const a = doc.atoms;
  const multiAtom = new Set();
  const starts = getResidueSegments(a);
  for (let r = 0; r < starts.length - 1; r += 1) if (starts[r + 1] - starts[r] > 1) multiAtom.add(a.res_name[starts[r]]);
  const needed = componentsNeedingCcd(multiAtom, doc.cifBlock);
  if (!needed.length || !fetchFn) return doc;
  const { blocks } = await fetchCcdComponents(needed, { fetchFn });
  return blocks.size ? new StudioDoc(text, name, { ...opts, ccdBlocks: blocks }) : doc;
}

/** Each residue's place in its chain's source ("38:"), this page's bookkeeping (not written to files). */
const SRC_KEY = '__src';

/** Buffer ions a fetched structure drops, on top of AtomWorks' rcsb `remove_ccds`. */
const BUFFER_IONS = ['NA', 'CL'];

export class StudioDoc {
  constructor(cifText, name = 'structure.cif', opts = {}) {
    this.name = name;
    const decoded = decodeTextOrZstd(cifText);
    const isPdb = !decoded.includes('_atom_site.') && /(?:^|\n)\s*(?:ATOM  |HETATM)/.test(decoded);
    const normalizedCif = isPdb ? pdbTextToCifText(decoded) : decoded;
    this.rawCif = isPdb ? null : normalizedCif;
    this.cifBlock = parseCifBlock(normalizedCif);
    if (opts.ccdBlocks?.size) mergeCcdIntoBlock(this.cifBlock, opts.ccdBlocks);
    this.atoms = parseCifToAtomArray(normalizedCif, { fixArginines: opts.preset === 'rcsb', ccdBlocks: opts.ccdBlocks });
    this.importNotes = [];
    if (opts.preset === 'rcsb') {
      // "Unannotated input": the AtomWorks rcsb preset's assembly / water / additive / MSE preparation.
      const prepared = prepareRcsbStructure(this.atoms, this.cifBlock, { assemblyId: opts.assembly ?? '1' });
      // Beyond AtomWorks' additive list: sodium and chloride from the crystallisation buffer, never part of a design.
      this.atoms = removeResidues(prepared.atoms, BUFFER_IONS);
      // Residues the entry did not resolve go too (`keepUnresolved` keeps them: the example recipes, which reproduce
      // upstream's examples as they were set up).
      if (!opts.keepUnresolved) this.atoms = removeUnresolvedResidues(this.atoms);
      this.importNotes = prepared.notes;
      computeIdAndEntityAnnotations(this.atoms);
      // Prepared (water, additives and other chains gone, the assembly built, all fixed), it is no longer the file:
      // the file's own text would show the viewer, and give a run, the waters and conditions it no longer has.
      this.rawCif = null;
    }
    this.atoms.distanceRestraints = this._parseInitialRestraints(this.cifBlock);
    this.atoms.cathText = this._parseInitialCathText(this.cifBlock, this.atoms.nAtoms);
    if (opts.preset === 'rcsb' && opts.loadStandardAnnotations === false) {
      this.atoms.annotations = {};
      this.atoms.distanceRestraints = [];
      this.atoms.cathText = null;
      this.revision = 1;
    } else {
      this.revision = 0;
    }
    this.undoStack = [];
    this.trajectoryCif = opts.trajectoryCif ?? null;
    // A plain structure opens with everything fixed (coordinates and sequence): a user starts from what they loaded
    // and marks what to design, rather than starting from "redesign all of it".
    if (opts.preset === 'rcsb' && opts.loadStandardAnnotations !== false && this.atoms.nAtoms) {
      const all = Array.from({ length: this.atoms.nAtoms }, (_, k) => k);
      this.setCondition('coordinate', all, { snapshot: false });
      this.setCondition('sequence', all, { snapshot: false });
    }
    // Each chain as it was loaded (this page only), so residues cropped from it can be brought back.
    this.chainSources = new Map();
    // Every document's chains as opened, not only a fetched entry's: each chain can then be edited as a plan.
    if (this.atoms.nAtoms) {
      const loaded = cloneAtoms(this.atoms);
      for (const cid of new Set(this.atoms.chain_id.map(String))) this.chainSources.set(cid, { atoms: loaded, chain: cid });
      // Each residue tagged with its place in that source ("38:", number and insertion code), carried by every edit:
      // a chain's plan reads residues by it, never by their numbers now (which a renumbering edit changes).
      const tag = this._ensureAnnotationArray(SRC_KEY, { default: '' });
      for (let i = 0; i < this.atoms.nAtoms; i += 1) if (this._isPolymerAtom(i)) tag[i] = `${this.atoms.res_id[i]}:${this.atoms.ins_code[i] ?? ''}`;
    }
  }

  _parseInitialRestraints(block) {
    const cat = block['condition_distance_2_atom'];
    if (!cat || !cat.idx0) return [];
    const out = [];
    for (let i = 0; i < cat.idx0.length; i += 1) {
      const a1 = Number.parseInt(cat.idx0[i], 10);
      const a2 = Number.parseInt(cat.idx1?.[i], 10);
      const d = Number.parseFloat(cat.val0?.[i]);
      if (Number.isFinite(a1) && Number.isFinite(a2) && Number.isFinite(d) && a1 !== a2) {
        out.push({ atom1: Math.min(a1, a2), atom2: Math.max(a1, a2), distance: d });
      }
    }
    return out;
  }

  _parseInitialCathText(block, nAtoms) {
    const atomSite = block.atom_site;
    if (atomSite?.['condition_cath-label_1_residue']) {
      return atomSite['condition_cath-label_1_residue'].slice(0, nAtoms);
    }
    return null;
  }

  snapshot() {
    // Inside `compound` the outer call already took the one snapshot for the whole operation.
    if (this._compound) return;
    this.undoStack.push({
      atoms: cloneAtoms(this.atoms),
      rawCif: this.rawCif,
      revision: this.revision,
    });
    if (this.undoStack.length > 10) this.undoStack.shift();
    this.revision += 1;
    this.rawCif = null;
  }

  undo() {
    if (this.undoStack.length === 0) return false;
    const prev = this.undoStack.pop();
    this.atoms = prev.atoms;
    this.rawCif = prev.rawCif;
    this.revision += 1;
    return true;
  }

  chainKeyArray() {
    return this.atoms.chain_id;
  }

  nextChainId() {
    const used = new Set(this.atoms.chain_id.map(String));
    const alpha = [...used].filter((c) => /^[A-Z]+$/.test(c));
    alpha.sort((a, b) => a.length - b.length || a.localeCompare(b));
    const highest = alpha.at(-1) || '';
    let num = 0;
    for (let i = 0; i < highest.length; i += 1) {
      num = num * 26 + (highest.charCodeAt(i) - 64);
    }
    num += 1;
    let res = '';
    while (num > 0) {
      const rem = (num - 1) % 26;
      res = String.fromCharCode(65 + rem) + res;
      num = Math.floor((num - 1) / 26);
    }
    return res || 'A';
  }

  resolveIndices(indices, scope = 'atom') {
    const n = this.atoms.nAtoms;
    const uniq = [...new Set((indices || []).map((i) => Number(i)).filter((i) => Number.isInteger(i) && i >= 0 && i < n))].sort((a, b) => a - b);
    if (uniq.length === 0) return [];
    if (scope === 'atom') return uniq;
    if (scope === 'chain') {
      const chains = new Set(uniq.map((i) => this.atoms.chain_id[i]));
      const out = [];
      for (let i = 0; i < n; i += 1) {
        if (chains.has(this.atoms.chain_id[i])) out.push(i);
      }
      return out;
    }
    if (scope === 'residue') {
      const keys = new Set(uniq.map((i) => `${this.atoms.chain_id[i]}:${this.atoms.res_id[i]}:${this.atoms.ins_code[i]}`));
      const out = [];
      for (let i = 0; i < n; i += 1) {
        if (keys.has(`${this.atoms.chain_id[i]}:${this.atoms.res_id[i]}:${this.atoms.ins_code[i]}`)) {
          out.push(i);
        }
      }
      return out;
    }
    return uniq;
  }

  queryIndices(expression, syntax = 'expression', scope = 'atom') {
    const expr = (expression || '').trim();
    if (!expr) throw new Error('selection query must not be empty');
    const n = this.atoms.nAtoms;
    if (syntax === 'path' || /^[A-Za-z0-9*,._\[\]-]+\/[A-Za-z0-9*,._\[\]-]*\/[A-Za-z0-9*,._\[\]-]*\/[A-Za-z0-9*,._'\[\]-]*$/.test(expr)) {
      const mask = evalPathSelection(this.atoms, expr);
      const selected = [];
      for (let i = 0; i < n; i += 1) {
        if (mask[i]) selected.push(i);
      }
      return selected.length ? this.resolveIndices(selected, scope) : [];
    }

    try {
      const mask = evalSelection(this.atoms, expr);
      const selected = [];
      for (let i = 0; i < n; i += 1) {
        if (mask[i]) selected.push(i);
      }
      return selected.length ? this.resolveIndices(selected, scope) : [];
    } catch {
      // Fallback for custom studio functions like `has_bonds()`
    }

    const bondedSet = new Set();
    for (const [u, v] of this.atoms.bonds || []) {
      bondedSet.add(u);
      bondedSet.add(v);
    }
    const selected = [];
    for (let i = 0; i < n; i += 1) {
      if (this._evalAtomExpr(expr, i, bondedSet)) selected.push(i);
    }
    return selected.length ? this.resolveIndices(selected, scope) : [];
  }

  _evalAtomExpr(expr, i, bondedSet) {
    try {
      const a = this.atoms;
      const ctx = {
        atom_id: a.atom_id[i],
        atom_index: i,
        atom_name: a.atom_name[i],
        res_name: a.res_name[i],
        res_id: a.res_id[i],
        chain_id: a.chain_id[i],
        chain_iid: a.chain_iid?.[i] ?? `${a.chain_id[i]}_1`,
        element: a.element[i],
        hetero: a.hetero[i],
        is_polymer: a.is_polymer[i],
        charge: a.charge[i],
        b_factor: a.b_factor[i],
        has_bonds: () => bondedSet.has(i),
      };
      const jsExpr = expr
        .replace(/\band\b/g, '&&')
        .replace(/\bor\b/g, '||')
        .replace(/\bnot\b/g, '!')
        .replace(/\bTrue\b/g, 'true')
        .replace(/\bFalse\b/g, 'false');
      const fn = new Function(...Object.keys(ctx), `return Boolean(${jsExpr});`);
      return fn(...Object.values(ctx));
    } catch {
      return false;
    }
  }

  _getConditionMask(key) {
    const n = this.atoms.nAtoms;
    const ann = this.atoms.annotations || {};
    if (key === 'sequence') {
      const arr = ann['condition_sequence_1_residue'];
      if (!arr) return new Uint8Array(n);
      return Uint8Array.from(arr, (v) => (v && v !== '<M>' && v !== '.' && v !== '?' ? 1 : 0));
    }
    if (key === 'coordinate') {
      const arr = ann['condition_coordinate_1_atom'];
      const grp = ann['condition_CoordinateGroup_1_atom'];
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i += 1) {
        if ((arr && Array.isArray(arr[i]) && Number.isFinite(arr[i][0])) || (grp && grp[i] === 0)) {
          out[i] = 1;
        }
      }
      return out;
    }
    if (key === 'internal_coordinate') {
      const grp = ann['condition_CoordinateGroup_1_atom'];
      const intCrd = ann['condition_InternalCoordinate_1_atom'];
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i += 1) {
        if ((grp && grp[i] > 0) || (intCrd && Array.isArray(intCrd[i]) && Number.isFinite(intCrd[i][0]))) {
          out[i] = 1;
        }
      }
      return out;
    }
    if (key === 'index') {
      const arr = ann['condition_index_1_residue'];
      if (!arr) return new Uint8Array(n).fill(1);
      return Uint8Array.from(arr, (v) => (v === false || v === 0 ? 0 : 1));
    }
    const spec = CONDITION_BY_KEY.get(key);
    if (!spec) return new Uint8Array(n);
    const arr = ann[spec.col];
    if (!arr) return new Uint8Array(n);
    if (spec.kind === 'mask') {
      return Uint8Array.from(arr, (v) => (v ? 1 : 0));
    }
    if (spec.kind === 'enum' || spec.kind === 'integer' || spec.kind === 'scalar') {
      return Uint8Array.from(arr, (v) => (v !== undefined && v !== null && v !== spec.default && !Number.isNaN(v) ? 1 : 0));
    }
    if (spec.kind === 'vector3') {
      return Uint8Array.from(arr, (v) => (Array.isArray(v) && Number.isFinite(v[0]) ? 1 : 0));
    }
    return new Uint8Array(n);
  }

  annotationDetails(conditionName = null) {
    const n = this.atoms.nAtoms;
    const resStarts = getResidueSegments(this.atoms);
    const numRes = resStarts.length - 1;
    const entries = [];

    const countResiduesInMask = (mask) => {
      let c = 0;
      for (let r = 0; r < numRes; r += 1) {
        if (mask[resStarts[r]]) c += 1;
      }
      return c;
    };

    for (const spec of CONDITION_SPECS) {
      if (!spec.visible && !conditionName) continue;
      if (spec.kind === 'mask' || spec.kind === 'operation') {
        const rawMask = this._getConditionMask(spec.key);
        const indices = [];
        for (let i = 0; i < n; i += 1) {
          const active = spec.invert_mask ? !rawMask[i] : Boolean(rawMask[i]);
          if (active) indices.push(i);
        }
        if (indices.length === 0) continue;
        const activeMask = new Uint8Array(n);
        for (const idx of indices) activeMask[idx] = 1;
        const count = spec.unit === 'residues' ? countResiduesInMask(activeMask) : indices.length;
        entries.push({
          name: spec.key,
          label: spec.invert_mask ? 'Unindexed residues' : spec.label,
          count,
          unit: spec.unit,
          values: [
            {
              label: spec.invert_mask ? 'Unindexed' : 'Active',
              count: indices.length,
              atom_indices: indices,
            },
          ],
        });
        continue;
      }

      if (spec.kind === 'pair') {
        const restraints = this.atoms.distanceRestraints || [];
        if (restraints.length === 0) continue;
        entries.push({
          name: spec.key,
          label: spec.label,
          count: restraints.length,
          unit: 'pairs',
          values: restraints.map((r) => ({
            label: `Atoms ${r.atom1}-${r.atom2}`,
            value: r.distance,
            count: 1,
            atom_indices: [r.atom1, r.atom2],
          })),
        });
        continue;
      }

      const ann = this.atoms.annotations?.[spec.col];
      if (!ann && !(spec.key === 'cath_label' && this.atoms.cathText)) continue;
      const values = [];

      if (spec.kind === 'enum') {
        for (const [memberName, memberVal] of Object.entries(spec.values || {})) {
          if (memberVal === spec.default) continue;
          const indices = [];
          for (let i = 0; i < n; i += 1) {
            if (ann[i] === memberVal) indices.push(i);
          }
          if (indices.length > 0) {
            values.push({
              label: memberName,
              value: memberVal,
              count: indices.length,
              atom_indices: indices,
            });
          }
        }
      } else if (spec.kind === 'integer') {
        const groupMap = new Map();
        for (let i = 0; i < n; i += 1) {
          const g = ann[i];
          if (g === undefined || g === null || g === spec.default) continue;
          if (spec.key === 'index_group' && g === 0) continue;
          if (!groupMap.has(g)) groupMap.set(g, []);
          groupMap.get(g).push(i);
        }
        for (const [g, indices] of [...groupMap.entries()].sort((a, b) => a[0] - b[0])) {
          values.push({
            label: spec.key === 'coordinate_group' && g === 0 ? 'Absolute coordinates (group 0)' : `Group ${g}`,
            value: g,
            count: indices.length,
            atom_indices: indices,
          });
        }
      } else if (spec.kind === 'text' && spec.key === 'cath_label') {
        const byLabel = new Map();
        for (let i = 0; i < n; i += 1) {
          let lbl = this.atoms.cathText?.[i];
          if (!lbl && typeof ann?.[i] === 'string') {
            lbl = ann[i];
          } else if (!lbl && Array.isArray(ann?.[i]) && ann[i][0] >= 0) {
            lbl = `${ann[i][0]}.${ann[i][1]}.${ann[i][2]}.x`;
          }
          if (!lbl || lbl === 'x.x.x.x' || lbl === 'x.x.x' || lbl === '.') continue;
          if (!byLabel.has(lbl)) byLabel.set(lbl, []);
          byLabel.get(lbl).push(i);
        }
        for (const [lbl, indices] of byLabel.entries()) {
          values.push({ label: lbl, value: lbl, count: indices.length, atom_indices: indices });
        }
      } else if (spec.kind === 'scalar') {
        const byVal = new Map();
        for (let i = 0; i < n; i += 1) {
          const v = Number(ann[i]);
          if (!Number.isFinite(v)) continue;
          const k = v.toFixed(4);
          if (!byVal.has(k)) byVal.set(k, { value: v, indices: [] });
          byVal.get(k).indices.push(i);
        }
        for (const { value: v, indices } of byVal.values()) {
          values.push({ label: String(v), value: v, count: indices.length, atom_indices: indices });
        }
      } else if (spec.kind === 'vector3') {
        const byChain = new Map();
        for (let i = 0; i < n; i += 1) {
          const v = ann[i];
          if (!Array.isArray(v) || !Number.isFinite(v[0])) continue;
          const cid = this.atoms.chain_id[i];
          if (!byChain.has(cid)) byChain.set(cid, { xyz: v.slice(0, 3), indices: [] });
          byChain.get(cid).indices.push(i);
        }
        for (const [cid, item] of byChain.entries()) {
          values.push({
            label: `Chain ${cid}`,
            value: item.xyz,
            count: item.indices.length,
            atom_indices: item.indices,
          });
        }
      }

      if (values.length > 0) {
        const activeMask = new Uint8Array(n);
        for (const v of values) {
          for (const idx of v.atom_indices) activeMask[idx] = 1;
        }
        const count =
          spec.unit === 'residues'
            ? countResiduesInMask(activeMask)
            : spec.unit === 'groups' || spec.unit === 'chains'
            ? values.length
            : values.reduce((acc, v) => acc + v.count, 0);
        entries.push({
          name: spec.key,
          label: spec.label,
          count,
          unit: spec.unit,
          values,
        });
      }
    }

    if (conditionName !== null) {
      const spec = CONDITION_BY_KEY.get(conditionName);
      if (!spec) throw new Error(`unsupported condition: ${conditionName}`);
      return entries.filter((e) => e.name === spec.key);
    }
    return entries;
  }

  storedAnnotationFields(fieldName = null) {
    const n = this.atoms.nAtoms;
    const fields = [];
    const cols = Object.keys(this.atoms.annotations || {}).sort();
    for (const col of cols) {
      const arr = this.atoms.annotations[col];
      if (!Array.isArray(arr) || arr.length === 0) continue;
      if (Array.isArray(arr[0])) {
        const indices = [];
        for (let i = 0; i < n; i += 1) {
          if (arr[i].some((x) => Number.isFinite(x) && x !== -1)) indices.push(i);
        }
        fields.push({
          name: col,
          label: col,
          count: n,
          unit: 'rows',
          values: [{ label: `Stored (${arr[0].length}) values`, count: indices.length, atom_indices: indices }],
          truncated: false,
        });
      } else {
        const byVal = new Map();
        for (let i = 0; i < n; i += 1) {
          const v = arr[i];
          const k = Number.isNaN(v) ? 'NaN' : String(v);
          if (!byVal.has(k)) byVal.set(k, { raw: Number.isNaN(v) ? null : v, indices: [] });
          byVal.get(k).indices.push(i);
        }
        const entries = [...byVal.entries()];
        const values = entries.slice(0, 64).map(([k, item]) => ({
          label: k,
          value: item.raw,
          count: item.indices.length,
          atom_indices: item.indices,
        }));
        fields.push({
          name: col,
          label: col,
          count: n,
          unit: 'rows',
          values,
          truncated: entries.length > 64,
        });
      }
    }
    if (fieldName !== null) return fields.filter((f) => f.name === fieldName);
    return fields;
  }

  scalarAnnotationFields(fieldName = null) {
    const n = this.atoms.nAtoms;
    const b = this.atoms.b_factor;
    if (!b || b.length === 0) return [];
    const finite = b.filter((v) => Number.isFinite(v));
    if (finite.length === 0 || Math.max(...finite.map(Math.abs)) === 0) return [];
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    const palette = [
      { name: 'red', value: 0xef5350 },
      { name: 'orange', value: 0xffa726 },
      { name: 'yellow', value: 0xffd54f },
      { name: 'blue', value: 0x42a5f5 },
    ];
    const groups = [];
    for (let k = 0; k < 4; k += 1) {
      const low = min + ((max - min) * k) / 4;
      const high = min + ((max - min) * (k + 1)) / 4;
      const indices = [];
      for (let i = 0; i < n; i += 1) {
        const v = b[i];
        if (!Number.isFinite(v)) continue;
        if (v >= low && (k === 3 ? v <= high : v < high)) indices.push(i);
      }
      if (indices.length > 0) {
        groups.push({
          label: `${low.toFixed(2)} to ${high.toFixed(2)}`,
          count: indices.length,
          atom_indices: indices,
          color: palette[k],
        });
      }
    }
    const fields = groups.length
      ? [{ name: 'b_factor', label: 'B-factor (CIF)', count: finite.length, unit: 'atoms', values: groups }]
      : [];
    if (fieldName !== null) return fields.filter((f) => f.name === fieldName);
    return fields;
  }

  atomAnnotationDetails(atomIndex) {
    const idx = Number(atomIndex);
    if (!Number.isInteger(idx) || idx < 0 || idx >= this.atoms.nAtoms) {
      throw new Error(`atom index outside [0, ${this.atoms.nAtoms - 1}]`);
    }
    const annotations = [];
    for (const entry of this.annotationDetails()) {
      const matching = entry.values
        .filter((v) => v.atom_indices?.includes(idx))
        .map(({ atom_indices, ...rest }) => rest);
      if (matching.length > 0) {
        annotations.push({ name: entry.name, label: entry.label, values: matching });
      }
    }
    const stored_fields = [];
    for (const field of this.storedAnnotationFields()) {
      for (const v of field.values) {
        if (v.atom_indices?.includes(idx)) {
          stored_fields.push({ name: field.name, value: v.value, label: v.label });
          break;
        }
      }
    }
    const scalar_fields = [];
    if (Number.isFinite(this.atoms.b_factor[idx])) {
      scalar_fields.push({ name: 'b_factor', label: 'B-factor (CIF)', value: this.atoms.b_factor[idx] });
    }
    return {
      atom_index: idx,
      chain: String(this.atoms.chain_id[idx]),
      chain_id: String(this.atoms.chain_id[idx]),
      residue_id: Number(this.atoms.res_id[idx]),
      residue_name: String(this.atoms.res_name[idx]),
      atom_name: String(this.atoms.atom_name[idx]),
      formal_charge: Number(this.atoms.charge[idx] || 0),
      annotations,
      stored_fields,
      scalar_fields,
    };
  }

  conditionGroups(name) {
    const spec = CONDITION_BY_KEY.get(name);
    if (!spec || !spec.group_editor) throw new Error(`${name} is not a group condition`);
    const ann = this.atoms.annotations?.[spec.col];
    if (!ann) return [];
    const counts = new Map();
    for (let i = 0; i < this.atoms.nAtoms; i += 1) {
      const g = ann[i];
      if (g === undefined || g === null || spec.reserved_group_ids?.includes(g)) continue;
      if (spec.key === 'index_group' && g < 0) continue;
      counts.set(g, (counts.get(g) || 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([id, count]) => ({ id, count, can_delete: id !== spec.default && !(spec.key === 'index_group' && id === 0) }));
  }

  chainCentroids(name = 'chain_centroid') {
    const ann = this.atoms.annotations?.['condition_ChainCentroid_1_atom'];
    const chains = [...new Set(this.atoms.chain_id.map(String))];
    return chains.map((cid) => {
      let sx = 0, sy = 0, sz = 0, cnt = 0;
      let target = null;
      for (let i = 0; i < this.atoms.nAtoms; i += 1) {
        if (String(this.atoms.chain_id[i]) !== cid) continue;
        const c = this.atoms.coord[i];
        if (Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2])) {
          sx += c[0]; sy += c[1]; sz += c[2]; cnt += 1;
        }
        if (!target && ann && Array.isArray(ann[i]) && Number.isFinite(ann[i][0])) {
          target = ann[i].slice(0, 3);
        }
      }
      return {
        chain: cid,
        current: cnt > 0 ? [sx / cnt, sy / cnt, sz / cnt] : null,
        target,
      };
    });
  }

  atomLabel(index) {
    const a = this.atoms;
    const ins = a.ins_code[index] || '';
    return `${a.chain_id[index]}/${a.res_name[index]} ${a.res_id[index]}${ins}/${a.atom_name[index]}`;
  }

  atomDistance(atom1, atom2) {
    const c1 = this.atoms.coord[atom1];
    const c2 = this.atoms.coord[atom2];
    if (!c1 || !c2) return null;
    const d = Math.hypot(c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]);
    return Number.isFinite(d) ? d : null;
  }

  distanceRestraints() {
    return (this.atoms.distanceRestraints || []).map((r) => ({
      atom1: r.atom1,
      atom2: r.atom2,
      atom1_label: this.atomLabel(r.atom1),
      atom2_label: this.atomLabel(r.atom2),
      distance: r.distance,
      current_distance: this.atomDistance(r.atom1, r.atom2),
    }));
  }

  conditionEditorState() {
    return {
      groups: {
        index_group: this.conditionGroups('index_group'),
        coordinate_group: this.conditionGroups('coordinate_group'),
      },
      centroids: {
        chain_centroid: this.chainCentroids('chain_centroid'),
      },
      restraints: this.distanceRestraints(),
    };
  }

  sequenceView() {
    const a = this.atoms;
    const n = a.nAtoms;
    if (n === 0) return [];
    const resStarts = getResidueSegments(a);
    const numRes = resStarts.length - 1;

    const seqMask = this._getConditionMask('sequence');
    const crdMask = this._getConditionMask('coordinate');
    const idxMask = this._getConditionMask('index');
    const hotMask = this._getConditionMask('hotspot');

    const segMin = a.annotations?.['annotation_expsegmin_1_residue'];
    const segMax = a.annotations?.['annotation_expsegmax_1_residue'];

    const chains = [];
    let r = 0;
    while (r < numRes) {
      const s0 = resStarts[r];
      const chainId = String(a.chain_id[s0]);
      const items = [];
      while (r < numRes) {
        const start = resStarts[r];
        const stop = resStarts[r + 1];
        if (String(a.chain_id[start]) !== chainId) break;

        const minVal = segMin ? segMin[start] : -1;
        const maxVal = segMax ? segMax[start] : -1;
        const isSeg = minVal > 0 && maxVal > 0;
        if (isSeg) {
          const realized = a.res_name[start] === '<M>';
          const runStart = r;
          while (r + 1 < numRes) {
            const ns = resStarts[r + 1];
            if (
              String(a.chain_id[ns]) !== chainId ||
              (segMin ? segMin[ns] : -1) !== minVal ||
              (segMax ? segMax[ns] : -1) !== maxVal ||
              (a.res_name[ns] === '<M>') !== realized
            ) {
              break;
            }
            r += 1;
          }
          const runStop = resStarts[r + 1];
          const atom_indices = [];
          for (let i = resStarts[runStart]; i < runStop; i += 1) atom_indices.push(i);
          items.push({
            kind: 'segment',
            atom_indices,
            minimum: minVal,
            maximum: maxVal,
            realized,
            realized_length: realized ? r - runStart + 1 : null,
          });
        } else {
          const atom_indices = [];
          let hasSeq = false, hasCrd = false, hasIdx = false, hasHot = false;
          for (let i = start; i < stop; i += 1) {
            atom_indices.push(i);
            if (seqMask[i]) hasSeq = true;
            if (crdMask[i]) hasCrd = true;
            if (idxMask[i]) hasIdx = true;
            if (hotMask[i]) hasHot = true;
          }
          const conditions = [];
          if (hasSeq) conditions.push('sequence');
          if (hasCrd) conditions.push('coordinate');
          if (hasIdx) conditions.push('index');
          if (hasHot) conditions.push('hotspot');
          const resName = String(a.res_name[start]);
          items.push({
            kind: 'residue',
            atom_indices,
            residue_id: Number(a.res_id[start]),
            residue_name: resName,
            code: RESIDUE_CODES[resName] || resName,
            conditions,
          });
        }
        r += 1;
      }
      chains.push({ chain_id: chainId, items });
    }
    return chains;
  }

  validate() {
    const issues = [];
    if (this.atoms.nAtoms === 0) {
      issues.push({ severity: 'error', check: 'empty_structure', message: 'Structure contains no atoms.' });
    }
    return issues;
  }

  summary() {
    const details = this.annotationDetails();
    const stripIndices = (list) =>
      list.map(({ values, ...rest }) => ({
        ...rest,
        values: values.map(({ atom_indices, ...vRest }) => vRest),
      }));
    const charged_atoms = [];
    for (let i = 0; i < this.atoms.nAtoms; i += 1) {
      if (this.atoms.charge[i]) {
        charged_atoms.push({ atom_index: i, charge: Number(this.atoms.charge[i]) });
      }
    }
    return {
      name: this.name,
      revision: this.revision,
      atom_count: this.atoms.nAtoms,
      charged_atoms,
      condition_editors: this.conditionEditorState(),
      chain_ids: [...new Set(this.atoms.chain_id.map(String))].sort(),
      next_chain_id: this.nextChainId(),
      sequence: this.sequenceView(),
      condition_counts: Object.fromEntries(details.map((d) => [d.name, d.count])),
      annotations: stripIndices(details),
      stored_fields: stripIndices(this.storedAnnotationFields()),
      scalar_fields: stripIndices(this.scalarAnnotationFields()),
      undo_depth: this.undoStack.length,
      validation: this.validate(),
    };
  }

  _ensureAnnotationArray(col, spec) {
    if (!this.atoms.annotations[col]) {
      const n = this.atoms.nAtoms;
      if (NONSCALAR_1BODY_ANNOTATIONS[col]) {
        const ns = NONSCALAR_1BODY_ANNOTATIONS[col];
        this.atoms.annotations[col] = Array.from({ length: n }, () => new Array(ns.cols).fill(ns.default));
      } else {
        const sc = SCALAR_1BODY_ANNOTATIONS[col] || { default: spec?.default ?? 0 };
        this.atoms.annotations[col] = new Array(n).fill(sc.default);
      }
    }
    return this.atoms.annotations[col];
  }

  /**
   * Several condition edits as one undoable step (the design panel's chain roles and guided set-ups). Each edit is
   * `{condition, atom_indices, enabled, value, scope}` as for `setCondition`.
   */
  setConditions(edits) {
    if (!Array.isArray(edits) || !edits.length) throw new Error('no condition edits');
    return this.compound(() => {
      let affected = 0;
      for (const e of edits) {
        affected += this.setCondition(e.condition, e.atom_indices, {
          enabled: e.enabled ?? true, value: e.value ?? null, scope: e.scope || 'atom', snapshot: false,
        }).affected_count;
      }
      return { affected_count: affected, changed: true, message: `Updated conditions on ${affected} atoms.` };
    });
  }

  /** Run several edits as one undo step. */
  compound(fn) {
    // Nested (a rebuild inside a larger edit): part of the outer step, which took the snapshot and restores on failure.
    if (this._compound) return fn();
    this.snapshot();
    this._compound = true;
    try {
      return fn();
    } catch (err) {
      // Nothing half-applied and no empty undo step: restore the state taken above.
      const prev = this.undoStack.pop();
      this.atoms = prev.atoms;
      this.rawCif = prev.rawCif;
      this.revision = prev.revision;
      throw err;
    } finally {
      this._compound = false;
    }
  }

  _isSegmentAtom(i) {
    return /^\d+-\d+$/.test(String(this.atoms.res_name[i]));
  }

  /** Polymer residue atom (modified residues stored as HETATM included), not a ligand or a segment sentinel. */
  _isPolymerAtom(i) {
    const a = this.atoms;
    const polymer = a.is_polymer?.length === a.nAtoms ? Boolean(a.is_polymer[i]) : !a.hetero[i];
    return polymer && !this._isSegmentAtom(i);
  }

  /**
   * Polymer residues of `chain` in order, as {key, resId, atoms}; with `segments`, the chain's new-residue segments
   * too, as {segment: true, atoms, length} (their mean length).
   */
  _chainResidues(chain, { segments = false } = {}) {
    const a = this.atoms;
    const out = [];
    for (let i = 0; i < a.nAtoms; i += 1) {
      if (String(a.chain_id[i]) !== chain) continue;
      if (segments && this._isSegmentAtom(i)) {
        const min = Number(a.annotations?.annotation_expsegmin_1_residue?.[i]) || 1;
        const max = Number(a.annotations?.annotation_expsegmax_1_residue?.[i]) || min;
        out.push({ key: `seg:${i}`, segment: true, atoms: [i], length: Math.round((min + max) / 2) });
        continue;
      }
      if (!this._isPolymerAtom(i)) continue;
      const key = `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}`;
      if (out[out.length - 1]?.key !== key) out.push({ key, resId: a.res_id[i], atoms: [] });
      out[out.length - 1].atoms.push(i);
    }
    return out;
  }

  /**
   * Insert one segment per gap `{chain, resId, before, min, max}` (anchored on residue `resId` of `chain`, or a new
   * chain `chain` when `resId` is null). C-terminal gaps go first so renumbering never moves an anchor still to be used.
   */
  _insertGapSegments(gaps) {
    const sorted = [...gaps].sort((g, h) => (g.chain === h.chain ? (h.resId ?? 0) - (g.resId ?? 0) || (g.before ? 1 : -1) : 0));
    for (const g of sorted) {
      if (g.resId == null) {
        this.insertSegment({ minimum: g.min, maximum: g.max, newChain: true, chainId: g.chain });
        continue;
      }
      const anchor = this._chainResidues(g.chain).find((r) => r.resId === g.resId)?.atoms[0];
      if (anchor != null) this.insertSegment({ minimum: g.min, maximum: g.max, atomIndex: anchor, before: g.before });
    }
  }

  /**
   * Motif scaffolding around a selection: keep the selected residues (sequence + coordinates of the whole residue, or
   * of its side chain only), delete the other polymer residues of their chains and rebuild them as new residues.
   * Indexed (default): one segment per gap (N-terminal, between motif pieces, C-terminal), each its original length
   * +/- `spread`. Unindexed: the motif may sit anywhere, so the chain becomes one segment of `totalMin`-`totalMax`
   * residues (default: the chain's length +/- `spread`) plus the motif residues.
   */
  scaffoldMotif(indices, { spread = 0.25, sidechainsOnly = false, unindexed = false, totalMin = null, totalMax = null, ordered = false } = {}) {
    const keep = this.resolveIndices(indices, 'residue');
    // With `ordered`, the motif residues take the order of `indices` (as typed), each chain renumbered after.
    const typedOrder = ordered ? [...new Set((Array.isArray(indices) ? indices : []).map((i) => `${this.atoms.chain_id[i]}:${this.atoms.res_id[i]}:${this.atoms.ins_code[i]}`))] : [];
    if (!keep.length) throw new Error('select the motif residues first');
    const a = this.atoms;
    const resKey = (i) => `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}`;
    // Selected ligands are only held in place; chains are rebuilt around the selected polymer residues.
    const keepPolymer = keep.filter((i) => this._isPolymerAtom(i));
    if (!keepPolymer.length) throw new Error('select at least one protein or nucleic-acid residue as the motif');
    const kept = new Set(keepPolymer.map(resKey));
    const chains = [...new Set(keepPolymer.map((i) => String(a.chain_id[i])))];
    const range = (len) => {
      const min = Math.max(1, Math.round(len * (1 - spread)));
      return { min, max: Math.max(min, Math.round(len * (1 + spread))) };
    };

    const gaps = [];
    const drop = [];
    for (const chain of chains) {
      // Existing segments are replaced too, their length counted into the gap they sit in.
      const residues = this._chainResidues(chain, { segments: true });
      const others = residues.filter((r) => !kept.has(r.key));
      for (const r of others) drop.push(...r.atoms);
      const first = residues.find((r) => kept.has(r.key));
      if (unindexed) {
        const def = range(residues.reduce((s, r) => s + (r.segment ? r.length : kept.has(r.key) ? 0 : 1), 0));
        const min = Math.max(1, Math.round(totalMin ?? def.min));
        gaps.push({ chain, resId: first.resId, before: true, min, max: Math.max(min, Math.round(totalMax ?? def.max)) });
        continue;
      }
      let run = 0;
      let lastKept = null;
      const flush = (next) => {
        if (!run) return;
        const anchor = lastKept ?? next;
        if (anchor) gaps.push({ chain, resId: anchor.resId, before: !lastKept, ...range(run) });
        run = 0;
      };
      for (const r of residues) {
        if (kept.has(r.key)) {
          flush(r);
          lastKept = r;
        } else {
          run += r.segment ? r.length : 1;
        }
      }
      flush(null);
    }

    const backbone = new Set(['N', 'CA', 'C', 'O', 'OXT']);
    const crdAtoms = sidechainsOnly
      ? keep.filter((i) => !this._isPolymerAtom(i) || !backbone.has(String(a.atom_name[i])))
      : keep;
    return this.compound(() => {
      this.setCondition('coordinate', keep, { enabled: false, snapshot: false });
      if (crdAtoms.length) this.setCondition('coordinate', crdAtoms, { snapshot: false });
      this.setCondition('sequence', keep, { snapshot: false });
      // "Index conditioning" is stored inverted: enabling it marks residues as unindexed (placed anywhere).
      this.setCondition('index', keepPolymer, { enabled: unindexed, snapshot: false });
      if (drop.length) this.deleteAtoms(drop);
      this._insertGapSegments(gaps);
      if (typedOrder.length > 1) this._orderMotif(typedOrder, chains);
      return {
        changed: true,
        message: `Kept ${kept.size} motif residue${kept.size === 1 ? '' : 's'}${unindexed ? ' (placed anywhere)' : ''}; ` +
          `${gaps.length} new-residue segment${gaps.length === 1 ? '' : 's'}.`,
      };
    });
  }

  /**
   * After a scaffold: the kept residues of `chains` take the order of `keys` ("A:38:" style), each in the slot of a kept
   * residue (new-residue segments stay where they are), and the chains are numbered 1, 2, 3… again.
   */
  _orderMotif(keys, chains) {
    const a = this.atoms;
    const resKey = (i) => `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}`;
    const starts = getResidueSegments(a);
    const residues = [];
    for (let r = 0; r < starts.length - 1; r += 1) {
      const atoms = [];
      for (let i = starts[r]; i < starts[r + 1]; i += 1) atoms.push(i);
      residues.push({ key: resKey(starts[r]), chain: String(a.chain_id[starts[r]]), atoms, segment: this._isSegmentAtom(starts[r]) });
    }
    let changed = false;
    for (const chain of chains) {
      const slots = residues.filter((r) => r.chain === String(chain) && !r.segment);
      const want = keys.filter((k) => slots.some((r) => r.key === k));
      if (want.length !== slots.length) continue;
      const byKey = new Map(slots.map((r) => [r.key, r]));
      slots.forEach((slot, k) => { slot.next = byKey.get(want[k]); if (slot.next !== slot) changed = true; });
    }
    if (!changed) return;
    // Renumbered, each kept residue remembers where it came from ("A38"): the card names it so, as AtomWorks does.
    const src = this._ensureAnnotationArray('annotation_src-component_1_atom', { default: '' });
    for (const r of residues) if (!r.segment && !src[r.atoms[0]]) for (const i of r.atoms) src[i] = `${a.chain_id[i]}${a.res_id[i]}`;
    this._permuteAtoms(residues.flatMap((r) => (r.next ?? r).atoms));
    for (const chain of chains) this._renumberChain(chain);
    // The new residues' bonds went with the residues they joined: join each to its new neighbours instead.
    const mine = new Set(chains.map(String));
    const sentinels = [];
    for (let i = 0; i < a.nAtoms; i += 1) if (mine.has(String(a.chain_id[i])) && this._isSegmentAtom(i)) sentinels.push(i);
    const sentinelSet = new Set(sentinels);
    a.bonds = (a.bonds || []).filter(([u, v]) => !sentinelSet.has(u) && !sentinelSet.has(v));
    for (const i of sentinels) this._installSentinelBonds(i, a.chain_type?.[i]);
    computeIdAndEntityAnnotations(a);
  }

  /**
   * Bring residues back from their source: `other` holds them (an entry opened as the job's chain was, cut down to
   * those residues, its conditions set), each labelled with its source residue (`annotation_src-component_1_atom`,
   * "A38"). They join chain `chainId` next to their source neighbours: after the residue before them in the source, or
   * before the one after them, else in source order among the chain's residues. A chain renumbered by an ordered motif
   * is numbered 1, 2, 3… again; peptide and new-residue bonds are rebuilt. One undo step.
   */
  insertResiduesFrom(other, chainId, srcChain, at = null) {
    const cid = String(chainId);
    const SRC = 'annotation_src-component_1_atom';
    const numOf = (label) => {
      const s = String(label ?? '');
      return s.startsWith(String(srcChain)) && /^-?\d+$/.test(s.slice(String(srcChain).length)) ? Number(s.slice(String(srcChain).length)) : null;
    };
    return this.compound(() => {
      const a = this.atoms;
      const before = a.nAtoms;
      const resKey = (i) => `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}`;
      // Renumbered: some residue's number is not its source's.
      const srcArr0 = a.annotations?.[SRC];
      let renumbered = false;
      for (let i = 0; i < before; i += 1) {
        if (String(a.chain_id[i]) !== cid || this._isSegmentAtom(i)) continue;
        const n = numOf(srcArr0?.[i]);
        if (n !== null && n !== Number(a.res_id[i])) { renumbered = true; break; }
      }
      const temp = `__ins${Date.now().toString(36)}`;
      for (let i = 0; i < other.atoms.nAtoms; i += 1) other.atoms.chain_id[i] = temp;
      this.combine([other]);
      for (let i = before; i < a.nAtoms; i += 1) a.chain_id[i] = cid;
      const src = this._ensureAnnotationArray(SRC, { default: '' });
      // The chain's residues in order, and the new ones grouped into runs of consecutive source numbers.
      const starts = getResidueSegments(a);
      const residues = [];
      for (let r = 0; r < starts.length - 1; r += 1) {
        const s0 = starts[r];
        const atoms = [];
        for (let i = s0; i < starts[r + 1]; i += 1) atoms.push(i);
        // A residue with no source label is where it was loaded (a chain not renumbered): its number is its source's.
        const seg = this._isSegmentAtom(s0);
        const num = numOf(src[s0]) ?? (s0 < before && !seg && !renumbered ? Number(a.res_id[s0]) : null);
        residues.push({ atoms, chain: String(a.chain_id[s0]), added: s0 >= before, segment: seg, num, key: resKey(s0) });
      }
      const mine = residues.filter((r) => r.chain === cid);
      const old = mine.filter((r) => !r.added);
      const added = mine.filter((r) => r.added).sort((x, y) => x.num - y.num);
      const runs = [];
      for (const r of added) {
        const run = runs.at(-1);
        if (run && r.num === run.at(-1).num + 1) run.push(r); else runs.push([r]);
      }
      // Where each run goes: an index into `old` to insert before. With `at` ({ atom, before }: the "+" pressed), all of
      // them go there, in source order, beside the residue holding that atom.
      const placed = old.slice();
      if (at && Number.isInteger(at.atom)) {
        const k = placed.findIndex((r) => r.atoms.includes(at.atom));
        placed.splice(k < 0 ? placed.length : (at.before ? k : k + 1), 0, ...added);
        runs.length = 0;
        // Out of source order now (a motif in its own order): each residue keeps its source name and the chain is
        // numbered again, as the motif set-up does.
        const nums = placed.filter((r) => !r.segment && r.num !== null).map((r) => r.num);
        if (nums.some((n, i) => i > 0 && n <= nums[i - 1])) {
          for (const r of placed) if (!r.segment && r.num !== null && !src[r.atoms[0]]) for (const i of r.atoms) src[i] = `${srcChain}${r.num}`;
          renumbered = true;
        }
      }
      for (const run of runs) {
        const lo = run[0].num;
        const hi = run.at(-1).num;
        let at = placed.findIndex((r) => r.num === lo - 1);
        if (at >= 0) at += 1;
        else {
          at = placed.findIndex((r) => r.num === hi + 1);
          if (at < 0) {
            let last = -1;
            placed.forEach((r, k) => { if (r.num !== null && r.num < lo) last = k; });
            const next = placed.findIndex((r) => r.num !== null && r.num > hi);
            at = last >= 0 ? last + 1 : next >= 0 ? next : placed.length;
          }
        }
        placed.splice(at, 0, ...run);
      }
      const order = [];
      let done = false;
      for (const r of residues) {
        if (r.chain !== cid) { order.push(...r.atoms); continue; }
        if (!done) { for (const p of placed) order.push(...p.atoms); done = true; }
      }
      this._permuteAtoms(order);
      if (renumbered) this._renumberChain(cid);
      else {
        // In the source's order, each residue is numbered as in the source: a label saying so would only mark the chain
        // as no longer the one loaded (no drag bar), so the residues brought back carry none, as if never removed.
        const lab = a.annotations?.[SRC];
        for (let i = 0; i < a.nAtoms; i += 1) {
          if (String(a.chain_id[i]) === cid && lab?.[i] && numOf(lab[i]) === Number(a.res_id[i])) lab[i] = '';
        }
      }
      if (!renumbered) this._renumberPlaceholders(cid);
      this._relinkChain(cid);
      computeIdAndEntityAnnotations(a);
      return { changed: true, message: `Brought back ${added.length} residue${added.length === 1 ? '' : 's'} to chain ${cid}.` };
    });
  }

  /**
   * New-residue placeholders of chain `cid` numbered between their neighbours again after residues came back around
   * them, as `insertSegment` numbers them: a numbering gap kept, otherwise the residue before's number with the next
   * insertion code ("40A"), so a placeholder at the chain's end numbered 41 does not clash with the residue 41 returned.
   */
  _renumberPlaceholders(cid) {
    const a = this.atoms;
    const starts = getResidueSegments(a);
    const res = [];
    for (let r = 0; r < starts.length - 1; r += 1) {
      const s0 = starts[r];
      if (String(a.chain_id[s0]) !== cid) continue;
      res.push({ s0, e: starts[r + 1], seg: this._isSegmentAtom(s0) });
    }
    res.forEach((r, k) => {
      if (!r.seg) return;
      const prev = res.slice(0, k).reverse().find((x) => !x.seg);
      const next = res.slice(k + 1).find((x) => !x.seg);
      const id = Number(a.res_id[r.s0]);
      const p = prev ? Number(a.res_id[prev.s0]) : null;
      const q = next ? Number(a.res_id[next.s0]) : null;
      if ((p === null || id > p) && (q === null || id < q)) return;
      // The residue before's number, the next insertion code after any placeholder already on it.
      const base = p ?? (q !== null ? q - 1 : id);
      const used = res.slice(0, k).filter((x) => x.seg && Number(a.res_id[x.s0]) === base).map((x) => String(a.ins_code[x.s0] || ''));
      const code = used.length ? String.fromCharCode(Math.max(...used.map((c) => c.charCodeAt(0) || 64)) + 1) : 'A';
      for (let i = r.s0; i < r.e; i += 1) { a.res_id[i] = base; a.ins_code[i] = p === null ? '' : code; }
    });
  }

  /** Chain `cid`'s links after its residues moved: peptide bonds between consecutive numbers, new residues joined to their neighbours. */
  _relinkChain(cid) {
    const a = this.atoms;
    const starts = getResidueSegments(a);
    const resOf = new Int32Array(a.nAtoms);
    for (let r = 0; r < starts.length - 1; r += 1) for (let i = starts[r]; i < starts[r + 1]; i += 1) resOf[i] = r;
    const inChain = (i) => String(a.chain_id[i]) === cid;
    const sentinels = [];
    for (let i = 0; i < a.nAtoms; i += 1) if (inChain(i) && this._isSegmentAtom(i)) sentinels.push(i);
    const sentinelSet = new Set(sentinels);
    // Drop the chain's links between residues; keep bonds within residues and to other chains.
    a.bonds = (a.bonds || []).filter(([u, v]) => !(inChain(u) && inChain(v) && resOf[u] !== resOf[v]) && !sentinelSet.has(u) && !sentinelSet.has(v));
    const find = (r, name) => { for (let i = starts[r]; i < starts[r + 1]; i += 1) if (a.atom_name[i] === name) return i; return -1; };
    for (let r = 0; r < starts.length - 2; r += 1) {
      const s1 = starts[r];
      const s2 = starts[r + 1];
      if (!inChain(s1) || !inChain(s2) || this._isSegmentAtom(s1) || this._isSegmentAtom(s2)) continue;
      if (!this._isPolymerAtom(s1) || !this._isPolymerAtom(s2) || Number(a.res_id[s2]) - Number(a.res_id[s1]) !== 1) continue;
      const na = a.chain_type?.[s1] === CHAIN_TYPE.DNA || a.chain_type?.[s1] === CHAIN_TYPE.RNA;
      const out = find(r, na ? "O3'" : 'C');
      const inn = find(r + 1, na ? 'P' : 'N');
      if (out >= 0 && inn >= 0) a.bonds.push([out, inn, BOND_TYPE.SINGLE]);
    }
    for (const i of sentinels) this._installSentinelBonds(i, a.chain_type?.[i]);
  }

  /** Replace the selected polymer residues with new residues: one `min`-`max` segment per contiguous run. */
  replaceWithSegments(indices, minimum, maximum) {
    const sel = this.resolveIndices(indices, 'residue');
    const a = this.atoms;
    const keys = new Set(sel.map((i) => `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}`));
    const min = Math.max(1, Math.round(Number(minimum)) || 1);
    const max = Math.max(min, Math.round(Number(maximum)) || min);
    const gaps = [];
    const drop = [];
    const runs = [];
    for (const chain of new Set(sel.map((i) => String(a.chain_id[i])))) {
      const residues = this._chainResidues(chain);
      let inRun = false;
      residues.forEach((r, k) => {
        if (!keys.has(r.key)) {
          inRun = false;
          return;
        }
        drop.push(...r.atoms);
        if (inRun) {
          runs.at(-1).atoms.push(...r.atoms);
          return;
        }
        inRun = true;
        const prev = residues.slice(0, k).reverse().find((x) => !keys.has(x.key));
        const next = residues.slice(k + 1).find((x) => !keys.has(x.key));
        runs.push({ chain, prev: prev?.key ?? null, next: next?.key ?? null, atoms: [...r.atoms] });
        gaps.push(prev ? { chain, resId: prev.resId, before: false, min, max }
          : next ? { chain, resId: next.resId, before: true, min, max }
          : { chain, resId: null, min, max });
      });
    }
    if (!drop.length) throw new Error('select polymer residues to replace');
    // Each run's residues are remembered (this page only), so the new residues can be turned back into them.
    const origins = runs.map((run) => ({ chain: run.chain, prev: run.prev, next: run.next, atoms: this._sliceAtoms(run.atoms) }));
    return this.compound(() => {
      // Through each chain's plan: the selected runs become new blocks in their place, the chain re-joined by one rule.
      for (const chain of new Set(runs.map((r) => r.chain))) {
        const plan = this.chainPlan(chain);
        const info = this._chainSourceIndex(chain);
        const parts = [];
        let inRun = false;
        for (const q of plan.parts) {
          if (q.kind === 'new') { parts.push({ kind: 'new', atom: q.atom, min: q.min, max: q.max }); inRun = false; continue; }
          if (q.kind === 'own') {
            let lastKey = null;
            for (const i of q.atoms) {
              const key = `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}`;
              if (key === lastKey) { if (parts.at(-1)?.kind === 'own' && !inRun) parts.at(-1).atoms.push(i); continue; }
              lastKey = key;
              if (keys.has(key)) { if (!inRun) { parts.push({ kind: 'new', min, max }); inRun = true; } }
              else { inRun = false; const last = parts.at(-1); if (last?.kind === 'own') last.atoms.push(i); else parts.push({ kind: 'own', atoms: [i] }); }
            }
            continue;
          }
          let lastK = null;
          for (const i of q.atoms) {
            const k = this._sourceIndexOf(i, info);
            if (k === lastK) continue;
            lastK = k;
            if (keys.has(`${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}`)) {
              if (!inRun) { parts.push({ kind: 'new', min, max }); inRun = true; }
            } else {
              inRun = false;
              const last = parts.at(-1);
              if (last?.kind === 'kept' && last.to === k - 1) last.to = k; else parts.push({ kind: 'kept', from: k, to: k });
            }
          }
        }
        this.rebuildChain(chain, parts);
      }
      (this.segmentOrigins ??= []).push(...origins);
      return { changed: true, message: `Replaced ${keys.size} residues with ${gaps.length} new-residue segment${gaps.length === 1 ? '' : 's'}.` };
    });
  }

  /** A copy of atoms `keep` (conditions, bonds and restraints among them included), as their own atom set. */
  _sliceAtoms(keep) {
    const part = Object.create(StudioDoc.prototype);
    part.atoms = cloneAtoms(this.atoms);
    part.undoStack = [];
    part._compound = true;
    const kept = new Set(keep);
    part.deleteAtoms(Array.from({ length: part.atoms.nAtoms }, (_, i) => i).filter((i) => !kept.has(i)));
    return part.atoms;
  }

  /** The residues new-residue segment `atomIndex` replaced, while this page remembers them and they still fit there. */
  _segmentOrigin(atomIndex) {
    const a = this.atoms;
    if (!this.segmentOrigins?.length || !this._isSegmentAtom(atomIndex)) return null;
    const chain = String(a.chain_id[atomIndex]);
    const residues = this._chainResidues(chain, { segments: true });
    const k = residues.findIndex((r) => r.segment && r.atoms[0] === atomIndex);
    const near = (r) => (!r ? null : r.segment ? undefined : r.key);
    const prev = near(residues[k - 1]);
    const next = near(residues[k + 1]);
    const taken = new Set(residues.filter((r) => !r.segment).map((r) => r.key));
    for (let o = this.segmentOrigins.length - 1; o >= 0; o -= 1) {
      const origin = this.segmentOrigins[o];
      if (origin.chain !== chain || origin.prev !== prev || origin.next !== next) continue;
      const b = origin.atoms;
      const keys = [...new Set(Array.from({ length: b.nAtoms }, (_, i) => `${chain}:${b.res_id[i]}:${b.ins_code[i]}`))];
      if (keys.some((key) => taken.has(key))) continue;
      return { origin, keys, count: keys.length };
    }
    return null;
  }

  /** How many residues segment `atomIndex` replaced, and their keys ("A:41:"), or null when they are not known. */
  segmentOrigin(atomIndex) {
    const found = this._segmentOrigin(atomIndex);
    return found ? { count: found.count, keys: found.keys } : null;
  }

  /**
   * Turn new-residue segment `atomIndex` back into the residues it replaced, in its place, with the conditions they
   * had and their coordinates (so partial diffusion can start from them again). One undo step.
   */
  restoreSegment(atomIndex) {
    const found = this._segmentOrigin(atomIndex);
    if (!found) throw new Error('these new residues did not replace residues this page remembers');
    const chain = String(this.atoms.chain_id[atomIndex]);
    return this.compound(() => {
      const before = this.atoms.nAtoms;
      const part = { atoms: cloneAtoms(found.origin.atoms) };
      // A chain id of its own, or `combine` would rename it.
      const temp = `__res${Date.now().toString(36)}`;
      for (let i = 0; i < part.atoms.nAtoms; i += 1) part.atoms.chain_id[i] = temp;
      this.combine([part]);
      const a = this.atoms;
      for (let i = before; i < a.nAtoms; i += 1) a.chain_id[i] = chain;
      if (a.cathText) for (let j = 0; a.cathText.length < a.nAtoms; j += 1) a.cathText.push(part.atoms.cathText?.[j] ?? 'x.x.x.x');
      // The residues where the segment was, the segment last (then removed).
      const order = [];
      for (let i = 0; i < before; i += 1) {
        if (i === atomIndex) for (let j = before; j < a.nAtoms; j += 1) order.push(j);
        else order.push(i);
      }
      order.push(atomIndex);
      this._permuteAtoms(order);
      this.deleteAtoms([a.nAtoms - 1]);
      this._relinkChain(chain);
      computeIdAndEntityAnnotations(this.atoms);
      return { changed: true, message: `Brought back ${found.count} residue${found.count === 1 ? '' : 's'} to chain ${chain}.` };
    });
  }

  /**
   * Chain `chain`'s residues as it was loaded, in order ({num, ins, name}), or null when this page does not know them
   * or the chain no longer follows their numbering (an ordered motif renumbered it).
   */
  sourceResidues(chain) {
    const src = this.chainSources?.get(String(chain));
    if (!src) return null;
    const b = src.atoms;
    const list = [];
    for (let i = 0; i < b.nAtoms; i += 1) {
      if (String(b.chain_id[i]) !== src.chain) continue;
      const polymer = b.is_polymer?.length === b.nAtoms ? Boolean(b.is_polymer[i]) : !b.hetero[i];
      if (!polymer || /^\d+-\d+$/.test(String(b.res_name[i]))) continue;
      const last = list.at(-1);
      if (last && last.num === Number(b.res_id[i]) && last.ins === String(b.ins_code[i] ?? '')) continue;
      list.push({ num: Number(b.res_id[i]), ins: String(b.ins_code[i] ?? ''), name: String(b.res_name[i]) });
    }
    const known = new Map(list.map((r) => [`${r.num}:${r.ins}`, r.name]));
    const a = this.atoms;
    const SRC = a.annotations?.['annotation_src-component_1_atom'];
    for (let i = 0; i < a.nAtoms; i += 1) {
      if (String(a.chain_id[i]) !== String(chain) || !this._isPolymerAtom(i)) continue;
      if (SRC?.[i] || known.get(`${a.res_id[i]}:${a.ins_code[i] ?? ''}`) !== String(a.res_name[i])) return null;
    }
    return list.length ? list : null;
  }

  /**
   * Chain `cid`'s source (the chain as it was loaded): its polymer residues in order ({ num, ins, name }), their index
   * by "num:ins", and the source chain's own id. Null for a chain not loaded here (a bundled example, + seq).
   */
  _chainSourceIndex(cid) {
    const src = this.chainSources?.get(String(cid));
    // No source (a chain made here, + seq): an empty one, so a chain of new residues only still reads as a plan.
    if (!src) return { list: [], index: new Map(), chain: String(cid), atoms: null };
    const b = src.atoms;
    const list = [];
    for (let i = 0; i < b.nAtoms; i += 1) {
      if (String(b.chain_id[i]) !== src.chain) continue;
      const polymer = b.is_polymer?.length === b.nAtoms ? Boolean(b.is_polymer[i]) : !b.hetero[i];
      if (!polymer || /^\d+-\d+$/.test(String(b.res_name[i]))) continue;
      const last = list.at(-1);
      const ins = String(b.ins_code[i] ?? '');
      if (last && last.num === Number(b.res_id[i]) && last.ins === ins) continue;
      // The name the residue had there ("A38" in a file whose motif was renumbered), if any.
      const label = String(b.annotations?.['annotation_src-component_1_atom']?.[i] ?? '');
      list.push({ num: Number(b.res_id[i]), ins, name: String(b.res_name[i]), label: /^[A-Za-z]+-?\d+[A-Za-z]?$/.test(label) ? label : '' });
    }
    const index = new Map(list.map((r, k) => [`${r.num}:${r.ins}`, k]));
    return { list, index, chain: src.chain, atoms: b };
  }

  /** The source index (into `_chainSourceIndex(cid).list`) of polymer residue atom `i`: by its source tag only. */
  _sourceIndexOf(i, info) {
    const key = String(this.atoms.annotations?.[SRC_KEY]?.[i] ?? '');
    return key ? info.index.get(key) ?? null : null;
  }

  /**
   * Chain `cid` as a plan: its parts in order, each a run of the source kept ({ kind: 'kept', from, to, lo, hi }:
   * source indices and numbers, consecutive in the source and in the chain) or a block of new residues ({ kind: 'new',
   * atom, min, max }), or residues the source does not have ({ kind: 'own', atoms }), kept as they are.
   */
  chainPlan(cid) {
    cid = String(cid);
    const info = this._chainSourceIndex(cid);
    const a = this.atoms;
    const segMin = a.annotations?.annotation_expsegmin_1_residue;
    const segMax = a.annotations?.annotation_expsegmax_1_residue;
    const parts = [];
    let lastKey = null;
    for (let i = 0; i < a.nAtoms; i += 1) {
      if (String(a.chain_id[i]) !== cid) continue;
      if (this._isSegmentAtom(i)) {
        const [lo, hi] = String(a.res_name[i]).split('-').map(Number);
        parts.push({ kind: 'new', atom: i, min: Number(segMin?.[i]) > 0 ? Number(segMin[i]) : lo, max: Number(segMax?.[i]) > 0 ? Number(segMax[i]) : hi });
        lastKey = null;
        continue;
      }
      if (!this._isPolymerAtom(i)) continue;
      const key = `${a.res_id[i]}:${a.ins_code[i]}`;
      if (key === lastKey) { parts.at(-1).atoms.push(i); continue; }
      lastKey = key;
      const k = this._sourceIndexOf(i, info);
      const last = parts.at(-1);
      // A residue the source does not have (made from new residues, say): kept as it is, in place.
      if (k === null) { if (last?.kind === 'own') last.atoms.push(i); else parts.push({ kind: 'own', atoms: [i] }); continue; }
      if (last?.kind === 'kept' && k === last.to + 1) { last.to = k; last.atoms.push(i); } else parts.push({ kind: 'kept', from: k, to: k, atoms: [i] });
    }
    for (const p of parts) if (p.kind === 'kept') { p.lo = info.list[p.from].num; p.hi = info.list[p.to].num; }
    return { chain: info.chain, source: info.list, parts };
  }

  /**
   * Chain `cid` rebuilt to `parts`, in that order: each kept part ({ kind: 'kept', from, to } source indices, or
   * { lo, hi } source numbers) as the residues already in the chain where they are (their conditions with them) and the
   * rest from the chain as loaded (fixed, as loaded); each new part as its block of new residues ({ atom } an existing
   * one, kept with its settings; else { min, max } a new one). Then, all in one place: numbered as the source when the
   * parts run in the source's order (new blocks between, as `insertSegment` numbers them), else 1, 2, 3... with each
   * residue named by its source ("A38", a motif in its own order); bonded along the plan (consecutive source residues
   * by peptide bonds, new blocks to both neighbours, a gap left a gap). One undo step.
   */
  rebuildChain(cid, parts) {
    cid = String(cid);
    const info = this._chainSourceIndex(cid);
    const toIndex = (num) => {
      const k = info.list.findIndex((r) => r.num === Number(num));
      if (k < 0) throw new Error(`chain ${cid} as loaded has no residue ${num}`);
      return k;
    };
    const plan = parts.map((p) => (p.kind === 'kept'
      ? { kind: 'kept', from: Number.isInteger(p.from) ? p.from : toIndex(p.lo), to: Number.isInteger(p.to) ? p.to : toIndex(p.hi) }
      : p.kind === 'own' ? { kind: 'own', atoms: [...p.atoms] }
        : { kind: 'new', atom: Number.isInteger(p.atom) ? p.atom : null, min: Math.max(1, Number(p.min) || 1), max: Math.max(Number(p.min) || 1, Number(p.max) || 1) }));
    const ownOf = new Map();
    plan.forEach((p, k) => { if (p.kind === 'own') for (const i of p.atoms) ownOf.set(i, k); });
    const want = new Set();
    for (const p of plan) if (p.kind === 'kept') for (let k = Math.min(p.from, p.to); k <= Math.max(p.from, p.to); k += 1) want.add(k);
    const keepSeg = new Set(plan.filter((p) => p.kind === 'new' && p.atom !== null).map((p) => p.atom));
    const TAG = '__plan_tag';
    const SRC = 'annotation_src-component_1_atom';
    return this.compound(() => {
      const a = this.atoms;
      // Each atom of the chain tagged with what it is, so the steps below can move atoms around freely.
      const tag = this._ensureAnnotationArray(TAG, { default: '' });
      const present = new Set();
      const drop = [];
      for (let i = 0; i < a.nAtoms; i += 1) {
        if (String(a.chain_id[i]) !== cid) continue;
        if (this._isSegmentAtom(i)) {
          tag[i] = `s${i}`;
          if (!keepSeg.has(i)) drop.push(i);
        } else if (this._isPolymerAtom(i)) {
          const k = this._sourceIndexOf(i, info);
          if (k !== null) { tag[i] = `r${k}`; if (want.has(k)) present.add(k); else drop.push(i); }
          else if (ownOf.has(i)) tag[i] = `p${ownOf.get(i)}`;
          else { tag[i] = 'x'; drop.push(i); }
        } else tag[i] = 'o';
      }
      const segTag = new Map(plan.filter((p) => p.kind === 'new' && p.atom !== null).map((p) => [p, `s${p.atom}`]));
      if (drop.length) this.deleteAtoms(drop);
      // Residues of the plan not in the chain, from the chain as loaded.
      const missing = new Set([...want].filter((k) => !present.has(k)));
      if (missing.size && !info.atoms) throw new Error(`chain ${cid} has no source to bring residues back from`);
      if (missing.size) {
        const b = info.atoms;
        const take = [];
        for (let i = 0; i < b.nAtoms; i += 1) {
          if (String(b.chain_id[i]) !== info.chain) continue;
          const polymer = b.is_polymer?.length === b.nAtoms ? Boolean(b.is_polymer[i]) : !b.hetero[i];
          if (!polymer) continue;
          const k = info.index.get(`${Number(b.res_id[i])}:${String(b.ins_code[i] ?? '')}`);
          if (k !== undefined && missing.has(k)) take.push(i);
        }
        const holder = Object.create(StudioDoc.prototype);
        holder.atoms = b;
        const part = { atoms: holder._sliceAtoms(take) };
        const pa = part.atoms;
        pa.annotations = { ...(pa.annotations ?? {}) };
        pa.annotations[TAG] = Array.from({ length: pa.nAtoms }, (_, i) => `r${info.index.get(`${Number(pa.res_id[i])}:${String(pa.ins_code[i] ?? '')}`)}`);
        pa.annotations[SRC_KEY] = Array.from({ length: pa.nAtoms }, (_, i) => `${Number(pa.res_id[i])}:${String(pa.ins_code[i] ?? '')}`);
        const temp = `__plan${Date.now().toString(36)}`;
        for (let i = 0; i < pa.nAtoms; i += 1) pa.chain_id[i] = temp;
        const before = this.atoms.nAtoms;
        this.combine([part]);
        const c = this.atoms;
        for (let i = before; i < c.nAtoms; i += 1) c.chain_id[i] = cid;
        if (c.cathText) for (let j = 0; c.cathText.length < c.nAtoms; j += 1) c.cathText.push(pa.cathText?.[j] ?? 'x.x.x.x');
      }
      // New blocks: a placeholder each, tagged by its place in the plan.
      plan.forEach((p, k) => {
        if (p.kind !== 'new' || p.atom !== null) return;
        const anchor = this.atoms.chain_id.findIndex((x) => String(x) === cid);
        this.insertSegment({ minimum: p.min, maximum: p.max, atomIndex: anchor, before: true });
        const c = this.atoms;
        const tt = c.annotations[TAG];
        // The block just made: the chain's only new-residue placeholder not yet tagged as one (it may copy its
        // neighbour's annotations, this tag included).
        const made = Array.from({ length: c.nAtoms }, (_, i) => i).find((i) => String(c.chain_id[i]) === cid && this._isSegmentAtom(i) && !/^[sn]/.test(String(tt[i] ?? '')));
        if (made === undefined) throw new Error('could not add new residues');
        tt[made] = `n${k}`;
        segTag.set(p, `n${k}`);
      });
      // The chain's atoms in the plan's order (anything else in the chain, a ligand, after them).
      const c = this.atoms;
      // Residues already linked (own residues among themselves, say) stay linked while they stay next to each other.
      const RID = '__plan_res';
      const rid = this._ensureAnnotationArray(RID, { default: '' });
      for (let i = 0; i < c.nAtoms; i += 1) if (String(c.chain_id[i]) === cid) rid[i] = `${c.annotations[TAG][i]}|${c.res_id[i]}|${c.ins_code[i]}|${c.res_name[i]}`;
      const t = c.annotations[TAG];
      const byTag = new Map();
      for (let i = 0; i < c.nAtoms; i += 1) if (String(c.chain_id[i]) === cid) { const key = t[i] || 'o'; if (!byTag.has(key)) byTag.set(key, []); byTag.get(key).push(i); }
      const block = [];
      const realOrder = [];
      for (const p of plan) {
        if (p.kind === 'kept') {
          const step = p.from <= p.to ? 1 : -1;
          for (let k = p.from; step > 0 ? k <= p.to : k >= p.to; k += step) { block.push(...(byTag.get(`r${k}`) ?? [])); realOrder.push(k); }
        } else if (p.kind === 'own') block.push(...(byTag.get(`p${plan.indexOf(p)}`) ?? []));
        else block.push(...(byTag.get(segTag.get(p)) ?? []));
      }
      block.push(...(byTag.get('o') ?? []));
      const order = [];
      let placed = false;
      for (let i = 0; i < c.nAtoms; i += 1) {
        if (String(c.chain_id[i]) !== cid) { order.push(i); continue; }
        if (!placed) { order.push(...block); placed = true; }
      }
      this._permuteAtoms(order);
      // Numbering and source names, one rule: in the source's order, the source's numbers (new blocks between them as
      // insertSegment numbers them); otherwise 1, 2, 3... with each residue named by its source.
      const d = this.atoms;
      const tg = d.annotations[TAG];
      const src = this._ensureAnnotationArray(SRC, { default: '' });
      const linkedBefore = new Set();
      {
        const r = d.annotations[RID];
        for (const [u, v] of d.bonds || []) if (r[u] && r[v] && r[u] !== r[v]) linkedBefore.add(`${r[u]}>${r[v]}`).add(`${r[v]}>${r[u]}`);
      }
      const starts = getResidueSegments(d);
      const res = [];
      for (let r = 0; r < starts.length - 1; r += 1) {
        const s0 = starts[r];
        if (String(d.chain_id[s0]) !== cid) continue;
        const tv = String(tg[s0] || 'o');
        res.push({ s0, e: starts[r + 1], k: tv.startsWith('r') ? Number(tv.slice(1)) : null, seg: this._isSegmentAtom(s0) });
      }
      const setRes = (r, id, ins) => { for (let i = r.s0; i < r.e; i += 1) { d.res_id[i] = id; d.ins_code[i] = ins; } };
      // The number each residue keeps: its source's, or (one the source does not have) its own; in the source's order
      // and strictly increasing, the chain is numbered so, new blocks between as `insertSegment` numbers them;
      // otherwise 1, 2, 3... with each source residue named by its source ("A38", a motif in its own order).
      const fixedNum = (r) => (r.k !== null ? [info.list[r.k].num, info.list[r.k].ins] : [Number(d.res_id[r.s0]), String(d.ins_code[r.s0] || '')]);
      const real = res.filter((r) => !r.seg);
      const cmp = ([x, xi], [y, yi]) => (x !== y ? x - y : (xi || '').localeCompare(yi || ''));
      const inOrder = real.every((r, j) => j === 0 || cmp(fixedNum(r), fixedNum(real[j - 1])) > 0);
      if (inOrder) {
        for (const r of res) if (r.k !== null) { const x = info.list[r.k]; setRes(r, x.num, x.ins); for (let i = r.s0; i < r.e; i += 1) src[i] = x.label; }
        let lastId = -Infinity;
        let lastIns = '';
        res.forEach((r, j) => {
          if (!r.seg) { lastId = Number(d.res_id[r.s0]); lastIns = String(d.ins_code[r.s0] || ''); return; }
          const q = res.slice(j + 1).find((x) => !x.seg);
          const qId = q ? fixedNum(q)[0] : null;
          let id;
          let ins = '';
          if (lastId === -Infinity) id = qId !== null ? qId - 1 : 1;
          else if (qId === null) id = lastId + 1;
          else if (qId - 1 > lastId) id = qId - 1;
          else { id = lastId; ins = lastIns ? String.fromCharCode(lastIns.charCodeAt(0) + 1) : 'A'; }
          setRes(r, id, ins);
          lastId = id;
          lastIns = ins;
        });
      } else {
        for (const r of res) if (r.k !== null) { const x = info.list[r.k]; for (let i = r.s0; i < r.e; i += 1) src[i] = x.label || `${info.chain}${x.num}${x.ins}`; }
        this._renumberChain(cid);
      }
      // Bonds along the plan: the chain's backbone links rebuilt (side-chain links such as disulfides kept).
      const resOf = new Int32Array(d.nAtoms).fill(-1);
      res.forEach((r, j) => { for (let i = r.s0; i < r.e; i += 1) resOf[i] = j; });
      const backbone = new Set(['C', 'N', "O3'", 'P']);
      d.bonds = (d.bonds || []).filter(([u, v]) => {
        const ru = resOf[u];
        const rv = resOf[v];
        if (ru < 0 || rv < 0 || ru === rv) return true;
        if (res[ru].seg || res[rv].seg) return false;
        return !(backbone.has(String(d.atom_name[u])) && backbone.has(String(d.atom_name[v])));
      });
      const find = (r, name) => { for (let i = r.s0; i < r.e; i += 1) if (d.atom_name[i] === name) return i; return -1; };
      for (let j = 0; j + 1 < res.length; j += 1) {
        const x = res[j];
        const y = res[j + 1];
        const wasLinked = linkedBefore.has(`${d.annotations[RID][x.s0]}>${d.annotations[RID][y.s0]}`);
        if (!wasLinked && (x.k === null || y.k === null || y.k !== x.k + 1)) continue;
        if (x.seg || y.seg) continue;
        const na = d.chain_type?.[x.s0] === CHAIN_TYPE.DNA || d.chain_type?.[x.s0] === CHAIN_TYPE.RNA;
        const u = find(x, na ? "O3'" : 'C');
        const v = find(y, na ? 'P' : 'N');
        if (u >= 0 && v >= 0) d.bonds.push([u, v, BOND_TYPE.SINGLE]);
      }
      for (const r of res) if (r.seg) this._installSentinelBonds(r.s0, d.chain_type?.[r.s0]);
      delete d.annotations[TAG];
      delete d.annotations[RID];
      computeIdAndEntityAnnotations(d);
      return { changed: true, message: `Chain ${cid} rebuilt.` };
    });
  }

  /**
   * Atoms `indices` removed. Whole residues and new-residue blocks taken out of a chain that stays are a plan edit
   * (`rebuildChain` without them: the chain re-joined and numbered by the one rule); anything else (a whole chain, a
   * ligand, some atoms of a residue) is deleted as it is. One undo step.
   */
  removeAtoms(indices) {
    const del = new Set(this.resolveIndices(indices, 'atom'));
    if (!del.size) return { changed: false, message: 'Nothing to remove.' };
    const a = this.atoms;
    const resKey = (i) => `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}:${a.res_name[i]}`;
    const byRes = new Map();
    for (let i = 0; i < a.nAtoms; i += 1) { const k = resKey(i); if (!byRes.has(k)) byRes.set(k, []); byRes.get(k).push(i); }
    const planned = new Set();
    for (const cid of new Set([...del].map((i) => String(a.chain_id[i])))) {
      const mine = [];
      for (let i = 0; i < a.nAtoms; i += 1) if (String(a.chain_id[i]) === cid) mine.push(i);
      if (mine.every((i) => del.has(i))) continue;
      const gone = mine.filter((i) => del.has(i));
      const whole = gone.every((i) => (this._isSegmentAtom(i) || this._isPolymerAtom(i)) && byRes.get(resKey(i)).every((j) => del.has(j)));
      if (whole && this.chainPlan(cid)) planned.add(cid);
    }
    if (!planned.size) return this.deleteAtoms([...del]);
    const TAG = '__remove';
    return this.compound(() => {
      const tag = this._ensureAnnotationArray(TAG, { default: 0 });
      for (const i of del) tag[i] = 1;
      for (const cid of planned) {
        const d = this.atoms;
        const t = d.annotations[TAG];
        const plan = this.chainPlan(cid);
        const out = new Set();
        for (let i = 0; i < d.nAtoms; i += 1) if (String(d.chain_id[i]) === cid && t[i]) out.add(i);
        const parts = [];
        for (const q of plan.parts) {
          if (q.kind === 'new') { if (!out.has(q.atom)) parts.push({ kind: 'new', atom: q.atom, min: q.min, max: q.max }); continue; }
          if (q.kind === 'own') { const left = q.atoms.filter((i) => !out.has(i)); if (left.length) parts.push({ kind: 'own', atoms: left }); continue; }
          // The source indices of this run still wanted, as consecutive runs.
          const keep = [];
          for (const i of q.atoms) if (!out.has(i)) { const k = this._sourceIndexOf(i, this._chainSourceIndex(cid)); if (k !== null && keep.at(-1) !== k) keep.push(k); }
          for (const k of keep) { const last = parts.at(-1); if (last?.kind === 'kept' && last.to === k - 1 && last.run === q) last.to = k; else parts.push({ kind: 'kept', from: k, to: k, run: q }); }
        }
        this.rebuildChain(cid, parts.map(({ run, ...p }) => p));
      }
      const d = this.atoms;
      const rest = [];
      for (let i = 0; i < d.nAtoms; i += 1) if (d.annotations[TAG]?.[i]) rest.push(i);
      if (rest.length) this.deleteAtoms(rest);
      delete this.atoms.annotations[TAG];
      return { changed: true, message: `Removed ${del.size} atoms.` };
    });
  }

  /**
   * A copy of this document with chain `cid` rebuilt to `parts` (`rebuildChain`), this one unchanged: what Apply would
   * give, for a preview.
   */
  previewRebuild(cid, parts) {
    const copy = Object.create(StudioDoc.prototype);
    Object.assign(copy, this, { atoms: cloneAtoms(this.atoms), undoStack: [], rawCif: null, revision: this.revision + 1, _compound: false });
    copy.rebuildChain(cid, parts);
    return copy;
  }

  /**
   * Bring residues `resIds` of chain `chain` back from the chain as it was loaded, as they were loaded (conditions
   * included), each run next to its neighbour by number; peptide and new-residue bonds rebuilt. One undo step.
   */
  restoreFromSource(chain, resIds, at = null) {
    const cid = String(chain);
    const src = this.chainSources?.get(cid);
    if (!src || !this.sourceResidues(cid)) throw new Error(`chain ${cid} has no source to bring residues back from`);
    const want = new Set((resIds || []).map(Number));
    const a0 = this.atoms;
    for (let i = 0; i < a0.nAtoms; i += 1) if (String(a0.chain_id[i]) === cid && this._isPolymerAtom(i)) want.delete(Number(a0.res_id[i]));
    const b = src.atoms;
    const take = [];
    for (let i = 0; i < b.nAtoms; i += 1) if (String(b.chain_id[i]) === src.chain && want.has(Number(b.res_id[i]))) take.push(i);
    if (!take.length) throw new Error('no residues to bring back');
    const holder = Object.create(StudioDoc.prototype);
    holder.atoms = b;
    const part = { atoms: holder._sliceAtoms(take) };
    return this.compound(() => {
      const before = this.atoms.nAtoms;
      const temp = `__src${Date.now().toString(36)}`;
      for (let i = 0; i < part.atoms.nAtoms; i += 1) part.atoms.chain_id[i] = temp;
      this.combine([part]);
      const a = this.atoms;
      for (let i = before; i < a.nAtoms; i += 1) a.chain_id[i] = cid;
      if (a.cathText) for (let j = 0; a.cathText.length < a.nAtoms; j += 1) a.cathText.push(part.atoms.cathText?.[j] ?? 'x.x.x.x');
      // Runs of consecutive numbers, each after the residue numbered just before it, or before the one just after.
      const added = [];
      for (let i = before; i < a.nAtoms; i += 1) {
        const n = Number(a.res_id[i]);
        const run = added.at(-1);
        if (run && (run.hi === n || run.hi + 1 === n)) { run.hi = n; run.atoms.push(i); } else added.push({ lo: n, hi: n, atoms: [i] });
      }
      const after = new Map();
      const ahead = new Map();
      const tail = [];
      for (const run of added) {
        let at = -1;
        for (let i = 0; i < before; i += 1) if (String(a.chain_id[i]) === cid && this._isPolymerAtom(i) && Number(a.res_id[i]) === run.lo - 1) at = i;
        if (at >= 0) { after.set(at, [...(after.get(at) ?? []), ...run.atoms]); continue; }
        for (let i = 0; i < before && at < 0; i += 1) if (String(a.chain_id[i]) === cid && this._isPolymerAtom(i) && Number(a.res_id[i]) === run.hi + 1) at = i;
        if (at >= 0) ahead.set(at, [...(ahead.get(at) ?? []), ...run.atoms]);
        else tail.push(...run.atoms);
      }
      // `at` ({ atom, before }: a piece dragged out): all of them beside that piece, on the side it grew, whatever is
      // past it (new residues after 36 with 37-42 brought back to 36's piece go between 36 and them, not after them).
      if (at && Number.isInteger(at.atom) && at.atom < before) {
        const key = (i) => `${a.res_id[i]}:${a.ins_code[i]}`;
        let s0 = at.atom;
        while (s0 > 0 && String(a.chain_id[s0 - 1]) === cid && key(s0 - 1) === key(at.atom)) s0 -= 1;
        let e0 = at.atom;
        while (e0 + 1 < before && String(a.chain_id[e0 + 1]) === cid && key(e0 + 1) === key(at.atom)) e0 += 1;
        after.clear();
        ahead.clear();
        tail.length = 0;
        const all = added.flatMap((run) => run.atoms);
        if (at.before) ahead.set(s0, all); else after.set(e0, all);
      }
      let lastOfChain = -1;
      for (let i = 0; i < before; i += 1) if (String(a.chain_id[i]) === cid) lastOfChain = i;
      const order = [];
      for (let i = 0; i < before; i += 1) {
        if (ahead.has(i)) order.push(...ahead.get(i));
        order.push(i);
        // After the last atom of residue `i`.
        if (after.has(i)) order.push(...after.get(i));
        if (i === lastOfChain) order.push(...tail);
      }
      if (lastOfChain < 0) order.push(...tail);
      this._permuteAtoms(order);
      this._relinkChain(cid);
      computeIdAndEntityAnnotations(this.atoms);
      const n = new Set(added.flatMap((r) => r.atoms.map((i) => `${a0.res_id[i] ?? ''}`))).size && new Set(take.map((i) => `${b.res_id[i]}:${b.ins_code[i]}`)).size;
      return { changed: true, message: `Brought back ${n} residue${n === 1 ? '' : 's'} to chain ${cid}.` };
    });
  }

  /**
   * Small-molecule binder set-up: optionally remove every polymer chain, keep the ligands (identity always; pose fixed
   * or predicted) and add a new protein chain of `minimum`-`maximum` residues.
   */
  designAroundLigand({ removeProtein = true, poseFixed = true, minimum = 100, maximum = 150 } = {}) {
    const a = this.atoms;
    const ligand = [];
    const polymer = [];
    for (let i = 0; i < a.nAtoms; i += 1) (this._isPolymerAtom(i) || this._isSegmentAtom(i) ? polymer : ligand).push(i);
    if (!ligand.length) throw new Error('this structure has no ligand');
    return this.compound(() => {
      if (removeProtein && polymer.length) this.deleteAtoms(polymer);
      const lig = [];
      for (let i = 0; i < this.atoms.nAtoms; i += 1) if (!this._isPolymerAtom(i) && !this._isSegmentAtom(i)) lig.push(i);
      this.setCondition('sequence', lig, { snapshot: false });
      this.setCondition('coordinate', lig, { enabled: poseFixed, snapshot: false });
      this.insertSegment({ minimum, maximum, newChain: true });
      return { changed: true, message: `Set up a protein of ${minimum}-${maximum} residues around ${new Set(lig.map((i) => this.atoms.res_name[i])).size} ligand(s).` };
    });
  }

  /** Change an expandable segment's length range (its sentinel atom's min/max, and its `min-max` name). */
  setSegmentRange(atomIndex, minimum, maximum) {
    const a = this.atoms;
    const segMin = a.annotations?.annotation_expsegmin_1_residue;
    if (!segMin || !(atomIndex >= 0 && atomIndex < a.nAtoms) || !(Number(segMin[atomIndex]) > 0)) {
      throw new Error('not an expandable segment');
    }
    const minLen = Math.max(1, Math.round(Number(minimum)) || 1);
    const maxLen = Math.max(minLen, Math.round(Number(maximum)) || minLen);
    this.snapshot();
    a.annotations.annotation_expsegmin_1_residue[atomIndex] = minLen;
    this._ensureAnnotationArray('annotation_expsegmax_1_residue', { default: -1 })[atomIndex] = maxLen;
    if (/^\d+-\d+$/.test(String(a.res_name[atomIndex]))) a.res_name[atomIndex] = `${minLen}-${maxLen}`;
    return { changed: true, message: `Segment length set to ${minLen}-${maxLen}.` };
  }

  setCondition(name, indices, { enabled = true, value = null, scope = 'atom', snapshot = true } = {}) {
    const spec = CONDITION_BY_KEY.get(name);
    if (!spec) throw new Error(`unsupported condition: ${name}`);
    const spreadScope = spec.level === 'chain' ? 'chain' : spec.level === 'residue' ? 'residue' : scope;
    const selected = this.resolveIndices(indices, spreadScope);
    if (selected.length === 0) throw new Error('at least one atom must be selected');

    if (snapshot) this.snapshot();
    const n = this.atoms.nAtoms;

    if (spec.key === 'sequence') {
      const arr = this._ensureAnnotationArray('condition_sequence_1_residue', spec);
      for (const idx of selected) {
        arr[idx] = enabled ? this.atoms.res_name[idx] : '<M>';
      }
    } else if (spec.key === 'coordinate') {
      const crd = this._ensureAnnotationArray('condition_coordinate_1_atom', spec);
      const grp = this._ensureAnnotationArray('condition_CoordinateGroup_1_atom', { default: -1 });
      for (const idx of selected) {
        crd[idx] = enabled ? this.atoms.coord[idx].slice(0, 3) : [NaN, NaN, NaN];
        grp[idx] = enabled ? 0 : -1;
      }
    } else if (spec.key === 'index') {
      const arr = this._ensureAnnotationArray('condition_index_1_residue', spec);
      const targetVal = spec.invert_mask ? !enabled : Boolean(enabled);
      for (const idx of selected) arr[idx] = targetVal;
    } else if (spec.key === 'cath_label') {
      if (!this.atoms.cathText) this.atoms.cathText = new Array(n).fill('x.x.x.x');
      let lbl = enabled ? String(value || 'x.x.x.x').trim() : 'x.x.x.x';
      if (enabled && lbl && lbl !== 'x.x.x.x' && lbl !== 'x.x.x' && lbl !== '.') {
        const parts = lbl.split('.');
        while (parts.length < 4) parts.push('x');
        lbl = parts.slice(0, 4).join('.');
      } else {
        lbl = 'x.x.x.x';
      }
      if (
        !this.atoms.annotations['condition_cath-label_1_residue'] ||
        Array.isArray(this.atoms.annotations['condition_cath-label_1_residue'][0])
      ) {
        this.atoms.annotations['condition_cath-label_1_residue'] = this.atoms.cathText.slice();
      }
      for (const idx of selected) {
        this.atoms.cathText[idx] = lbl;
        this.atoms.annotations['condition_cath-label_1_residue'][idx] = lbl;
      }
    } else if (spec.key === 'chain_centroid') {
      const arr = this._ensureAnnotationArray('condition_ChainCentroid_1_atom', spec);
      const vec = enabled && Array.isArray(value) ? value.map(Number) : [NaN, NaN, NaN];
      for (const idx of selected) arr[idx] = vec.slice(0, 3);
    } else if (spec.kind === 'mask') {
      const arr = this._ensureAnnotationArray(spec.col, spec);
      for (const idx of selected) arr[idx] = Boolean(enabled);
    } else if (spec.kind === 'enum' || spec.kind === 'integer' || spec.kind === 'scalar') {
      const arr = this._ensureAnnotationArray(spec.col, spec);
      const targetVal = enabled ? Number(value) : spec.default;
      for (const idx of selected) arr[idx] = targetVal;
      if (spec.key === 'coordinate_group' && enabled && targetVal > 0) {
        const crd = this._ensureAnnotationArray('condition_coordinate_1_atom', { default: NaN });
        const intCrd = this._ensureAnnotationArray('condition_InternalCoordinate_1_atom', { default: NaN });
        for (const idx of selected) {
          crd[idx] = [NaN, NaN, NaN];
          intCrd[idx] = this.atoms.coord[idx].slice(0, 3);
        }
      }
    }

    return {
      affected_count: selected.length,
      changed: true,
      message: `${enabled ? 'Applied' : 'Removed'} ${spec.label} on ${selected.length} atoms.`,
    };
  }

  /** `groupId`: join that rigid group (default: a new one). */
  setInternalCoordinateConditioning(indices, enabled = true, scope = 'atom', groupId = null) {
    const selected = this.resolveIndices(indices, scope);
    if (selected.length === 0) throw new Error('at least one atom must be selected');
    this.snapshot();
    const grp = this._ensureAnnotationArray('condition_CoordinateGroup_1_atom', { default: -1 });
    const crd = this._ensureAnnotationArray('condition_coordinate_1_atom', { default: NaN });
    const intCrd = this._ensureAnnotationArray('condition_InternalCoordinate_1_atom', { default: NaN });
    let nextGroup = 1;
    for (let i = 0; i < this.atoms.nAtoms; i += 1) {
      if (grp[i] >= nextGroup) nextGroup = grp[i] + 1;
    }
    if (Number.isInteger(groupId) && groupId > 0) nextGroup = groupId;
    let affected = 0;
    let skipped = 0;
    for (const idx of selected) {
      const c = this.atoms.coord[idx];
      const finite = Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2]);
      if (enabled && !finite) {
        skipped += 1;
        continue;
      }
      affected += 1;
      crd[idx] = [NaN, NaN, NaN];
      grp[idx] = enabled ? nextGroup : -1;
      intCrd[idx] = enabled ? c.slice(0, 3) : [NaN, NaN, NaN];
    }
    return {
      affected_count: affected,
      skipped_count: skipped,
      changed: affected > 0,
      message: enabled
        ? `Marked ${affected} atoms for internal coordinates (group ${nextGroup}).`
        : `Unset internal coordinates on ${affected} atoms.`,
    };
  }

  editConditionGroup(condition, action, indices, { groupId = null, scope = 'atom' } = {}) {
    const spec = CONDITION_BY_KEY.get(condition);
    if (!spec || !spec.group_editor) throw new Error(`${condition} is not a group condition`);
    const ann = this._ensureAnnotationArray(spec.col, spec);
    if (action === 'new') {
      let nextId = 1;
      for (const v of ann) {
        if (typeof v === 'number' && v >= nextId) nextId = v + 1;
      }
      while (spec.reserved_group_ids?.includes(nextId)) nextId += 1;
      const res = this.setCondition(spec.key, indices, { enabled: true, value: nextId, scope });
      return {
        group_id: nextId,
        affected_count: res.affected_count,
        changed: true,
        message: `Created group ${nextId} with ${res.affected_count} atoms.`,
      };
    }
    if (action === 'add') {
      const res = this.setCondition(spec.key, indices, { enabled: true, value: Number(groupId), scope });
      return {
        group_id: Number(groupId),
        affected_count: res.affected_count,
        changed: true,
        message: `Added ${res.affected_count} atoms to group ${groupId}.`,
      };
    }
    if (action === 'remove') {
      const res = this.setCondition(spec.key, indices, { enabled: false, value: spec.default, scope });
      return {
        affected_count: res.affected_count,
        changed: true,
        message: `Restored default group assignments for ${res.affected_count} atoms.`,
      };
    }
    if (action === 'delete') {
      const target = Number(groupId);
      const selected = [];
      for (let i = 0; i < this.atoms.nAtoms; i += 1) {
        if (ann[i] === target) selected.push(i);
      }
      const res = this.setCondition(spec.key, selected, { enabled: false, value: spec.default, scope: 'atom' });
      return {
        affected_count: res.affected_count,
        changed: true,
        message: `Deleted group ${groupId} (${res.affected_count} atoms reassigned).`,
      };
    }
    throw new Error(`Unsupported group action: ${action}`);
  }

  editChainCentroid(condition, chain, value) {
    const indices = [];
    for (let i = 0; i < this.atoms.nAtoms; i += 1) {
      if (String(this.atoms.chain_id[i]) === String(chain)) indices.push(i);
    }
    if (indices.length === 0) throw new Error(`Unknown chain: ${chain}`);
    this.setCondition(condition, indices, {
      enabled: value !== null && value !== undefined,
      value,
      scope: 'chain',
    });
    return {
      changed: true,
      message: `${value !== null ? 'Set' : 'Cleared'} centroid for chain '${chain}'.`,
    };
  }

  replaceDistanceRestraints(pairs, values) {
    this.snapshot();
    const out = [];
    const seen = new Set();
    for (let i = 0; i < (pairs || []).length; i += 1) {
      const a1 = Math.min(Number(pairs[i][0]), Number(pairs[i][1]));
      const a2 = Math.max(Number(pairs[i][0]), Number(pairs[i][1]));
      const d = Number(values[i]);
      const key = `${a1}:${a2}`;
      if (a1 !== a2 && !seen.has(key) && Number.isFinite(d) && d > 0) {
        seen.add(key);
        out.push({ atom1: a1, atom2: a2, distance: d });
      }
    }
    this.atoms.distanceRestraints = out;
    return { changed: true, message: `Saved ${out.length} distance restraints.` };
  }

  resetConditions() {
    this.snapshot();
    const keep = {};
    for (const [k, v] of Object.entries(this.atoms.annotations || {})) {
      // New-residue lengths and this page's own bookkeeping (`__src`: where each residue came from) are not conditions.
      if (k.startsWith('annotation_expseg') || k.startsWith('__')) keep[k] = v;
    }
    this.atoms.annotations = keep;
    this.atoms.distanceRestraints = [];
    this.atoms.cathText = null;
    return { changed: true, message: 'All conditions reset to defaults.' };
  }

  setCharge(indices, { value = null, delta = null } = {}) {
    const selected = this.resolveIndices(indices, 'atom');
    this.snapshot();
    for (const idx of selected) {
      if (delta !== null && delta !== undefined) {
        this.atoms.charge[idx] = (this.atoms.charge[idx] || 0) + Number(delta);
      } else {
        this.atoms.charge[idx] = Number(value || 0);
      }
    }
    return { changed: true, message: `Updated formal charge on ${selected.length} atoms.` };
  }

  editBond(atom1, atom2, action, bondType = 'SINGLE') {
    const u = Math.min(Number(atom1), Number(atom2));
    const v = Math.max(Number(atom1), Number(atom2));
    if (u === v) throw new Error('a bond requires two different atoms');
    this.snapshot();
    const bt = BOND_NAME_TO_TYPE[bondType] ?? BOND_TYPE.SINGLE;
    const bonds = (this.atoms.bonds || []).filter(([a, b]) => !(Math.min(a, b) === u && Math.max(a, b) === v));
    if (action === 'add') bonds.push([u, v, bt]);
    this.atoms.bonds = bonds;
    computeIdAndEntityAnnotations(this.atoms);
    return { changed: true, message: `${action === 'add' ? 'Added' : 'Removed'} bond ${u}-${v}.` };
  }

  inspectBonds(atomIndices = null) {
    const filterSet = atomIndices ? new Set(this.resolveIndices(atomIndices, 'atom')) : null;
    const out = [];
    for (const [u, v, bt] of this.atoms.bonds || []) {
      if (filterSet && !filterSet.has(u) && !filterSet.has(v)) continue;
      out.push({
        atom1: u,
        atom2: v,
        bond_type: BOND_TYPE_TO_NAME[bt] || 'SINGLE',
        distance: this.atomDistance(u, v) ?? 0,
      });
    }
    return out;
  }

  inferBondsFromCcd() {
    this.snapshot();
    buildBondsForAtomArray(this.atoms, this.cifBlock);
    computeIdAndEntityAnnotations(this.atoms);
    return { changed: true, message: 'Inferred CCD bonds.' };
  }

  protonate(ph = 7.4) {
    this.snapshot();
    const a = this.atoms;
    const pHVal = Number.isFinite(Number(ph)) ? Number(ph) : 7.4;
    let chargedCount = 0;
    for (let i = 0; i < a.nAtoms; i += 1) {
      const res = a.res_name[i];
      const aname = a.atom_name[i];
      if ((res === 'ASP' && aname === 'OD2') || (res === 'GLU' && aname === 'OE2')) {
        a.charge[i] = pHVal >= 4.2 ? -1 : 0;
        if (a.charge[i] !== 0) chargedCount += 1;
      } else if (res === 'LYS' && aname === 'NZ') {
        a.charge[i] = pHVal <= 10.5 ? 1 : 0;
        if (a.charge[i] !== 0) chargedCount += 1;
      } else if (res === 'ARG' && aname === 'NH2') {
        a.charge[i] = pHVal <= 12.0 ? 1 : 0;
        if (a.charge[i] !== 0) chargedCount += 1;
      } else if (res === 'HIS' && aname === 'ND1') {
        a.charge[i] = pHVal <= 6.0 ? 1 : 0;
        if (a.charge[i] !== 0) chargedCount += 1;
      }
    }
    const hCount = a.element.filter((e) => e === 'H').length;
    return {
      changed: true,
      message: `Prepared protonation state at pH ${pHVal.toFixed(1)} (${chargedCount} ionizable sites, ${hCount} explicit H atoms).`,
      hydrogen_count: hCount,
    };
  }

  deleteAtoms(indices, closeResidueGaps = false) {
    const delSet = new Set(this.resolveIndices(indices, 'atom'));
    if (delSet.size >= this.atoms.nAtoms) throw new Error('cannot delete every atom');
    this.snapshot();
    const touched = new Set([...delSet].map((i) => String(this.atoms.chain_id[i])));
    const oldToNew = new Int32Array(this.atoms.nAtoms).fill(-1);
    const keepIndices = [];
    for (let i = 0; i < this.atoms.nAtoms; i += 1) {
      if (!delSet.has(i)) {
        oldToNew[i] = keepIndices.length;
        keepIndices.push(i);
      }
    }
    const m = keepIndices.length;
    const pick = (arr) => (Array.isArray(arr) ? keepIndices.map((i) => arr[i]) : arr);
    const a = this.atoms;
    a.nAtoms = m;
    for (const key of [
      'hetero', 'element', 'atom_name', 'res_name', 'chain_id', 'label_entity_id',
      'res_id', 'ins_code', 'auth_seq_id', 'atom_id', 'b_factor', 'occupancy',
      'charge', 'label_alt_id', 'coord', 'transformation_id', 'is_polymer',
      'chain_type', 'chem_comp_type', 'atomic_number', 'is_aromatic', 'is_leaving_atom',
    ]) {
      a[key] = pick(a[key]);
    }
    for (const [k, v] of Object.entries(a.annotations || {})) {
      a.annotations[k] = pick(v);
    }
    if (a.cathText) a.cathText = pick(a.cathText);
    a.bonds = (a.bonds || [])
      .filter(([u, v]) => oldToNew[u] >= 0 && oldToNew[v] >= 0)
      .map(([u, v, bt]) => [oldToNew[u], oldToNew[v], bt]);
    a.distanceRestraints = (a.distanceRestraints || [])
      .filter((r) => oldToNew[r.atom1] >= 0 && oldToNew[r.atom2] >= 0)
      .map((r) => ({ atom1: oldToNew[r.atom1], atom2: oldToNew[r.atom2], distance: r.distance }));
    // New residues next to residues removed keep a bond to whatever is next to them now: new residues added after 29,
    // then 30-32 removed, were otherwise left bonded to 29 alone, and the model read a chain break before 33 (a design
    // came back 29 A apart there).
    const sentinels = [];
    for (let i = 0; i < m; i += 1) if (touched.has(String(a.chain_id[i])) && this._isSegmentAtom(i)) sentinels.push(i);
    if (sentinels.length) {
      const own = new Set(sentinels);
      a.bonds = a.bonds.filter(([u, v]) => !own.has(u) && !own.has(v));
      for (const i of sentinels) this._installSentinelBonds(i, a.chain_type?.[i]);
    }

    if (closeResidueGaps) {
      this._renumberChainResidues();
    }
    computeIdAndEntityAnnotations(a);
    return { changed: true, message: `Deleted ${delSet.size} atoms.` };
  }

  /** Put the atoms in `order` (every index once): annotations, bonds and restraints follow. */
  _permuteAtoms(order) {
    const a = this.atoms;
    if (order.length !== a.nAtoms) throw new Error('permutation must name every atom once');
    const oldToNew = new Int32Array(a.nAtoms);
    order.forEach((oldIndex, k) => { oldToNew[oldIndex] = k; });
    const pick = (arr) => (Array.isArray(arr) ? order.map((i) => arr[i]) : arr);
    for (const key of [
      'hetero', 'element', 'atom_name', 'res_name', 'chain_id', 'label_entity_id',
      'res_id', 'ins_code', 'auth_seq_id', 'atom_id', 'b_factor', 'occupancy',
      'charge', 'label_alt_id', 'coord', 'transformation_id', 'is_polymer',
      'chain_type', 'chem_comp_type', 'atomic_number', 'is_aromatic', 'is_leaving_atom',
    ]) {
      a[key] = pick(a[key]);
    }
    for (const [k, v] of Object.entries(a.annotations || {})) a.annotations[k] = pick(v);
    if (a.cathText) a.cathText = pick(a.cathText);
    a.bonds = (a.bonds || []).map(([u, v, bt]) => [oldToNew[u], oldToNew[v], bt]);
    a.distanceRestraints = (a.distanceRestraints || [])
      .map((r) => ({ atom1: oldToNew[r.atom1], atom2: oldToNew[r.atom2], distance: r.distance }));
  }

  /** Chains in the order given (chains not named keep their place after those named). */
  reorderChains(chainIds) {
    const a = this.atoms;
    const want = (chainIds || []).map(String);
    const present = [...new Set(a.chain_id.map(String))];
    const missing = want.filter((c) => !present.includes(c));
    if (missing.length) throw new Error(`no chain ${missing.join(', ')}`);
    const sequence = [...want, ...present.filter((c) => !want.includes(c))];
    const order = [];
    for (const cid of sequence) for (let i = 0; i < a.nAtoms; i += 1) if (String(a.chain_id[i]) === cid) order.push(i);
    this.snapshot();
    this._permuteAtoms(order);
    computeIdAndEntityAnnotations(a);
    return { changed: true, message: `Chains in order ${sequence.join(', ')}.` };
  }

  /**
   * One chain's residues numbered consecutively in document order, its first residue that is not new residues being 1
   * (new residues before it count down from 0), as AtomWorks numbers a scaffolded motif.
   */
  _renumberChain(chainId) {
    const resStarts = getResidueSegments(this.atoms);
    const mine = [];
    for (let r = 0; r < resStarts.length - 1; r += 1) if (String(this.atoms.chain_id[resStarts[r]]) === String(chainId)) mine.push(r);
    const firstKept = Math.max(0, mine.findIndex((r) => !this._isSegmentAtom(resStarts[r])));
    let next = -firstKept;
    for (const r of mine) {
      const s = resStarts[r];
      next += 1;
      for (let i = s; i < resStarts[r + 1]; i += 1) {
        this.atoms.res_id[i] = next;
        this.atoms.auth_seq_id[i] = next;
        this.atoms.ins_code[i] = '';
      }
    }
  }

  /**
   * Forget where `indices` are: no coordinates, no occupancy, no coordinate conditions. Atoms in `lead` move to the
   * front of their residues (in their order), so a residue that keeps only some atoms is read from one of those.
   */
  clearCoordinates(indices, lead = []) {
    const selected = this.resolveIndices(indices, 'atom');
    if (!selected.length) throw new Error('at least one atom must be selected');
    this.snapshot();
    const a = this.atoms;
    const crd = this._ensureAnnotationArray('condition_coordinate_1_atom', { default: NaN });
    const grp = this._ensureAnnotationArray('condition_CoordinateGroup_1_atom', { default: -1 });
    const intCrd = this._ensureAnnotationArray('condition_InternalCoordinate_1_atom', { default: NaN });
    for (const i of selected) {
      a.coord[i] = [NaN, NaN, NaN];
      a.occupancy[i] = 0;
      crd[i] = [NaN, NaN, NaN];
      grp[i] = -1;
      intCrd[i] = [NaN, NaN, NaN];
    }
    const first = new Set(lead);
    if (first.size) {
      const starts = getResidueSegments(a);
      const order = [];
      for (let r = 0; r < starts.length - 1; r += 1) {
        const atoms = [];
        for (let i = starts[r]; i < starts[r + 1]; i += 1) atoms.push(i);
        order.push(...atoms.filter((i) => first.has(i)), ...atoms.filter((i) => !first.has(i)));
      }
      this._permuteAtoms(order);
      computeIdAndEntityAnnotations(a);
    }
    return { changed: true, affected_count: selected.length, message: `Cleared the coordinates of ${selected.length} atoms.` };
  }

  _renumberChainResidues() {
    const resStarts = getResidueSegments(this.atoms);
    const numRes = resStarts.length - 1;
    const chainCounters = new Map();
    for (let r = 0; r < numRes; r += 1) {
      const s = resStarts[r];
      const e = resStarts[r + 1];
      const cid = this.atoms.chain_id[s];
      const nextId = (chainCounters.get(cid) || 0) + 1;
      chainCounters.set(cid, nextId);
      for (let i = s; i < e; i += 1) {
        this.atoms.res_id[i] = nextId;
        this.atoms.auth_seq_id[i] = nextId;
      }
    }
  }

  resetResidueIds() {
    this.snapshot();
    const chains = [...new Set(this.atoms.chain_id)];
    for (const cid of chains) {
      let minRes = Infinity;
      for (let i = 0; i < this.atoms.nAtoms; i += 1) {
        if (this.atoms.chain_id[i] === cid && this.atoms.res_id[i] < minRes) {
          minRes = this.atoms.res_id[i];
        }
      }
      if (Number.isFinite(minRes)) {
        const shift = minRes - 1;
        for (let i = 0; i < this.atoms.nAtoms; i += 1) {
          if (this.atoms.chain_id[i] === cid) {
            this.atoms.res_id[i] -= shift;
            this.atoms.auth_seq_id[i] -= shift;
          }
        }
      }
    }
    return { changed: true, message: 'Reset residue IDs to start at 1 per chain.' };
  }

  insertSegment({ minimum = 5, maximum = 10, atomIndex = null, before = true, newChain = false, chainId = null, chainType = 'polypeptide(L)' } = {}) {
    const minLen = Math.max(1, Number(minimum) || 5);
    const maxLen = Math.max(minLen, Number(maximum) || minLen);
    this.snapshot();

    const a = this.atoms;
    const n = a.nAtoms;
    const targetChain = newChain ? ((chainId || '').trim() || this.nextChainId()) : a.chain_id[atomIndex ?? 0];
    let insertPos = n;
    let resId = 1;
    let insCode = '';
    let shiftDownstream = false;

    if (!newChain && atomIndex !== null && atomIndex >= 0 && atomIndex < n) {
      const resStarts = getResidueSegments(a);
      let rIdx = 0;
      for (let r = 0; r < resStarts.length - 1; r += 1) {
        if (atomIndex >= resStarts[r] && atomIndex < resStarts[r + 1]) {
          rIdx = r;
          break;
        }
      }
      insertPos = before ? resStarts[rIdx] : resStarts[rIdx + 1];
      // Numbered so that no residue already in the chain is renumbered (atomworks `_compute_insert_res_id` shifts
      // the rest of the chain up by one, so a target's numbering - and its hotspots - changed under the user): a
      // numbering gap if there is one; else the residue before's number with the next insertion code ("40A"), which
      // shared/rfd4p/pipeline/transforms.js `realizeExpandableSegments` numbers past when it expands the segment;
      // before the chain's first residue, first - 1 (zero or negative is fine).
      const chainIdx = [];
      for (let i = 0; i < n; i += 1) if (a.chain_id[i] === targetChain) chainIdx.push(i);
      const nBefore = chainIdx.filter((i) => i < insertPos).length;
      if (nBefore > 0) {
        const prevAtom = chainIdx[nBefore - 1];
        const prevResId = a.res_id[prevAtom];
        const nextAtom = nBefore < chainIdx.length ? chainIdx[nBefore] : null;
        const nextResId = nextAtom === null ? null : a.res_id[nextAtom];
        const prevIns = String(a.ins_code[prevAtom] || '');
        const code = prevIns ? String.fromCharCode(prevIns.charCodeAt(0) + 1) : 'A';
        const nextIns = nextAtom === null ? '' : String(a.ins_code[nextAtom] || '');
        if (nextResId === null || nextResId > prevResId + 1) {
          resId = nextResId === null ? prevResId + 1 : nextResId - 1;
        } else if (!(nextResId === prevResId && nextIns && nextIns <= code) && /^[A-Z]$/.test(code)) {
          resId = prevResId;
          insCode = code;
        } else {
          // No room between two insertion codes: the upstream rule.
          resId = prevResId + 1;
          shiftDownstream = true;
        }
      } else {
        resId = a.res_id[chainIdx[0]] - 1;
      }
      if (shiftDownstream) {
        for (let i = insertPos; i < n; i += 1) {
          if (a.chain_id[i] === targetChain && a.res_id[i] >= resId) {
            a.res_id[i] += 1;
            a.auth_seq_id[i] += 1;
          }
        }
      }
    }

    const compName = `${minLen}-${maxLen}`;
    const ct =
      chainType === 'polydeoxyribonucleotide'
        ? CHAIN_TYPE.DNA
        : chainType === 'polyribonucleotide'
        ? CHAIN_TYPE.RNA
        : chainType === 'polypeptide(D)'
        ? CHAIN_TYPE.POLYPEPTIDE_D
        : CHAIN_TYPE.POLYPEPTIDE_L;

    const spliceAt = (arr, val) => arr.splice(insertPos, 0, val);
    spliceAt(a.hetero, true);
    spliceAt(a.element, 'X');
    spliceAt(a.atom_name, 'UNK');
    spliceAt(a.res_name, compName);
    spliceAt(a.chain_id, targetChain);
    spliceAt(a.label_entity_id, 0);
    spliceAt(a.res_id, resId);
    spliceAt(a.ins_code, insCode);
    spliceAt(a.auth_seq_id, resId);
    spliceAt(a.atom_id, n + 1);
    spliceAt(a.b_factor, NaN);
    spliceAt(a.occupancy, 0.0);
    spliceAt(a.charge, 0);
    spliceAt(a.label_alt_id, '.');
    spliceAt(a.coord, [NaN, NaN, NaN]);
    spliceAt(a.transformation_id, '1');
    spliceAt(a.is_polymer, true);
    spliceAt(a.chain_type, ct);
    spliceAt(a.chem_comp_type, 'NON-POLYMER');
    spliceAt(a.atomic_number, 0);
    spliceAt(a.is_aromatic, false);
    spliceAt(a.is_leaving_atom, false);

    a.nAtoms = n + 1;

    const segMin = this._ensureAnnotationArray('annotation_expsegmin_1_residue', { default: -1 });
    const segMax = this._ensureAnnotationArray('annotation_expsegmax_1_residue', { default: -1 });
    for (const [k, arr] of Object.entries(a.annotations)) {
      if (arr.length === n) {
        const ns = NONSCALAR_1BODY_ANNOTATIONS[k];
        const sc = SCALAR_1BODY_ANNOTATIONS[k];
        const isVec = ns && (arr.length === 0 || Array.isArray(arr[0]));
        const defVal = isVec
          ? new Array(ns.cols).fill(ns.default)
          : k.startsWith('__')
          ? ''
          : typeof arr[0] === 'string'
          ? 'x.x.x.x'
          : sc
          ? sc.default
          : 0;
        arr.splice(insertPos, 0, defVal);
      }
    }
    if (Array.isArray(a.cathText) && a.cathText.length === n) {
      a.cathText.splice(insertPos, 0, 'x.x.x.x');
    }
    segMin[insertPos] = minLen;
    segMax[insertPos] = maxLen;

    // Shift bond indices >= insertPos
    a.bonds = (a.bonds || []).map(([u, v, bt]) => [
      u >= insertPos ? u + 1 : u,
      v >= insertPos ? v + 1 : v,
      bt,
    ]);
    this._installSentinelBonds(insertPos, ct);
    computeIdAndEntityAnnotations(a);

    return {
      changed: true,
      message: `Inserted expandable segment ${compName} on chain ${targetChain}.`,
    };
  }

  /**
   * Wire an expandable-segment sentinel into the polymer bond graph (atomworks `_install_sentinel_bonds`): single
   * bonds to the previous residue's terminal polymer atom and the next residue's start atom in the same chain (an
   * adjacent one-atom sentinel stands in for either), replacing any direct bond between those two. Without these the
   * realized segment is a separate fragment to the model.
   */
  _installSentinelBonds(sentinelIdx, chainType) {
    const a = this.atoms;
    const isNa = chainType === CHAIN_TYPE.DNA || chainType === CHAIN_TYPE.RNA;
    const [termName, startName] = isNa ? ["O3'", 'P'] : ['C', 'N'];
    const segMin = a.annotations['annotation_expsegmin_1_residue'];
    const resStarts = getResidueSegments(a);
    const r = resStarts.findIndex((s, k) => k < resStarts.length - 1 && sentinelIdx >= s && sentinelIdx < resStarts[k + 1]);
    const chain = a.chain_id[sentinelIdx];
    const pick = (k, name) => {
      if (k < 0 || k >= resStarts.length - 1) return null;
      const s = resStarts[k];
      const e = resStarts[k + 1];
      if (a.chain_id[s] !== chain) return null;
      for (let i = s; i < e; i += 1) {
        if (a.atom_name[i] === name || (e - s === 1 && segMin && segMin[i] !== -1)) return i;
      }
      return null;
    };
    const prev = pick(r - 1, termName);
    const next = pick(r + 1, startName);
    const key = (u, v) => `${Math.min(u, v)}-${Math.max(u, v)}`;
    let bonds = a.bonds || [];
    if (prev !== null && next !== null) bonds = bonds.filter(([u, v]) => key(u, v) !== key(prev, next));
    const have = new Set(bonds.map(([u, v]) => key(u, v)));
    if (prev !== null && !have.has(key(prev, sentinelIdx))) bonds.push([prev, sentinelIdx, BOND_TYPE.SINGLE]);
    if (next !== null && !have.has(key(sentinelIdx, next))) bonds.push([sentinelIdx, next, BOND_TYPE.SINGLE]);
    a.bonds = bonds;
  }

  clearInternalCoordinateTargets(indices, scope = 'atom') {
    const selected = this.resolveIndices(indices, scope);
    if (selected.length === 0) throw new Error('at least one atom must be selected');
    const grp = this.atoms.annotations?.['condition_CoordinateGroup_1_atom'];
    const intCrd = this.atoms.annotations?.['condition_InternalCoordinate_1_atom'];
    if (!grp || !intCrd) {
      throw new Error('The selected internal-coordinate groups have no saved targets to clear.');
    }
    const selectedGroups = new Set();
    for (const idx of selected) {
      if (grp[idx] > 0) selectedGroups.add(grp[idx]);
    }
    let cleared = 0;
    this.snapshot();
    for (let i = 0; i < this.atoms.nAtoms; i += 1) {
      if (selectedGroups.has(grp[i]) && Array.isArray(intCrd[i]) && Number.isFinite(intCrd[i][0])) {
        intCrd[i] = [NaN, NaN, NaN];
        cleared += 1;
      }
    }
    return {
      affected_count: cleared,
      changed: cleared > 0,
      message: `Cleared internal coordinate targets on ${cleared} atoms.`,
    };
  }

  editDistanceRestraint(atom1, atom2, action = 'set', distance = null) {
    const u = Math.min(Number(atom1), Number(atom2));
    const v = Math.max(Number(atom1), Number(atom2));
    if (u === v) throw new Error('a distance restraint requires two different atoms');
    this.snapshot();
    const list = (this.atoms.distanceRestraints || []).filter((r) => !(r.atom1 === u && r.atom2 === v));
    if (action === 'set') {
      const d = Number(distance);
      if (!Number.isFinite(d) || d <= 0) throw new Error('distance is required when action is set');
      list.push({ atom1: u, atom2: v, distance: d });
    }
    this.atoms.distanceRestraints = list;
    return {
      changed: true,
      message: action === 'set' ? `Set distance restraint ${u}-${v} (${distance} Å).` : `Removed distance restraint ${u}-${v}.`,
    };
  }

  assignChain(indices, chainId) {
    const selected = this.resolveIndices(indices, 'atom');
    const cid = String(chainId || '').trim();
    if (!cid) throw new Error('chain_id must not be empty');
    this.snapshot();
    const from = [...new Set(selected.map((i) => String(this.atoms.chain_id[i])))];
    for (const idx of selected) {
      this.atoms.chain_id[idx] = cid;
    }
    // A chain renamed whole keeps its source.
    if (from.length === 1 && !this.atoms.chain_id.some((c) => String(c) === from[0]) && this.chainSources?.has(from[0])) {
      this.chainSources.set(cid, this.chainSources.get(from[0]));
    }
    computeIdAndEntityAnnotations(this.atoms);
    return {
      affected_count: selected.length,
      changed: true,
      message: `Assigned chain '${cid}' to ${selected.length} atoms.`,
    };
  }

  realizeSegments(seed = 42) {
    const a = this.atoms;
    const segMin = a.annotations?.['annotation_expsegmin_1_residue'];
    const segMax = a.annotations?.['annotation_expsegmax_1_residue'];
    if (!segMin || !segMax) return { changed: false, message: 'No expandable segments to realize.' };
    let hasUnrealized = false;
    for (let i = 0; i < a.nAtoms; i += 1) {
      if (segMin[i] > 0 && segMax[i] > 0 && a.res_name[i] !== '<M>') {
        hasUnrealized = true;
        break;
      }
    }
    if (!hasUnrealized) return { changed: false, message: 'Expandable segments are already realized.' };
    this.snapshot();
    let s = (Number(seed) || 42) >>> 0;
    const nextRand = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 0x100000000;
    };
    let i = 0;
    while (i < a.nAtoms) {
      if (segMin[i] > 0 && segMax[i] > 0 && a.res_name[i] !== '<M>') {
        const mn = segMin[i];
        const mx = segMax[i];
        const len = mn + Math.floor(nextRand() * (mx - mn + 1));
        a.res_name[i] = '<M>';
        a.atom_name[i] = 'CA';
        a.element[i] = 'C';
        for (let k = 1; k < len; k += 1) {
          const pos = i + k;
          const spliceAt = (arr, val) => arr.splice(pos, 0, val);
          spliceAt(a.hetero, true);
          spliceAt(a.element, 'C');
          spliceAt(a.atom_name, 'CA');
          spliceAt(a.res_name, '<M>');
          spliceAt(a.chain_id, a.chain_id[i]);
          spliceAt(a.label_entity_id, a.label_entity_id[i]);
          spliceAt(a.res_id, a.res_id[i] + k);
          spliceAt(a.ins_code, '');
          spliceAt(a.auth_seq_id, a.auth_seq_id[i] + k);
          spliceAt(a.atom_id, a.nAtoms + k);
          spliceAt(a.b_factor, NaN);
          spliceAt(a.occupancy, 0.0);
          spliceAt(a.charge, 0);
          spliceAt(a.label_alt_id, '.');
          spliceAt(a.coord, [NaN, NaN, NaN]);
          spliceAt(a.transformation_id, '1');
          spliceAt(a.is_polymer, true);
          spliceAt(a.chain_type, a.chain_type[i]);
          spliceAt(a.chem_comp_type, a.chem_comp_type[i]);
          spliceAt(a.atomic_number, 6);
          spliceAt(a.is_aromatic, false);
          spliceAt(a.is_leaving_atom, false);
          for (const [, arr] of Object.entries(a.annotations)) {
            const val = Array.isArray(arr[i]) ? arr[i].slice() : arr[i];
            arr.splice(pos, 0, val);
          }
          if (Array.isArray(a.cathText)) {
            a.cathText.splice(pos, 0, a.cathText[i] || 'x.x.x.x');
          }
          a.nAtoms += 1;
        }
        i += len;
      } else {
        i += 1;
      }
    }
    this._renumberChainResidues();
    computeIdAndEntityAnnotations(a);
    return { changed: true, message: 'Realized expandable segments.' };
  }

  collapseSegments() {
    const a = this.atoms;
    const segMin = a.annotations?.['annotation_expsegmin_1_residue'];
    const segMax = a.annotations?.['annotation_expsegmax_1_residue'];
    if (!segMin || !segMax) return { changed: false, message: 'No realized segments to collapse.' };
    const toDelete = [];
    let collapsedAny = false;
    let i = 0;
    while (i < a.nAtoms) {
      if (segMin[i] > 0 && segMax[i] > 0 && a.res_name[i] === '<M>') {
        collapsedAny = true;
        const mn = segMin[i];
        const mx = segMax[i];
        a.res_name[i] = `${mn}-${mx}`;
        a.atom_name[i] = 'UNK';
        a.element[i] = 'X';
        let j = i + 1;
        while (
          j < a.nAtoms &&
          a.chain_id[j] === a.chain_id[i] &&
          segMin[j] === mn &&
          segMax[j] === mx &&
          a.res_name[j] === '<M>'
        ) {
          toDelete.push(j);
          j += 1;
        }
        i = j;
      } else {
        i += 1;
      }
    }
    if (!collapsedAny) return { changed: false, message: 'No realized segments to collapse.' };
    if (toDelete.length > 0) {
      this.deleteAtoms(toDelete, true);
    } else {
      this.snapshot();
    }
    return { changed: true, message: 'Collapsed realized segments.' };
  }

  combine(otherDocs) {
    this.snapshot();
    const a = this.atoms;
    const usedChains = new Set(a.chain_id.map(String));
    for (const otherDoc of otherDocs) {
      const b = cloneAtoms(otherDoc.atoms);
      const offset = a.nAtoms;
      const chainRename = new Map();
      for (const cid of new Set(b.chain_id.map(String))) {
        if (usedChains.has(cid)) {
          let next = this.nextChainId();
          while (usedChains.has(next)) {
            usedChains.add(next);
            next = `${next}A`;
          }
          chainRename.set(cid, next);
          usedChains.add(next);
        } else {
          usedChains.add(cid);
        }
      }
      for (let i = 0; i < b.nAtoms; i += 1) {
        const cid = String(b.chain_id[i]);
        if (chainRename.has(cid)) b.chain_id[i] = chainRename.get(cid);
      }
      for (const key of [
        'hetero', 'element', 'atom_name', 'res_name', 'chain_id', 'label_entity_id',
        'res_id', 'ins_code', 'auth_seq_id', 'atom_id', 'b_factor', 'occupancy',
        'charge', 'label_alt_id', 'coord', 'transformation_id', 'is_polymer',
        'chain_type', 'chem_comp_type', 'atomic_number', 'is_aromatic', 'is_leaving_atom',
      ]) {
        a[key].push(...b[key]);
      }
      const allAnnKeys = new Set([...Object.keys(a.annotations || {}), ...Object.keys(b.annotations || {})]);
      for (const k of allAnnKeys) {
        const spec = CONDITION_BY_KEY.get(k);
        const arrA = this._ensureAnnotationArray(k, spec);
        const arrB = b.annotations?.[k];
        if (arrB && arrB.length === b.nAtoms) {
          arrA.push(...arrB);
        } else {
          const ns = NONSCALAR_1BODY_ANNOTATIONS[k];
          const sc = SCALAR_1BODY_ANNOTATIONS[k] || { default: spec?.default ?? 0 };
          for (let i = 0; i < b.nAtoms; i += 1) {
            arrA.push(ns ? new Array(ns.cols).fill(ns.default) : sc.default);
          }
        }
      }
      a.nAtoms += b.nAtoms;
      // Where the chains came from goes with them, under their new names.
      for (const cid of new Set(b.chain_id.map(String))) {
        const from = [...chainRename].find(([, to]) => to === cid)?.[0] ?? cid;
        const src = otherDoc.chainSources?.get(from);
        if (src) (this.chainSources ??= new Map()).set(cid, src);
      }
      for (const [u, v, bt] of b.bonds || []) {
        a.bonds.push([u + offset, v + offset, bt]);
      }
      for (const r of b.distanceRestraints || []) {
        a.distanceRestraints.push({ atom1: r.atom1 + offset, atom2: r.atom2 + offset, distance: r.distance });
      }
    }
    computeIdAndEntityAnnotations(a);
    return { changed: true, message: `Combined structure (${a.nAtoms} atoms total).` };
  }

  analyzeInteractions({ interaction_types = ['hbond', 'hydrophobic', 'saltbridge', 'metal'], inter_chain_only = true, source_atom_indices = [], target_atom_indices = [] } = {}) {
    const a = this.atoms;
    const n = a.nAtoms;
    const srcSet = source_atom_indices?.length ? new Set(this.resolveIndices(source_atom_indices, 'atom')) : null;
    const tgtSet = target_atom_indices?.length ? new Set(this.resolveIndices(target_atom_indices, 'atom')) : null;
    const types = new Set(interaction_types);
    const interactions = [];

    const METAL_ELEMENTS = new Set([
      'MG', 'ZN', 'CA', 'MN', 'FE', 'CU', 'CO', 'NI', 'NA', 'K', 'CD', 'HG', 'PT', 'AU',
    ]);
    const HALOGEN_ELEMENTS = new Set(['F', 'CL', 'BR', 'I']);

    const isPositiveAtom = (idx) => {
      if (a.charge[idx] > 0) return true;
      const res = a.res_name[idx];
      const aname = a.atom_name[idx];
      return (
        (res === 'LYS' && aname === 'NZ') ||
        (res === 'ARG' && (aname === 'NH1' || aname === 'NH2' || aname === 'NE')) ||
        (res === 'HIS' && (aname === 'ND1' || aname === 'NE2'))
      );
    };
    const isNegativeAtom = (idx) => {
      if (a.charge[idx] < 0) return true;
      const res = a.res_name[idx];
      const aname = a.atom_name[idx];
      return (
        (res === 'ASP' && (aname === 'OD1' || aname === 'OD2')) ||
        (res === 'GLU' && (aname === 'OE1' || aname === 'OE2')) ||
        aname === 'OP1' ||
        aname === 'OP2'
      );
    };
    const isAromaticAtom = (idx) => {
      if (a.is_aromatic?.[idx]) return true;
      const ringSet = STD_RESIDUE_AROMATIC?.[a.res_name[idx]];
      return Boolean(ringSet && ringSet.has(a.atom_name[idx]));
    };

    const seenResiduePairs = new Set();

    for (let i = 0; i < n; i += 1) {
      const ci = a.coord[i];
      if (!Number.isFinite(ci[0])) continue;
      const ei = (a.element[i] || '').toUpperCase();
      if (ei === 'H' || ei === 'X') continue;
      for (let j = i + 1; j < n; j += 1) {
        if (inter_chain_only && a.chain_id[i] === a.chain_id[j]) continue;
        if (!inter_chain_only && a.chain_id[i] === a.chain_id[j] && Math.abs(a.res_id[i] - a.res_id[j]) <= 1) continue;
        if (srcSet && tgtSet) {
          if (!((srcSet.has(i) && tgtSet.has(j)) || (srcSet.has(j) && tgtSet.has(i)))) continue;
        } else if (srcSet && !srcSet.has(i) && !srcSet.has(j)) {
          continue;
        }
        const cj = a.coord[j];
        if (!Number.isFinite(cj[0])) continue;
        const ej = (a.element[j] || '').toUpperCase();
        if (ej === 'H' || ej === 'X') continue;
        const d = Math.hypot(ci[0] - cj[0], ci[1] - cj[1], ci[2] - cj[2]);
        if (d > 5.0 || d < 0.5) continue;

        const posI = isPositiveAtom(i);
        const posJ = isPositiveAtom(j);
        const negI = isNegativeAtom(i);
        const negJ = isNegativeAtom(j);
        const aromI = isAromaticAtom(i);
        const aromJ = isAromaticAtom(j);

        if (types.has('metal') && d <= 3.0 && ((METAL_ELEMENTS.has(ei) && (ej === 'O' || ej === 'N' || ej === 'S')) || (METAL_ELEMENTS.has(ej) && (ei === 'O' || ei === 'N' || ei === 'S')))) {
          interactions.push({
            type: 'metal',
            interaction_type: 'metal',
            atom1: i,
            atom2: j,
            atom1_role: METAL_ELEMENTS.has(ei) ? 'metal' : 'ligand',
            atom2_role: METAL_ELEMENTS.has(ej) ? 'metal' : 'ligand',
            distance: d,
          });
        } else if (types.has('saltbridge') && d <= 4.2 && ((posI && negJ) || (negI && posJ))) {
          interactions.push({
            type: 'salt',
            interaction_type: 'saltbridge',
            atom1: i,
            atom2: j,
            atom1_role: posI ? 'positive' : 'negative',
            atom2_role: posJ ? 'positive' : 'negative',
            distance: d,
          });
        } else if (types.has('hbond') && (ei === 'N' || ei === 'O') && (ej === 'N' || ej === 'O') && d <= 3.5) {
          interactions.push({
            type: 'hbond',
            interaction_type: 'hbond',
            atom1: i,
            atom2: j,
            atom1_role: ei === 'N' ? 'donor' : 'acceptor',
            atom2_role: ej === 'O' ? 'acceptor' : 'donor',
            distance: d,
          });
        } else if (types.has('pistacking') && aromI && aromJ && d <= 4.5) {
          const pairKey = `pi:${a.chain_id[i]}:${a.res_id[i]}:${a.chain_id[j]}:${a.res_id[j]}`;
          if (!seenResiduePairs.has(pairKey)) {
            seenResiduePairs.add(pairKey);
            interactions.push({
              type: 'pistacking',
              interaction_type: 'pistacking',
              atom1: i,
              atom2: j,
              atom1_role: 'aromatic',
              atom2_role: 'aromatic',
              distance: d,
            });
          }
        } else if (types.has('pication') && d <= 4.5 && ((aromI && posJ) || (posI && aromJ))) {
          const pairKey = `picat:${a.chain_id[i]}:${a.res_id[i]}:${a.chain_id[j]}:${a.res_id[j]}`;
          if (!seenResiduePairs.has(pairKey)) {
            seenResiduePairs.add(pairKey);
            interactions.push({
              type: 'pication',
              interaction_type: 'pication',
              atom1: i,
              atom2: j,
              atom1_role: aromI ? 'aromatic' : 'cation',
              atom2_role: aromJ ? 'aromatic' : 'cation',
              distance: d,
            });
          }
        } else if (types.has('halogen') && d <= 3.8 && ((HALOGEN_ELEMENTS.has(ei) && (ej === 'O' || ej === 'N' || ej === 'S')) || (HALOGEN_ELEMENTS.has(ej) && (ei === 'O' || ei === 'N' || ei === 'S')))) {
          interactions.push({
            type: 'halogen',
            interaction_type: 'halogen',
            atom1: i,
            atom2: j,
            atom1_role: HALOGEN_ELEMENTS.has(ei) ? 'donor' : 'acceptor',
            atom2_role: HALOGEN_ELEMENTS.has(ej) ? 'donor' : 'acceptor',
            distance: d,
          });
        } else if (types.has('hydrophobic') && ei === 'C' && ej === 'C' && d <= 4.0 && a.atom_name[i] !== 'CA' && a.atom_name[i] !== 'C' && a.atom_name[j] !== 'CA' && a.atom_name[j] !== 'C') {
          interactions.push({
            type: 'hydrophobic',
            interaction_type: 'hydrophobic',
            atom1: i,
            atom2: j,
            atom1_role: 'hydrophobic',
            atom2_role: 'hydrophobic',
            distance: d,
          });
        }
      }
    }

    return {
      revision: this.revision,
      cached: false,
      interaction_count: interactions.length,
      interactions,
    };
  }

  toCif() {
    if (this.revision === 0 && this.rawCif) {
      return this.rawCif;
    }
    const a = this.atoms;
    const n = a.nAtoms;
    const lines = [`data_${(this.name || 'structure').replace(/\s+/g, '_')}`, '#'];

    // Entity & polymer categories
    const chains = [...new Set(a.chain_id.map(String))];
    lines.push('loop_');
    lines.push('_entity.id');
    lines.push('_entity.type');
    for (let idx = 0; idx < chains.length; idx += 1) {
      const firstAtom = a.chain_id.indexOf(chains[idx]);
      const isPoly = a.is_polymer[firstAtom];
      lines.push(`${idx} ${isPoly ? 'polymer' : 'non-polymer'}`);
    }
    lines.push('#');

    const polyChains = chains
      .map((cid, idx) => ({ cid, idx, firstAtom: a.chain_id.indexOf(cid) }))
      .filter((item) => a.is_polymer[item.firstAtom]);
    if (polyChains.length > 0) {
      lines.push('loop_');
      lines.push('_entity_poly.entity_id');
      lines.push('_entity_poly.type');
      lines.push('_entity_poly.pdbx_strand_id');
      for (const item of polyChains) {
        const ct = a.chain_type[item.firstAtom];
        const pt =
          ct === CHAIN_TYPE.DNA
            ? 'polydeoxyribonucleotide'
            : ct === CHAIN_TYPE.RNA
            ? 'polyribonucleotide'
            : ct === CHAIN_TYPE.POLYPEPTIDE_D
            ? "'polypeptide(D)'"
            : "'polypeptide(L)'";
        lines.push(`${item.idx} ${pt} ${item.cid}`);
      }
      lines.push('#');
    }

    // _chem_comp table
    const uniqueComps = [...new Set(a.res_name.map(String))];
    if (uniqueComps.length > 0) {
      lines.push('loop_');
      lines.push('_chem_comp.id');
      lines.push('_chem_comp.name');
      lines.push('_chem_comp.type');
      for (const comp of uniqueComps) {
        const idx = a.res_name.indexOf(comp);
        const cct = fmtCifVal(a.chem_comp_type[idx] || 'NON-POLYMER');
        lines.push(`${fmtCifVal(comp)} ${fmtCifVal(comp)} ${cct}`);
      }
      lines.push('#');
    }

    // Preserve _chem_comp_bond and _chem_comp_atom tables from original CIF block when present
    for (const catName of ['chem_comp_bond', 'chem_comp_atom']) {
      const cat = this.cifBlock?.[catName];
      const cols = cat ? Object.keys(cat) : [];
      const nCatRows = cols.length > 0 ? cat[cols[0]].length : 0;
      if (nCatRows > 0) {
        lines.push('loop_');
        for (const col of cols) lines.push(`_${catName}.${col}`);
        for (let r = 0; r < nCatRows; r += 1) {
          lines.push(cols.map((col) => fmtCifVal(cat[col][r])).join(' '));
        }
        lines.push('#');
      }
    }

    // _struct_conn table for inter-residue or expandable-segment bonds
    const resStarts = getResidueSegments(a);
    const atomToRes = new Int32Array(n);
    for (let r = 0; r < resStarts.length - 1; r += 1) {
      for (let i = resStarts[r]; i < resStarts[r + 1]; i += 1) atomToRes[i] = r;
    }
    const structConnRows = [];
    for (const [u, v, bt] of a.bonds || []) {
      const ru = atomToRes[u];
      const rv = atomToRes[v];
      if (ru === rv) continue;
      const isStdPoly =
        a.chain_id[u] === a.chain_id[v] &&
        Math.abs(a.res_id[u] - a.res_id[v]) === 1 &&
        a.atom_name[u] !== 'UNK' &&
        a.atom_name[v] !== 'UNK';
      if (isStdPoly) continue;
      const ord = bt === BOND_TYPE.DOUBLE ? 'doub' : bt === BOND_TYPE.TRIPLE ? 'trip' : 'sing';
      structConnRows.push(
        `${structConnRows.length + 1} covale ${ord} ${a.chain_id[u]} ${a.chain_id[v]} ${fmtCifVal(a.res_name[u])} ${fmtCifVal(a.res_name[v])} ${a.res_id[u]} ${a.res_id[v]} ${fmtCifVal(a.atom_name[u])} ${fmtCifVal(a.atom_name[v])} ${a.ins_code[u] || '.'} ${a.ins_code[v] || '.'} 1 1`
      );
    }
    if (structConnRows.length > 0) {
      lines.push('loop_');
      lines.push('_struct_conn.id');
      lines.push('_struct_conn.conn_type_id');
      lines.push('_struct_conn.pdbx_value_order');
      lines.push('_struct_conn.ptnr1_label_asym_id');
      lines.push('_struct_conn.ptnr2_label_asym_id');
      lines.push('_struct_conn.ptnr1_label_comp_id');
      lines.push('_struct_conn.ptnr2_label_comp_id');
      lines.push('_struct_conn.ptnr1_label_seq_id');
      lines.push('_struct_conn.ptnr2_label_seq_id');
      lines.push('_struct_conn.ptnr1_label_atom_id');
      lines.push('_struct_conn.ptnr2_label_atom_id');
      lines.push('_struct_conn.pdbx_ptnr1_PDB_ins_code');
      lines.push('_struct_conn.pdbx_ptnr2_PDB_ins_code');
      lines.push('_struct_conn.ptnr1_transformation_id');
      lines.push('_struct_conn.ptnr2_transformation_id');
      lines.push(...structConnRows);
      lines.push('#');
    }

    // Scalar 1-body annotation columns on _atom_site
    const scalarCols = Object.keys(a.annotations || {}).filter(
      (col) =>
        // "__" names are this page's own bookkeeping (a residue's place in its source), never part of the file.
        !col.startsWith('__') &&
        (!NONSCALAR_1BODY_ANNOTATIONS[col] || typeof a.annotations[col]?.[0] === 'string') &&
        Array.isArray(a.annotations[col]) &&
        a.annotations[col].length === n
    );
    if (this.atoms.cathText && this.atoms.cathText.length === n && !scalarCols.includes('condition_cath-label_1_residue')) {
      a.annotations['condition_cath-label_1_residue'] = this.atoms.cathText.slice();
      scalarCols.push('condition_cath-label_1_residue');
    }

    lines.push('loop_');
    const baseCols = [
      '_atom_site.group_PDB',
      '_atom_site.id',
      '_atom_site.type_symbol',
      '_atom_site.label_atom_id',
      '_atom_site.label_alt_id',
      '_atom_site.label_comp_id',
      '_atom_site.label_asym_id',
      '_atom_site.label_entity_id',
      '_atom_site.label_seq_id',
      '_atom_site.pdbx_PDB_ins_code',
      '_atom_site.Cartn_x',
      '_atom_site.Cartn_y',
      '_atom_site.Cartn_z',
      '_atom_site.occupancy',
      '_atom_site.B_iso_or_equiv',
      '_atom_site.pdbx_formal_charge',
      '_atom_site.auth_seq_id',
      '_atom_site.auth_comp_id',
      '_atom_site.auth_asym_id',
      '_atom_site.auth_atom_id',
      '_atom_site.pdbx_PDB_model_num',
      '_atom_site.atom_array_index',
    ];
    for (const col of baseCols) lines.push(col);
    for (const col of scalarCols) lines.push(`_atom_site.${col}`);

    const chainToEntityIdx = new Map(chains.map((cid, idx) => [cid, idx]));
    for (let i = 0; i < n; i += 1) {
      const c = a.coord[i];
      const row = [
        a.hetero[i] ? 'HETATM' : 'ATOM',
        i + 1,
        fmtCifVal(a.element[i] || 'C'),
        fmtCifVal(a.atom_name[i] || 'CA'),
        fmtCifVal(a.label_alt_id[i] || '.'),
        fmtCifVal(a.res_name[i] || 'ALA'),
        fmtCifVal(a.chain_id[i] || 'A'),
        chainToEntityIdx.get(String(a.chain_id[i])) ?? 0,
        a.res_id[i],
        fmtCifVal(a.ins_code[i] || '.'),
        fmtCifVal(c[0]),
        fmtCifVal(c[1]),
        fmtCifVal(c[2]),
        fmtCifVal(a.occupancy[i] ?? 1.0),
        fmtCifVal(a.b_factor[i]),
        a.charge[i] ? String(a.charge[i]) : '?',
        a.auth_seq_id[i] ?? a.res_id[i],
        fmtCifVal(a.res_name[i] || 'ALA'),
        fmtCifVal(a.chain_id[i] || 'A'),
        fmtCifVal(a.atom_name[i] || 'CA'),
        1,
        i,
      ];
      for (const col of scalarCols) {
        row.push(fmtCifVal(a.annotations[col][i]));
      }
      lines.push(row.join(' '));
    }
    lines.push('#');

    // Non-scalar 1-body annotation tables
    for (const [catName, spec] of Object.entries(NONSCALAR_1BODY_ANNOTATIONS)) {
      const arr = a.annotations?.[catName];
      if (!Array.isArray(arr) || arr.length !== n || !Array.isArray(arr[0])) continue;
      lines.push('loop_');
      lines.push(`_${catName}.idx0`);
      for (let c = 0; c < spec.cols; c += 1) {
        lines.push(`_${catName}.val${c}`);
      }
      for (let i = 0; i < n; i += 1) {
        const vals = arr[i].map(fmtCifVal).join(' ');
        lines.push(`${i} ${vals}`);
      }
      lines.push('#');
    }

    return lines.join('\n') + '\n';
  }

  toPdb() {
    const a = this.atoms;
    const records = [];
    for (let i = 0; i < a.nAtoms; i += 1) {
      const c = a.coord[i];
      if (!c || !Number.isFinite(c[0]) || !Number.isFinite(c[1]) || !Number.isFinite(c[2])) continue;
      records.push({
        serial: i + 1,
        groupPdb: a.hetero[i] ? 'HETATM' : 'ATOM',
        atomName: a.atom_name[i] || 'CA',
        resName: a.res_name[i] || 'ALA',
        chainId: a.chain_id[i] || 'A',
        resId: a.res_id[i] ?? 1,
        insCode: a.ins_code[i] && a.ins_code[i] !== '.' ? a.ins_code[i] : ' ',
        element: a.element[i] || 'C',
        x: c[0],
        y: c[1],
        z: c[2],
        occupancy: Number.isFinite(a.occupancy[i]) ? a.occupancy[i] : 1.0,
        bFactor: Number.isFinite(a.b_factor[i]) ? a.b_factor[i] : 0.0,
      });
    }
    return toPdbString(records, { title: this.name || 'AtomWorks Studio Structure' });
  }
}

function buildBackboneRowsFromCa(caNm, tokens, finalResidueType = null, vocabTokens = null) {
  const n = tokens.length;
  const rows = [];
  for (let i = 0; i < n; i += 1) {
    const tok = tokens[i];
    const rt = finalResidueType ? finalResidueType[i] : -1;
    const resName =
      rt >= 0 && vocabTokens && vocabTokens[rt] && vocabTokens[rt] !== '<M>' && vocabTokens[rt] !== '<A>'
        ? vocabTokens[rt]
        : tok.resName || 'ALA';
    const cid = String(tok.chainId || 'A');
    const seqId = Number.isFinite(tok.resId) ? tok.resId : i + 1;
    const cx = (caNm[i * 3 + 0] || 0) * 10.0;
    const cy = (caNm[i * 3 + 1] || 0) * 10.0;
    const cz = (caNm[i * 3 + 2] || 0) * 10.0;

    if (tok.isAtomized) {
      rows.push({
        groupPdb: 'HETATM',
        element: tok.element || 'C',
        atomName: tok.atomName || 'CA',
        resName,
        chainId: cid,
        seqId,
        x: cx,
        y: cy,
        z: cz,
      });
      continue;
    }

    const hasPrev = i > 0 && String(tokens[i - 1].chainId || 'A') === cid && !tokens[i - 1].isAtomized;
    const hasNext = i + 1 < n && String(tokens[i + 1].chainId || 'A') === cid && !tokens[i + 1].isAtomized;
    let ux = 1, uy = 0, uz = 0;
    if (hasPrev) {
      const dx = cx - (caNm[(i - 1) * 3 + 0] || 0) * 10.0;
      const dy = cy - (caNm[(i - 1) * 3 + 1] || 0) * 10.0;
      const dz = cz - (caNm[(i - 1) * 3 + 2] || 0) * 10.0;
      const len = Math.hypot(dx, dy, dz) || 1;
      ux = dx / len; uy = dy / len; uz = dz / len;
    } else if (hasNext) {
      const dx = (caNm[(i + 1) * 3 + 0] || 0) * 10.0 - cx;
      const dy = (caNm[(i + 1) * 3 + 1] || 0) * 10.0 - cy;
      const dz = (caNm[(i + 1) * 3 + 2] || 0) * 10.0 - cz;
      const len = Math.hypot(dx, dy, dz) || 1;
      ux = dx / len; uy = dy / len; uz = dz / len;
    }

    let vx = ux, vy = uy, vz = uz;
    if (hasNext) {
      const dx = (caNm[(i + 1) * 3 + 0] || 0) * 10.0 - cx;
      const dy = (caNm[(i + 1) * 3 + 1] || 0) * 10.0 - cy;
      const dz = (caNm[(i + 1) * 3 + 2] || 0) * 10.0 - cz;
      const len = Math.hypot(dx, dy, dz) || 1;
      vx = dx / len; vy = dy / len; vz = dz / len;
    }

    let tx = ux + vx, ty = uy + vy, tz = uz + vz;
    const tLen = Math.hypot(tx, ty, tz);
    if (tLen > 1e-4) {
      tx /= tLen; ty /= tLen; tz /= tLen;
    } else {
      tx = ux; ty = uy; tz = uz;
    }

    let bx = vx - ux, by = vy - uy, bz = vz - uz;
    const bLen = Math.hypot(bx, by, bz);
    if (bLen > 1e-4) {
      bx /= bLen; by /= bLen; bz /= bLen;
    } else {
      // Orthogonal unit vector to t
      if (Math.abs(tx) < 0.9) {
        bx = 0; by = -tz; bz = ty;
      } else {
        bx = -tz; by = 0; bz = tx;
      }
      const nLen = Math.hypot(bx, by, bz) || 1;
      bx /= nLen; by /= nLen; bz /= nLen;
    }

    let nx = ty * bz - tz * by;
    let ny = tz * bx - tx * bz;
    let nz = tx * by - ty * bx;
    const nLen = Math.hypot(nx, ny, nz) || 1;
    nx /= nLen; ny /= nLen; nz /= nLen;

    const cAtomX = cx + 1.25 * vx - 0.87 * bx;
    const cAtomY = cy + 1.25 * vy - 0.87 * by;
    const cAtomZ = cz + 1.25 * vz - 0.87 * bz;

    rows.push(
      {
        groupPdb: 'ATOM',
        element: 'N',
        atomName: 'N',
        resName,
        chainId: cid,
        seqId,
        x: cx - 1.20 * ux - 0.82 * bx,
        y: cy - 1.20 * uy - 0.82 * by,
        z: cz - 1.20 * uz - 0.82 * bz,
      },
      {
        groupPdb: 'ATOM',
        element: 'C',
        atomName: 'CA',
        resName,
        chainId: cid,
        seqId,
        x: cx,
        y: cy,
        z: cz,
      },
      {
        groupPdb: 'ATOM',
        element: 'C',
        atomName: 'C',
        resName,
        chainId: cid,
        seqId,
        x: cAtomX,
        y: cAtomY,
        z: cAtomZ,
      },
      {
        groupPdb: 'ATOM',
        element: 'O',
        atomName: 'O',
        resName,
        chainId: cid,
        seqId,
        x: cAtomX + 1.05 * bx + 0.64 * nx,
        y: cAtomY + 1.05 * by + 0.64 * ny,
        z: cAtomZ + 1.05 * bz + 0.64 * nz,
      },
    );
  }
  return rows;
}

const SS_PROTEIN_RESIDUES = new Set([
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'MSE', 'UNK',
]);

/**
 * `_struct_conf` / `_struct_sheet_range` loops from the Cα trace alone, using py2Dmol's assignment (backbone predicted
 * from Cα, then DSSP hydrogen-bond rules; `web/vendor/py2dmol-ss.js`). Mid-diffusion frames have a nearly final Cα
 * trace but unconverged N/O, so Mol*'s own DSSP finds no helices; file-provided secondary structure takes precedence.
 * @param {Array<{atomName: string, resName: string, chainId: string, resId: number, x: number, y: number, z: number}>} atoms (Å)
 */
export function caSecondaryStructureCif(atoms) {
  const ca = atoms.filter((a) => a.atomName === 'CA' && SS_PROTEIN_RESIDUES.has(a.resName));
  const n = ca.length;
  if (n < 5) return '';
  // Chain-contiguous runs (break on chain change or a Cα gap): residues may only be sequence neighbours within one.
  const seg = new Int32Array(n);
  for (let i = 1; i < n; i += 1) {
    const gap = Math.hypot(ca[i].x - ca[i - 1].x, ca[i].y - ca[i - 1].y, ca[i].z - ca[i - 1].z);
    seg[i] = seg[i - 1] + (ca[i].chainId !== ca[i - 1].chainId || gap > 4.2 ? 1 : 0);
  }
  const { sec } = assignSecondary(ca, n, null, { names: ca.map((a) => a.resName), links: seg });
  const helix = Uint8Array.from(sec, (c) => (c === 'H' ? 1 : 0));
  const paired = Uint8Array.from(sec, (c) => (c === 'E' ? 1 : 0));
  const runs = (flags, minLen) => {
    const out = [];
    for (let i = 0; i < n; ) {
      if (!flags[i]) { i += 1; continue; }
      let j = i;
      while (j + 1 < n && flags[j + 1] && seg[j + 1] === seg[i]) j += 1;
      if (j - i + 1 >= minLen) out.push([i, j]);
      i = j + 1;
    }
    return out;
  };
  const helices = runs(helix, 1);
  const strands = runs(paired, 1);
  const ends = (i, j) => [ca[i], ca[j]].flatMap((a) => [fmtCifVal(a.resName), fmtCifVal(a.chainId), a.resId]);
  const lines = [];
  if (helices.length) {
    lines.push('loop_', ...['conf_type_id', 'id', 'beg_label_comp_id', 'beg_label_asym_id', 'beg_label_seq_id',
      'end_label_comp_id', 'end_label_asym_id', 'end_label_seq_id', 'beg_auth_comp_id', 'beg_auth_asym_id',
      'beg_auth_seq_id', 'end_auth_comp_id', 'end_auth_asym_id', 'end_auth_seq_id', 'pdbx_PDB_helix_class']
      .map((f) => `_struct_conf.${f}`));
    helices.forEach(([i, j], k) => {
      const e = ends(i, j);
      lines.push(['HELX_P', `HELX_P${k + 1}`, ...e, ...e, 1].join(' '));
    });
    lines.push('#');
  }
  if (strands.length) {
    lines.push('loop_', ...['sheet_id', 'id', 'beg_label_comp_id', 'beg_label_asym_id', 'beg_label_seq_id',
      'end_label_comp_id', 'end_label_asym_id', 'end_label_seq_id', 'beg_auth_comp_id', 'beg_auth_asym_id',
      'beg_auth_seq_id', 'end_auth_comp_id', 'end_auth_asym_id', 'end_auth_seq_id']
      .map((f) => `_struct_sheet_range.${f}`));
    strands.forEach(([i, j], k) => {
      const e = ends(i, j);
      lines.push(['A', k + 1, ...e, ...e].join(' '));
    });
    lines.push('#');
  }
  return lines.length ? lines.join('\n') + '\n' : '';
}

/**
 * Build a single-model mmCIF string from a denoised `caCoordsNm` frame for live diffusion rendering.
 */
export function buildLiveFrameCifFromCa(caCoordsNm, tokens, finalResidueType = null, vocabTokens = null) {
  if (!caCoordsNm || !Array.isArray(tokens) || tokens.length === 0) return null;
  const chains = [...new Set(tokens.map((t) => String(t.chainId || 'A')))];
  const chainToEntity = new Map(chains.map((c, idx) => [c, idx + 1]));
  const lines = [
    'data_live_frame',
    '#',
    'loop_',
    '_entity.id',
    '_entity.type',
    ...chains.map((_, idx) => `${idx + 1} polymer`),
    '#',
    'loop_',
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.type_symbol',
    '_atom_site.label_atom_id',
    '_atom_site.label_alt_id',
    '_atom_site.label_comp_id',
    '_atom_site.label_asym_id',
    '_atom_site.label_entity_id',
    '_atom_site.label_seq_id',
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.auth_seq_id',
    '_atom_site.auth_comp_id',
    '_atom_site.auth_asym_id',
    '_atom_site.auth_atom_id',
    '_atom_site.pdbx_PDB_model_num',
  ];
  const rows = buildBackboneRowsFromCa(caCoordsNm, tokens, finalResidueType, vocabTokens);
  let atomId = 1;
  for (const a of rows) {
    const eid = chainToEntity.get(a.chainId) ?? 1;
    lines.push(
      `${a.groupPdb} ${atomId++} ${fmtCifVal(a.element)} ${fmtCifVal(a.atomName)} . ${fmtCifVal(a.resName)} ${fmtCifVal(a.chainId)} ${eid} ${a.seqId} ? ${a.x.toFixed(3)} ${a.y.toFixed(3)} ${a.z.toFixed(3)} 1.00 80.00 ${a.seqId} ${fmtCifVal(a.resName)} ${fmtCifVal(a.chainId)} ${fmtCifVal(a.atomName)} 1`
    );
  }
  lines.push('#');
  return lines.join('\n') + '\n' + caSecondaryStructureCif(rows.map((r) => ({ ...r, resId: r.seqId })));
}

/**
 * Build a multi-model Cα/backbone trajectory mmCIF string from recorded flow-matching frames
 * (`trajectoryFrames[m].caCoordsNm`) so AtomWorks Studio's `TrajectoryController` (`#trajectory-bar`)
 * can scrub and play back the predicted clean structures in Mol* WebGL or Patinae WebGPU.
 */
export function buildTrajectoryCifFromFrames(trajectoryFrames, tokens, finalResidueType = null, vocabTokens = null) {
  if (!Array.isArray(trajectoryFrames) || trajectoryFrames.length < 2 || !Array.isArray(tokens) || tokens.length === 0) {
    return null;
  }
  const maxFrames = 30;
  const stride = Math.max(1, Math.floor(trajectoryFrames.length / maxFrames));
  const sampled = [];
  for (let i = 0; i < trajectoryFrames.length; i += stride) {
    sampled.push(trajectoryFrames[i]);
  }
  if (sampled[sampled.length - 1] !== trajectoryFrames[trajectoryFrames.length - 1]) {
    sampled.push(trajectoryFrames[trajectoryFrames.length - 1]);
  }

  const chains = [...new Set(tokens.map((t) => String(t.chainId || 'A')))];
  const chainToEntity = new Map(chains.map((c, idx) => [c, idx + 1]));
  const lines = [
    'data_trajectory',
    '#',
    'loop_',
    '_entity.id',
    '_entity.type',
    ...chains.map((_, idx) => `${idx + 1} polymer`),
    '#',
    'loop_',
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.type_symbol',
    '_atom_site.label_atom_id',
    '_atom_site.label_alt_id',
    '_atom_site.label_comp_id',
    '_atom_site.label_asym_id',
    '_atom_site.label_entity_id',
    '_atom_site.label_seq_id',
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.auth_seq_id',
    '_atom_site.auth_comp_id',
    '_atom_site.auth_asym_id',
    '_atom_site.auth_atom_id',
    '_atom_site.pdbx_PDB_model_num',
  ];

  let atomId = 1;
  for (let m = 0; m < sampled.length; m += 1) {
    const modelNum = m + 1;
    const rows = buildBackboneRowsFromCa(sampled[m].caCoordsNm, tokens, finalResidueType, vocabTokens);
    for (const a of rows) {
      const eid = chainToEntity.get(a.chainId) ?? 1;
      lines.push(
        `${a.groupPdb} ${atomId++} ${fmtCifVal(a.element)} ${fmtCifVal(a.atomName)} . ${fmtCifVal(a.resName)} ${fmtCifVal(a.chainId)} ${eid} ${a.seqId} ? ${a.x.toFixed(3)} ${a.y.toFixed(3)} ${a.z.toFixed(3)} 1.00 80.00 ${a.seqId} ${fmtCifVal(a.resName)} ${fmtCifVal(a.chainId)} ${fmtCifVal(a.atomName)} ${modelNum}`
      );
    }
  }
  lines.push('#');
  return lines.join('\n') + '\n';
}

/**
 * Shared in-browser Studio store (kept on the top window, so a test harness that mounts the page in an iframe
 * still sees one store).
 */
export function getSharedStudioStore() {
  // Sessions live in memory for this page load, so anything Studio remembered from a previous one is already gone.
  // Clearing it here keeps Studio from reporting a lost document on every reload.
  if (!globalThis.__rfd4pStudioStore) {
    try {
      sessionStorage.removeItem('annotation-studio-session');
      sessionStorage.removeItem('annotation-studio-name');
    } catch { /* private window */ }
  }
  let rootWin = globalThis;
  try {
    if (globalThis.top && globalThis.top.location?.origin === globalThis.location?.origin) {
      rootWin = globalThis.top;
    }
  } catch {
    rootWin = globalThis;
  }
  if (!rootWin.__rfd4pStudioStore) {
    rootWin.__rfd4pStudioStore = {
      sessions: new Map(),
      schema: null,
      examples: null,
      exampleCifs: new Map(),
      counter: 0,
      createSession(cifText, name = 'structure.cif', opts = {}) {
        this.counter += 1;
        const sessionId = opts.sessionId || `studio_session_${String(this.counter).padStart(4, '0')}_${Date.now().toString(36)}`;
        const doc = new StudioDoc(cifText, name, opts);
        this.sessions.set(sessionId, doc);
        return { sessionId, doc };
      },
      getSession(sessionId) {
        return this.sessions.get(sessionId) || null;
      },
    };
  }
  return rootWin.__rfd4pStudioStore;
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function textResponse(text, status = 200, contentType = 'text/plain; charset=utf-8') {
  return new Response(text, {
    status,
    headers: { 'Content-Type': contentType },
  });
}

export async function handleStudioApiRequest(urlObj, init = {}, nativeFetch = globalThis.fetch) {
  const store = getSharedStudioStore();
  const method = (init.method || 'GET').toUpperCase();
  const path = urlObj.pathname.replace(/^\/api\/(?!v1\/)/, '/api/v1/');

  try {
    if (path === '/api/v1/health' || path === '/api/v1/ready') {
      return jsonResponse({ status: 'ok' });
    }

    if (path === '/api/v1/schema' && method === 'GET') {
      if (!store.schema) {
        const resp = await nativeFetch(new URL('./studio/schema.json', import.meta.url));
        store.schema = await resp.json();
      }
      return jsonResponse(store.schema);
    }

    if (path === '/api/v1/examples' && method === 'GET') {
      if (!store.examples) {
        const resp = await nativeFetch(new URL('./studio/examples/examples.json', import.meta.url));
        store.examples = await resp.json();
      }
      return jsonResponse(store.examples);
    }

    const exMatch = path.match(/^\/api\/v1\/examples\/([^/]+)\/cif$/);
    if (exMatch && method === 'GET') {
      const exId = decodeURIComponent(exMatch[1]);
      if (!store.examples) {
        const resp = await nativeFetch(new URL('./studio/examples/examples.json', import.meta.url));
        store.examples = await resp.json();
      }
      const ex = store.examples.find((e) => e.id === exId);
      if (!ex) return jsonResponse({ detail: `Unknown example: ${exId}` }, 404);
      if (!store.exampleCifs.has(ex.file)) {
        const resp = await nativeFetch(new URL(`./studio/examples/${ex.file}`, import.meta.url));
        store.exampleCifs.set(ex.file, await resp.text());
      }
      return textResponse(store.exampleCifs.get(ex.file), 200, 'chemical/x-mmcif');
    }

    if (path === '/api/v1/import/inspect' && method === 'POST') {
      const form = init.body instanceof FormData ? init.body : null;
      const file = form?.get('file');
      const text = file && typeof file.text === 'function' ? await file.text() : String(file || '');
      const isAnnotated = text.includes('condition_') || text.includes('annotation_expseg') || text.includes('mask_');
      return jsonResponse({
        preset: isAnnotated ? 'minimal' : 'rcsb',
        message: isAnnotated
          ? 'Detected saved AtomWorks annotations; using Annotated CIF preset.'
          : 'Unannotated structure; using RCSB preparation preset.',
      });
    }

    if (path === '/api/v1/sessions' && method === 'POST') {
      const form = init.body instanceof FormData ? init.body : null;
      const file = form?.get('file');
      const preset = form?.get('parse_preset') || form?.get('preset') || 'minimal';
      const rawOptions = form?.get('advanced_options') || form?.get('options');
      const parsedOptions = rawOptions ? JSON.parse(String(rawOptions)) : {};
      const name = ((file && file.name) || 'structure.cif').replace(/\.(cif|mmcif|pdb|ent|zst)+$/i, '');
      const text = file && typeof file.text === 'function' ? await file.text() : String(file || '');
      const sessionOpts = {
        preset,
        assembly: String(form?.get('assembly') ?? parsedOptions.assembly ?? '1'),
        loadStandardAnnotations: parsedOptions.load_standard_annotations,
      };
      const { sessionId } = store.createSession(text, name, sessionOpts);
      const doc = await createStudioDocWithCcd(text, name, sessionOpts, nativeFetch);
      store.sessions.set(sessionId, doc);
      store.activeSessionId = sessionId;
      return jsonResponse({
        session_id: sessionId,
        editor_path: `/?session=${encodeURIComponent(sessionId)}&embed=1`,
        structure: doc.summary(),
      });
    }

    const sessionMatch = path.match(/^\/api\/v1\/sessions\/([^/]+)(\/.*)?$/);
    if (sessionMatch) {
      const sessionId = decodeURIComponent(sessionMatch[1]);
      const subpath = sessionMatch[2] || '';
      const doc = store.getSession(sessionId);
      if (!doc) return jsonResponse({ detail: `Session not found: ${sessionId}` }, 404);
      store.activeSessionId = sessionId;

      if (subpath === '' && method === 'GET') {
        return jsonResponse({
          session_id: sessionId,
          editor_path: `/?session=${encodeURIComponent(sessionId)}&embed=1`,
          structure: doc.summary(),
        });
      }
      if (subpath === '' && method === 'DELETE') {
        store.sessions.delete(sessionId);
        return new Response(null, { status: 204 });
      }
      if (subpath === '/cif' && method === 'GET') {
        return textResponse(doc.toCif(), 200, 'chemical/x-mmcif');
      }
      if (subpath === '/pdb' && method === 'GET') {
        return textResponse(doc.toPdb(), 200, 'chemical/x-pdb');
      }
      if (subpath === '/trajectory.cif' && method === 'GET') {
        if (!doc.trajectoryCif) return jsonResponse({ detail: 'No trajectory available' }, 404);
        return textResponse(doc.trajectoryCif, 200, 'chemical/x-mmcif');
      }
      if (subpath === '/export' && method === 'POST') {
        return jsonResponse({
          session_id: sessionId,
          name: doc.name.replace(/\.(cif|mmcif)$/i, ''),
          revision: doc.revision,
          cif: doc.toCif(),
          issues: doc.validate(true),
        });
      }
      if (subpath === '/validate' && method === 'POST') {
        return jsonResponse({ issues: doc.validate(true) });
      }
      if (subpath === '/distance-restraints' && method === 'GET') {
        const restraints = doc.distanceRestraints();
        return jsonResponse({ restraints, count: restraints.length });
      }

      const annMatch = subpath.match(/^\/annotations\/([^/]+)$/);
      if (annMatch && method === 'GET') {
        const cond = decodeURIComponent(annMatch[1]);
        const details = doc.annotationDetails(cond);
        return jsonResponse({ annotation: details[0] ?? null });
      }

      const fieldMatch = subpath.match(/^\/annotation-fields\/([^/]+)$/);
      if (fieldMatch && method === 'GET') {
        const fname = decodeURIComponent(fieldMatch[1]);
        const fields = doc.storedAnnotationFields(fname);
        return jsonResponse({ annotation: fields[0] ?? null, field: fields[0] ?? null });
      }

      const scalarMatch = subpath.match(/^\/scalar-fields\/([^/]+)$/);
      if (scalarMatch && method === 'GET') {
        const sname = decodeURIComponent(scalarMatch[1]);
        const fields = doc.scalarAnnotationFields(sname);
        return jsonResponse({ annotation: fields[0] ?? null, field: fields[0] ?? null });
      }

      const atomAnnMatch = subpath.match(/^\/atoms\/(\d+)\/annotations$/);
      if (atomAnnMatch && method === 'GET') {
        return jsonResponse(doc.atomAnnotationDetails(Number(atomAnnMatch[1])));
      }

      if (subpath === '/atom-distance' && method === 'GET') {
        const a1 = Number(urlObj.searchParams.get('atom1'));
        const a2 = Number(urlObj.searchParams.get('atom2'));
        return jsonResponse({
          revision: doc.revision,
          distance: doc.atomDistance(a1, a2),
          atom1_label: doc.atomLabel(a1),
          atom2_label: doc.atomLabel(a2),
        });
      }

      const body = init.body && typeof init.body === 'string' ? JSON.parse(init.body) : {};

      if (subpath === '/selections/resolve' && method === 'POST') {
        const indices = doc.resolveIndices(body.atom_indices || [], body.scope || 'atom');
        return jsonResponse({ atom_indices: indices, count: indices.length });
      }
      if (subpath === '/selections/query' && method === 'POST') {
        const indices = doc.queryIndices(body.expression || '', body.syntax || 'expression', body.scope || 'atom');
        return jsonResponse({ atom_indices: indices, count: indices.length });
      }
      if (subpath === '/interactions/analyze' && method === 'POST') {
        return jsonResponse(doc.analyzeInteractions(body));
      }
      if (subpath === '/bonds/inspect' && method === 'POST') {
        const bonds = doc.inspectBonds(body.atom_indices);
        return jsonResponse({ bonds, count: bonds.length });
      }
      if (subpath === '/bonds' && method === 'GET') {
        const atomIdxParam = urlObj.searchParams.get('atom_index');
        const filterIndices =
          atomIdxParam !== null && atomIdxParam !== '' && Number.isFinite(Number(atomIdxParam))
            ? [Number(atomIdxParam)]
            : null;
        const bonds = doc.inspectBonds(filterIndices);
        return jsonResponse({ bonds, count: bonds.length });
      }
      if (subpath === '/conditions/batch' && method === 'POST') {
        const editResult = doc.setConditions(body.edits);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/conditions' && method === 'POST') {
        const editResult = doc.setCondition(body.condition, body.atom_indices, {
          enabled: body.enabled ?? true,
          value: body.value ?? null,
          scope: body.scope || 'atom',
        });
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/internal-coords' && method === 'POST') {
        const editResult = doc.setInternalCoordinateConditioning(body.atom_indices, body.enabled ?? true, body.scope || 'atom', body.group_id ?? null);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/internal-coords/clear-targets' && method === 'POST') {
        const editResult = doc.clearInternalCoordinateTargets(body.atom_indices, body.scope || 'atom');
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/groups' && method === 'POST') {
        const editResult = doc.editConditionGroup(body.condition, body.action, body.atom_indices || [], {
          groupId: body.group_id,
          scope: body.scope || 'atom',
        });
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/centroids' && method === 'POST') {
        const editResult = doc.editChainCentroid(body.condition || 'chain_centroid', body.chain, body.value);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/distance-restraints' && method === 'POST') {
        const editResult = doc.editDistanceRestraint(body.atom1, body.atom2, body.action || 'set', body.distance);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/distance-restraints/replace' && method === 'POST') {
        const editResult = doc.replaceDistanceRestraints(body.pairs || [], body.values || []);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/conditions/reset' && method === 'POST') {
        const editResult = doc.resetConditions();
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/charges' && method === 'POST') {
        const editResult = doc.setCharge(body.atom_indices || [], { value: body.value, delta: body.delta });
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/chains/rebuild' && method === 'POST') {
        const editResult = doc.rebuildChain(body.chain_id, body.parts || []);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/residues/restore' && method === 'POST') {
        const editResult = doc.restoreFromSource(body.chain_id, body.res_ids || [],
          Number.isInteger(body.at_atom) ? { atom: body.at_atom, before: Boolean(body.before) } : null);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/segments/restore' && method === 'POST') {
        const editResult = doc.restoreSegment(body.atom_index);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/residues/insert' && method === 'POST') {
        const other = store.sessions.get(body.from_session_id);
        if (!other) return jsonResponse({ detail: 'no such session to take residues from' }, 404);
        const editResult = doc.insertResiduesFrom(other, body.chain_id, body.source_chain,
          Number.isInteger(body.at_atom) ? { atom: body.at_atom, before: Boolean(body.before) } : null);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/coordinates/clear' && method === 'POST') {
        const editResult = doc.clearCoordinates(body.atom_indices || [], body.lead || []);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/chains/order' && method === 'POST') {
        const editResult = doc.reorderChains(body.chain_ids || []);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/chains' && method === 'POST') {
        const editResult = doc.assignChain(body.atom_indices || [], body.chain_id);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/bonds' && method === 'POST') {
        const editResult = doc.editBond(body.atom1, body.atom2, body.action, body.bond_type || 'SINGLE');
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/bonds/infer-ccd' && method === 'POST') {
        const editResult = doc.inferBondsFromCcd();
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/delete' && method === 'POST') {
        const editResult = body.close_residue_gaps ? doc.deleteAtoms(body.atom_indices || [], true) : doc.removeAtoms(body.atom_indices || []);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/reset-residue-ids' && method === 'POST') {
        const editResult = doc.resetResidueIds();
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/segments' && method === 'POST') {
        const editResult = doc.insertSegment({
          minimum: body.minimum,
          maximum: body.maximum,
          atomIndex: body.atom_index,
          before: body.before ?? true,
          newChain: body.new_chain ?? false,
          chainId: body.chain_id,
          chainType: body.chain_type,
        });
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/segments/scaffold' && method === 'POST') {
        const editResult = doc.scaffoldMotif(body.atom_indices, {
          spread: body.spread ?? 0.25,
          sidechainsOnly: Boolean(body.sidechains_only),
          unindexed: Boolean(body.unindexed), ordered: Boolean(body.ordered),
          totalMin: body.total_min ?? null,
          totalMax: body.total_max ?? null,
        });
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/segments/replace' && method === 'POST') {
        const editResult = doc.replaceWithSegments(body.atom_indices, body.minimum, body.maximum);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/segments/ligand-binder' && method === 'POST') {
        const editResult = doc.designAroundLigand({
          removeProtein: body.remove_protein ?? true,
          poseFixed: body.pose_fixed ?? true,
          minimum: body.minimum ?? 100,
          maximum: body.maximum ?? 150,
        });
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/segments/range' && method === 'POST') {
        const editResult = doc.setSegmentRange(body.atom_index, body.minimum, body.maximum);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/segments/realize' && method === 'POST') {
        const editResult = doc.realizeSegments(body.seed);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/segments/collapse' && method === 'POST') {
        const editResult = doc.collapseSegments();
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/combine' && method === 'POST') {
        const otherDocs = (body.session_ids || [])
          .map((sid) => store.getSession(sid))
          .filter(Boolean);
        const editResult = doc.combine(otherDocs);
        return jsonResponse({ session_id: sessionId, structure: doc.summary(), edit_result: editResult });
      }
      if (subpath === '/undo' && method === 'POST') {
        const ok = doc.undo();
        if (!ok) return jsonResponse({ detail: 'undo history is empty' }, 409);
        return jsonResponse({
          session_id: sessionId,
          structure: doc.summary(),
          edit_result: { changed: true, message: 'Undid last edit.' },
        });
      }
      if (subpath === '/protonate' && method === 'POST') {
        const editResult = doc.protonate(body.ph ?? 7.4);
        return jsonResponse({
          session_id: sessionId,
          structure: doc.summary(),
          edit_result: editResult,
        });
      }
    }

    return jsonResponse({ detail: `Unhandled Studio API route: ${method} ${path}` }, 404);
  } catch (err) {
    return jsonResponse({ detail: err?.message || String(err) }, 422);
  }
}

export function installStudioApiInterceptor() {
  if (globalThis.__rfd4pStudioFetchInstalled) return;
  globalThis.__rfd4pStudioFetchInstalled = true;

  // Transparently support .pdb / .ent and .zst files in Studio's client-side WASM `inspect_cif(await file.text())`
  if (typeof File !== 'undefined' && File.prototype && !File.prototype.__rfd4pTextWrapped) {
    const nativeBlobText = Blob.prototype.text;
    const nativeArrayBuffer = Blob.prototype.arrayBuffer;
    Object.defineProperty(File.prototype, '__rfd4pTextWrapped', { value: true });
    File.prototype.text = async function wrappedFileText() {
      const fname = (this.name || '').toLowerCase();
      let rawText;
      if (fname.endsWith('.zst')) {
        const bytes = new Uint8Array(await nativeArrayBuffer.call(this));
        rawText = decodeTextOrZstd(bytes);
      } else {
        rawText = await nativeBlobText.call(this);
      }
      if (
        !rawText.includes('_atom_site.') &&
        (fname.endsWith('.pdb') ||
          fname.endsWith('.ent') ||
          /^\s*(?:ATOM  |HETATM|HEADER|TITLE |REMARK|CRYST1|MODEL )/m.test(rawText))
      ) {
        const stem = (this.name || 'structure').replace(/\.(pdb|ent|zst)+$/i, '');
        return new StudioDoc(rawText, stem).toCif();
      }
      return rawText;
    };
  }

  const nativeFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (input, init) => {
    try {
      const rawUrl = typeof input === 'string' ? input : input instanceof URL ? input.href : input?.url || '';
      const urlObj = new URL(rawUrl, globalThis.location?.href || 'http://localhost/');
      if (urlObj.origin === (globalThis.location?.origin || 'http://localhost') && urlObj.pathname.startsWith('/api/')) {
        return await handleStudioApiRequest(urlObj, init, nativeFetch);
      }
    } catch {
      // Fall through to native fetch
    }
    return nativeFetch(input, init);
  };
}

if (typeof window !== 'undefined') {
  installStudioApiInterceptor();
}
