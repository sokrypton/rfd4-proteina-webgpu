/**
 * The RFD4-Proteina card's layer over AtomWorks Studio: a "what do you want to design?" start screen, a Design
 * panel that states what the open document will generate (task, chain roles, lengths, contig) and edits it, and the
 * simple/advanced switch that hides Studio's expert tools.
 *
 * Every edit goes through Studio's own mutate path (`window.atomworksStudio?.applyEdit`), so Undo, the viewer and the
 * `atomworks-studio` "changed" event behave exactly as for edits made in the Studio panels.
 */
import { INTERACTIONS, designSpec, formatResidueList, parseResidueList } from './design-spec.js';
import { StudioDoc, createStudioDocWithCcd, getSharedStudioStore, handleStudioApiRequest } from './studio-api.js';
import { isCcdCode, ligandCifFromCcd } from './ligand-ccd.js';
import { fetchCcdComponents } from '../shared/rfd4p/pipeline/ccd.js';

const ADVANCED_KEY = 'rfd4p-advanced-editor';

/**
 * Start-screen tasks. Each with a `recipe` is built from its PDB entry with the card's own controls, out of sight
 * (`_buildRecipe`), and shown finished: the set-up a user would make, captioned with its entry. Each matches the bundled
 * Studio example of the same name (`example`, web/studio/examples) model input for model input: the same entries,
 * residues, held atoms, chain names and order. The two 2KL8 examples are the exception: the bundled file lacks residues
 * 8–27 (a bug in it), so they keep the whole protein and drop only the His tag. If an entry cannot be fetched the
 * example does not open (the page says why); there is no stand-in.
 * Steps: `fetch` (ID and chains), `start` (a bundled Studio example instead, no download), `add` (another entry's chains,
 * as + pdb), `piece` (open the piece holding `at`, fill `set`, `from`/`to` its Residues, Apply), `pick` (residues picked
 * as in the viewer, "H99-109", and that popup's `set`: structure, sequence, length), `removePiece` (× the piece holding
 * `at`), `removeNew` (× the new residues left in a chain), `addSeq`,
 * `newLength` (the chain's `k`-th new piece), `ligand` (pose, held atoms), `chain` (its popup: `name`, `order` from 1),
 * `blocks` (a new chain laid out as blocks, [length, shape] each: "+" after the last, then each block's popup),
 * `partial`.
 */
const TASKS = [
  { example: 'unconditional', title: 'Protein from scratch', desc: 'A new fold of the length you choose' },
  {
    // The blocks idea: a ferredoxin-like fold drawn as strands, helices and loops, each its own block of new residues.
    example: 'blueprint', title: 'Blueprint', desc: 'A fold drawn as blocks: E-H-E-E-H-E',
    recipe: [
      { start: 'unconditional' },
      { blocks: [[6, 'E'], [3, 'C'], [14, 'H'], [3, 'C'], [6, 'E'], [3, 'C'],
        [6, 'E'], [3, 'C'], [14, 'H'], [3, 'C'], [6, 'E']] },
    ],
  },
  {
    example: 'protein_binder', title: 'Protein binder', desc: 'Bind PD-L1 (PDB 5O45)',
    recipe: [
      { fetch: '5O45 A' },
      { piece: 'A', at: 10, set: { hotspots: 'A40(CG,OH), A99(CG,SD), A107(CD2,OH)', keepOnly: 'A1-115' } },
      { removeNew: 'A' },
      { addSeq: true },
      { newLength: [50, 120] },
    ],
  },
  {
    example: 'small_molecule_binder', title: 'Small-molecule binder', desc: 'A pocket around SAM (PDB 1XVA)',
    recipe: [
      { fetch: '1XVA D_1' },
      { addSeq: true },
      { newLength: [150, 150] },
      { ligand: 'SAM', pose: 'free' },
      { chain: 'D_1', name: 'B' },
      { chain: 'A', order: 1 },
    ],
  },
  {
    example: 'motif_scaffolding', title: 'Motif scaffolding', desc: 'Keep three residues, build the rest (PDB 1QJG)',
    recipe: [
      { fetch: '1QJG A' },
      { piece: 'A', at: 10, set: { keepOnly: 'A38, A14, A99' } },
      { newLength: [10, 20], chain: 'A', k: 0 },
      { newLength: [15, 30], chain: 'A', k: 1 },
      { newLength: [15, 30], chain: 'A', k: 2 },
      { newLength: [10, 20], chain: 'A', k: 3 },
    ],
  },
  {
    example: 'enzyme_design', title: 'Enzyme', desc: 'A catalytic Asp around Zn and orotate (PDB 1J79)',
    recipe: [
      { fetch: '1J79 A C E' },
      { piece: 'A', at: 10, set: { atoms: 'CG, OD1', position: 'anywhere', keepOnly: 'A250' } },
      { newLength: [180, 180], chain: 'A' },
      { ligand: 'ORO', pose: 'fixed', atoms: 'N1, C2, N3, C4' },
      { chain: 'C', name: 'B' },
      { chain: 'E', name: 'C' },
      { chain: 'C', order: 2 },
    ],
  },
  {
    example: 'antibody_design', title: 'Antibody loop', desc: 'Redesign CDR H3 against HER2 (PDB 1N8Z)',
    recipe: [
      { fetch: '1N8Z A B C' },
      { chain: 'A', name: 'L' },
      { chain: 'B', name: 'H' },
      { chain: 'C', name: 'T' },
      { piece: 'H', at: 10, set: { structure: 'rigid', keepOnly: 'H1-120' } },
      { removeNew: 'H' },
      { piece: 'L', at: 10, set: { structure: 'rigid', keepOnly: 'L1-107' } },
      { removeNew: 'L' },
      { pick: 'H99-109', set: { structure: 'designed', sequence: 'designed' } },
      { piece: 'T', at: 550, set: { keepOnly: 'T545-580, T591-607', hotspots: 'T557-573, T591-605' } },
      { removeNew: 'T' },
      // The frameworks' fold, as upstream's antibody examples give it (Ig-like, CATH 2.60.40).
      { chain: 'H', fold: '2.60.40' },
      { chain: 'L', fold: '2.60.40' },
    ],
  },
  {
    example: 'nanobody', title: 'Nanobody (VHH)', desc: 'Design CDRs against PD-L1 (PDB 4Z18, 3EAK)',
    recipe: [
      { fetch: '4Z18 A' },
      { piece: 'A', at: 10, set: { structure: 'rigid', atoms: 'backbone', keepOnly: 'A2-114', hotspots: 'A39(CZ), A98(CE), A106(CZ)' } },
      { removeNew: 'A' },
      { add: '3EAK B' },
      { chain: 'B', name: 'H' },
      { piece: 'H', at: 10, set: { structure: 'rigid', atoms: 'backbone', keepOnly: 'H1-127' } },
      { removeNew: 'H' },
      { pick: 'H23-38', set: { structure: 'designed', sequence: 'designed', length: [11, 16] } },
      { pick: 'H53-62', set: { structure: 'designed', sequence: 'designed', length: [8, 12] } },
      { pick: 'H100-117', set: { structure: 'designed', sequence: 'designed', length: [8, 22] } },
    ],
  },
  {
    example: 'inverse_folding', title: 'Redesign a sequence', desc: 'New sequence for a fixed backbone (PDB 2KL8)', then: 'sequence',
    recipe: [
      { fetch: '2KL8' },
      { piece: 'A', at: 10, set: { from: 1, to: 79 } },
      { piece: 'A', at: 10, set: { sequence: 'designed' } },
    ],
  },
  {
    example: 'partial_diffusion', title: 'Diversify a structure', desc: 'Variants close to a starting fold (PDB 2KL8)',
    recipe: [
      { fetch: '2KL8' },
      { piece: 'A', at: 10, set: { from: 1, to: 79 } },
      { partial: true },
    ],
  },
];

const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
};

/** Hotspots' one colour, wherever they are shown: sequence bar, viewer, chain bar, chip (styles.css `--rfd4p-hotspot`). */
const HOTSPOT_COLOR = 0xf59e0b;

/** How faded residues the model rebuilds are in the input structure (0 solid, 1 gone). */
const FREE_TRANSPARENCY = 0.65;

// The three properties of residues, the same in a sub-card's popup and the selection's: each fixed or free (the model
// decides), and a structure may also be rigid (kept, moving as one body).
const STRUCTURE_OPTIONS = [['fixed', 'fixed'], ['rigid', 'rigid'], ['designed', 'free']];
// Which atoms a fixed structure holds; "named" takes atom names typed beside it (CG, OD1).
const ATOM_OPTIONS = [['all', 'all'], ['backbone', 'backbone'], ['sidechain', 'side chains'], ['named', 'named']];
const SEQUENCE_OPTIONS = [['fixed', 'fixed'], ['designed', 'free']];
const POSITION_OPTIONS = [['fixed', 'fixed'], ['anywhere', 'free']];
// What else can be asked of residues: their secondary structure, how exposed they are, an interaction to make. Every
// list ends with "free", the model decides, as the three properties do.
// Secondary structure as DSSP's letters, H helix, E strand, C coil (loop): a card reads "6 · E · new", the popup says both.
const SHAPE_OPTIONS = [['1', 'H helix'], ['2', 'E strand'], ['3', 'C loop'], ['0', 'free']];
const SHAPE_LETTER = { 1: 'H', 2: 'E', 3: 'C' };

/** A minimal mmCIF of chain `chain`'s residues numbered `nums` in `atoms` (an overlay: the viewer draws it as given). */
function atomsCif(atoms, chain, nums) {
  const rows = ['data_overlay', 'loop_', ...['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_comp_id', 'label_asym_id',
    'label_entity_id', 'label_seq_id', 'auth_seq_id', 'auth_asym_id', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv',
    'pdbx_PDB_model_num'].map((k) => `_atom_site.${k}`)];
  let id = 0;
  for (let i = 0; i < atoms.nAtoms; i += 1) {
    if (String(atoms.chain_id[i]) !== chain || !nums.has(Number(atoms.res_id[i]))) continue;
    const c = atoms.coord[i];
    if (!c || !Number.isFinite(c[0])) continue;
    const el = String(atoms.element?.[i] ?? String(atoms.atom_name[i])[0]).toUpperCase() || 'C';
    const name = String(atoms.atom_name[i]);
    rows.push(`ATOM ${++id} ${el} ${/['"\s]/.test(name) ? `"${name}"` : name} ${atoms.res_name[i]} ${chain} 1 ${atoms.res_id[i]} ${atoms.res_id[i]} ${chain} ${c[0].toFixed(3)} ${c[1].toFixed(3)} ${c[2].toFixed(3)} 1.00 0.00 1`);
  }
  return id ? `${rows.join('\n')}\n` : null;
}
const EXPOSURE_OPTIONS = [['1', 'buried'], ['2', 'intermediate'], ['3', 'exposed'], ['0', 'free']];
const INTERACTION_OPTIONS = [
  ['hbond:1', 'H-bond donor'], ['hbond:2', 'H-bond acceptor'], ['hbond:3', 'H-bond donor or acceptor'],
  ['saltbridge:1', 'salt bridge, positive'], ['saltbridge:2', 'salt bridge, negative'], ['pistack:1', 'π-stacking'],
  ['pication:1', 'π-cation, aromatic'], ['pication:2', 'π-cation, cation'], ['hydrophobic:1', 'hydrophobic'],
  ['metal_coord:2', 'metal-coordinating'], ['', 'free'],
];
// Folds a chain can be asked for, by name (CATH class, architecture, topology); "other" takes any code.
/** What + pdb takes, the same on the start screen and under a job. */
const STRUCTURE_NOTE = 'Its ID in the PDB, then any chains to keep (5O45 A, 1J79 A C) with the ligands filed under them, or a ligand by its code (9FJX B LVY), or a file. It comes in fixed; free what you want designed.';

/**
 * What "1J79 A C" or "9FJX B LVY" keeps of an entry: the chains named, with the ligands the entry files under them
 * (`pdbx_nonpoly_scheme.pdb_strand_id`: lenalidomide and zinc are label chains J and I, author chain B), and any
 * ligand named by its code. Throws, naming the chains and ligands there are, for a name that is neither.
 */
export function namedAtoms(doc, names) {
  const a = doc.atoms;
  const col = (cat, key) => { const t = doc.cifBlock?.[cat]; const v = t?.[key] ?? t?.columns?.[key]; return v == null ? [] : (Array.isArray(v) ? v : [v]); };
  const asym = col('pdbx_nonpoly_scheme', 'asym_id');
  const strand = col('pdbx_nonpoly_scheme', 'pdb_strand_id');
  const strandOf = new Map(asym.map((id, k) => [String(id).toUpperCase(), String(strand[k] ?? '').toUpperCase()]));
  const chainOf = (i) => String(a.chain_id[i]).toUpperCase();
  const present = new Set(Array.from(a.chain_id, (c) => String(c).toUpperCase()));
  const codes = new Set(Array.from(a.res_name, (r) => String(r).toUpperCase()));
  const keepChains = new Set();
  const keepCodes = new Set();
  const missing = [];
  for (const raw of names) {
    const n = String(raw).toUpperCase();
    if (present.has(n)) {
      keepChains.add(n);
      for (const [label, auth] of strandOf) if (auth === n && present.has(label)) keepChains.add(label);
    } else if (codes.has(n)) keepCodes.add(n);
    else missing.push(raw);
  }
  if (missing.length) {
    const ligands = [...present].filter((c) => strandOf.has(c)).map((c) => {
      const i = Array.from(a.chain_id, (x) => String(x).toUpperCase()).indexOf(c);
      return String(a.res_name[i]);
    });
    const chains = [...present].filter((c) => !strandOf.has(c));
    throw new Error(`no chain or ligand ${missing.join(', ')} (chains ${chains.join(', ')}${ligands.length ? `; ligands ${[...new Set(ligands)].join(', ')}` : ''})`);
  }
  const keep = [];
  for (let i = 0; i < a.nAtoms; i += 1) if (keepChains.has(chainOf(i)) || keepCodes.has(String(a.res_name[i]).toUpperCase())) keep.push(i);
  return keep;
}

const FOLD_OPTIONS = [
  ['1', 'mainly α'], ['2', 'mainly β'], ['3', 'α/β'],
  ['1.10', 'helix bundle (orthogonal)'], ['1.20', 'up-down helix bundle'], ['2.40', 'β-barrel'], ['2.60', 'β-sandwich'], ['2.60.40', 'immunoglobulin-like'],
  ['3.30.70', 'ferredoxin-like'], ['3.40.50', 'Rossmann-like'], ['3.20.20', 'TIM barrel'], ['other', 'other code'], ['', 'free'],
];
const CONDITION_KEY = { hbond: 'hbond', saltbridge: 'saltbridge', pistack: 'pistack', pication: 'pication', hydrophobic: 'hydrophobic', metal_coord: 'metal_coord' };

export class DesignPanel {
  /** @param {object} app the `Rfd4pStudioApp` (input session, partial-diffusion controls, run state) */
  constructor(app) {
    this.app = app;
    this.selectionCount = 0;
    this.busy = false;
    const $ = (id) => document.getElementById(id);
    this.els = {
      start: $('rfd4p-start'),
      taskGrid: $('rfd4p-task-grid'),
      fetchStatus: $('rfd4p-fetch-status'),
      design: $('rfd4p-design'),
      taskLabel: $('rfd4p-task-label'),
      taskDetail: $('rfd4p-task-detail'),
      newDoc: $('rfd4p-new'),
      chainList: $('rfd4p-chain-list'),
      addSeq: $('rfd4p-add-seq'),
      startSeq: $('rfd4p-start-seq'),
      startPdb: $('rfd4p-start-pdb'),
      startLig: $('rfd4p-start-lig'),
      addPdb: $('rfd4p-add-pdb'),
      addLig: $('rfd4p-add-lig'),
      undo: $('rfd4p-undo'),
      warnings: $('rfd4p-warnings'),
      runSettings: $('rfd4p-run-settings'),
      advanced: $('rfd4p-advanced'),
    };
    // Floating actions next to the pointer for the current selection (viewer or sequence bar).
    this.pop = el('div', { class: 'rfd4p-piece-pop rfd4p-sel-pop', role: 'dialog', 'aria-label': 'Selected residues', hidden: true });
    document.body.append(this.pop);
    this._movable(this.pop);
    this.lastPointer = null;
    this._buildAddStructure();
    this._buildStartStructure();
    this._buildTaskGrid();
    this._bind();
    this._applyAdvanced(this._readAdvanced());
    this.render();
  }

  /**
   * "+ pdb": a structure's chains added to the open job, fixed as they are (a target). Opened from a source caption,
   * the same form replaces that source's chains instead, so an example's target is swapped without setting the job up
   * again.
   */
  _buildAddStructure() {
    const actions = this.els.addPdb?.parentElement;
    if (!actions) return;
    const id = this._structureInput();
    const fileInput = el('input', { type: 'file', accept: '.cif,.mmcif,.pdb,.ent,.cif.zst', hidden: true });
    const fileBtn = el('button', { type: 'button', class: 'secondary compact', text: 'Open a file…', onclick: () => fileInput.click() });
    const { form, status, title, submit: fetchBtn } = this._addForm({
      label: 'Add a structure', button: 'Add', input: id, extra: [fileBtn, fileInput], note: STRUCTURE_NOTE,
      onClose: () => { form.hidden = true; },
      onSubmit: async () => {
        const m = /^\s*([0-9][A-Za-z0-9]{3})(?:[\s:_,]+(.*))?$/.exec(id.value);
        if (!m) { status.textContent = 'Type a 4-character PDB ID, optionally followed by chains: 5O45 A'; return; }
        const pdb = m[1].toUpperCase();
        const chains = (m[2] ?? '').split(/[\s,;]+/).filter(Boolean);
        status.textContent = `Downloading ${pdb} from RCSB…`;
        try {
          const res = await fetch(`https://files.rcsb.org/download/${pdb}.cif`);
          if (!res.ok) throw new Error(res.status === 404 ? `${pdb} was not found` : `RCSB returned ${res.status}`);
          const text = await res.text();
          this._rememberTitle(`PDB ${pdb}`, text);
          await this._addStructure(text, pdb, chains, this.addStructure.replace);
        } catch (err) {
          status.textContent = `Could not add ${pdb}: ${err?.message ?? err}`;
        }
      },
    });
    form.hidden = true;
    this.addStructure = { form, title, fetchBtn, status, id, replace: null };
    fileInput.addEventListener('change', () => {
      const file = fileInput.files?.[0];
      fileInput.value = '';
      if (file) void this._addFile(file);
    });
    actions.after(form);
  }

  /** A structure file's chains added to the open job (the + pdb form's file, or one dropped on the page). */
  async _addFile(file) {
    const { id, status, replace } = this.addStructure;
    const chains = id.value.trim().split(/[\s,;]+/).filter((t) => /^[A-Za-z0-9]{1,4}$/.test(t) && !/^[0-9][A-Za-z0-9]{3}$/.test(t));
    try {
      const text = file.name.endsWith('.zst') ? new Uint8Array(await file.arrayBuffer()) : await file.text();
      await this._addStructure(text, file.name.replace(/\.(cif|mmcif|pdb|ent|zst)+$/i, ''), chains, replace);
    } catch (err) {
      status.textContent = `Could not add ${file.name}: ${err?.message ?? err}`;
      this.app.setStatus?.(status.textContent, true);
    }
  }

  /**
   * A structure file dropped anywhere on the page does what the visible "Open a file…" would: on the start screen it
   * opens as a new job (Studio's own file input), with a job open its chains are added (+ pdb). Taken in the capture
   * phase, so Mol*'s own drop (which loads the file into the viewer only, outside any job) never sees it, nor the
   * dragenter that shows its "Load Files" overlay: the overlay goes away on Mol*'s drop, so it stayed up.
   */
  _bindDrop() {
    const isFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files');
    for (const type of ['dragenter', 'dragover']) {
      window.addEventListener(type, (e) => {
        if (!isFiles(e)) return;
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = 'copy';
      }, true);
    }
    window.addEventListener('drop', (e) => {
      if (!isFiles(e)) return;
      e.preventDefault();
      e.stopPropagation();
      const file = [...(e.dataTransfer.files ?? [])][0];
      if (!file) return;
      const onStart = !document.getElementById('rfd4p-start')?.hidden;
      if (!onStart && this.addStructure) { void this._addFile(file); return; }
      // Studio opens mmCIF only; + pdb, once a job is open, also takes PDB files.
      if (!/\.(cif|mmcif)$/i.test(file.name)) {
        this.app.setStatus?.(`${file.name}: a new job opens from an mmCIF file (.cif); PDB files can be added to a job with + pdb`, true);
        return;
      }
      const input = document.getElementById('file-input');
      if (!input) return;
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    }, true);
  }

  /** The start screen's + pdb: the same form, opening the structure as a new job. */
  _buildStartStructure() {
    const actions = this.els.startPdb?.parentElement;
    if (!actions) return;
    const id = this._structureInput();
    const fileBtn = el('button', { type: 'button', class: 'secondary compact', text: 'Open a file…',
      onclick: () => document.getElementById('file-input')?.click() });
    const { form, status } = this._addForm({
      label: 'Add a structure', button: 'Add', input: id, extra: [fileBtn], note: STRUCTURE_NOTE,
      onClose: () => { form.hidden = true; },
      onSubmit: (st) => this._fetchPdb(id.value, st),
    });
    form.hidden = true;
    this.startStructure = { form, id, status };
    actions.after(form);
  }

  /** The + pdb form; with `group` (a source caption's chains), it replaces them. */
  _openAddStructure(group = null) {
    const { form, title, fetchBtn, status, id } = this.addStructure ?? {};
    if (!form) return;
    this.addStructure.replace = group ? group.chains.flatMap((c) => c.atomIndices) : null;
    title.textContent = group ? `Replace ${group.label}` : 'Add a structure';
    fetchBtn.textContent = group ? 'Replace' : 'Add';
    // The source's own ID and chains, ready to edit ("5O45 A" for "PDB 5O45 chain A").
    const m = group ? /^PDB (\S+)(?: chains? (.+))?$/.exec(group.label) : null;
    id.value = m ? [m[1], m[2] ?? ''].join(' ').trim() : '';
    status.textContent = '';
    form.hidden = false;
    id.focus();
    id.select();
  }

  /** Add `chains` (all when empty) of a structure to the job, fixed as they are; first remove the atoms in `replace`. */
  /**
   * An entry opened as AtomWorks does (rcsb preset), in its first biological assembly that has all of `chains` (3EAK's
   * chain B is in assembly 2); the first assembly when no chains are named or none has them all.
   */
  async _openEntry(text, name, chains = [], extra = {}) {
    const wanted = chains.map((c) => String(c).toUpperCase());
    const has = (doc) => {
      const present = new Set(Array.from(doc.atoms.chain_id, (c) => String(c).toUpperCase()));
      const codes = new Set(Array.from(doc.atoms.res_name, (r) => String(r).toUpperCase()));
      return wanted.every((c) => present.has(c) || codes.has(c));
    };
    const first = await createStudioDocWithCcd(text, name, { preset: 'rcsb', assembly: '1', ...extra });
    if (!wanted.length || has(first)) return first;
    const gen = first.cifBlock?.pdbx_struct_assembly_gen;
    const raw = gen?.assembly_id ?? gen?.columns?.assembly_id ?? [];
    const ids = [...new Set((Array.isArray(raw) ? raw : [raw]).map(String))].filter((id) => id !== '1');
    for (const id of ids) {
      const doc = await createStudioDocWithCcd(text, name, { preset: 'rcsb', assembly: id, ...extra });
      if (has(doc)) return doc;
    }
    return first;
  }

  /** Keep only what `names` ("B", "LVY") names of an opened entry (`namedAtoms`); all of it when none are named. */
  _keepNamed(doc, names) {
    if (!names.length) return;
    const keep = new Set(namedAtoms(doc, names));
    const drop = [];
    for (let i = 0; i < doc.atoms.nAtoms; i += 1) if (!keep.has(i)) drop.push(i);
    if (drop.length) doc.deleteAtoms(drop);
  }

  async _addStructure(text, name, chains, replace) {
    const { status } = this.addStructure;
    const doc = await this._openEntry(text, name, chains);
    this._keepNamed(doc, chains);
    if (!doc.atoms.nAtoms) throw new Error('nothing left to add');
    // Replaced: the new structure goes in first and the old one comes out after, so a job that was only the old one is
    // never emptied (Studio refuses that; an example with one chain could not be replaced). The old atoms are found by
    // chain afterwards, since merging can reorder atoms.
    const a0 = this._inputDoc()?.atoms;
    const oldChains = new Set((replace ?? []).map((i) => String(a0.chain_id[i])));
    // Opened as a plain structure, it is already fixed throughout, so adding it is one Undo step.
    const added = await this._combine(doc);
    if (oldChains.size) {
      const a = this._inputDoc().atoms;
      const addedSet = new Set(added);
      const old = [];
      for (let i = 0; i < a.nAtoms; i += 1) if (!addedSet.has(i) && oldChains.has(String(a.chain_id[i]))) old.push(i);
      if (old.length && !await this._mutate('/delete', { atom_indices: old })) throw new Error(this.lastError ?? 'could not remove the old chains');
      added.splice(0, added.length, ...this._chainAtoms(new Set(added.map((i) => String(a.chain_id[i])))));
    }
    const newChains = [...new Set(added.map((i) => String(this._inputDoc().atoms.chain_id[i])))];
    const original = chains.length === 1 ? ` chain ${chains[0].toUpperCase()}` : '';
    this._rememberSource(newChains, /^[0-9][A-Za-z0-9]{3}$/.test(name) ? `PDB ${name}${original}` : `${name}${original}`);
    // Added beside a chain the job designs from scratch (a binder), the structure is its target, set up as upstream's.
    const binder = (this._spec()?.chains ?? []).some((c) => !newChains.includes(c.id) && c.role === 'new');
    if (binder && !this.building) await this._makeTargets(this._asLoaded(newChains));
    this.addStructure.form.hidden = true;
    this.addStructure.replace = null;
    this.render();
  }

  /** + new: a new chain to design, 60–100 residues; structures kept as loaded beside it become its targets. */
  async _addSeq() {
    const targets = this._asLoaded();
    if (!await this._mutate('/segments', {
      minimum: 60, maximum: 100, atom_index: null, before: false, new_chain: true, chain_id: null,
      chain_type: 'polypeptide(L)',
    })) return;
    if (targets.length) await this._makeTargets(targets);
  }

  /**
   * Chains still as loaded (`ids`, or all of the job's): every piece fixed, all atoms, in place, but for residues the
   * entry has no coordinates for, which come in with their sequence only. Chains set up otherwise are left as they
   * are (upstream: preserve explicit choices rather than converting existing inputs silently).
   */
  _asLoaded(ids = null) {
    const fixedAll = (now) => now.structure === 'fixed' && now.atoms === 'all' && now.sequence === 'fixed' && now.position === 'fixed';
    const seqOnly = (now) => now.structure === 'designed' && now.sequence === 'fixed' && now.position === 'fixed';
    return (this._spec()?.chains ?? []).filter((c) => (!ids || ids.includes(c.id)) && c.role !== 'ligand' && c.pieces.length
      && c.pieces.every((p) => p.kind === 'kept' && (fixedAll(this._properties(p)) || seqOnly(this._properties(p))))
      && c.pieces.some((p) => fixedAll(this._properties(p)))).map((c) => c.id);
  }

  /**
   * A binder's targets as upstream sets them up (RFD4-Proteina's protein-binder task and PD-L1 example): the backbone
   * held as one rigid body with internal coordinates, so its shape is kept but not its place, which reduces binders
   * that copy the target; the sequence fixed; side chains free; hotspots as they are. Shown on the card.
   */
  async _makeTargets(ids) {
    const parts = (this._spec()?.chains ?? []).filter((c) => ids.includes(c.id))
      .flatMap((c) => c.pieces.filter((p) => p.kind === 'kept').map((p) => ({
        p, atoms: p.atomIndices, want: { structure: 'rigid', atoms: 'backbone', sequence: 'fixed', position: 'fixed' },
      })));
    if (parts.length) await this._applyPieces(parts);
  }

  /** Merge a document into the job (one undo step); resolves to the added atoms' indices (the chains that are new). */
  /** Atom indices of the input in these chains. */
  _chainAtoms(chains) {
    const a = this._inputDoc()?.atoms;
    const out = [];
    for (let i = 0; i < (a?.nAtoms ?? 0); i += 1) if (chains.has(String(a.chain_id[i]))) out.push(i);
    return out;
  }

  async _combine(doc) {
    const store = getSharedStudioStore();
    const chainsOf = () => new Set(Array.from(this._inputDoc()?.atoms?.chain_id ?? [], String));
    const before = chainsOf();
    const tmp = `rfd4p_add_${Date.now().toString(36)}`;
    store.sessions.set(tmp, doc);
    try {
      if (!await this._mutate('/combine', { session_ids: [tmp] })) throw new Error(this.lastError ?? 'could not add it');
    } finally {
      store.sessions.delete(tmp);
    }
    // Merged chains are not always appended at the end (Studio orders polymers before ligands): find them by chain.
    const a = this._inputDoc().atoms;
    const added = [];
    for (let i = 0; i < a.nAtoms; i += 1) if (!before.has(String(a.chain_id[i]))) added.push(i);
    return added;
  }

  /** Centre of `indices` (all atoms when null) with coordinates, or the origin. */
  _centroid(indices = null) {
    const a = this._inputDoc()?.atoms;
    const sum = [0, 0, 0];
    let n = 0;
    for (const i of indices ?? Array.from({ length: a?.nAtoms ?? 0 }, (_, k) => k)) {
      const c = a.coord[i];
      if (!c || !Number.isFinite(c[0]) || !Number.isFinite(c[1]) || !Number.isFinite(c[2])) continue;
      sum[0] += c[0]; sum[1] += c[1]; sum[2] += c[2];
      n += 1;
    }
    return n ? sum.map((v) => v / n) : [0, 0, 0];
  }

  /**
   * A ligand by its dictionary code, its pose free (the model places it). With `replace` (a ligand's atoms), it
   * takes that ligand's place: same chain letter, started from the same spot.
   */
  async _addLigand(rawCode, { replace = null, chainId = 'L' } = {}) {
    const code = String(rawCode ?? '').trim().toUpperCase();
    if (!isCcdCode(code)) throw new Error('type the ligand\'s code in the PDB: 1–5 letters or digits, e.g. HEM');
    const { blocks } = await fetchCcdComponents([code]);
    const block = blocks.get(code);
    if (!block) throw new Error(`the PDB has no ligand called ${code}`);
    const center = this._centroid(replace);
    const { cif } = ligandCifFromCcd(block, { chainId, center });
    const doc = await createStudioDocWithCcd(cif, code);
    if (replace && !await this._mutate('/delete', { atom_indices: replace })) throw new Error(this.lastError ?? 'could not remove the old ligand');
    const added = await this._combine(doc);
    // A ligand added by its code comes from no structure: it sits outside the source groups.
    this._rememberSource([...new Set(added.map((i) => String(this._inputDoc().atoms.chain_id[i])))], null);
    this.render();
    return code;
  }

  /**
   * The one form + pdb and + lig open, on the start screen and under a job: a title with its × (as popups close), one
   * line of what it takes, the input and its button (and any more, a file for a structure), and a status line.
   */
  _addForm({ label, note, input, button, extra = [], onSubmit, onClose }) {
    const status = el('p', { class: 'rfd4p-add-status muted', 'aria-live': 'polite' });
    const title = el('strong', { text: label });
    const submit = el('button', { type: 'submit', class: 'primary compact', text: button });
    const form = el('form', { class: 'rfd4p-add-structure' },
      el('div', { class: 'rfd4p-add-head' }, title,
        el('button', { type: 'button', class: 'rfd4p-pop-x', 'aria-label': 'Close', title: 'Close', text: '×', onclick: onClose })),
      el('span', { class: 'muted', text: note }),
      el('div', { class: 'rfd4p-add-row' }, input, submit),
      extra.length ? el('div', { class: 'rfd4p-add-row' }, ...extra) : null,
      status);
    form.addEventListener('submit', (ev) => {
      ev.preventDefault();
      void onSubmit(status);
    });
    return { form, status, title, submit };
  }

  /** + lig's form: a ligand by its code. */
  _ligandCodeForm({ label, button, onSubmit, onCancel }) {
    const input = el('input', {
      type: 'text', class: 'rfd4p-add-id', spellcheck: 'false', autocomplete: 'off', maxlength: 5,
      placeholder: 'Code, e.g. HEM', 'aria-label': 'Ligand code',
    });
    const { form } = this._addForm({
      label, button, input, onClose: onCancel,
      note: 'Its code in the PDB (HEM, ATP, NAD, ZN…). The model places it; fix its pose afterwards to keep it where it is.',
      onSubmit: async (status) => {
        status.textContent = `Looking up ${input.value.trim().toUpperCase()}…`;
        try {
          await onSubmit(input.value);
          status.textContent = '';
        } catch (err) {
          status.textContent = `Could not add it: ${String(err?.message ?? err).replace(/^Error:\s*/, '')}`;
        }
      },
    });
    queueMicrotask(() => input.focus());
    return form;
  }

  /** + pdb's input: a PDB ID, then any chains. */
  _structureInput() {
    return el('input', {
      type: 'text', class: 'rfd4p-add-id', spellcheck: 'false', autocomplete: 'off', maxlength: 40,
      placeholder: 'ID, then chains: 5O45 A', 'aria-label': 'PDB ID and chains',
    });
  }

  // ---- wiring -------------------------------------------------------------------------------------------------

  _bind() {
    const { els } = this;
    this._bindDrop();
    // The start screen's parts start a new job: a free chain, a structure (the form below), or a ligand.
    els.startSeq?.addEventListener('click', () => void this._startNew(async () => {
      const res = await fetch(new URL('./studio/examples/unconditional.cif', import.meta.url));
      if (!res.ok) throw new Error('no empty chain to start from');
      const doc = new StudioDoc(await res.text(), 'design', {});
      const seg = doc.atoms.res_name.findIndex((r) => /^\d+-\d+$/.test(String(r)));
      if (seg >= 0) doc.setSegmentRange(seg, 60, 100);
      return doc;
    }));
    els.startPdb?.addEventListener('click', () => {
      const { form, id } = this.startStructure ?? {};
      if (!form) return;
      this.startLigandForm?.remove();
      this.startLigandForm = null;
      form.hidden = false;
      id.focus();
    });
    els.startLig?.addEventListener('click', () => {
      if (this.startStructure) this.startStructure.form.hidden = true;
      this.startLigandForm?.remove();
      this.startLigandForm = this._ligandCodeForm({
        label: 'Add a ligand',
        button: 'Add',
        onSubmit: (code) => this._startNew(async () => {
          const want = String(code ?? '').trim().toUpperCase();
          if (!isCcdCode(want)) throw new Error('type the ligand\'s code in the PDB: 1–5 letters or digits, e.g. HEM');
          const { blocks } = await fetchCcdComponents([want]);
          const block = blocks.get(want);
          if (!block) throw new Error(`the PDB has no ligand called ${want}`);
          return createStudioDocWithCcd(ligandCifFromCcd(block, { chainId: 'L', center: [0, 0, 0] }).cif, want);
        }, { rethrow: true }),
        onCancel: () => { this.startLigandForm?.remove(); this.startLigandForm = null; },
      });
      els.startLig.parentElement.after(this.startLigandForm);
    });
    // Clear empties the viewer and the sequence and shows the start screen; opening anything from it replaces the
    // document, and Undo there brings back what was shown (the session is kept).
    els.newDoc?.addEventListener('click', () => {
      this.closedSession = this.app.studioSessionId ?? this.app.inputSessionId;
      this.showStart = true;
      this.render();
      void window.atomworksStudio?.closeDocument?.();
    });
    const back = () => {
      const sid = this.closedSession;
      this.closedSession = null;
      this.showStart = false;
      this.render();
      if (sid) void window.atomworksStudio?.openSession?.(sid);
    };
    document.getElementById('file-input')?.addEventListener('change', () => { this.showStart = false; });
    // Kept as the pending edit, so an example being built waits for the whole of it (the chain and its targets).
    els.addSeq?.addEventListener('click', () => { this.applying = this._addSeq(); });
    els.addPdb?.addEventListener('click', () => {
      this.addLigandForm?.remove();
      this.addLigandForm = null;
      this._openAddStructure();
    });
    els.addLig?.addEventListener('click', () => {
      if (this.addStructure) this.addStructure.form.hidden = true;
      this.addLigandForm?.remove();
      this.addLigandForm = this._ligandCodeForm({
        label: 'Add a ligand',
        button: 'Add',
        onSubmit: async (code) => {
          await this._addLigand(code);
          this.addLigandForm?.remove();
          this.addLigandForm = null;
        },
        onCancel: () => { this.addLigandForm?.remove(); this.addLigandForm = null; },
      });
      els.addLig.parentElement.after(this.addLigandForm);
    });
    // Undo on the start screen undoes Clear; otherwise it is Studio's Undo.
    els.undo?.addEventListener('click', () => {
      if (this.showStart && this.closedSession) back();
      else document.getElementById('undo')?.click();
    });
    els.advanced?.addEventListener('change', () => this._applyAdvanced(els.advanced.checked, true));
    window.addEventListener('atomworks-studio', (ev) => {
      if (ev.detail?.type === 'selection') return;
      queueMicrotask(() => this.render());
    });
    document.addEventListener('pointerdown', (ev) => {
      if (this.pop.contains(ev.target)) return;
      const src = ev.target.closest?.('#viewer, #sequence-view');
      this.lastPointer = src ? { x: ev.clientX, y: ev.clientY } : null;
      if (!src) this.pop.hidden = true;
    }, true);
    document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') this.pop.hidden = true; });
    window.addEventListener('atomworks-studio', (ev) => {
      if (ev.detail?.type !== 'selection') return;
      this.selectionCount = ev.detail?.count ?? 0;
      queueMicrotask(() => {
        this._markSelected();
        // A row's own pick has its popup open already.
        if (this.cardPicking) { this.cardPicking = false; this.pop.hidden = true; return; }
        // A click in the viewer while a row's popup is open (turning the structure) opens no other popup; on a preview
        // frame its atoms are not the document's, so the pick is let go.
        if (this.piecePop && !this.piecePop.hidden) {
          this.pop.hidden = true;
          if (this.planPreview && this.selectionCount) window.atomworksStudio?.setSelection?.([]);
          return;
        }
        this._renderPop();
      });
    });
    // Starting from the input diversifies what is designed; with nothing designed yet, that is all the protein.
    document.getElementById('partial-diffusion')?.addEventListener('change', (ev) => {
      if (ev.target.checked) document.getElementById('sampling-accordion')?.setAttribute('open', '');
      if (!ev.target.checked) return;
      const doc = this._inputDoc();
      const spec = doc?.atoms ? designSpec(doc.atoms) : null;
      if (spec && !spec.designedMax) {
        void this._batch(spec.chains.filter((c) => c.residues > 0).flatMap((c) => this._roleEdits(c, 'free')));
      }
    });
    for (const id of ['partial-diffusion', 'partial-diffusion-percent', 'partial-diffusion-percent-num']) {
      document.getElementById(id)?.addEventListener('input', () => this.render());
      document.getElementById(id)?.addEventListener('change', () => this.render());
    }
  }

  _readAdvanced() {
    try { return localStorage.getItem(ADVANCED_KEY) === '1'; } catch { return false; }
  }

  _applyAdvanced(on, persist = false) {
    document.body.classList.toggle('rfd4p-advanced', on);
    if (this.els.advanced) this.els.advanced.checked = on;
    if (persist) {
      try { localStorage.setItem(ADVANCED_KEY, on ? '1' : '0'); } catch { /* private mode */ }
    }
  }

  _buildTaskGrid() {
    const grid = this.els.taskGrid;
    if (!grid) return;
    grid.replaceChildren(...TASKS.map((t) => el('button', {
      type: 'button',
      class: 'rfd4p-task-card',
      'data-example': t.example,
      onclick: () => void this._startTask(t),
    }, el('strong', { text: t.title }), el('span', { text: t.desc }))));
  }

  // ---- state --------------------------------------------------------------------------------------------------

  /** The input document (what Generate will use), if the Studio has one open. */
  _inputDoc() {
    // An example being built lives in a session Studio does not show until it is finished.
    if (this.buildSession) return getSharedStudioStore().getSession(this.buildSession);
    const sid = this.app.useStudioSession ? this.app.inputSessionId : null;
    return sid ? getSharedStudioStore().getSession(sid) : null;
  }

  _spec() {
    const doc = this._inputDoc();
    if (!doc?.atoms) return null;
    const partial = Boolean(this.app.els.partialDiffusion?.checked);
    const percent = Number.parseFloat(this.app.els.partialPercent?.value ?? '40') || 40;
    return designSpec(doc.atoms, { partialDiffusion: partial, partialPercent: percent });
  }

  /** Whether the viewer shows a generated design rather than the input (edits would then fork from that design). */
  _viewingDesign() {
    const { studioSessionId, inputSessionId, generatedViews } = this.app;
    return Boolean(studioSessionId && studioSessionId !== inputSessionId && generatedViews?.get(studioSessionId));
  }

  render() {
    const spec = this._spec();
    const { els } = this;
    const hasDoc = Boolean(spec && !spec.empty);
    // An example being built keeps the start screen up; its chains are laid out (hidden) for the build to use.
    const starting = !hasDoc || Boolean(this.showStart) || Boolean(this.building);
    if (els.start) els.start.hidden = !starting;
    if (els.design) els.design.hidden = starting;
    if (els.runSettings) els.runSettings.hidden = starting;
    // A run's results and status belong to the document, so they go with it.
    els.start?.closest('#rfd4p-studio-card')?.classList.toggle('rfd4p-starting', starting);
    if (!hasDoc || (starting && !this.building)) {
      // The start screen sits under the Input tab too, with no Clear, and Undo only when there is a job to go back to.
      if (els.newDoc) els.newDoc.hidden = true;
      if (els.undo) {
        els.undo.hidden = !this.closedSession;
        els.undo.disabled = !this.closedSession;
      }
      this.app._renderDesigns?.();
      return;
    }

    const viewingDesign = this._viewingDesign();
    els.design.classList.toggle('rfd4p-readonly', viewingDesign);
    els.taskLabel.textContent = spec.task.label;
    els.taskDetail.textContent = viewingDesign ? 'A generated design (read-only). Select Input under Designs to edit.' : '';
    els.taskDetail.hidden = !viewingDesign;
    if (els.newDoc) els.newDoc.hidden = viewingDesign;
    const actions = els.addSeq?.parentElement;
    if (actions) actions.hidden = viewingDesign;
    if (viewingDesign && this.addStructure) this.addStructure.form.hidden = true;
    els.undo.hidden = viewingDesign;
    els.undo.disabled = document.getElementById('undo')?.disabled ?? true;

    els.chainList.replaceChildren(...this._sourceGroups(spec.chains).map((g) => this._groupBlock(g, viewingDesign)));
    this._markSelected();

    const warnings = this.lastError ? [this.lastError, ...spec.warnings] : [...spec.warnings];
    const notes = warnings.map((w) => el('p', { class: 'rfd4p-warning', text: w }));
    if (spec.task.key === 'partial' && spec.chains.some((c) => c.segments.length)) {
      // New residues that replaced residues can be those residues again, free in place, which partial diffusion takes.
      const doc = this._inputDoc();
      let known = false;
      for (let i = 0; i < doc.atoms.nAtoms && !known; i += 1) known = Boolean(doc.segmentOrigin?.(i));
      notes.push(el('p', { class: 'rfd4p-warning' }, 'Partial diffusion needs every residue defined; this input has new residues.',
        known && !viewingDesign ? el('button', { type: 'button', class: 'secondary compact rfd4p-warning-fix',
          onclick: () => void this._restoreAll(), text: 'Back to the residues they replaced' }) : null));
    }
    els.warnings.replaceChildren(...notes);

    this.showHotspots();
    // An edit to the input changes which Input rows the Designs table shows (the input as it is now, or as a run had it).
    // Also with no runs: the Input / Outputs tabs are there from the start.
    this.app._renderDesigns?.();
    // A job is open: have the model ready before Generate, if its weights are already cached (nothing is downloaded).
    this.app.preloadWeights?.();

    // Nothing to generate: keep Generate off (a running job keeps its Stop button).
    if (!this.app.activeAbortController && this.app.els.predictBtn) {
      this.app.els.predictBtn.disabled = Boolean(this.app.noWebGpu) || spec.task.key === 'nothing' || spec.task.key === 'empty';
    }
  }

  /**
   * Chains by where their coordinates come from, in the job's order (as the sequence bar shows them, and as they were
   * added); chains made here are uncaptioned, in place.
   */
  _sourceGroups(chains) {
    const groups = [];
    const bySource = new Map();
    for (const c of chains) {
      const label = this._sourceOf(c);
      let g = label ? bySource.get(label) : groups[groups.length - 1];
      if (!g || (!label && g.label)) {
        g = { label, chains: [] };
        groups.push(g);
        if (label) bySource.set(label, g);
      }
      g.chains.push(c);
    }
    return groups;
  }

  /** A source's caption (click: replace it with another structure), then its chains. */
  _groupBlock(g, readonly) {
    // "PDB 5O45 · A": the entry and the chains of it in the job; its title when pointed at.
    const entry = g.label?.replace(/ chains? .*$/, '');
    const title = entry ? this.titles?.get(entry) : null;
    const text = entry ? `${entry} · ${g.chains.map((c) => c.id).join(' ')}` : null;
    // The caption replaces its structure (another entry or file in its place) when clicked, and × beside it, shown
    // while the caption is pointed at, takes its chains out.
    const atoms = g.chains.flatMap((c) => c.atomIndices);
    const caption = !g.label ? null : readonly
      ? el('span', { class: 'rfd4p-source', title: title ?? null, text })
      : el('div', { class: 'rfd4p-source-row' },
        el('button', { type: 'button', class: 'rfd4p-source', onclick: () => this._openAddStructure(g), text,
          title: [title, `Click to replace ${g.label}`].filter(Boolean).join('\n') }),
        el('button', { type: 'button', class: 'rfd4p-x rfd4p-source-x', text: '×', title: `Remove ${g.label}`,
          'aria-label': `Remove ${g.label}`, onclick: () => {
            // The last thing in the job: removing it is Clear (the start screen, with Back), not an empty document.
            if (atoms.length >= (this._inputDoc()?.atoms?.nAtoms ?? 0)) this.els.newDoc?.click();
            else void this._mutate('/delete', { atom_indices: atoms });
          } }));
    return el('div', { class: 'rfd4p-group' }, caption, ...g.chains.map((c) => this._chainRow(c, readonly)));
  }

  /**
   * A chain as its pieces, in order (the contig, spelled out): the chain letter on the left, then one sub-card per
   * piece and ligand. A sub-card opens its popup and has its own ×; a "+" between sub-cards inserts new residues there.
   */
  _chainRow(c, readonly) {
    const remove = (title, atoms) => (readonly ? null : el('button', {
      type: 'button', class: 'rfd4p-x', title, 'aria-label': title, text: '×',
      onclick: () => void this._mutate('/delete', { atom_indices: atoms }),
    }));
    const pieces = c.pieces;
    const many = pieces.length + c.ligands.length > 1;
    const items = [];
    // A slot between any two pieces and at both ends: new residues next to new residues are a block of their own, so
    // a free chain can be built as blocks of different shapes (a helix, a loop, a strand).
    // A chain from a PDB entry: "+" asks whether to add new residues or bring back residues of the entry (a motif
    // residue removed, say); otherwise it adds new residues straight away.
    const entry = this._entryOf(c);
    const slot = (atom, before) => (readonly ? null : el('button', {
      type: 'button', class: 'rfd4p-slot', title: entry ? 'Add residues here' : 'New residues here', 'aria-label': entry ? 'Add residues here' : 'New residues here',
      onclick: (ev) => {
        if (entry) { this._openPop(ev.currentTarget, () => this._slotPop(c, entry, atom, before)); return; }
        void this._mutate('/segments', {
          minimum: 5, maximum: 10, atom_index: atom, before, new_chain: false, chain_id: null, chain_type: 'polypeptide(L)',
        });
      },
    }, el('span', { text: '+' })));
    pieces.forEach((p, k) => {
      const prev = pieces[k - 1];
      items.push(prev ? slot(prev.atomIndices[prev.atomIndices.length - 1], false) : slot(p.atomIndices[0], true));
      items.push(this._pieceCard(c, p, readonly, remove(many ? 'Remove these residues' : `Remove chain ${c.id}`,
        p.atomIndices)));
      if (k === pieces.length - 1) items.push(slot(p.atomIndices[p.atomIndices.length - 1], false));
    });
    for (const l of c.ligands) {
      items.push(this._ligandCard(c, l, readonly, remove(many ? `Remove ${l.name}` : `Remove chain ${c.id}`, l.atomIndices)));
    }
    const gutter = el('div', { class: 'rfd4p-chain-gutter' },
      el('button', {
        type: 'button', class: 'rfd4p-chain-id', text: c.id, disabled: readonly,
        title: `Chain ${c.id}${c.fold ? ` · fold ${c.fold.replace(/(\.x)+$/, '')}` : ''}: name, order and fold`,
        onclick: (ev) => this._openPop(ev.currentTarget, () => this._chainPop(c)),
      }),
      many ? remove(`Remove chain ${c.id}`, c.atomIndices) : null);
    const row = el('div', { class: `rfd4p-chain rfd4p-role-${c.role}` }, gutter, el('div', { class: 'rfd4p-chain-items' }, ...items.filter(Boolean)));
    // The chain's colour, as in the viewer and the sequence bar.
    const color = window.atomworksStudio?.chainColor?.(c.id);
    if (color) row.style.setProperty('--chain-color', color);
    return row;
  }

  _range(c, p) {
    return p.start === p.end ? `${c.id}${p.start}` : `${c.id}${p.start}–${p.end}`;
  }

  /** A piece's name, on the left of its row: its residues (where a set-up took them from, if renumbered), or for
   * new residues their length. */
  _pieceText(c, p) {
    if (p.kind === 'new') return p.min === p.max ? `${p.min}` : `${p.min}–${p.max}`;
    // A count only where numbering skips (it would repeat the range otherwise).
    const n = p.count !== p.end - p.start + 1 ? ` (${p.count})` : '';
    return p.source ?? `${this._range(c, p)}${n}`;
  }

  /** What a piece row says after its residues: what is kept, in a few words. */
  _status(p) {
    const word = (options, v) => (v ? options.find(([k]) => k === String(v))?.[1] ?? '' : '');
    // "new" last, so it lines up down a chain of blocks ("H · new", "C · new").
    if (p.kind === 'new') {
      const code = String(p.fold ?? '').replace(/(\.x)+$/, '');
      const fold = code ? FOLD_OPTIONS.find(([k]) => k === code)?.[1] ?? code : '';
      return [fold, SHAPE_LETTER[p.ss] ?? '', word(EXPOSURE_OPTIONS, p.rasa), 'new'].filter(Boolean).join(' · ');
    }
    const { structure, atoms, names, sequence, position } = this._properties(p);
    // The popup's words: both fixed reads "fixed", both free "free", otherwise which one is fixed; part of the atoms
    // says which.
    const part = atoms === 'backbone' ? 'backbone' : atoms === 'sidechain' ? 'side chains' : atoms === 'named' ? names : '';
    const what = structure === 'designed' && sequence === 'designed' ? 'free'
      : structure === 'rigid' ? (part ? `${part} rigid` : 'rigid')
        : structure === 'fixed' && sequence === 'fixed' ? (part ? `${part} fixed` : 'fixed')
          : structure === 'fixed' ? 'structure fixed' : 'sequence fixed';
    return [what, position === 'anywhere' ? 'position free' : '', SHAPE_LETTER[p.ss] ?? '', word(EXPOSURE_OPTIONS, p.rasa), word(INTERACTION_OPTIONS, p.inter),
      p.hotspots.length ? `${p.hotspots.length} hotspot${p.hotspots.length === 1 ? '' : 's'}` : ''].filter(Boolean).join(' · ');
  }

  /** A sub-card: name and status (opens its popup), then its ×. */
  _card(name, status, kindClass, readonly, onOpen, remove, atoms = null) {
    const card = el('div', { class: `rfd4p-sub ${kindClass}` },
      el('button', {
        type: 'button', class: 'rfd4p-sub-open', disabled: readonly, 'aria-haspopup': 'dialog',
        onclick: (ev) => onOpen(ev.currentTarget),
      },
      el('span', { class: 'rfd4p-sub-name', text: name }),
      el('span', { class: 'rfd4p-sub-status', text: status })),
      remove);
    // Its atoms, so the card can show which sub-cards hold selected residues (`_markSelected`).
    card.rfd4pAtoms = atoms;
    return card;
  }

  /** Outline the sub-cards holding selected residues, as the viewer and the sequence bar outline them. */
  _markSelected() {
    const sel = new Set(this._viewingDesign() ? [] : window.atomworksStudio?.getSelection?.() ?? []);
    for (const card of this.els.chainList?.querySelectorAll('.rfd4p-sub') ?? []) {
      card.classList.toggle('rfd4p-sub-selected', Boolean(sel.size && card.rfd4pAtoms?.some((i) => sel.has(i))));
    }
  }

  _pieceCard(c, p, readonly, remove) {
    return this._card(this._pieceText(c, p), this._status(p), `rfd4p-sub-${p.kind === 'kept' ? 'kept' : 'designed'}`, readonly,
      (anchor) => {
        if (p.kind !== 'new') this._selectFromCard(p.atomIndices);
        this._openPop(anchor, () => (p.kind === 'new' ? this._newPop(c, p) : this._piecePop(c, p)));
      }, remove, p.atomIndices);
  }

  /** A row's residues selected in the viewer and the sequence too, while its popup is open (not the viewer's popup). */
  _selectFromCard(atoms, { center = true } = {}) {
    if (typeof window.atomworksStudio?.setSelection !== 'function' || !atoms?.length || this.building) return;
    this.cardPicking = true;
    this.cardSelected = true;
    clearTimeout(this.cardPickTimer);
    this.cardPickTimer = setTimeout(() => { this.cardPicking = false; }, 400);
    window.atomworksStudio.setSelection(atoms);
    // Brought to the middle of the view (the zoom kept), so the popup's residues are in sight.
    if (center) window.atomworksStudio.centerOn?.(atoms);
  }

  _ligandCard(c, l, readonly, remove) {
    // Some atoms held: named, as a residue's are ("N1, C2, N3, C4 fixed"), or counted when the names would not fit.
    const part = l.heldNames && l.heldNames.length <= 24 ? `${l.heldNames} fixed` : `${l.fixedAtoms} of ${l.atoms} atoms fixed`;
    const status = l.pose === 'fixed' ? 'fixed' : l.pose === 'predicted' ? 'free' : part;
    return this._card(l.name, status, 'rfd4p-sub-ligand', readonly, (anchor) => {
      this._selectFromCard(l.atomIndices);
      this._openPop(anchor, () => this._ligandPop(c, l));
    }, remove, l.atomIndices);
  }

  /** The PDB entry chain `c` came from, and its chain there ("1QJG", "A"); null when it did not come from one. */
  _entryOf(c) {
    const m = /^PDB (\w{4})(?: chains? (\S+))?/.exec(this._sourceOf(c) ?? '');
    if (!m) return null;
    const a = this._inputDoc()?.atoms;
    const SRC = a?.annotations?.['annotation_src-component_1_atom'];
    // A residue's name in the entry ("A38"), not a new residue's placeholder text.
    const label = c.atomIndices?.map((i) => String(SRC?.[i] ?? '')).find((x) => /^[A-Za-z]+-?\d/.test(x));
    const chain = /^([A-Za-z]+)/.exec(label ?? '')?.[1] ?? m[2] ?? c.id;
    return { id: m[1].toUpperCase(), chain };
  }

  /** "+" in a chain from an entry: new residues of a length, or residues of the entry brought back where it was pressed. */
  _slotPop(c, entry, atom, before) {
    const lenMin = el('input', { type: 'number', min: 1, max: 1000, value: 5, 'aria-label': 'Fewest new residues' });
    const lenMax = el('input', { type: 'number', min: 1, max: 1000, value: 10, 'aria-label': 'Most new residues' });
    // Changed through the chain's plan: what is added goes exactly here, and the preview is the chain as Add would
    // leave it. Residues of the entry not in the chain, as from – to boxes like a piece's Residues: the stretch missing
    // here to start with, with a drag bar over it; any others typed (a motif residue removed, put back in place), the
    // boxes empty then.
    const plan = this._plan(c);
    const src = plan.source;
    const gap = this._missingAt(plan, atom, before);
    const keptAt = (x, y) => this._partsWithInsert(plan, atom, before, { kind: 'kept', lo: x, hi: y });
    const inChain = new Set(plan.parts.filter((q) => q.kind === 'kept')
      .flatMap((q) => src.slice(Math.min(q.from, q.to), Math.max(q.from, q.to) + 1).map((r) => r.num)));
    const missing = src.filter((r) => !inChain.has(r.num));
    const from = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      value: gap ? `${gap.lo}–${gap.hi}` : '', 'aria-label': `Residues of PDB ${entry.id}` });
    // The range typed, if every residue in it is the entry's and not in the chain; else why not.
    const pick = () => {
      const m = /^\s*(-?\d+)\s*(?:[-–]\s*(-?\d+))?\s*$/.exec(from.value);
      if (!m) return { error: 'Residues as from – to' };
      const x = Math.min(Number(m[1]), Number(m[2] ?? m[1]));
      const y = Math.max(Number(m[1]), Number(m[2] ?? m[1]));
      const run = src.filter((r) => r.num >= x && r.num <= y);
      if (!run.length) return { error: `PDB ${entry.id} chain ${entry.chain} has no residues ${x}–${y}` };
      const there = run.find((r) => inChain.has(r.num));
      if (there) return { error: `${entry.chain}${there.num} is in the chain already` };
      return { lo: run[0].num, hi: run.at(-1).num };
    };
    const ranges = () => { const r = pick(); return r.error ? null : [[r.lo, r.hi]]; };
    let bar = null;
    if (gap) {
      const iLo = src.findIndex((r) => r.num === gap.lo);
      const iHi = src.findLastIndex((r) => r.num === gap.hi);
      const piece = { start: gap.lo, end: gap.hi, atomIndices: [] };
      bar = this._cropBar(c, piece, src, { lo: gap.lo, hi: gap.hi, iLo, iHi, present: inChain }, from, ranges);
    }
    // Nothing is shown until there is something to add: where none of the entry is missing beside the "+" (after the
    // chain's last residue), the boxes start empty, or the preview joined some far residue onto this end.
    const show = () => {
      const r = from.value.trim() ? pick() : { error: 'none' };
      // Not a range yet (a box emptied to type a new number): the last preview stays, as a piece's popup keeps it;
      // going back to the document and building the preview again on the next key was a blink.
      if (!r.error) this._previewParts(c, keptAt(r.lo, r.hi));
    };
    if (missing.length) {
      from.addEventListener('input', show);
      queueMicrotask(show);
    }
    const reach = { lo: missing[0]?.num, hi: missing.at(-1)?.num };
    const note = el('p', { class: 'rfd4p-pop-note', 'aria-live': 'polite' });
    const go = (fn) => async () => {
      note.textContent = '';
      try { await fn(); this._closePop(); } catch (err) { note.textContent = String(err?.message ?? err).replace(/^Error:\s*/, ''); }
    };
    const rebuild = async (parts, what) => {
      if (!await this._mutate('/chains/rebuild', { chain_id: c.id, parts })) throw new Error(this.lastError ?? `could not ${what}`);
    };
    const addNew = go(async () => {
      const lo = Math.max(1, Number(lenMin.value) || 1);
      const hi = Math.max(lo, Number(lenMax.value) || lo);
      await rebuild(this._partsWithInsert(plan, atom, before, { kind: 'new', min: lo, max: hi }), 'add them');
    });
    const addBack = go(async () => {
      const r = pick();
      if (r.error) throw new Error(r.error);
      await rebuild(keptAt(r.lo, r.hi), 'bring them back');
    });
    from.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); void addBack(); } });
    const row = (label, ...kids) => el('div', { class: 'rfd4p-pop-row rfd4p-slot-row' }, el('span', { class: 'rfd4p-pop-label', text: label }), ...kids);
    return [
      this._popHead('Add here'),
      // A blank as wide as the entry row's chain letter, so both rows' boxes line up.
      row('New', el('span', { class: 'rfd4p-pop-inline' }, missing.length ? el('span', { text: entry.chain, style: 'visibility:hidden' }) : null,
        lenMin, el('span', { text: '–' }), lenMax),
        el('button', { type: 'button', class: 'secondary compact', text: 'Add', onclick: addNew })),
      // Residues of the entry are offered only when some are not in the chain.
      missing.length ? row(`PDB ${entry.id}`, el('span', { class: 'rfd4p-pop-inline' }, el('span', { text: entry.chain }), this._rangeBoxes(from, reach)),
        el('button', { type: 'button', class: 'secondary compact', text: 'Add', onclick: addBack })) : null,
      bar,
      note,
    ];
  }

  /**
   * The entry's residues missing where a "+" sits in `plan` ({ lo, hi, src }): between the nearest kept runs either
   * side (past new residues), after the one before it (to the chain's end) or before the one after it (from the
   * chain's start). Null when none are missing there, or the runs either side are out of the entry's order (a motif).
   */
  _missingAt(plan, atom, before) {
    const src = plan?.source;
    if (!src?.length) return null;
    const parts = plan.parts;
    const k = parts.findIndex((q) => (q.kind === 'new' ? q.atom === atom : q.atoms?.includes(atom)));
    if (k < 0) return null;
    // The nearest kept runs either side, past any new residues between (A1–40, 5–10 new, "+": A41 on are missing).
    const kept = (q) => q?.kind === 'kept';
    let i = before ? k - 1 : k;
    while (i >= 0 && !kept(parts[i])) i -= 1;
    let j = before ? k : k + 1;
    while (j < parts.length && !kept(parts[j])) j += 1;
    if (!kept(parts[i]) && !kept(parts[j])) return null;
    const lo = kept(parts[i]) ? Math.max(parts[i].from, parts[i].to) + 1 : 0;
    const hi = kept(parts[j]) ? Math.min(parts[j].from, parts[j].to) - 1 : src.length - 1;
    if (lo > hi) return null;
    // None of them anywhere else in the chain (in the entry's order, none would be).
    if (parts.some((q) => kept(q) && Math.max(q.from, q.to) >= lo && Math.min(q.from, q.to) <= hi)) return null;
    return { lo: src[lo].num, hi: src[hi].num, src };
  }

  /** Where a chain's coordinates come from, as the page knows it; null for chains made here (+ new, + lig). */
  _sourceOf(c) {
    if (c.role === 'new') return null;
    const doc = this._inputDoc();
    const added = this.sources?.get(doc);
    if (added?.has(c.id)) return added.get(c.id);
    const name = String(doc?.name ?? '').replace(/\.(cif|mmcif|pdb|zst)+$/i, '');
    const task = TASKS.find((t) => t.example === name);
    if (task) return `${task.title} example`;
    return /^[0-9][A-Za-z0-9]{3}$/.test(name) ? `PDB ${name.toUpperCase()}` : name || null;
  }

  /** The entry's title (`_struct.title`), shown when pointing at its caption. */
  _rememberTitle(label, cifText) {
    // A value on the same line or the next, quoted, bare, or a ;-delimited block.
    const m = /^_struct\.title\s+(?:'([^']*)'|"([^"]*)"|;([\s\S]*?)\n;|([^\s?'";_][^\n]*))/m.exec(cifText);
    const title = (m?.[1] ?? m?.[2] ?? m?.[3] ?? m?.[4] ?? '').replace(/\s+/g, ' ').trim();
    if (title) (this.titles ??= new Map()).set(label, title);
  }

  _rememberSource(chains, label) {
    const doc = this._inputDoc();
    if (!doc) return;
    this.sources ??= new WeakMap();
    if (!this.sources.has(doc)) this.sources.set(doc, new Map());
    for (const id of chains) this.sources.get(doc).set(id, label);
  }

  /**
   * The one popup, next to the row that opened it. `build` makes its content; closed by its ×, Escape or a click outside
   * it other than in the viewer (× on a row means delete).
   */
  _openPop(anchor, build) {
    if (!this.piecePop) {
      this.piecePop = el('div', { class: 'rfd4p-piece-pop', role: 'dialog', hidden: true });
      document.body.append(this.piecePop);
      this._movable(this.piecePop);
      document.addEventListener('pointerdown', (ev) => {
        // Not the viewer: the structure is turned to see what the popup changes from another side.
        if (!this.piecePop.hidden && !this.piecePop.contains(ev.target) && !ev.target.closest?.('.rfd4p-sub-open, #viewer')) this._closePop();
      }, true);
      document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') this._closePop(); });
    }
    const pop = this.piecePop;
    this.pop.hidden = true;
    pop.replaceChildren(...build().filter(Boolean));
    pop.hidden = false;
    const r = anchor.getBoundingClientRect();
    const card = anchor.closest('#rfd4p-studio-card')?.getBoundingClientRect() ?? r;
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;
    // Beside the card (it sits at the right edge), level with the row, kept on screen.
    const left = card.left - w - 8 >= 8 ? card.left - w - 8 : Math.max(8, Math.min(window.innerWidth - w - 8, r.left));
    pop.style.left = `${left}px`;
    pop.style.top = `${Math.max(8, Math.min(window.innerHeight - h - 8, r.top - 8))}px`;
    this._placeMoved(pop);
    pop.querySelector('input:not([type=radio]):not([type=checkbox]), button.primary')?.focus();
  }

  /** Popups move by their title bar; once moved, popups open where it was left (until the page is reloaded). */
  _movable(pop) {
    pop.addEventListener('pointerdown', (ev) => {
      const head = ev.target.closest?.('.rfd4p-pop-title');
      if (!head || ev.button !== 0 || ev.target.closest('button, input, select')) return;
      ev.preventDefault();
      const r = pop.getBoundingClientRect();
      const dx = ev.clientX - r.left;
      const dy = ev.clientY - r.top;
      try { head.setPointerCapture(ev.pointerId); } catch { /* not an active pointer */ }
      pop.classList.add('rfd4p-pop-moving');
      const move = (e) => {
        this.popSpot = { left: e.clientX - dx, top: e.clientY - dy };
        this._placeMoved(pop);
      };
      const up = () => {
        pop.classList.remove('rfd4p-pop-moving');
        head.removeEventListener('pointermove', move);
        head.removeEventListener('pointerup', up);
        head.removeEventListener('pointercancel', up);
      };
      head.addEventListener('pointermove', move);
      head.addEventListener('pointerup', up);
      head.addEventListener('pointercancel', up);
    });
  }

  /** Where the user last left a popup, kept on screen. */
  _placeMoved(pop) {
    if (!this.popSpot) return;
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;
    pop.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, this.popSpot.left))}px`;
    pop.style.top = `${Math.max(8, Math.min(window.innerHeight - Math.min(h, 40) - 8, this.popSpot.top))}px`;
  }

  _closePop() {
    if (this.piecePop) this.piecePop.hidden = true;
    this._endPreview();
    // A row's residues were selected only while its popup was open.
    if (this.cardSelected) {
      this.cardSelected = false;
      if (!this.building) window.atomworksStudio?.setSelection?.([]);
    }
  }

  /** Chain `c` as a plan (studio-api.js `chainPlan`): kept runs of the chain as loaded and new blocks, in order; or null. */
  _plan(c) {
    return this._inputDoc()?.chainPlan?.(c.id) ?? null;
  }

  /** `plan`'s parts as `rebuildChain` takes them (kept runs by source number, new blocks by their placeholder). */
  _parts(plan) {
    return plan.parts.map((q) => (q.kind === 'kept' ? { kind: 'kept', lo: q.lo, hi: q.hi }
      : q.kind === 'own' ? { kind: 'own', atoms: q.atoms } : { kind: 'new', atom: q.atom, min: q.min, max: q.max }));
  }

  /** `plan` with piece `p`'s residues (source numbers `p.start`-`p.end`) replaced by `lo`-`hi`, in the same place. */
  _partsWithRange(plan, p, lo, hi) {
    const out = [];
    let placed = false;
    for (const q of this._parts(plan)) {
      if (q.kind !== 'kept' || q.hi < p.start || q.lo > p.end) { out.push(q); continue; }
      if (q.lo < p.start) out.push({ kind: 'kept', lo: q.lo, hi: p.start - 1 });
      if (!placed && lo <= hi) { out.push({ kind: 'kept', lo, hi }); placed = true; }
      if (q.hi > p.end) out.push({ kind: 'kept', lo: p.end + 1, hi: q.hi });
    }
    return out;
  }

  /** `plan` with `part` added where a "+" sits (beside the part holding `atom`, before or after it). */
  _partsWithInsert(plan, atom, before, part) {
    const parts = this._parts(plan);
    const k = plan.parts.findIndex((q) => (q.kind === 'new' ? q.atom === atom : q.atoms?.includes(atom)));
    const at = k < 0 ? parts.length : before ? k : k + 1;
    parts.splice(at, 0, part);
    return parts;
  }

  /**
   * The chain as Apply would leave it (`parts`), shown in the viewer in the document's place, one connected structure,
   * with the chain as loaded faint behind it: its residues not in `parts`. Adding and editing look the same: what
   * Apply keeps solid, the rest faint. `_endPreview` puts the document back.
   */
  _previewParts(c, parts) {
    const studio = window.atomworksStudio;
    if (typeof studio?.renderFrame !== 'function') return;
    // Drawn as soon as asked; while a draw is under way only the latest request is kept, drawn when it is done, so a
    // drag never queues up frames behind it.
    this.previewNext = { c, parts, gen: this.previewGen ?? 0 };
    if (this.previewBusy) return;
    this.previewBusy = true;
    void (async () => {
      try {
        while (this.previewNext) {
          const { c: chain, parts: want, gen } = this.previewNext;
          this.previewNext = null;
          if (gen !== (this.previewGen ?? 0)) continue;
          const doc = this._inputDoc();
          if (!doc) break;
          let copy;
          try { copy = doc.previewRebuild(chain.id, want); } catch { continue; }
          // The first frame of a preview loads a scene of its own, which takes the faint chain with it: drawn again then.
          if (!this.planPreview) this.ghostKey = null;
          this.planPreview = true;
          // The card's selection would draw over the preview (an outline, and sticks where atoms are picked).
          if (this.cardSelected) { this.cardSelected = false; studio.setSelection?.([]); }
          await studio.renderFrame(copy.toCif(), doc.name ?? 'preview', { keepCamera: true });
          // Closed while it was drawn: the document back, not this frame (then the next popup's preview, if one is waiting).
          if (gen !== (this.previewGen ?? 0)) { this.planPreview = false; await studio.showDocument?.(); continue; }
          // The chain as loaded, faint and a little thinner, drawn once per popup and never changed: the residues Apply
          // keeps hide it inside their own cartoon, so only those left out show, and a drag swaps nothing (an overlay
          // changed per step blinked, or pulsed while two were up; one updated in place, or a lone residue, came out as
          // solid sticks).
          const source = doc.chainSources?.get(chain.id);
          if (!source) continue;
          const key = chain.id;
          if (key === this.ghostKey) continue;
          this.ghostKey = key;
          this.ghostOn = true;
          const all = new Set();
          for (let i = 0; i < source.atoms.nAtoms; i += 1) if (String(source.atoms.chain_id[i]) === source.chain) all.add(Number(source.atoms.res_id[i]));
          const color = Number.parseInt(String(studio.chainColor?.(chain.id) ?? '#8899aa').replace('#', ''), 16);
          await studio.setOverlay('loadedChain', null);
          await studio.setOverlay('loadedChain', atomsCif(source.atoms, source.chain, all),
            { representation: 'cartoon', alpha: 0.22, color, sizeFactor: 0.17, label: 'The chain as loaded' });
        }
      } finally {
        this.previewBusy = false;
      }
    })();
  }

  /** The document back in the viewer after a preview, and the faint chain away. */
  _endPreview() {
    this.previewNext = null;
    this.previewGen = (this.previewGen ?? 0) + 1;
    this.ghostKey = null;
    const studio = window.atomworksStudio;
    if (this.ghostOn) { this.ghostOn = false; void studio?.setOverlay?.('loadedChain', null); }
    if (this.planPreview) { this.planPreview = false; void studio?.showDocument?.(); }
  }

  /** A popup's title, and its × (top right) to close it. */
  _popHead(title, close = () => this._closePop(), detail = null) {
    return el('div', { class: 'rfd4p-pop-title' },
      el('strong', { text: title }),
      detail ? el('span', { class: 'muted', text: detail }) : null,
      el('button', { type: 'button', class: 'rfd4p-pop-x', 'aria-label': 'Close', title: 'Close', text: '×', onclick: close }));
  }

  _popActions(apply) {
    return el('div', { class: 'rfd4p-pop-actions' },
      el('span', { class: 'rfd4p-pop-spacer' }),
      el('button', { type: 'button', class: 'primary compact', text: 'Apply', onclick: () => { this.applying = Promise.resolve(apply()); } }));
  }

  /** The rest of a popup, folded: opened by More. Three rows or fewer are not worth folding, and are shown as they are. */
  _moreFold(...children) {
    const rows = children.filter(Boolean);
    if (rows.length <= 3) {
      const frag = document.createDocumentFragment();
      frag.append(...rows);
      return frag;
    }
    return el('details', { class: 'rfd4p-pop-more' }, el('summary', { text: 'More' }), el('div', { class: 'rfd4p-pop-more-body' }, ...children.filter(Boolean)));
  }

  /**
   * Shape (secondary structure), Exposure and Interaction rows, set from `now` ({ss, rasa, inter}, "mixed" where the
   * residues differ). `edits(residueAtoms, interactionAtoms)` gives the condition edits for what changed.
   */
  _extraFields(now, { interaction = true } = {}) {
    const select = (label, options, value, title) => {
      const s = el('select', { 'aria-label': label, title });
      if (value === 'mixed') s.append(el('option', { value: 'mixed', text: 'mixed', selected: true }));
      for (const [v, text] of options) s.append(el('option', { value: v, text, selected: v === String(value) }));
      return s;
    };
    const shape = select('Shape', SHAPE_OPTIONS, now.ss ?? 0, 'The secondary structure these residues should take');
    const exposure = select('Exposure', EXPOSURE_OPTIONS, now.rasa ?? 0, 'How buried in the design these residues should be');
    const inter = select('Interaction', INTERACTION_OPTIONS, now.inter ?? '', 'An interaction their side chains (or the atoms named) should make');
    const row = (label, field) => el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: label }), field);
    return {
      rows: [row('Shape', shape), row('Exposure', exposure), interaction ? row('Interaction', inter) : null],
      edits: (atoms, interactionAtoms = atoms) => {
        const edits = [];
        if (!atoms.length) return edits;
        if (shape.value !== 'mixed' && shape.value !== String(now.ss ?? 0)) {
          const v = Number(shape.value);
          edits.push({ condition: 'secondary_structure', atom_indices: atoms, enabled: v > 0, value: v, scope: 'residue' });
        }
        if (exposure.value !== 'mixed' && exposure.value !== String(now.rasa ?? 0)) {
          const v = Number(exposure.value);
          edits.push({ condition: 'rasa', atom_indices: atoms, enabled: v > 0, value: v });
        }
        if (interaction && inter.value !== 'mixed' && inter.value !== (now.inter ?? '')) {
          for (const [key] of INTERACTIONS) edits.push({ condition: CONDITION_KEY[key], atom_indices: atoms, enabled: false });
          if (inter.value && interactionAtoms.length) {
            const [key, v] = inter.value.split(':');
            edits.push({ condition: CONDITION_KEY[key], atom_indices: interactionAtoms, enabled: true, value: Number(v) });
          }
        }
        return edits;
      },
    };
  }

  /**
   * The Fold row for chain `c` (named folds, or a CATH code typed under "other"), and `apply()`, which sets it on the
   * chain's residues and new residues (each new residue takes it when the job is built). Resolves false on a bad code.
   */
  /**
   * The Fold row (CATH): of the whole chain from its chain popup, or with `block` (a new-residues piece) of those new
   * residues only, so a chain can hold domains of different folds (a kept domain and a β-barrel built onto it). The
   * label is per residue, as the model reads it.
   */
  _foldField(c, note, block = null) {
    const now = String((block ? block.fold : c.fold) ?? '').replace(/(\.x)+$/, '');
    const known = FOLD_OPTIONS.some(([v]) => v && v !== 'other' && v === now);
    const select = el('select', { 'aria-label': 'Fold',
      title: block ? 'A fold for these new residues to take (CATH); the rest of the chain keeps its own' : 'A fold for the whole chain to take (CATH)' });
    for (const [v, text] of FOLD_OPTIONS) {
      select.append(el('option', { value: v, text: v && v !== 'other' ? `${text} (${v})` : text,
        selected: v === (known ? now : now ? 'other' : '') }));
    }
    const code = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      value: known ? '' : now, placeholder: 'e.g. 3.30.70', 'aria-label': 'Fold code' });
    const show = () => { code.hidden = select.value !== 'other'; };
    select.addEventListener('change', show);
    show();
    const row = el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Fold' }), select, code);
    const apply = async () => {
      const want = select.value === 'other' ? code.value.trim() : select.value;
      if (want && !/^\d+(\.\d+){0,3}$/.test(want)) { note.textContent = 'A fold code is numbers with dots, e.g. 3.30.70'; return false; }
      const atoms = block ? block.atomIndices : [...c.polymerAtomIndices, ...c.segments.map((g) => g.atomIndex)];
      if (want === now || !atoms.length) return true;
      return this._batch([{ condition: 'cath_label', atom_indices: atoms, enabled: Boolean(want), value: want, scope: 'residue' }]);
    };
    return { row, apply };
  }

  /** The atoms an interaction is asked of: those named, else the side chains (a glycine's CA). */
  _interactionAtoms(atoms, which) {
    const a = this._inputDoc()?.atoms;
    if (!a) return [];
    if (Array.isArray(which)) {
      const names = new Set(which);
      return atoms.filter((i) => names.has(String(a.atom_name[i]).toUpperCase()));
    }
    const heavy = (i) => String(a.element?.[i] ?? '').toUpperCase() !== 'H';
    const bb = new Set(['N', 'CA', 'C', 'O', 'OXT']);
    // Each residue's side chain, or its CA where it has none (a glycine).
    const byRes = new Map();
    for (const i of atoms) {
      const k = `${a.chain_id[i]}|${a.res_id[i]}|${a.ins_code?.[i] ?? ''}`;
      byRes.set(k, [...(byRes.get(k) ?? []), i]);
    }
    return [...byRes.values()].flatMap((res) => {
      const side = res.filter((i) => heavy(i) && !bb.has(String(a.atom_name[i])));
      return side.length ? side : res.filter((i) => String(a.atom_name[i]) === 'CA');
    });
  }

  /** Existing residues: which of them, then three properties (structure, sequence, position), each fixed or designed. */
  _piecePop(c, p) {
    const a = this._inputDoc()?.atoms;
    // Which residues: one range of the chain as loaded, from – to and a drag bar. Narrowed, the piece is cropped;
    // widened, residues come back as loaded, up to its neighbours. Applied as the chain's plan (`/chains/rebuild`).
    const residuesText = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      value: `${p.start}–${p.end}`, 'aria-label': 'Residues' });
    const now = this._properties(p);
    const select = (label, options, value, title) => {
      const s = el('select', { 'aria-label': label, title });
      for (const [v, text] of options) s.append(el('option', { value: v, text, selected: v === value }));
      return s;
    };
    const structure = select('Structure', STRUCTURE_OPTIONS,
      now.structure, 'Fixed: kept where it is. Rigid: kept as it is, moving as one body (a framework). Free: the model builds it.');
    const sequence = select('Sequence', SEQUENCE_OPTIONS, now.sequence,
      'Fixed: these amino acids. Free: the model writes them.');
    const position = select('Position in chain', POSITION_OPTIONS, now.position,
      'Free: the model chooses where in the chain these residues go.');
    // A residue taken from the entry and renumbered (a motif's "A38") has no range here: its number would not match.
    const plan = p.source ? null : this._plan(c);
    const src = plan?.source?.length ? plan.source : null;
    const reach = (() => {
      const k = c.pieces.indexOf(p);
      const iS = src ? src.findIndex((r) => r.num === p.start) : -1;
      const iE = src ? src.findLastIndex((r) => r.num === p.end) : -1;
      if (iS < 0 || iE < 0 || k < 0) return { lo: p.start, hi: p.end, iLo: -1, iHi: -1 };
      const present = new Set((a ? c.polymerAtomIndices : []).map((i) => Number(a.res_id[i])));
      // Up to the next residue in the job, past new residues too: brought back, they go beside this piece.
      let iLo = iS;
      while (iLo > 0 && !present.has(src[iLo - 1].num)) iLo -= 1;
      let iHi = iE;
      while (iHi < src.length - 1 && !present.has(src[iHi + 1].num)) iHi += 1;
      return { lo: src[iLo].num, hi: src[iHi].num, iLo, iHi, present };
    })();
    const editable = reach.iLo >= 0;
    // The range typed ([[lo, hi]], clamped to what the piece can reach); null when not understood.
    const parse = () => {
      const m = /^\s*(?:[A-Za-z_]+)?(-?\d+)\s*(?:[-–]\s*(-?\d+))?\s*$/.exec(residuesText.value);
      return m ? [Number(m[1]), Number(m[2] ?? m[1])] : null;
    };
    const ranges = () => {
      if (!editable) return [[p.start, p.end]];
      const xy = parse();
      if (!xy) return null;
      const lo = Math.max(reach.lo, Math.min(...xy));
      const hi = Math.min(reach.hi, Math.max(...xy));
      return lo <= hi ? [[lo, hi]] : null;
    };
    // Residues the range holds once applied: the chain's as loaded from lo to hi.
    const count = () => {
      const [[lo, hi]] = ranges() ?? [[p.start, p.end]];
      return src && editable ? src.filter((r) => r.num >= lo && r.num <= hi).length : p.count;
    };
    const lenMin = el('input', { type: 'number', min: 1, max: 1000, value: count(), 'aria-label': 'Fewest residues' });
    const lenMax = el('input', { type: 'number', min: 1, max: 1000, value: count(), 'aria-label': 'Most residues' });
    let lenTouched = false;
    for (const input of [lenMin, lenMax]) input.addEventListener('input', () => { lenTouched = true; });
    // The chain as Apply would leave it, drawn as the range is edited, the chain as loaded faint behind it.
    const preview = () => {
      const rs = ranges();
      if (rs && plan && editable) this._previewParts(c, this._partsWithRange(plan, p, rs[0][0], rs[0][1]));
    };
    residuesText.addEventListener('input', () => {
      if (!lenTouched) lenMin.value = lenMax.value = String(count());
      preview();
    });
    const lengthRow = el('div', { class: 'rfd4p-pop-row', title: 'Each design picks one length from this range' },
      el('span', { class: 'rfd4p-pop-label', text: 'Length' }), el('span', { class: 'rfd4p-pop-inline' }, lenMin, el('span', { text: '–' }), lenMax));
    const atomsField = this._atomsField(now.atoms, now.names);
    const atomsRow = el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Atoms' }), atomsField.select, atomsField.text);
    // Side chains held means those side chains, so their amino acids; a fully designed piece has no position to keep,
    // only a length.
    const sync = () => {
      const sidechain = atomsField.select.value === 'sidechain' && structure.value !== 'designed';
      if (sidechain) sequence.value = 'fixed';
      sequence.disabled = sidechain;
      atomsRow.hidden = structure.value === 'designed';
      const free = structure.value === 'designed' && sequence.value === 'designed';
      if (free) position.value = 'fixed';
      positionRow.hidden = free;
      lengthRow.hidden = !free;
    };
    structure.addEventListener('change', sync);
    sequence.addEventListener('change', sync);
    atomsField.select.addEventListener('change', sync);

    const hot = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      value: formatResidueList(c.id, c.hotspotResIds, c.hotspotAtoms), placeholder: 'none', 'aria-label': 'Hotspots',
      title: 'Residues of this chain a binder should touch, e.g. A56, A60-62, or some of their atoms: A56(CG,OH)' });
    const keepOnly = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      placeholder: 'all', 'aria-label': 'Keep only these residues',
      title: 'Type some residues (e.g. A38, A99) to keep only those, in that order; the rest of the chain becomes new residues (a motif)' });
    const note = el('p', { class: 'rfd4p-pop-note', 'aria-live': 'polite' });
    const extra = this._extraFields({ ss: p.ss, rasa: p.rasa, inter: p.inter });
    const residuesIn = (text, inChain) => {
      const { residues, bad } = parseResidueList(text, c.id);
      const mine = residues.filter((r) => r.chain === c.id);
      // In the order typed; a residue may name some of its atoms.
      const want = new Set(mine.map((r) => r.resId));
      const atomsOf = new Map(mine.filter((r) => r.atoms).map((r) => [r.resId, r.atoms]));
      const byRes = new Map();
      for (const i of (a ? inChain : [])) {
        const id = Number(a.res_id[i]);
        if (want.has(id)) byRes.set(id, [...(byRes.get(id) ?? []), i]);
      }
      const atoms = [...want].flatMap((id) => byRes.get(id) ?? []);
      return { want, atomsOf, atoms, bad, missing: [...want].filter((id) => !byRes.has(id)) };
    };

    // Residues typed that this piece cannot reach (they are another piece's): say whose, and offer to open it.
    const elsewhere = () => {
      const xy = parse();
      const out = xy?.find((x) => x < reach.lo || x > reach.hi);
      if (out == null) return null;
      return { num: out, piece: c.pieces.find((q) => q !== p && q.kind === 'kept' && out >= q.start && out <= q.end) ?? null };
    };
    const apply = async () => {
      const away = !ranges() && elsewhere();
      if (away) {
        const name = (q) => `${c.id}${q.start}${q.end !== q.start ? `–${q.end}` : ''}`;
        note.replaceChildren(away.piece
          ? `${c.id}${away.num} is in ${name(away.piece)}, not this piece. `
          : `${c.id}${away.num} is not in this piece (${name(p)}).`);
        if (away.piece) {
          note.append(el('button', { type: 'button', class: 'rfd4p-link-btn', text: `Open ${name(away.piece)}`, onclick: () => {
            const card = [...(this.els.chainList?.querySelectorAll('.rfd4p-sub') ?? [])].find((x) => x.rfd4pAtoms?.[0] === away.piece.atomIndices[0]);
            this._closePop?.();
            card?.querySelector('.rfd4p-sub-open')?.click();
          } }));
        }
        return;
      }
      const rs = ranges();
      if (!rs) { note.textContent = `Residues from ${reach.lo} to ${reach.hi}`; return; }
      // Hotspots are the chain's, whichever of its pieces the popup was opened from.
      const hotspots = residuesIn(hot.value, c.polymerAtomIndices);
      if (hotspots.bad.length || hotspots.missing.length) {
        note.textContent = hotspots.bad.length ? `Hotspots not understood: ${hotspots.bad.join(' ')}`
          : `Hotspots not in chain ${c.id}: ${formatResidueList(c.id, hotspots.missing)}`;
        return;
      }
      const hotChanged = formatResidueList(c.id, [...hotspots.want], hotspots.atomsOf) !== formatResidueList(c.id, c.hotspotResIds, c.hotspotAtoms);
      // Re-read after each edit: atom indices move when residues are added or removed.
      const chainNow = () => this._spec()?.chains.find((x) => x.id === c.id);
      const setHotspots = async () => {
        const now = chainNow();
        if (!hotChanged || !now) return true;
        const { res_id: ids, atom_name: names } = this._inputDoc().atoms;
        const edits = [{ condition: 'hotspot', atom_indices: now.polymerAtomIndices, enabled: false }];
        const on = now.polymerAtomIndices.filter((i) => hotspots.want.has(Number(ids[i]))
          && (!hotspots.atomsOf.has(Number(ids[i])) || hotspots.atomsOf.get(Number(ids[i])).includes(String(names[i]).toUpperCase())));
        if (on.length) edits.push({ condition: 'hotspot', atom_indices: on, enabled: true });
        return this._batch(edits);
      };
      if (keepOnly.value.trim()) {
        const only = residuesIn(keepOnly.value, c.polymerAtomIndices);
        if (only.bad.length || only.missing.length || !only.atoms.length) {
          note.textContent = only.bad.length ? `Not understood: ${only.bad.join(' ')}` : `Not in chain ${c.id}: ${formatResidueList(c.id, only.missing)}`;
          return;
        }
        this._closePop();
        // Shape, exposure and interaction first: they change no atoms, so the edits after see the same indices.
        const extraEdits = extra.edits(p.atomIndices, this._interactionAtoms(p.atomIndices, atomsField.value()));
        if (extraEdits.length && !await this._batch(extraEdits)) return;
        // The residues kept, in the order typed, take this popup's properties: side chains and anywhere are part of the
        // set-up (an enzyme's site); "can move", other atoms or a designed sequence are set on them after.
        const which = atomsField.value();
        await this._runFlow('scaffold', this._spec(), {
          atoms: only.atoms, ordered: true, sidechainsOnly: which === 'sidechain', unindexed: position.value === 'anywhere',
        });
        if (structure.value === 'rigid' || sequence.value === 'designed' || Array.isArray(which) || which === 'backbone') {
          // An ordered motif is renumbered: the kept residues are the chain's non-new residues.
          const now = chainNow();
          const kept = (now?.pieces ?? []).filter((x) => x.kind !== 'new').flatMap((x) => x.atomIndices);
          if (kept.length) await this._applyPiece({ kind: 'kept', keep: 'all' }, kept, { structure: structure.value, atoms: which, sequence: sequence.value, position: position.value });
        }
        await setHotspots();
        return;
      }
      this._closePop();
      const [[lo, hi]] = rs;
      // A new range: the chain rebuilt to its plan (order, numbering and bonds all from it); residues brought back come
      // in as loaded and take the popup's settings below with the rest.
      const grew = lo < p.start || hi > p.end;
      if (editable && (lo !== p.start || hi !== p.end)) {
        if (!await this._mutate('/chains/rebuild', { chain_id: c.id, parts: this._partsWithRange(plan, p, lo, hi) })) return;
      }
      const mine = () => {
        const doc = this._inputDoc().atoms;
        return (chainNow()?.polymerAtomIndices ?? [])
          .filter((i) => !/^\d+-\d+$/.test(String(doc.res_name[i])) && Number(doc.res_id[i]) >= lo && Number(doc.res_id[i]) <= hi);
      };
      const atoms = editable ? mine() : p.atomIndices;
      // Shape, exposure and interaction: they change no atoms, so the edits after see the same indices.
      const extraEdits = extra.edits(atoms, this._interactionAtoms(atoms, atomsField.value()));
      if (extraEdits.length && !await this._batch(extraEdits)) return;
      if (!await setHotspots()) return;
      const want = { structure: structure.value, atoms: atomsField.value(), sequence: sequence.value, position: position.value };
      const atomsNow = now.atoms === 'named' ? this._atomNames(now.names).join(',') : now.atoms;
      const atomsWant = Array.isArray(want.atoms) ? want.atoms.join(',') : want.atoms;
      const changed = want.structure !== now.structure || want.sequence !== now.sequence || want.position !== now.position
        || (want.structure !== 'designed' && atomsWant !== atomsNow);
      // Residues brought back come as loaded (fixed); they take the popup's settings when those are not that.
      const asLoaded = want.structure === 'fixed' && want.sequence === 'fixed' && want.position === 'fixed' && (want.atoms === 'all' || !want.atoms);
      if (atoms.length && (changed || (grew && !asLoaded)) && !await this._applyPieces([{ p, atoms: editable ? mine() : atoms, want }])) return;
      // Lengths last: free residues of another length become new residues.
      if (!lengthRow.hidden && lenTouched) {
        const n = count();
        const min = Math.max(1, Math.round(Number(lenMin.value)) || n);
        const max = Math.max(min, Math.round(Number(lenMax.value)) || n);
        if (min !== n || max !== n) await this._mutate('/segments/replace', { atom_indices: editable ? mine() : atoms, minimum: min, maximum: max });
      }
    };

    const row = (label, ...children) => el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: label }), ...children);
    const crop = editable && src.length > 1 ? this._cropBar(c, p, src, reach, residuesText, ranges) : null;
    const positionRow = row('Position', position);
    sync();
    // The chain as loaded, its residues not in the job see-through, from the moment the popup opens.
    if (plan && editable) queueMicrotask(() => this._previewParts(c, this._parts(plan)));
    const name = `${c.id}${p.start}${p.end !== p.start ? `–${p.end}` : ''}`;
    return [
      this._popHead(this._pieceText(c, p)),
      // The range first: narrowing it is how a loaded structure is cut to the part that matters.
      editable
        ? row('Residues', el('span', { class: 'rfd4p-pop-inline' }, el('span', { text: c.id }), this._rangeBoxes(residuesText, reach)))
        : row('Residues', el('span', { class: 'rfd4p-pop-static', text: p.source ?? name,
          title: p.source ? `From the entry's ${p.source}; ${name} in this chain` : undefined })),
      crop,
      row('Structure', structure),
      atomsRow,
      row('Sequence', sequence),
      positionRow,
      lengthRow,
      row('Hotspots', hot),
      this._moreFold(
        ...extra.rows,
        row('Keep only', keepOnly)),
      note,
      this._popActions(apply),
    ];
  }

  /** A piece's three properties, as the popup and the row name them. */
  _properties(p) {
    if (p.kind !== 'kept') return { structure: 'designed', atoms: 'all', names: '', sequence: 'designed', position: 'fixed' };
    const structure = { all: 'fixed', backbone: 'fixed', sidechain: 'fixed', shape: 'rigid' }[p.keep] ?? 'designed';
    const sequence = p.keep === 'backbone' ? 'designed' : 'fixed';
    // Backbone held with a designed sequence is the plain "structure fixed": the side chains change anyway.
    let atoms = p.atoms ?? 'all';
    if (sequence === 'designed' && atoms === 'backbone') atoms = 'all';
    const named = !['all', 'backbone', 'sidechain'].includes(atoms);
    return { structure, atoms: named ? 'named' : atoms, names: named ? atoms : '', sequence, position: p.anywhere ? 'anywhere' : 'fixed' };
  }

  /** Atom names typed as "CG, OD1" or "CG OD1". */
  _atomNames(text) {
    return String(text ?? '').split(/[\s,;]+/).filter(Boolean).map((n) => n.toUpperCase());
  }

  /** The Atoms row: a select, and the names it holds when "named". */
  _atomsField(now = 'all', names = '') {
    const select = el('select', { 'aria-label': 'Atoms', title: 'Which atoms are held: all of them, the backbone, the side chains, or atoms named here' });
    if (now === 'mixed') select.append(el('option', { value: 'mixed', text: 'mixed', selected: true }));
    for (const [v, text] of ATOM_OPTIONS) select.append(el('option', { value: v, text, selected: v === now }));
    const text = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      value: names, placeholder: 'CG, OD1', 'aria-label': 'Atom names' });
    const show = () => { text.hidden = select.value !== 'named'; };
    select.addEventListener('change', show);
    show();
    return { select, text, value: () => (select.value === 'named' ? this._atomNames(text.value) : select.value) };
  }

  /** Set structure / sequence / position on `atoms` (all or part of piece `p`). Rigid holds are internal coordinates. */
  async _applyPiece(p, atoms, want) {
    return this._applyPieces([{ p, atoms, want }]);
  }

  /** `_applyPiece` for several pieces at once: one condition edit, so one Undo. */
  async _applyPieces(parts) {
    // Internal coordinates are their own edit: clear them before anything else, set them before the rest.
    const off = parts.filter(({ p }) => p.kind === 'kept' && p.keep === 'shape').flatMap(({ atoms }) => atoms);
    if (off.length && !await this._mutate('/internal-coords', { atom_indices: off, enabled: false })) return false;
    // "Rigid" is one rigid body for the whole job (an antibody's two framework chains, a framework split by
    // CDRs): join the rigid group already there. Separate bodies are the Advanced editor's.
    // One rigid body per chain (an antibody's light and heavy frameworks move apart; a framework split by CDRs moves
    // as one), as the bundled examples set them. Only the held atoms join it.
    const rigid = parts.filter(({ want }) => want.structure === 'rigid');
    const chains = [...new Set(rigid.flatMap(({ atoms }) => atoms.map((i) => String(this._inputDoc().atoms.chain_id[i]))))];
    for (const chain of chains) {
      const a = this._inputDoc().atoms;
      const on = rigid.flatMap(({ atoms, want }) => this._heldAtoms(atoms.filter((i) => String(a.chain_id[i]) === chain), want));
      if (on.length && !await this._mutate('/internal-coords', { atom_indices: on, enabled: true, group_id: this._rigidGroup(chain) })) return false;
      // No saved targets: the body's shape is read from its coordinates when the job is built, centred and turned at
      // random, as AtomWorks does for a group without targets (and as the bundled examples are saved).
      if (on.length && !await this._mutate('/internal-coords/clear-targets', { atom_indices: on })) return false;
    }
    if (!await this._batch(parts.flatMap(({ atoms, want }) => this._pieceEdits(atoms, want)))) return false;
    // A residue placed anywhere keeps only its held atoms' coordinates (an enzyme's catalytic tips): where the rest of
    // it sits is the model's to choose, as the motif inputs of RFdiffusion benchmarks give it.
    const loose = parts.filter(({ want }) => want.position === 'anywhere' && want.structure === 'fixed')
      .flatMap(({ atoms, want }) => { const held = new Set(this._heldAtoms(atoms, want)); return held.size ? atoms.filter((i) => !held.has(i)) : []; });
    // The held atoms lead their residue then: a token is read from its first atom.
    const lead = parts.filter(({ want }) => want.position === 'anywhere' && want.structure === 'fixed').flatMap(({ atoms, want }) => this._heldAtoms(atoms, want));
    return loose.length ? this._mutate('/coordinates/clear', { atom_indices: loose, lead }) : true;
  }

  /** The atoms of `atoms` a structure setting holds. A fixed structure with a designed sequence holds the backbone. */
  _heldAtoms(atoms, { structure, atoms: which = 'all', sequence }) {
    if (structure === 'designed') return [];
    const a = this._inputDoc()?.atoms;
    const backbone = new Set(this._backboneOf(atoms));
    if (Array.isArray(which)) {
      const names = new Set(which);
      return atoms.filter((i) => names.has(String(a.atom_name[i]).toUpperCase()));
    }
    if (which === 'backbone' || (which === 'all' && sequence === 'designed')) return atoms.filter((i) => backbone.has(i));
    if (which === 'sidechain') return atoms.filter((i) => !backbone.has(i));
    return atoms;
  }

  _pieceEdits(atoms, want) {
    const { structure, sequence, position } = want;
    const edits = [];
    const held = this._heldAtoms(atoms, want);
    if (structure !== 'rigid') {
      edits.push({ condition: 'coordinate', atom_indices: atoms, enabled: false });
      if (held.length) edits.push({ condition: 'coordinate', atom_indices: held, enabled: true });
    } else {
      // The rigid body is the held atoms with coordinates (set before this edit); the others are free.
      const coord = this._inputDoc().atoms.coord;
      const inBody = new Set(held.filter((i) => coord[i]?.every?.(Number.isFinite)));
      const free = atoms.filter((i) => !inBody.has(i));
      if (free.length) edits.push({ condition: 'coordinate', atom_indices: free, enabled: false });
    }
    edits.push({ condition: 'sequence', atom_indices: atoms, enabled: sequence === 'fixed' });
    edits.push({ condition: 'index', atom_indices: atoms, enabled: position === 'anywhere' });
    return edits;
  }

  /**
   * The chain as it was loaded, as a bar: this piece's ranges on it, the chain's other residues muted, and a handle at
   * each end. Dragging a handle in crops the piece, out brings residues back (up to its neighbours); it edits the
   * Residues field, so Apply works as typed.
   */
  /**
   * A range of residues as two boxes (from – to), as new residues take their length, kept in step with `field`, the
   * hidden text the popup reads ("1–40") and the crop bar writes.
   */
  _rangeBoxes(field, reach) {
    field.hidden = true;
    const box = el('span', { class: 'rfd4p-ranges' });
    let writing = false;
    const read = () => {
      const m = /^\s*(-?\d+)\s*(?:[-–]\s*(-?\d+))?\s*$/.exec(String(field.value));
      return m ? [m[1], m[2] ?? m[1]] : ['', ''];
    };
    const write = () => {
      const [x, y] = [...box.querySelectorAll('input')].map((i) => i.value.trim());
      writing = true;
      field.value = !x && !y ? '' : x === y || !y ? x : !x ? y : `${x}–${y}`;
      field.dispatchEvent(new Event('input', { bubbles: true }));
      writing = false;
    };
    const draw = () => {
      const num = (v, label) => {
        const i = el('input', { type: 'number', value: v, min: reach.lo, max: reach.hi, step: 1, 'aria-label': label });
        i.addEventListener('input', write);
        return i;
      };
      const [x, y] = read();
      box.replaceChildren(el('span', { class: 'rfd4p-range' }, num(x, 'From residue'), el('span', { text: '–' }), num(y, 'To residue')));
    };
    field.addEventListener('input', () => { if (!writing) draw(); });
    draw();
    return el('span', { class: 'rfd4p-ranges-wrap' }, box, field);
  }

  _cropBar(c, p, src, reach, field, ranges) {
    const n = src.length;
    const pct = (k) => `${(k / n) * 100}%`;
    const first = (num) => src.findIndex((r) => r.num === num);
    const last = (num) => src.findLastIndex((r) => r.num === num);
    const track = el('div', { class: 'rfd4p-crop-track' });
    const bar = el('div', { class: 'rfd4p-crop', title: `Drag the ends to crop, or to bring back residues (${c.id}${src[0].num}–${src.at(-1).num} as loaded)` }, track);
    const block = (cls, i0, i1) => el('span', { class: cls, style: `left:${pct(i0)};width:${pct(i1 - i0 + 1)}` });
    const handles = ['lo', 'hi'].map((side) => el('span', { class: `rfd4p-crop-handle rfd4p-crop-${side}`, role: 'slider',
      'aria-label': side === 'lo' ? 'First residue' : 'Last residue' }));
    const draw = () => {
      const rs = ranges() ?? [[p.start, p.end]];
      const own = new Set(p.atomIndices.length ? src.filter((r) => r.num >= p.start && r.num <= p.end).map((r) => r.num) : []);
      const parts = [];
      // The chain's other residues, in runs.
      let run = null;
      src.forEach((r, k) => {
        const other = reach.present?.has(r.num) && !own.has(r.num);
        if (other) { if (run) run[1] = k; else run = [k, k]; }
        if ((!other || k === n - 1) && run) { parts.push(block('rfd4p-crop-other', ...run)); run = null; }
      });
      for (const [x, y] of rs) parts.push(block('rfd4p-crop-sel', first(x), last(y)));
      track.replaceChildren(...parts);
      handles[0].style.left = pct(first(rs[0][0]));
      handles[1].style.left = pct(last(rs.at(-1)[1]) + 1);
      handles[0].setAttribute('aria-valuenow', String(rs[0][0]));
      handles[1].setAttribute('aria-valuenow', String(rs.at(-1)[1]));
    };
    const set = (side, k) => {
      const rs = (ranges() ?? [[p.start, p.end]]).map((r) => r.slice());
      if (side === 'lo') rs[0][0] = src[Math.max(reach.iLo, Math.min(k, last(rs[0][1])))].num;
      else rs.at(-1)[1] = src[Math.min(reach.iHi, Math.max(k, first(rs.at(-1)[0])))].num;
      field.value = rs.map(([x, y]) => (x === y ? `${x}` : `${x}–${y}`)).join(', ');
      field.dispatchEvent(new Event('input', { bubbles: true }));
    };
    handles.forEach((h, j) => {
      const side = j ? 'hi' : 'lo';
      h.addEventListener('pointerdown', (ev) => {
        if (ev.button !== 0) return;
        ev.preventDefault();
        ev.stopPropagation();
        try { h.setPointerCapture(ev.pointerId); } catch { /* not an active pointer */ }
        const move = (e) => {
          const r = track.getBoundingClientRect();
          const f = (e.clientX - r.left) / r.width;
          set(side, side === 'lo' ? Math.round(f * n) : Math.round(f * n) - 1);
        };
        const up = () => { h.removeEventListener('pointermove', move); h.removeEventListener('pointerup', up); h.removeEventListener('pointercancel', up); };
        h.addEventListener('pointermove', move);
        h.addEventListener('pointerup', up);
        h.addEventListener('pointercancel', up);
      });
    });
    bar.append(...handles);
    field.addEventListener('input', draw);
    draw();
    return bar;
  }

  /**
   * New residues: how many, or remove them. New residues that replaced residues (a free piece given another length)
   * turn back into them, with their coordinates, at their old count or with a fixed Structure or Sequence.
   */
  _newPop(c, p) {
    const min = el('input', { type: 'number', min: 1, max: 1000, value: p.min, 'aria-label': 'Fewest new residues' });
    const max = el('input', { type: 'number', min: 1, max: 1000, value: p.max, 'aria-label': 'Most new residues' });
    const origin = this._inputDoc()?.segmentOrigin?.(p.atomIndex) ?? null;
    const select = (label, options, title) => {
      const s = el('select', { 'aria-label': label, title });
      for (const [v, text] of options) s.append(el('option', { value: v, text, selected: v === 'designed' }));
      return s;
    };
    const structure = origin && select('Structure', STRUCTURE_OPTIONS, 'Fixed or rigid: the residues these replaced, back in place');
    const sequence = origin && select('Sequence', SEQUENCE_OPTIONS, 'Fixed: the residues these replaced, back with their amino acids');
    // New residues can be asked a shape and an exposure: each of them takes it when the job is built.
    const extra = this._extraFields({ ss: p.ss, rasa: p.rasa }, { interaction: false });
    const note = el('p', { class: 'rfd4p-pop-note', 'aria-live': 'polite' });
    // The chain's fold, here too: where a new chain is laid out is where one looks for it.
    const fold = this._foldField(c, note, p);
    const apply = async () => {
      const lo = Math.max(1, Math.round(Number(min.value)) || p.min);
      const hi = Math.max(lo, Math.round(Number(max.value)) || p.max);
      if (!await fold.apply()) return false;
      this._closePop();
      const want = origin && { structure: structure.value, atoms: 'all', sequence: sequence.value, position: 'fixed' };
      if (origin && (want.structure !== 'designed' || want.sequence !== 'designed' || (lo === origin.count && hi === origin.count))) {
        const back = await this._restoreSegment(c.id, p.atomIndex);
        if (!back) return false;
        const edits = extra.edits(back);
        if (edits.length && !await this._batch(edits)) return false;
        const free = want.structure === 'designed' && want.sequence === 'designed';
        return free || this._applyPieces([{ p: { kind: 'kept' }, atoms: back, want }]);
      }
      // Atom indices may have moved if the fold was set: re-read this block's.
      const now = this._spec()?.chains.find((x) => x.id === c.id)?.pieces.find((x) => x.kind === 'new' && x.atomIndex === p.atomIndex) ?? p;
      const edits = extra.edits(now.atomIndices);
      if (edits.length && !await this._batch(edits)) return false;
      return lo !== p.min || hi !== p.max ? this._mutate('/segments/range', { atom_index: p.atomIndex, minimum: lo, maximum: hi }) : true;
    };
    const row = (label, ...children) => el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: label }), ...children);
    // The old count, one click away: set back to it, the residues come back in place.
    const was = origin && el('button', { type: 'button', class: 'rfd4p-pop-was', text: `was ${origin.count}`,
      title: 'The residues these replaced, back in place, free', onclick: () => { min.value = origin.count; max.value = origin.count; } });
    return [
      this._popHead(`Chain ${c.id}, new residues`),
      el('div', { class: 'rfd4p-pop-row', title: 'Each design picks one length from this range' },
        el('span', { class: 'rfd4p-pop-label', text: 'Length' }), el('span', { class: 'rfd4p-pop-inline' }, min, el('span', { text: '–' }), max, was)),
      origin && row('Structure', structure),
      origin && row('Sequence', sequence),
      // Shape first (blocks are drawn with it), then exposure and the chain's fold.
      extra.rows[0],
      this._moreFold(extra.rows[1], fold.row),
      note,
      this._popActions(apply),
    ].filter(Boolean);
  }

  /** New residues back into the residues they replaced; their atoms after, or null. */
  async _restoreSegment(chainId, atomIndex) {
    const keys = this._inputDoc()?.segmentOrigin?.(atomIndex)?.keys;
    if (!keys || !await this._mutate('/segments/restore', { atom_index: atomIndex })) return null;
    const a = this._inputDoc().atoms;
    const want = new Set(keys);
    const atoms = [];
    for (let i = 0; i < a.nAtoms; i += 1) if (want.has(`${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code[i]}`)) atoms.push(i);
    return atoms;
  }

  /** Every remembered segment back into its residues (partial diffusion's fix), last first so indices hold. */
  async _restoreAll() {
    const doc = this._inputDoc();
    const segs = [];
    for (let i = 0; i < (doc?.atoms.nAtoms ?? 0); i += 1) if (doc.segmentOrigin?.(i)) segs.push(i);
    for (const i of segs.reverse()) if (!await this._restoreSegment(String(doc.atoms.chain_id[i]), i)) return;
  }

  /** A ligand: where it goes, or another molecule in its place. */
  _ligandPop(c, l) {
    const atoms = l.atomIndices;
    const pose = el('select', { 'aria-label': 'Structure', title: 'Fixed: kept where it is. Free: the model places it.' },
      el('option', { value: 'fixed', text: 'fixed', selected: l.pose !== 'predicted' }),
      el('option', { value: 'free', text: 'free', selected: l.pose === 'predicted' }));
    // Which atoms a fixed pose holds: all, or some named (the rest placed by the model).
    const held = el('select', { 'aria-label': 'Atoms', title: 'Hold all of the ligand, or only the atoms named here' },
      el('option', { value: 'all', text: 'all', selected: !l.heldNames }),
      el('option', { value: 'named', text: 'named', selected: Boolean(l.heldNames) }));
    const names = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      value: l.heldNames ?? '', placeholder: 'N1, C2', 'aria-label': 'Atom names' });
    const code = el('input', { type: 'text', class: 'rfd4p-pop-text', maxlength: 5, spellcheck: 'false', autocomplete: 'off',
      placeholder: `another code, e.g. HEM`, 'aria-label': 'Swap for ligand code' });
    const note = el('p', { class: 'rfd4p-pop-note', 'aria-live': 'polite' });
    const atomsRow = el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Atoms' }), held, names);
    const sync = () => {
      atomsRow.hidden = pose.value !== 'fixed';
      names.hidden = held.value !== 'named';
    };
    pose.addEventListener('change', sync);
    held.addEventListener('change', sync);
    sync();
    const apply = async () => {
      if (code.value.trim()) {
        note.textContent = `Looking up ${code.value.trim().toUpperCase()}…`;
        try {
          await this._addLigand(code.value, { replace: atoms, chainId: c.id });
        } catch (err) {
          note.textContent = String(err?.message ?? err).replace(/^Error:\s*/, '');
          return;
        }
        this._closePop();
        return;
      }
      const a = this._inputDoc().atoms;
      const wanted = pose.value !== 'fixed' ? [] : held.value === 'named'
        ? (() => { const set = new Set(this._atomNames(names.value)); return atoms.filter((i) => set.has(String(a.atom_name[i]).toUpperCase())); })()
        : atoms;
      if (pose.value === 'fixed' && !wanted.length) {
        note.textContent = `No atoms named ${names.value.trim() || 'here'} in ${l.name}`;
        return;
      }
      this._closePop();
      const nowHeld = pose.value === 'fixed' && (held.value === 'named' ? names.value.trim() !== (l.heldNames ?? '') : Boolean(l.heldNames));
      if (nowHeld || (pose.value === 'fixed') !== (l.pose !== 'predicted')) {
        const edits = [{ condition: 'coordinate', atom_indices: atoms, enabled: false }];
        if (wanted.length) edits.push({ condition: 'coordinate', atom_indices: wanted, enabled: true });
        void this._batch(edits);
      }
    };
    return [
      this._popHead(`Ligand ${l.name}`),
      el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Structure' }), pose),
      atomsRow,
      el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Swap' }), code),
      note,
      this._popActions(apply),
    ].filter(Boolean);
  }

  /** A chain: its name and its place in the job (the order the model reads the chains in). */
  _chainPop(c) {
    const ids = (this._spec()?.chains ?? []).map((x) => x.id);
    const name = el('input', { type: 'text', class: 'rfd4p-pop-text', maxlength: 4, spellcheck: 'false', autocomplete: 'off',
      value: c.id, 'aria-label': 'Chain name' });
    const order = el('select', { 'aria-label': 'Chain order', title: 'Where this chain comes in the job' });
    ids.forEach((id, k) => order.append(el('option', { value: String(k), text: String(k + 1), selected: id === c.id })));
    const note = el('p', { class: 'rfd4p-pop-note', 'aria-live': 'polite' });
    const fold = c.role !== 'ligand' ? this._foldField(c, note) : null;
    const apply = async () => {
      if (fold && !await fold.apply()) return;
      const next = name.value.trim();
      if (!/^[A-Za-z0-9_]{1,4}$/.test(next)) { note.textContent = 'A chain name is 1–4 letters or digits'; return; }
      if (next !== c.id && ids.includes(next)) { note.textContent = `There is a chain ${next} already`; return; }
      this._closePop();
      if (next !== c.id) {
        if (!await this._mutate('/chains', { atom_indices: c.atomIndices, chain_id: next })) return;
        // Its caption ("PDB 3EAK · B") follows it too.
        const label = this.sources?.get(this._inputDoc())?.get(c.id);
        if (label) this._rememberSource([next], label);
      }
      const at = Number(order.value);
      if (at !== ids.indexOf(c.id)) {
        const rest = ids.filter((id) => id !== c.id).map((id) => (id === c.id ? next : id));
        rest.splice(at, 0, next);
        await this._mutate('/chains/order', { chain_ids: rest });
      }
    };
    return [
      this._popHead(`Chain ${c.id}`),
      el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Name' }), name),
      el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Order' }), order),
      fold?.row,
      note,
      this._popActions(apply),
    ];
  }

  /** Chain `chainId`'s rigid group ("fixed, can move"): the one it has, else the next free number. */
  _rigidGroup(chainId) {
    const a = this._inputDoc()?.atoms;
    const grp = a?.annotations?.condition_CoordinateGroup_1_atom;
    // Chains from the same structure share one body (an antibody's heavy and light chains keep their pairing), as
    // upstream's antibody and VHH examples are set; chains from different structures move apart.
    const chains = this._spec()?.chains ?? [];
    const source = this._sourceOf(chains.find((c) => c.id === String(chainId)) ?? { id: chainId, role: 'new' });
    const same = new Set(source ? chains.filter((c) => this._sourceOf(c) === source).map((c) => c.id) : [String(chainId)]);
    same.add(String(chainId));
    let own = null;
    let top = 0;
    for (let i = 0; i < (grp?.length ?? 0); i += 1) {
      const g = Number(grp[i]);
      if (!(g > 0)) continue;
      top = Math.max(top, g);
      if (same.has(String(a.chain_id[i])) && (own === null || g < own)) own = g;
    }
    return own ?? top + 1;
  }

  _backboneOf(indices) {
    const a = this._inputDoc()?.atoms;
    if (!a) return [];
    return indices.filter((i) => ['N', 'CA', 'C', 'O'].includes(String(a.atom_name[i])));
  }

  /**
   * The shown document's hotspot residues coloured amber in the viewer, in the structure's own cartoon (Studio's
   * `setHighlight`; side-chain sticks were too busy): the input's while editing and while a design runs, a
   * design's own when one is shown (designs carry the input's hotspots, web/app.js `markHotspots`). Studio keeps a
   * highlight across loads and live frames.
   */
  showHotspots() {
    if (typeof window.atomworksStudio?.setHighlight !== 'function') return;
    const shown = this.app.studioSessionId ? getSharedStudioStore().getSession(this.app.studioSessionId) : null;
    const atoms = (shown ?? this._inputDoc())?.atoms;
    const hot = atoms?.annotations?.condition_hotspot_1_atom;
    const residues = new Map();
    // A run in progress shows live frames over the input, numbered as the job is (past any new residues).
    const live = this.app.liveHotspotKeys;
    if (live && (!shown || shown === this._inputDoc())) {
      for (const key of live) {
        const [chain, residue] = key.split('|');
        residues.set(`${chain}|${residue}`, { chain, residue: Number(residue) });
      }
    } else if (hot) {
      for (let i = 0; i < atoms.nAtoms; i += 1) {
        // The viewer numbers residues as the document's CIF writes auth_seq_id (a fetched entry keeps its authors'
        // numbering there), not by res_id: 5O45's A40 is author residue 56.
        const residue = Number(atoms.auth_seq_id?.[i] ?? atoms.res_id[i]);
        if (hot[i]) residues.set(`${atoms.chain_id[i]}|${residue}`, { chain: String(atoms.chain_id[i]), residue });
      }
    }
    // Residues the model rebuilds (no atom of theirs held) are faded in the input: what stays is what is solid.
    const free = new Map();
    const crd = atoms?.annotations?.condition_coordinate_1_atom;
    // Not with partial diffusion: then the free residues' coordinates are where the model starts.
    if (atoms && !live && (!shown || shown === this._inputDoc()) && !this.app.els.partialDiffusion?.checked) {
      const held = new Set();
      const key = (i) => `${atoms.chain_id[i]}|${Number(atoms.auth_seq_id?.[i] ?? atoms.res_id[i])}`;
      // Held in place, or rigid (a body of its own: kept as it is, moved as one).
      const grp = atoms.annotations?.condition_CoordinateGroup_1_atom;
      for (let i = 0; i < atoms.nAtoms; i += 1) if (Number.isFinite(crd?.[i]?.[0]) || Number(grp?.[i]) > 0) held.add(key(i));
      for (let i = 0; i < atoms.nAtoms; i += 1) {
        const k = key(i);
        if (held.has(k) || free.has(k) || /^\d+-\d+$/.test(String(atoms.res_name[i]))) continue;
        free.set(k, { chain: String(atoms.chain_id[i]), residue: Number(atoms.auth_seq_id?.[i] ?? atoms.res_id[i]) });
      }
    }
    // Sent on every render rather than once: Studio ignores it until its viewer has started.
    this.hotspotQueue = (this.hotspotQueue ?? Promise.resolve())
      .then(() => window.atomworksStudio.setHighlight('hotspots', residues.size ? [...residues.values()] : null, { color: HOTSPOT_COLOR }))
      .then(() => window.atomworksStudio.setHighlight('free', free.size ? [...free.values()] : null, { transparency: FREE_TRANSPARENCY }))
      .catch((err) => console.warn('Hotspot highlight failed:', err));
  }

  // ---- selection popup ----------------------------------------------------------------------------------------

  /** The popup for the residues picked in the viewer or the sequence bar (or `picked`: atoms, as an example is built). */
  _renderPop(picked = null) {
    const pop = this.pop;
    const doc = this._inputDoc();
    const sel = picked ?? window.atomworksStudio?.getSelection?.() ?? [];
    if (!sel.length || !doc || (!picked && (this._viewingDesign() || document.body.classList.contains('rfd4p-advanced'))) || this.busy) {
      pop.hidden = true;
      return;
    }
    const a = doc.atoms;
    const residues = doc.resolveIndices(sel, 'residue');
    const sentinel = (i) => /^\d+-\d+$/.test(String(a.res_name[i]));
    const ligandAtoms = residues.filter((i) => a.hetero[i] && !sentinel(i));
    const polymerAtoms = residues.filter((i) => !a.hetero[i]);
    // "A60–65, B3" style summary of the selected residues.
    const byChain = new Map();
    for (const i of polymerAtoms) {
      const ids = byChain.get(a.chain_id[i]) ?? new Set();
      ids.add(a.res_id[i]);
      byChain.set(a.chain_id[i], ids);
    }
    const parts = [];
    let nRes = 0;
    for (const [chain, ids] of byChain) {
      const sorted = [...ids].sort((x, y) => x - y);
      nRes += sorted.length;
      let start = sorted[0];
      for (let k = 1; k <= sorted.length; k += 1) {
        if (k === sorted.length || sorted[k] !== sorted[k - 1] + 1) {
          parts.push(start === sorted[k - 1] ? `${chain}${start}` : `${chain}${start}–${sorted[k - 1]}`);
          start = sorted[k];
        }
      }
    }
    const ligNames = [...new Set(ligandAtoms.map((i) => a.res_name[i]))];
    const ann = a.annotations ?? {};
    const hasCrd = (i) => Array.isArray(ann.condition_coordinate_1_atom?.[i]) && Number.isFinite(ann.condition_coordinate_1_atom[i][0]);
    const hotN = polymerAtoms.filter((i) => ann.condition_hotspot_1_atom?.[i]).length;

    // The pieces the selection touches: each keeps what the popup leaves as "mixed".
    const inSel = new Set(polymerAtoms);
    const hits = [];
    for (const c of this._spec()?.chains ?? []) {
      for (const p of c.pieces ?? []) {
        if (p.kind === 'new' || !p.atomIndices?.length) continue;
        const atoms = p.atomIndices.filter((i) => inSel.has(i));
        if (atoms.length) hits.push({ c, p, atoms, now: this._properties(p) });
      }
    }
    const common = (key) => {
      const v = new Set(hits.map((h) => h.now[key]));
      return v.size === 1 ? [...v][0] : 'mixed';
    };
    const select = (label, options, value) => {
      const s = el('select', { 'aria-label': label });
      if (value === 'mixed') s.append(el('option', { value: 'mixed', text: 'mixed', selected: true }));
      for (const [v, text] of options) s.append(el('option', { value: v, text, selected: v === value }));
      return s;
    };
    const row = (label, ...children) => el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: label }), ...children);
    const rows = [];
    const more = [];
    const title = [parts.join(', '), ligNames.join(', ')].filter(Boolean).join(' · ');
    // Its × closes it and lets go of the selection.
    const head = this._popHead(title || `${sel.length} atoms`, () => document.getElementById('clear-selection')?.click(),
      nRes ? `${nRes} residue${nRes === 1 ? '' : 's'}` : null);
    const done = () => { pop.hidden = true; };
    let apply = async () => {};

    if (hits.length) {
      const structure = select('Structure', STRUCTURE_OPTIONS, common('structure'));
      const sequence = select('Sequence', SEQUENCE_OPTIONS, common('sequence'));
      const position = select('Position in chain', POSITION_OPTIONS, common('position'));
      // One stretch of one chain can become new residues of another length.
      const oneStretch = byChain.size === 1 && parts.length === 1;
      const lenMin = el('input', { type: 'number', min: 1, max: 1000, value: nRes, 'aria-label': 'Fewest residues' });
      const lenMax = el('input', { type: 'number', min: 1, max: 1000, value: nRes, 'aria-label': 'Most residues' });
      const positionRow = row('Position', position);
      const lengthRow = row('Length', el('span', { class: 'rfd4p-pop-inline' }, lenMin, el('span', { text: '–' }), lenMax));
      const atomKey = (h) => (h.now.atoms === 'named' ? `named:${h.now.names}` : h.now.atoms);
      const atomKeys = new Set(hits.map(atomKey));
      const oneAtoms = atomKeys.size === 1 ? hits[0].now : null;
      const atomsField = this._atomsField(oneAtoms ? oneAtoms.atoms : 'mixed', oneAtoms?.names ?? '');
      const atomsRow = row('Atoms', atomsField.select, atomsField.text);
      const sync = () => {
        const sidechain = atomsField.select.value === 'sidechain' && structure.value !== 'designed';
        if (sidechain) sequence.value = 'fixed';
        sequence.disabled = sidechain;
        atomsRow.hidden = structure.value === 'designed';
        const free = structure.value === 'designed' && sequence.value === 'designed';
        if (free && position.value !== 'mixed') position.value = 'fixed';
        positionRow.hidden = free;
        lengthRow.hidden = !free || !oneStretch;
      };
      structure.addEventListener('change', sync);
      sequence.addEventListener('change', sync);
      atomsField.select.addEventListener('change', sync);
      sync();
      const hot = el('input', { type: 'checkbox', checked: hotN > 0 && hotN === polymerAtoms.length, 'aria-label': 'Hotspot' });
      hot.indeterminate = hotN > 0 && hotN < polymerAtoms.length;
      let hotTouched = false;
      hot.addEventListener('change', () => { hotTouched = true; });
      const same = (key) => { const v = new Set(hits.map((h) => h.p[key] ?? (key === 'inter' ? '' : 0))); return v.size === 1 ? [...v][0] : 'mixed'; };
      const extra = this._extraFields({ ss: same('ss'), rasa: same('rasa'), inter: same('inter') });
      // A pick asks little at first: fixed or free, a hotspot, in or out; the finer settings are under More.
      rows.push(row('Structure', structure), row('Sequence', sequence), lengthRow,
        el('label', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Hotspot' }), hot));
      more.push(atomsRow, positionRow, ...extra.rows.filter(Boolean));
      apply = async () => {
        done();
        if (hotTouched && !await this._batch([{ condition: 'hotspot', atom_indices: polymerAtoms, enabled: hot.checked }])) return;
        const which = atomsField.select.value === 'mixed' ? 'all' : atomsField.value();
        const extraEdits = extra.edits(polymerAtoms, this._interactionAtoms(polymerAtoms, which));
        if (extraEdits.length && !await this._batch(extraEdits)) return;
        if (!lengthRow.hidden) {
          const min = Math.max(1, Math.round(Number(lenMin.value)) || nRes);
          const max = Math.max(min, Math.round(Number(lenMax.value)) || nRes);
          if (min !== nRes || max !== nRes) {
            await this._mutate('/segments/replace', { atom_indices: polymerAtoms, minimum: min, maximum: max });
            return;
          }
        }
        const pick = (s, h, key) => (s.value === 'mixed' ? h.now[key] : s.value);
        const changes = [];
        for (const h of hits) {
          const nowAtoms = h.now.atoms === 'named' ? this._atomNames(h.now.names) : h.now.atoms;
          const want = {
            structure: pick(structure, h, 'structure'), sequence: pick(sequence, h, 'sequence'), position: pick(position, h, 'position'),
            atoms: atomsField.select.value === 'mixed' ? nowAtoms : atomsField.value(),
          };
          if (want.atoms === 'sidechain' && want.structure !== 'designed') want.sequence = 'fixed';
          if (want.structure === 'designed' && want.sequence === 'designed') want.position = 'fixed';
          const sameAtoms = String(want.atoms) === String(nowAtoms) || want.structure === 'designed';
          if (want.structure !== h.now.structure || want.sequence !== h.now.sequence || want.position !== h.now.position || !sameAtoms) {
            changes.push({ p: h.p, atoms: h.atoms, want });
          }
        }
        if (changes.length) await this._applyPieces(changes);
      };
    }
    if (ligandAtoms.length) {
      const nFixed = ligandAtoms.filter(hasCrd).length;
      const pose = select('Structure', [['fixed', 'fixed'], ['free', 'free']],
        nFixed === ligandAtoms.length ? 'fixed' : nFixed === 0 ? 'free' : 'mixed');
      // Beside residues, the row names the ligand; alone, it is the ligand's Structure like any other.
      rows.push(row(!hits.length ? 'Structure' : ligNames.length === 1 ? ligNames[0] : 'Ligands', pose));
      const polymerApply = apply;
      apply = async () => {
        done();
        if (pose.value !== 'mixed') {
          const fix = pose.value === 'fixed';
          const want = fix ? nFixed !== ligandAtoms.length : nFixed !== 0;
          if (want && !await this._batch([{ condition: 'coordinate', atom_indices: ligandAtoms, enabled: fix }])) return;
        }
        await polymerApply();
      };
    }
    // Unticked, the selection leaves the job (nothing else is applied then).
    const include = el('input', { type: 'checkbox', checked: true, 'aria-label': 'Include' });
    rows.push(el('label', { class: 'rfd4p-pop-row', title: 'Untick to take these out of the job' },
      el('span', { class: 'rfd4p-pop-label', text: 'Include' }), include));
    const applyEdits = apply;
    apply = async () => {
      if (include.checked) return applyEdits();
      done();
      return this._mutate('/delete', { atom_indices: residues });
    };
    pop.replaceChildren(...[head, ...rows, more.length ? this._moreFold(...more) : null, this._popActions(apply)].filter(Boolean));
    pop.hidden = false;
    // Next to the pointer that made the selection, kept on screen.
    const p = this.lastPointer ?? (() => {
      const r = document.getElementById('viewer')?.getBoundingClientRect();
      return r ? { x: r.left + r.width / 2 - 140, y: r.top + 16 } : { x: 200, y: 120 };
    })();
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;
    pop.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, p.x + 14))}px`;
    pop.style.top = `${Math.max(8, Math.min(window.innerHeight - h - 8, p.y + 14))}px`;
    this._placeMoved(pop);
  }

  // ---- set-ups (the start screen's "then", and the selection popup's shortcuts) --------------------------------

  async _runFlow(key, spec, opts = {}) {
    const polymers = spec.chains.filter((c) => c.residues > 0);
    const roleEdits = (c, role) => this._roleEdits(c, role);
    const selection = opts.atoms ?? window.atomworksStudio?.getSelection?.() ?? [];
    // Each step must succeed before the next; a failure keeps the form open with its error shown.
    let ok = true;
    if (key === 'ligand') {
      const lo = Math.max(20, Math.round(opts.min) || 100);
      ok = await this._mutate('/segments/ligand-binder', {
        remove_protein: opts.removeProtein !== false, pose_fixed: opts.poseFixed !== false,
        minimum: lo, maximum: Math.max(lo, Math.round(opts.max) || 150),
      });
    } else if (key === 'scaffold') {
      if (!selection.length) return;
      if (opts.rebuild !== false) {
        ok = await this._mutate('/segments/scaffold', {
          atom_indices: selection, spread: opts.spread ?? 0.25, sidechains_only: Boolean(opts.sidechainsOnly),
          unindexed: Boolean(opts.unindexed), ordered: Boolean(opts.ordered),
          total_min: opts.unindexed ? opts.totalMin : null, total_max: opts.unindexed ? opts.totalMax : null,
        });
      } else {
        // In place: the rest of the selected chains is redesigned at its current length.
        const doc = this._inputDoc();
        const a = doc.atoms;
        const residues = doc.resolveIndices(selection, 'residue');
        const backbone = new Set(['N', 'CA', 'C', 'O', 'OXT']);
        const fixAtoms = opts.sidechainsOnly ? residues.filter((i) => !backbone.has(String(a.atom_name[i]))) : residues;
        const chainsOfSelection = new Set(residues.map((i) => String(a.chain_id[i])));
        const edits = polymers.filter((c) => chainsOfSelection.has(c.id)).flatMap((c) => roleEdits(c, 'free'));
        if (fixAtoms.length) edits.push({ condition: 'coordinate', atom_indices: fixAtoms, enabled: true });
        edits.push({ condition: 'sequence', atom_indices: residues, enabled: true, scope: 'residue' });
        ok = await this._batch(edits);
      }
    } else if (key === 'sequence') {
      ok = await this._batch(polymers.flatMap((c) => roleEdits(c, 'backbone')));
    }
    if (ok) this.app._setPartialDiffusion?.(false);
    this.render();
  }

  // ---- edits --------------------------------------------------------------------------------------------------

  _roleEdits(c, role) {
    const all = c.polymerAtomIndices;
    if (!all.length) return [];
    if (role === 'fixed') {
      return [
        { condition: 'coordinate', atom_indices: all, enabled: true },
        { condition: 'sequence', atom_indices: all, enabled: true },
      ];
    }
    if (role === 'sequence') {
      return [
        { condition: 'coordinate', atom_indices: all, enabled: false },
        { condition: 'sequence', atom_indices: all, enabled: true },
      ];
    }
    if (role === 'backbone') {
      return [
        { condition: 'coordinate', atom_indices: all, enabled: false },
        ...(c.backboneAtomIndices.length
          ? [{ condition: 'coordinate', atom_indices: c.backboneAtomIndices, enabled: true }] : []),
        { condition: 'sequence', atom_indices: all, enabled: false },
      ];
    }
    return [
      { condition: 'coordinate', atom_indices: all, enabled: false },
      { condition: 'sequence', atom_indices: all, enabled: false },
    ];
  }

  _setRole(c, role) {
    return this._batch(this._roleEdits(c, role));
  }

  _batch(edits) {
    return edits.length ? this._mutate('/conditions/batch', { edits }) : Promise.resolve(true);
  }

  /** One edit through Studio's mutate path; resolves true on success. A failure stays shown until the next success. */
  async _mutate(path, body) {
    if ((!this.buildSession && typeof window.atomworksStudio?.applyEdit !== 'function') || this.busy) return false;
    // A document cannot be emptied ("cannot delete every atom"): removing the last of it means starting over.
    const total = this._inputDoc()?.atoms?.nAtoms ?? 0;
    if (path === '/delete' && total && new Set(body.atom_indices).size >= total) {
      this.lastError = null;
      this.showStart = true;
      this.render();
      return false;
    }
    this.busy = true;
    this.els.design?.classList.add('rfd4p-busy');
    try {
      if (this.buildSession) {
        // Straight to the page's Studio API, so the viewer is not redrawn for every step of an example.
        const url = new URL(`/api/v1/sessions/${encodeURIComponent(this.buildSession)}${path}`, location.href);
        const res = await handleStudioApiRequest(url, { method: 'POST', body: JSON.stringify(body) });
        if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `edit failed (${res.status})`);
      } else {
        await window.atomworksStudio?.applyEdit(path, body);
      }
      this.lastError = null;
      return true;
    } catch (err) {
      console.warn('Design edit failed:', err);
      this.lastError = `Edit failed: ${String(err?.message ?? err).replace(/^Error:\s*/, '')}`;
      return false;
    } finally {
      this.busy = false;
      this.els.design?.classList.remove('rfd4p-busy');
      this.render();
    }
  }

  // ---- start screen -------------------------------------------------------------------------------------------

  async _startTask(task) {
    if (typeof window.atomworksStudio?.openExample !== 'function' || this.building) return;
    const button = this.els.taskGrid?.querySelector(`[data-example="${task.example}"]`);
    button?.classList.add('loading');
    try {
      // An example with a recipe is built from its entries, or not opened at all (the page says why).
      if (task.recipe) { await this._buildRecipe(task); return; }
      await window.atomworksStudio?.openExample(task.example);
      this.showStart = false;
      if (task.then) {
        const spec = this._spec();
        if (spec) await this._runFlow(task.then, spec);
      }
    } finally {
      button?.classList.remove('loading');
      this.render();
    }
  }

  /** "1QYS", or "1J79 A C E" to open only those chains (as the card names them). */
  /** A new job from `make` (a document), opened in place of what was shown. */
  async _startNew(make, { rethrow = false } = {}) {
    const status = this.els.fetchStatus;
    try {
      const doc = await make();
      doc.undoStack = [];
      const sid = `rfd4p_new_${Date.now().toString(36)}`;
      getSharedStudioStore().sessions.set(sid, doc);
      this.startLigandForm?.remove();
      this.startLigandForm = null;
      if (this.startStructure) this.startStructure.form.hidden = true;
      const opened = await window.atomworksStudio?.openSession?.(sid);
      if (opened !== false) this.showStart = false;
    } catch (err) {
      if (rethrow) throw err;
      if (status) status.textContent = String(err?.message ?? err).replace(/^Error:\s*/, '');
    } finally {
      this.render();
    }
  }

  async _fetchPdb(raw, status = this.els.fetchStatus) {
    const [id = '', ...chains] = String(raw ?? '').trim().toUpperCase().split(/[\s,;]+/).filter(Boolean);
    if (!/^[0-9][A-Z0-9]{3}$/.test(id)) {
      status.textContent = 'Enter a 4-character PDB ID, e.g. 1QYS, optionally followed by chains: 1J79 A C E.';
      return;
    }
    status.textContent = `Downloading ${id} from RCSB…`;
    try {
      const res = await fetch(`https://files.rcsb.org/download/${id}.cif`);
      if (!res.ok) throw new Error(res.status === 404 ? `${id} was not found` : `RCSB returned ${res.status}`);
      const text = await res.text();
      this._rememberTitle(`PDB ${id}`, text);
      // Opened here rather than by Studio's file loader, so the chains named come from an assembly that has them.
      const doc = await this._openEntry(text, id, chains);
      if (chains.length) {
        this._keepNamed(doc, chains);
        doc.undoStack = [];
      }
      const sid = `rfd4p_fetch_${id}_${Date.now().toString(36)}`;
      getSharedStudioStore().sessions.set(sid, doc);
      const opened = await window.atomworksStudio?.openSession?.(sid);
      status.textContent = opened === false ? `Opening ${id} was cancelled.` : '';
      if (opened !== false) this.showStart = false;
      return opened !== false;
    } catch (err) {
      status.textContent = `Could not fetch ${id}: ${err?.message ?? err}`;
      return false;
    } finally {
      this.render();
    }
  }

  // ---- examples, built from their PDB entries ----------------------------------------------------------------

  /**
   * Build `task` from its PDB entry with the card's own controls while the start screen stays up, then show it
   * finished, with no Undo history (an example is a starting point). Resolves true once built; false if an entry could
   * not be fetched or a step failed, and then nothing is shown but why.
   */
  async _buildRecipe(task) {
    const card = (id) => [...this.els.chainList.querySelectorAll('.rfd4p-chain')]
      .find((c) => c.querySelector('.rfd4p-chain-id')?.textContent === id);
    const subs = (chain) => [...(card(chain)?.querySelectorAll('.rfd4p-sub') ?? [])];
    const isNew = (sub) => /\bnew$/.test(sub.querySelector('.rfd4p-sub-status')?.textContent ?? '');
    const idle = async () => {
      await this.applying;
      this.applying = null;
      for (let i = 0; i < 600 && this.busy; i += 1) await new Promise((r) => setTimeout(r, 50));
    };
    const open = (sub) => { sub.querySelector('.rfd4p-sub-open').click(); return this.piecePop; };
    const fill = (pop, label, value, event = 'input') => {
      const field = pop.querySelector(`[aria-label="${label}"]`);
      if (!field) return;
      field.value = String(value);
      field.dispatchEvent(new Event(event));
    };
    const apply = async (pop) => {
      [...pop.querySelectorAll('button')].find((b) => b.textContent === 'Apply').click();
      await idle();
      const note = pop.hidden ? '' : pop.querySelector('.rfd4p-pop-note')?.textContent;
      if (note) throw new Error(note);
    };
    const click = async (node) => { node.click(); await idle(); };

    this.building = true;
    document.body.classList.add('rfd4p-building');
    try {
      for (const step of task.recipe) {
        if (step.fetch) {
          const [id, ...chains] = step.fetch.split(/\s+/);
          let text;
          try {
            const res = await fetch(`https://files.rcsb.org/download/${id}.cif`);
            if (!res.ok) throw new Error(`RCSB returned ${res.status}`);
            text = await res.text();
          } catch (err) {
            this.els.fetchStatus.textContent = `Could not fetch ${id}: ${err?.message ?? err}`;
            return false;
          }
          this._rememberTitle(`PDB ${id}`, text);
          // As upstream's example was set up: its unresolved residues kept, unlike a structure someone loads.
          const doc = await this._openEntry(text, id, chains, { keepUnresolved: true });
          if (chains.length) {
            const drop = [];
            for (let i = 0; i < doc.atoms.nAtoms; i += 1) if (!chains.includes(String(doc.atoms.chain_id[i]))) drop.push(i);
            if (drop.length) doc.deleteAtoms(drop);
          }
          this.buildSession = `rfd4p_example_${task.example}_${Date.now().toString(36)}`;
          getSharedStudioStore().sessions.set(this.buildSession, doc);
          this.render();
        } else if (step.start) {
          // A bundled example as the starting point (Protein from scratch's 250 new residues): nothing to download.
          const res = await fetch(new URL(`./studio/examples/${step.start}.cif`, import.meta.url));
          if (!res.ok) throw new Error(`no bundled example ${step.start}`);
          const doc = new StudioDoc(await res.text(), step.start, {});
          this.buildSession = `rfd4p_example_${task.example}_${Date.now().toString(36)}`;
          getSharedStudioStore().sessions.set(this.buildSession, doc);
          this.render();
        } else if (step.blocks) {
          // One block per entry: "+" after the last block until there are enough, then each block's length and shape.
          const chainCard = () => [...this.els.chainList.querySelectorAll('.rfd4p-chain')].at(-1);
          const blocksNow = () => [...chainCard().querySelectorAll('.rfd4p-sub')].filter(isNew);
          for (let i = 0; i < 100 && blocksNow().length < step.blocks.length; i += 1) {
            await click([...chainCard().querySelectorAll('.rfd4p-slot')].at(-1));
          }
          const shapes = Object.fromEntries(SHAPE_OPTIONS.map(([v, text]) => [text.split(' ')[0], v]));
          for (let k = 0; k < step.blocks.length; k += 1) {
            const [len, shape] = step.blocks[k];
            const pop = open(blocksNow()[k]);
            fill(pop, 'Fewest new residues', len);
            fill(pop, 'Most new residues', len);
            fill(pop, 'Shape', shapes[shape] ?? '0', 'change');
            await apply(pop);
          }
        } else if (step.add) {
          const [id, ...chains] = step.add.split(/\s+/);
          const res = await fetch(`https://files.rcsb.org/download/${id}.cif`);
          if (!res.ok) throw new Error(`RCSB returned ${res.status} for ${id}`);
          const text = await res.text();
          this._rememberTitle(`PDB ${id}`, text);
          await this._addStructure(text, id, chains, null);
          await idle();
        } else if (step.piece || step.removePiece) {
          const s = step.set ?? {};
          const chain = step.piece ?? step.removePiece;
          const sub = subs(chain).find((x) => {
            const m = /(\d+)(?:–(\d+))?/.exec(x.querySelector('.rfd4p-sub-name')?.textContent ?? '');
            return !isNew(x) && m && step.at >= Number(m[1]) && step.at <= Number(m[2] ?? m[1]);
          });
          if (!sub) throw new Error(`chain ${chain} has no residue ${step.at}`);
          if (step.removePiece) {
            await click(sub.querySelector('.rfd4p-x'));
            continue;
          }
          const pop = open(sub);
          if (s.from != null) fill(pop, 'Residues', `${s.from}-${s.to}`);
          if (s.structure) fill(pop, 'Structure', s.structure, 'change');
          if (s.atoms) {
            const named = !ATOM_OPTIONS.some(([v]) => v === s.atoms);
            fill(pop, 'Atoms', named ? 'named' : s.atoms, 'change');
            if (named) fill(pop, 'Atom names', s.atoms);
          }
          if (s.sequence) fill(pop, 'Sequence', s.sequence, 'change');
          if (s.position) fill(pop, 'Position in chain', s.position, 'change');
          if (s.length) { fill(pop, 'Fewest residues', s.length[0]); fill(pop, 'Most residues', s.length[1]); }
          if (s.hotspots) fill(pop, 'Hotspots', s.hotspots);
          if (s.keepOnly) fill(pop, 'Keep only these residues', s.keepOnly);
          await apply(pop);
        } else if (step.pick) {
          // Residues picked, as in the viewer, and the pick's popup: part of a piece freed, or given a length.
          const [, chain, x, y] = /^(\w+?)(-?\d+)-(-?\d+)$/.exec(step.pick);
          const a = this._inputDoc().atoms;
          const atoms = [];
          for (let i = 0; i < a.nAtoms; i += 1) {
            if (String(a.chain_id[i]) === chain && a.res_id[i] >= Number(x) && a.res_id[i] <= Number(y) && !/^\d+-\d+$/.test(String(a.res_name[i]))) atoms.push(i);
          }
          if (!atoms.length) throw new Error(`chain ${chain} has no residues ${x}–${y}`);
          this._renderPop(atoms);
          const pop = this.pop;
          const s = step.set ?? {};
          if (s.structure) fill(pop, 'Structure', s.structure, 'change');
          if (s.sequence) fill(pop, 'Sequence', s.sequence, 'change');
          if (s.length) { fill(pop, 'Fewest residues', s.length[0]); fill(pop, 'Most residues', s.length[1]); }
          await apply(pop);
        } else if (step.removeNew) {
          for (let sub = subs(step.removeNew).find(isNew); sub; sub = subs(step.removeNew).find(isNew)) {
            await click(sub.querySelector('.rfd4p-x'));
          }
        } else if (step.addSeq) {
          await click(this.els.addSeq);
        } else if (step.newLength) {
          const chains = step.chain ? [card(step.chain)] : [...this.els.chainList.querySelectorAll('.rfd4p-chain')].reverse();
          const sub = chains.flatMap((c) => [...(c?.querySelectorAll('.rfd4p-sub') ?? [])]).filter(isNew)[step.k ?? 0];
          if (!sub) throw new Error(`no new residues #${(step.k ?? 0) + 1} in chain ${step.chain ?? ''}`);
          const pop = open(sub);
          fill(pop, 'Fewest new residues', step.newLength[0]);
          fill(pop, 'Most new residues', step.newLength[1]);
          await apply(pop);
        } else if (step.ligand) {
          const sub = [...this.els.chainList.querySelectorAll('.rfd4p-sub')].find((x) => x.querySelector('.rfd4p-sub-name')?.textContent === step.ligand);
          const pop = open(sub);
          fill(pop, 'Structure', step.pose, 'change');
          if (step.atoms) { fill(pop, 'Atoms', 'named', 'change'); fill(pop, 'Atom names', step.atoms); }
          await apply(pop);
        } else if (step.chain && (step.name || step.order || step.fold)) {
          const chip = card(step.chain)?.querySelector('.rfd4p-chain-id');
          if (!chip) throw new Error(`no chain ${step.chain}`);
          chip.click();
          const pop = this.piecePop;
          if (step.name) fill(pop, 'Chain name', step.name);
          if (step.order) fill(pop, 'Chain order', step.order - 1, 'change');
          if (step.fold) fill(pop, 'Fold', step.fold, 'change');
          await apply(pop);
        } else if (step.partial) {
          const box = document.getElementById('partial-diffusion');
          box.checked = true;
          box.dispatchEvent(new Event('change'));
          await idle();
        }
      }
      // An example is a starting point: no Undo back through how it was built. Studio shows it once, finished.
      const sid = this.buildSession;
      getSharedStudioStore().getSession(sid).undoStack = [];
      this.buildSession = null;
      this.showStart = false;
      await window.atomworksStudio?.openSession?.(sid);
      return true;
    } catch (err) {
      // Nothing half built is shown: the start screen stays, saying what went wrong.
      console.warn('Example set-up stopped:', err);
      if (this.buildSession) getSharedStudioStore().sessions.delete(this.buildSession);
      this.buildSession = null;
      this.els.fetchStatus.textContent = `The ${task.title} example could not be set up: ${String(err?.message ?? err).replace(/^Error:\s*/, '')}`;
      return false;
    } finally {
      this.buildSession = null;
      this.building = false;
      document.body.classList.remove('rfd4p-building');
      this._closePop();
      this.render();
    }
  }
}
