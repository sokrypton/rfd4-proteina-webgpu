/**
 * The RFD4-Proteina card's layer over AtomWorks Studio: a "what do you want to design?" start screen, a Design
 * panel that states what the open document will generate (task, chain roles, lengths, contig) and edits it, and the
 * simple/advanced switch that hides Studio's expert tools.
 *
 * Every edit goes through Studio's own mutate path (`window.atomworksStudio?.applyEdit`), so Undo, the viewer and the
 * `atomworks-studio` "changed" event behave exactly as for edits made in the Studio panels.
 */
import { designSpec, formatResidueList, parseResidueList } from './design-spec.js';
import { createStudioDocWithCcd, getSharedStudioStore, handleStudioApiRequest } from './studio-api.js';
import { isCcdCode, ligandCifFromCcd } from './ligand-ccd.js';
import { fetchCcdComponents } from '../shared/rfd4p/pipeline/ccd.js';

const ADVANCED_KEY = 'rfd4p-advanced-editor';

/**
 * Start-screen tasks. Each with a `recipe` is built from its PDB entry with the card's own controls, out of sight
 * (`_buildRecipe`), and shown finished: the set-up a user would make, numbered as the entry is and captioned with it.
 * Without a network the bundled Studio example (`example`, web/studio/examples) opens instead.
 * Steps: `fetch` (ID and chains), `piece` (open the piece holding `at`, fill `set`, Apply), `removeNew` (× the new
 * residues left in a chain), `addSeq`, `newLength` (the chain's `k`-th new piece), `ligand` (pose), `partial`.
 */
const TASKS = [
  { example: 'unconditional', title: 'Protein from scratch', desc: 'A new fold of the length you choose' },
  {
    example: 'protein_binder', title: 'Protein binder', desc: 'Bind PD-L1 (PDB 5O45)',
    recipe: [
      { fetch: '5O45 A' },
      { piece: 'A', at: 10, set: { hotspots: 'A40, A99, A107', keepOnly: 'A1-115' } },
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
    ],
  },
  {
    example: 'motif_scaffolding', title: 'Motif scaffolding', desc: 'Keep three residues, build the rest (PDB 1QJG)',
    recipe: [
      { fetch: '1QJG C' },
      { piece: 'C', at: 10, set: { keepOnly: 'C14, C38, C99' } },
    ],
  },
  {
    example: 'enzyme_design', title: 'Enzyme', desc: 'A catalytic Asp around Zn and orotate (PDB 1J79)',
    recipe: [
      { fetch: '1J79 A C E' },
      { piece: 'A', at: 10, set: { structure: 'sidechain', position: 'anywhere', keepOnly: 'A250' } },
      { newLength: [180, 180], chain: 'A' },
    ],
  },
  {
    example: 'antibody_design', title: 'Antibody loop', desc: 'Redesign CDR H3 against HER2 (PDB 1N8Z)',
    recipe: [
      { fetch: '1N8Z A B C' },
      { piece: 'A', at: 10, set: { structure: 'rigid', keepOnly: 'A1-107' } },
      { removeNew: 'A' },
      { piece: 'B', at: 10, set: { structure: 'rigid', keepOnly: 'B1-120' } },
      { removeNew: 'B' },
      { piece: 'B', at: 100, set: { from: 99, to: 109, structure: 'designed', sequence: 'designed' } },
      { piece: 'C', at: 550, set: { keepOnly: 'C545-580, C591-607', hotspots: 'C557-573, C591-605' } },
      { removeNew: 'C' },
    ],
  },
  {
    example: 'nanobody', title: 'Nanobody (VHH)', desc: 'Design CDRs against PD-L1 (PDB 5JDS)',
    recipe: [
      { fetch: '5JDS' },
      { piece: 'A', at: 10, set: { structure: 'rigid', keepOnly: 'A3-125' } },
      { removeNew: 'A' },
      { piece: 'B', at: 10, set: { keepOnly: 'B3-115', hotspots: 'B40, B99, B107' } },
      { removeNew: 'B' },
      { piece: 'A', at: 100, set: { from: 99, to: 119, structure: 'designed', sequence: 'designed', length: [16, 26] } },
      { piece: 'A', at: 55, set: { from: 52, to: 61, structure: 'designed', sequence: 'designed', length: [8, 13] } },
      { piece: 'A', at: 30, set: { from: 28, to: 37, structure: 'designed', sequence: 'designed', length: [8, 13] } },
    ],
  },
  {
    example: 'inverse_folding', title: 'Redesign a sequence', desc: 'New sequence for a fixed backbone (PDB 2KL8)', then: 'sequence',
    recipe: [
      { fetch: '2KL8' },
      { piece: 'A', at: 10, set: { sequence: 'designed' } },
    ],
  },
  {
    example: 'partial_diffusion', title: 'Diversify a structure', desc: 'Variants close to a starting fold (PDB 2KL8)',
    recipe: [
      { fetch: '2KL8' },
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

// The three properties of residues, the same in a sub-card's popup and the selection's.
const STRUCTURE_OPTIONS = [['fixed', 'fixed'], ['rigid', 'fixed, can move'], ['sidechain', 'side chains only'], ['designed', 'designed']];
const SEQUENCE_OPTIONS = [['fixed', 'fixed'], ['designed', 'designed']];
const POSITION_OPTIONS = [['fixed', 'fixed'], ['anywhere', 'anywhere']];

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
      fetchForm: $('rfd4p-fetch-form'),
      pdbId: $('rfd4p-pdb-id'),
      fetchStatus: $('rfd4p-fetch-status'),
      openFile: $('rfd4p-open-file'),
      design: $('rfd4p-design'),
      taskLabel: $('rfd4p-task-label'),
      taskDetail: $('rfd4p-task-detail'),
      newDoc: $('rfd4p-new'),
      back: $('rfd4p-back'),
      chainList: $('rfd4p-chain-list'),
      addSeq: $('rfd4p-add-seq'),
      addPdb: $('rfd4p-add-pdb'),
      addLig: $('rfd4p-add-lig'),
      undo: $('rfd4p-undo'),
      contig: $('rfd4p-contig'),
      warnings: $('rfd4p-warnings'),
      runSettings: $('rfd4p-run-settings'),
      advanced: $('rfd4p-advanced'),
    };
    // Floating actions next to the pointer for the current selection (viewer or sequence bar).
    this.pop = el('div', { class: 'rfd4p-piece-pop rfd4p-sel-pop', role: 'dialog', 'aria-label': 'Selected residues', hidden: true });
    document.body.append(this.pop);
    this.lastPointer = null;
    this.partialRow = el('div', { class: 'rfd4p-partial-row', hidden: true });
    this.els.taskDetail?.after(this.partialRow);
    this._buildAddStructure();
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
    const id = el('input', {
      type: 'text', class: 'rfd4p-add-id', spellcheck: 'false', autocomplete: 'off', maxlength: 40,
      placeholder: 'PDB ID, then chains: 5O45 A', 'aria-label': 'PDB ID and chains',
    });
    const title = el('strong');
    const fetchBtn = el('button', { type: 'submit', class: 'primary compact' });
    const fileInput = el('input', { type: 'file', accept: '.cif,.mmcif,.pdb,.ent,.cif.zst', hidden: true });
    const fileBtn = el('button', { type: 'button', class: 'secondary compact', text: 'Open a file…', onclick: () => fileInput.click() });
    const status = el('p', { class: 'rfd4p-add-status muted', 'aria-live': 'polite' });
    const cancel = el('button', { type: 'button', class: 'secondary compact', text: 'Cancel', onclick: () => { form.hidden = true; } });
    const form = el('form', { class: 'rfd4p-add-structure', hidden: true },
      title,
      el('div', { class: 'rfd4p-add-row' }, id, fetchBtn), el('div', { class: 'rfd4p-add-row' }, fileBtn, fileInput, cancel),
      status);
    this.addStructure = { form, title, fetchBtn, status, id, replace: null };
    form.addEventListener('submit', (ev) => {
      ev.preventDefault();
      const m = /^\s*([0-9][A-Za-z0-9]{3})(?:[\s:_,]+(.*))?$/.exec(id.value);
      if (!m) { status.textContent = 'Type a 4-character PDB ID, optionally followed by chains: 5O45 A'; return; }
      const pdb = m[1].toUpperCase();
      const chains = (m[2] ?? '').split(/[\s,;]+/).filter(Boolean);
      void (async () => {
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
      })();
    });
    fileInput.addEventListener('change', () => {
      const file = fileInput.files?.[0];
      fileInput.value = '';
      if (!file) return;
      const chains = id.value.trim().split(/[\s,;]+/).filter((t) => /^[A-Za-z0-9]{1,4}$/.test(t) && !/^[0-9][A-Za-z0-9]{3}$/.test(t));
      void (async () => {
        try {
          const text = file.name.endsWith('.zst') ? new Uint8Array(await file.arrayBuffer()) : await file.text();
          await this._addStructure(text, file.name.replace(/\.(cif|mmcif|pdb|ent|zst)+$/i, ''), chains, this.addStructure.replace);
        } catch (err) {
          status.textContent = `Could not add ${file.name}: ${err?.message ?? err}`;
        }
      })();
    });
    actions.after(form);
  }

  /** The + pdb form; with `group` (a source caption's chains), it replaces them. */
  _openAddStructure(group = null) {
    const { form, title, fetchBtn, status, id } = this.addStructure ?? {};
    if (!form) return;
    this.addStructure.replace = group ? group.chains.flatMap((c) => c.atomIndices) : null;
    title.textContent = group ? `Replace ${group.label}` : 'Add chains from a structure';
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
  async _addStructure(text, name, chains, replace) {
    const { status } = this.addStructure;
    const doc = await createStudioDocWithCcd(text, name, { preset: 'rcsb', assembly: '1' });
    const a = doc.atoms;
    const present = [...new Set(a.chain_id.map(String))];
    const wanted = new Set(chains.map((c) => c.toUpperCase()));
    const missing = [...wanted].filter((c) => !present.some((p) => p.toUpperCase() === c));
    if (missing.length) throw new Error(`no chain ${missing.join(', ')} (it has ${present.join(', ')})`);
    if (wanted.size) {
      const drop = [];
      for (let i = 0; i < a.nAtoms; i += 1) if (!wanted.has(String(a.chain_id[i]).toUpperCase())) drop.push(i);
      if (drop.length) doc.deleteAtoms(drop);
    }
    if (!doc.atoms.nAtoms) throw new Error('nothing left to add');
    if (replace?.length && !await this._mutate('/delete', { atom_indices: replace })) throw new Error(this.lastError ?? 'could not remove the old chains');
    // Opened as a plain structure, it is already fixed throughout, so adding it is one Undo step.
    const added = await this._combine(doc);
    const newChains = [...new Set(added.map((i) => String(this._inputDoc().atoms.chain_id[i])))];
    const original = chains.length === 1 ? ` chain ${chains[0].toUpperCase()}` : '';
    this._rememberSource(newChains, /^[0-9][A-Za-z0-9]{3}$/.test(name) ? `PDB ${name}${original}` : `${name}${original}`);
    this.addStructure.form.hidden = true;
    this.addStructure.replace = null;
    this.render();
  }

  /** Merge a document into the job (one undo step); resolves to the added atoms' indices (the chains that are new). */
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

  /** A one-line "ligand code" form: [input] [button] and a status line. */
  _ligandCodeForm({ label, button, onSubmit, onCancel }) {
    const input = el('input', {
      type: 'text', class: 'rfd4p-add-id', spellcheck: 'false', autocomplete: 'off', maxlength: 5,
      placeholder: 'Code, e.g. HEM', 'aria-label': 'Ligand code',
    });
    const status = el('p', { class: 'rfd4p-add-status muted', 'aria-live': 'polite' });
    const form = el('form', { class: 'rfd4p-add-structure rfd4p-ligand-form' },
      el('strong', { text: label }),
      el('span', { class: 'muted', text: 'Its code in the PDB (HEM, ATP, NAD, ZN…). The model places it; fix its pose afterwards if you want it kept where it is.' }),
      el('div', { class: 'rfd4p-add-row' }, input, el('button', { type: 'submit', class: 'primary compact', text: button }),
        el('button', { type: 'button', class: 'secondary compact', text: 'Cancel', onclick: onCancel })),
      status);
    form.addEventListener('submit', (ev) => {
      ev.preventDefault();
      status.textContent = `Looking up ${input.value.trim().toUpperCase()}…`;
      void onSubmit(input.value).then(() => { status.textContent = ''; }, (err) => {
        status.textContent = `Could not add it: ${String(err?.message ?? err).replace(/^Error:\s*/, '')}`;
      });
    });
    queueMicrotask(() => input.focus());
    return form;
  }

  // ---- wiring -------------------------------------------------------------------------------------------------

  _bind() {
    const { els } = this;
    els.fetchForm?.addEventListener('submit', (ev) => {
      ev.preventDefault();
      void this._fetchPdb(els.pdbId.value);
    });
    els.openFile?.addEventListener('click', () => document.getElementById('file-input')?.click());
    // "New…" shows the start screen over the open document; opening anything from it replaces the document.
    els.newDoc?.addEventListener('click', () => { this.showStart = true; this.render(); });
    els.back?.addEventListener('click', () => { this.showStart = false; this.render(); });
    document.getElementById('file-input')?.addEventListener('change', () => { this.showStart = false; });
    els.addSeq?.addEventListener('click', () => void this._mutate('/segments', {
      minimum: 60, maximum: 100, atom_index: null, before: false, new_chain: true, chain_id: null,
      chain_type: 'polypeptide(L)',
    }));
    els.addPdb?.addEventListener('click', () => this._openAddStructure());
    els.addLig?.addEventListener('click', () => {
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
    els.undo?.addEventListener('click', () => document.getElementById('undo')?.click());
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
      queueMicrotask(() => this._renderPop());
    });
    // Starting from the input diversifies what is designed; with nothing designed yet, that is all the protein.
    document.getElementById('partial-diffusion')?.addEventListener('change', (ev) => {
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
    const percent = Number.parseFloat(this.app.els.partialPercent?.value ?? '70') || 70;
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
    if (els.back) els.back.hidden = !hasDoc;
    if (els.design) els.design.hidden = starting;
    if (els.runSettings) els.runSettings.hidden = starting;
    if (!hasDoc || (starting && !this.building)) return;

    const viewingDesign = this._viewingDesign();
    els.design.classList.toggle('rfd4p-readonly', viewingDesign);
    els.taskLabel.textContent = spec.task.label;
    els.taskDetail.textContent = viewingDesign ? 'A generated design (read-only). Select Input under View to edit.' : '';
    els.taskDetail.hidden = !viewingDesign;
    if (els.newDoc) els.newDoc.hidden = viewingDesign;
    const actions = els.addSeq?.parentElement;
    if (actions) actions.hidden = viewingDesign;
    if (viewingDesign && this.addStructure) this.addStructure.form.hidden = true;
    els.undo.hidden = viewingDesign;
    els.undo.disabled = document.getElementById('undo')?.disabled ?? true;

    els.chainList.replaceChildren(...this._sourceGroups(spec.chains).map((g) => this._groupBlock(g, viewingDesign)));
    els.contig.textContent = spec.contig || '—';

    const warnings = this.lastError ? [this.lastError, ...spec.warnings] : [...spec.warnings];
    if (spec.task.key === 'partial' && spec.chains.some((c) => c.segments.length)) {
      warnings.push('Partial diffusion needs every residue defined; this input has new-residue segments.');
    }
    els.warnings.replaceChildren(...warnings.map((w) => el('p', { class: 'rfd4p-warning', text: w })));

    this._renderPartial(spec, viewingDesign);
    this.showHotspots();

    // Nothing to generate: keep Generate off (a running job keeps its Stop button).
    if (!this.app.activeAbortController && this.app.els.predictBtn) {
      this.app.els.predictBtn.disabled = Boolean(this.app.noWebGpu) || spec.task.key === 'nothing' || spec.task.key === 'empty';
    }
  }

  /** Chains by where their coordinates come from, in order of first appearance; chains made here last, uncaptioned. */
  _sourceGroups(chains) {
    const groups = new Map();
    const made = { label: null, chains: [] };
    for (const c of chains) {
      const label = this._sourceOf(c);
      if (!label) { made.chains.push(c); continue; }
      if (!groups.has(label)) groups.set(label, { label, chains: [] });
      groups.get(label).chains.push(c);
    }
    return [...groups.values(), ...(made.chains.length ? [made] : [])];
  }

  /** A source's caption (click: replace it with another structure), then its chains. */
  _groupBlock(g, readonly) {
    // "PDB 5O45 · A": the entry and the chains of it in the job; its title when pointed at.
    const entry = g.label?.replace(/ chains? .*$/, '');
    const title = entry ? this.titles?.get(entry) : null;
    const text = entry ? `${entry} · ${g.chains.map((c) => c.id).join(' ')}` : null;
    const caption = !g.label ? null : readonly
      ? el('span', { class: 'rfd4p-source', title: title ?? null, text })
      : el('button', { type: 'button', class: 'rfd4p-source', onclick: () => this._openAddStructure(g), text,
        title: [title, `Click to replace ${g.label}`].filter(Boolean).join('\n') });
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
    // New residues next to new residues would be one piece: a slot only where a new piece is not already.
    const slot = (atom, before) => (readonly ? null : el('button', {
      type: 'button', class: 'rfd4p-slot', title: 'New residues here', 'aria-label': 'New residues here',
      onclick: () => void this._mutate('/segments', {
        minimum: 5, maximum: 10, atom_index: atom, before, new_chain: false, chain_id: null, chain_type: 'polypeptide(L)',
      }),
    }, el('span', { text: '+' })));
    pieces.forEach((p, k) => {
      const prev = pieces[k - 1];
      if (p.kind !== 'new' && (!prev || prev.kind !== 'new')) {
        items.push(prev ? slot(prev.atomIndices[prev.atomIndices.length - 1], false) : slot(p.atomIndices[0], true));
      }
      items.push(this._pieceCard(c, p, readonly, remove(many ? 'Remove these residues' : `Remove chain ${c.id}`,
        p.atomIndices)));
      if (k === pieces.length - 1 && p.kind !== 'new') items.push(slot(p.atomIndices[p.atomIndices.length - 1], false));
    });
    for (const l of c.ligands) {
      items.push(this._ligandCard(c, l, readonly, remove(many ? `Remove ${l.name}` : `Remove chain ${c.id}`, l.atomIndices)));
    }
    const gutter = el('div', { class: 'rfd4p-chain-gutter' },
      el('span', { class: 'rfd4p-chain-id', text: c.id }),
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
    if (p.kind === 'new') return 'new';
    const { structure, sequence, position } = this._properties(p);
    // The popup's words: both fixed reads "fixed", both designed "designed", otherwise which one is fixed.
    const what = structure === 'designed' && sequence === 'designed' ? 'designed'
      : structure === 'fixed' && sequence === 'fixed' ? 'fixed'
        : structure === 'rigid' ? 'fixed, can move'
          : structure === 'sidechain' ? 'side chains fixed'
            : structure === 'fixed' ? 'structure fixed' : 'sequence fixed';
    return [what, position === 'anywhere' ? 'anywhere' : '',
      p.hotspots.length ? `${p.hotspots.length} hotspot${p.hotspots.length === 1 ? '' : 's'}` : ''].filter(Boolean).join(' · ');
  }

  /** A sub-card: name and status (opens its popup), then its ×. */
  _card(name, status, kindClass, readonly, onOpen, remove) {
    return el('div', { class: `rfd4p-sub ${kindClass}` },
      el('button', {
        type: 'button', class: 'rfd4p-sub-open', disabled: readonly, 'aria-haspopup': 'dialog',
        onclick: (ev) => onOpen(ev.currentTarget),
      },
      el('span', { class: 'rfd4p-sub-name', text: name }),
      el('span', { class: 'rfd4p-sub-status', text: status })),
      remove);
  }

  _pieceCard(c, p, readonly, remove) {
    return this._card(this._pieceText(c, p), this._status(p), `rfd4p-sub-${p.kind === 'kept' ? 'kept' : 'designed'}`, readonly,
      (anchor) => this._openPop(anchor, () => (p.kind === 'new' ? this._newPop(c, p) : this._piecePop(c, p))), remove);
  }

  _ligandCard(c, l, readonly, remove) {
    const status = l.pose === 'fixed' ? 'fixed' : l.pose === 'predicted' ? 'free' : `${l.fixedAtoms} of ${l.atoms} atoms fixed`;
    return this._card(l.name, status, 'rfd4p-sub-ligand', readonly, (anchor) => this._openPop(anchor, () => this._ligandPop(c, l)), remove);
  }

  /** Where a chain's coordinates come from, as the page knows it; null for chains made here (+ seq, + lig). */
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

  /** The one popup, next to the row that opened it. `build` makes its content; closed by Cancel, Escape or a click outside (× on a row means delete). */
  _openPop(anchor, build) {
    if (!this.piecePop) {
      this.piecePop = el('div', { class: 'rfd4p-piece-pop', role: 'dialog', hidden: true });
      document.body.append(this.piecePop);
      document.addEventListener('pointerdown', (ev) => {
        if (!this.piecePop.hidden && !this.piecePop.contains(ev.target) && !ev.target.closest?.('.rfd4p-sub-open')) this._closePop();
      }, true);
      document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') this._closePop(); });
    }
    const pop = this.piecePop;
    this.pop.hidden = true;
    pop.replaceChildren(...build());
    pop.hidden = false;
    const r = anchor.getBoundingClientRect();
    const card = anchor.closest('#rfd4p-studio-card')?.getBoundingClientRect() ?? r;
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;
    // Beside the card (it sits at the right edge), level with the row, kept on screen.
    const left = card.left - w - 8 >= 8 ? card.left - w - 8 : Math.max(8, Math.min(window.innerWidth - w - 8, r.left));
    pop.style.left = `${left}px`;
    pop.style.top = `${Math.max(8, Math.min(window.innerHeight - h - 8, r.top - 8))}px`;
    pop.querySelector('input:not([type=radio]):not([type=checkbox]), button.primary')?.focus();
  }

  _closePop() {
    if (this.piecePop) this.piecePop.hidden = true;
  }

  _popHead(title) {
    return el('div', { class: 'rfd4p-pop-title' }, el('strong', { text: title }));
  }

  _popActions(apply) {
    return el('div', { class: 'rfd4p-pop-actions' },
      el('span', { class: 'rfd4p-pop-spacer' }),
      el('button', { type: 'button', class: 'secondary compact', text: 'Cancel', onclick: () => this._closePop() }),
      el('button', { type: 'button', class: 'primary compact', text: 'Apply', onclick: () => { this.applying = Promise.resolve(apply()); } }));
  }

  /** Existing residues: which of them, then three properties (structure, sequence, position), each fixed or designed. */
  _piecePop(c, p) {
    const a = this._inputDoc()?.atoms;
    const num = (value, label) => el('input', { type: 'number', min: p.start, max: p.end, value, 'aria-label': label });
    const from = num(p.start, 'First residue');
    const to = num(p.end, 'Last residue');
    const now = this._properties(p);
    const select = (label, options, value, title) => {
      const s = el('select', { 'aria-label': label, title });
      for (const [v, text] of options) s.append(el('option', { value: v, text, selected: v === value }));
      return s;
    };
    const structure = select('Structure', STRUCTURE_OPTIONS,
      now.structure, 'Fixed: kept where it is. Can move: kept rigid, free to move as a whole (a framework). Designed: the model builds it.');
    const sequence = select('Sequence', SEQUENCE_OPTIONS, now.sequence,
      'Fixed: these amino acids. Designed: the model writes them.');
    const position = select('Position in chain', POSITION_OPTIONS, now.position,
      'Anywhere: the model chooses where in the chain these residues go.');
    const span = () => {
      const lo = Math.max(p.start, Math.min(p.end, Math.round(Number(from.value)) || p.start));
      const hi = Math.max(lo, Math.min(p.end, Math.round(Number(to.value)) || p.end));
      const atoms = a ? p.atomIndices.filter((i) => Number(a.res_id[i]) >= lo && Number(a.res_id[i]) <= hi) : p.atomIndices;
      return { lo, hi, atoms, count: a ? new Set(atoms.map((i) => `${a.res_id[i]}${a.ins_code[i]}`)).size : p.count };
    };
    // Designed residues may change length: a length other than their count makes them new residues.
    const count = span().count;
    const lenMin = el('input', { type: 'number', min: 1, max: 1000, value: count, 'aria-label': 'Fewest residues' });
    const lenMax = el('input', { type: 'number', min: 1, max: 1000, value: count, 'aria-label': 'Most residues' });
    let lenTouched = false;
    for (const input of [lenMin, lenMax]) input.addEventListener('input', () => { lenTouched = true; });
    for (const input of [from, to]) {
      input.addEventListener('input', () => {
        if (lenTouched) return;
        lenMin.value = lenMax.value = String(span().count);
      });
    }
    const lengthRow = el('div', { class: 'rfd4p-pop-row', title: 'Each design picks one length from this range' },
      el('span', { class: 'rfd4p-pop-label', text: 'Length' }), el('span', { class: 'rfd4p-pop-inline' }, lenMin, el('span', { text: '–' }), lenMax));
    // Side chains only means those side chains, so their amino acids; a fully designed piece has no position to keep,
    // only a length.
    const sync = () => {
      if (structure.value === 'sidechain') sequence.value = 'fixed';
      sequence.disabled = structure.value === 'sidechain';
      const free = structure.value === 'designed' && sequence.value === 'designed';
      if (free) position.value = 'fixed';
      positionRow.hidden = free;
      lengthRow.hidden = !free;
    };
    structure.addEventListener('change', sync);
    sequence.addEventListener('change', sync);

    const hot = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      value: formatResidueList(c.id, c.hotspotResIds), placeholder: 'none', 'aria-label': 'Hotspots',
      title: 'Residues of this chain a binder should touch, e.g. A56, A60-62' });
    const keepOnly = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      placeholder: 'all', 'aria-label': 'Keep only these residues',
      title: 'Type some residues (e.g. A38, A99) to keep only those; the rest of the chain becomes new residues (a motif)' });
    const note = el('p', { class: 'rfd4p-pop-note', 'aria-live': 'polite' });
    const residuesIn = (text, inChain) => {
      const { residues, bad } = parseResidueList(text, c.id);
      const want = new Set(residues.filter((r) => r.chain === c.id).map((r) => r.resId));
      const atoms = a ? inChain.filter((i) => want.has(Number(a.res_id[i]))) : [];
      const found = new Set(atoms.map((i) => Number(a.res_id[i])));
      return { want, atoms, bad, missing: [...want].filter((id) => !found.has(id)) };
    };

    const apply = async () => {
      // Hotspots are the chain's, whichever of its pieces the popup was opened from.
      const hotspots = residuesIn(hot.value, c.polymerAtomIndices);
      if (hotspots.bad.length || hotspots.missing.length) {
        note.textContent = hotspots.bad.length ? `Hotspots not understood: ${hotspots.bad.join(' ')}`
          : `Hotspots not in chain ${c.id}: ${formatResidueList(c.id, hotspots.missing)}`;
        return;
      }
      const hotChanged = formatResidueList(c.id, [...hotspots.want]) !== formatResidueList(c.id, c.hotspotResIds);
      // Re-read after each edit: atom indices move when residues are added or removed.
      const chainNow = () => this._spec()?.chains.find((x) => x.id === c.id);
      const setHotspots = async () => {
        const now = chainNow();
        if (!hotChanged || !now) return true;
        const ids = this._inputDoc().atoms.res_id;
        const edits = [{ condition: 'hotspot', atom_indices: now.polymerAtomIndices, enabled: false }];
        const on = now.polymerAtomIndices.filter((i) => hotspots.want.has(Number(ids[i])));
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
        // The residues kept take this popup's properties: side chains only and anywhere are part of the set-up (an
        // enzyme's site); "can move" or a designed sequence are set on them after.
        await this._runFlow('scaffold', this._spec(), {
          atoms: only.atoms, sidechainsOnly: structure.value === 'sidechain', unindexed: position.value === 'anywhere',
        });
        if (structure.value === 'rigid' || (structure.value === 'fixed' && sequence.value === 'designed')) {
          const now = chainNow();
          const ids = this._inputDoc().atoms.res_id;
          const kept = (now?.polymerAtomIndices ?? []).filter((i) => only.want.has(Number(ids[i])));
          if (kept.length) await this._applyPiece({ kind: 'kept', keep: 'all' }, kept, { structure: structure.value, sequence: sequence.value, position: position.value });
        }
        await setHotspots();
        return;
      }
      this._closePop();
      if (!await setHotspots()) return;
      const { lo, hi, atoms, count: n } = span();
      const want = { structure: structure.value, sequence: sequence.value, position: position.value };
      if (!lengthRow.hidden && lenTouched) {
        const min = Math.max(1, Math.round(Number(lenMin.value)) || n);
        const max = Math.max(min, Math.round(Number(lenMax.value)) || n);
        if (min !== n || max !== n) {
          await this._mutate('/segments/replace', { atom_indices: atoms, minimum: min, maximum: max });
          return;
        }
      }
      const whole = lo === p.start && hi === p.end;
      const changed = want.structure !== now.structure || want.sequence !== now.sequence || want.position !== now.position;
      if (changed) {
        // New settings go to the residues in the range only; the rest of the piece stays as it is.
        await this._applyPiece(p, atoms, want);
      } else if (!whole && !(now.structure === 'designed' && now.sequence === 'designed')) {
        // Only the range changed: the piece is now those residues, and the ones cut off are designed (same length).
        const inRange = new Set(atoms);
        const rest = p.atomIndices.filter((i) => !inRange.has(i));
        if (rest.length) await this._applyPiece(p, rest, { structure: 'designed', sequence: 'designed', position: 'fixed' });
      }
    };

    const row = (label, ...children) => el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: label }), ...children);
    const positionRow = row('Position', position);
    sync();
    return [
      this._popHead(`Chain ${c.id}`),
      row('Residues', el('span', { class: 'rfd4p-pop-inline', title: 'Narrow the range to change only some of these residues' },
        el('span', { text: c.id }), from, el('span', { text: '–' }), to)),
      row('Structure', structure),
      row('Sequence', sequence),
      positionRow,
      lengthRow,
      row('Hotspots', hot),
      row('Keep only', keepOnly),
      note,
      this._popActions(apply),
    ];
  }

  /** A piece's three properties, as the popup and the row name them. */
  _properties(p) {
    if (p.kind !== 'kept') return { structure: 'designed', sequence: 'designed', position: 'fixed' };
    const structure = { all: 'fixed', backbone: 'fixed', sidechain: 'sidechain', shape: 'rigid' }[p.keep] ?? 'designed';
    const sequence = p.keep === 'backbone' ? 'designed' : 'fixed';
    return { structure, sequence, position: p.anywhere ? 'anywhere' : 'fixed' };
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
    // "Fixed, can move" is one rigid body for the whole job (an antibody's two framework chains, a framework split by
    // CDRs): join the rigid group already there. Separate bodies are the Advanced editor's.
    const on = parts.filter(({ want }) => want.structure === 'rigid').flatMap(({ atoms }) => atoms);
    if (on.length && !await this._mutate('/internal-coords', { atom_indices: on, enabled: true, group_id: this._rigidGroup() })) return false;
    return this._batch(parts.flatMap(({ atoms, want }) => this._pieceEdits(atoms, want)));
  }

  _pieceEdits(atoms, { structure, sequence, position }) {
    const backbone = this._backboneOf(atoms);
    const side = atoms.filter((i) => !backbone.includes(i));
    const edits = [];
    if (structure !== 'rigid') {
      edits.push({ condition: 'coordinate', atom_indices: atoms, enabled: false });
      // Fixed with a designed sequence holds the backbone only: the side chains will change.
      const held = structure === 'fixed' ? (sequence === 'fixed' ? atoms : backbone) : structure === 'sidechain' ? side : [];
      if (held.length) edits.push({ condition: 'coordinate', atom_indices: held, enabled: true });
    }
    edits.push({ condition: 'sequence', atom_indices: atoms, enabled: sequence === 'fixed' });
    edits.push({ condition: 'index', atom_indices: atoms, enabled: position === 'anywhere' });
    return edits;
  }

  /** New residues: how many, or remove them. */
  _newPop(c, p) {
    const min = el('input', { type: 'number', min: 1, max: 1000, value: p.min, 'aria-label': 'Fewest new residues' });
    const max = el('input', { type: 'number', min: 1, max: 1000, value: p.max, 'aria-label': 'Most new residues' });
    const apply = () => {
      const lo = Math.max(1, Math.round(Number(min.value)) || p.min);
      const hi = Math.max(lo, Math.round(Number(max.value)) || p.max);
      this._closePop();
      return lo !== p.min || hi !== p.max ? this._mutate('/segments/range', { atom_index: p.atomIndex, minimum: lo, maximum: hi }) : true;
    };
    return [
      this._popHead(`Chain ${c.id}, new residues`),
      el('div', { class: 'rfd4p-pop-row', title: 'Each design picks one length from this range' },
        el('span', { class: 'rfd4p-pop-label', text: 'Length' }), el('span', { class: 'rfd4p-pop-inline' }, min, el('span', { text: '–' }), max)),
      this._popActions(apply),
    ];
  }

  /** A ligand: where it goes, or another molecule in its place. */
  _ligandPop(c, l) {
    const atoms = l.atomIndices;
    const pose = el('select', { 'aria-label': 'Pose', title: 'Fixed: kept where it is. Free: the model places it.' },
      el('option', { value: 'fixed', text: 'fixed', selected: l.pose !== 'predicted' }),
      el('option', { value: 'free', text: 'free', selected: l.pose === 'predicted' }));
    const code = el('input', { type: 'text', class: 'rfd4p-pop-text', maxlength: 5, spellcheck: 'false', autocomplete: 'off',
      placeholder: `another code, e.g. HEM`, 'aria-label': 'Swap for ligand code' });
    const note = el('p', { class: 'rfd4p-pop-note', 'aria-live': 'polite' });
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
      this._closePop();
      const fix = pose.value === 'fixed';
      if (fix !== (l.pose === 'fixed')) {
        void this._batch([{ condition: 'coordinate', atom_indices: atoms, enabled: fix }]);
      }
    };
    return [
      this._popHead(`Ligand ${l.name}`),
      el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Pose' }), pose),
      l.pose === 'partly fixed' ? el('p', { class: 'muted', title: 'Set atom by atom in the Advanced editor', text: `${l.fixedAtoms} of ${l.atoms} atoms fixed now` }) : null,
      el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Swap' }), code),
      note,
      this._popActions(apply),
    ].filter(Boolean);
  }

  /** The job's rigid group ("fixed, can move"), if it has one. */
  _rigidGroup() {
    const grp = this._inputDoc()?.atoms?.annotations?.condition_CoordinateGroup_1_atom;
    let id = null;
    for (const g of grp ?? []) if (g > 0 && (id === null || g < id)) id = g;
    return id;
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
    if (hot) {
      for (let i = 0; i < atoms.nAtoms; i += 1) {
        // The viewer numbers residues as the document's CIF writes auth_seq_id (a fetched entry keeps its authors'
        // numbering there), not by res_id: 5O45's A40 is author residue 56.
        const residue = Number(atoms.auth_seq_id?.[i] ?? atoms.res_id[i]);
        if (hot[i]) residues.set(`${atoms.chain_id[i]}|${residue}`, { chain: String(atoms.chain_id[i]), residue });
      }
    }
    // Sent on every render rather than once: Studio ignores it until its viewer has started.
    this.hotspotQueue = (this.hotspotQueue ?? Promise.resolve())
      .then(() => window.atomworksStudio.setHighlight('hotspots', residues.size ? [...residues.values()] : null, { color: HOTSPOT_COLOR }))
      .catch((err) => console.warn('Hotspot highlight failed:', err));
  }

  _renderPartial(spec, readonly) {
    const row = this.partialRow;
    row.hidden = spec.task.key !== 'partial';
    if (row.hidden) return;
    const pct = el('input', {
      type: 'number', min: 1, max: 100, step: 1, class: 'rfd4p-pct', disabled: readonly,
      value: Number.parseFloat(this.app.els.partialPercent?.value ?? '70') || 70, 'aria-label': 'Partial diffusion percent',
    });
    pct.addEventListener('change', () => {
      this.app._setPartialDiffusion?.(true, Math.max(1, Math.min(100, Number(pct.value) || 70)));
      this.render();
    });
    row.replaceChildren(
      el('span', { text: 'Noise added to this structure ' }), pct, el('span', { text: '% (higher = more different)' }),
      readonly ? null : el('button', {
        type: 'button', class: 'secondary compact', text: 'Turn off',
        onclick: () => { this.app._setPartialDiffusion?.(false); this.render(); },
      }));
  }

  // ---- selection popup ----------------------------------------------------------------------------------------

  _renderPop() {
    const pop = this.pop;
    const doc = this._inputDoc();
    const sel = window.atomworksStudio?.getSelection?.() ?? [];
    if (!sel.length || !doc || this._viewingDesign() || document.body.classList.contains('rfd4p-advanced') || this.busy) {
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
    const title = [parts.join(', '), ligNames.join(', ')].filter(Boolean).join(' · ');
    rows.push(el('div', { class: 'rfd4p-pop-title' },
      el('strong', { text: title || `${sel.length} atoms` }),
      nRes ? el('span', { class: 'muted', text: `${nRes} residue${nRes === 1 ? '' : 's'}` }) : null));
    const done = () => { pop.hidden = true; };
    let apply = async () => {};
    let keepOnly = null;

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
      const sync = () => {
        if (structure.value === 'sidechain') sequence.value = 'fixed';
        sequence.disabled = structure.value === 'sidechain';
        const free = structure.value === 'designed' && sequence.value === 'designed';
        if (free && position.value !== 'mixed') position.value = 'fixed';
        positionRow.hidden = free;
        lengthRow.hidden = !free || !oneStretch;
      };
      structure.addEventListener('change', sync);
      sequence.addEventListener('change', sync);
      sync();
      const hot = el('input', { type: 'checkbox', checked: hotN > 0 && hotN === polymerAtoms.length, 'aria-label': 'Hotspot' });
      hot.indeterminate = hotN > 0 && hotN < polymerAtoms.length;
      let hotTouched = false;
      hot.addEventListener('change', () => { hotTouched = true; });
      rows.push(row('Structure', structure), row('Sequence', sequence), positionRow, lengthRow,
        el('label', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Hotspot' }), hot));
      apply = async () => {
        done();
        if (hotTouched && !await this._batch([{ condition: 'hotspot', atom_indices: polymerAtoms, enabled: hot.checked }])) return;
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
          const want = { structure: pick(structure, h, 'structure'), sequence: pick(sequence, h, 'sequence'), position: pick(position, h, 'position') };
          if (want.structure === 'sidechain') want.sequence = 'fixed';
          if (want.structure === 'designed' && want.sequence === 'designed') want.position = 'fixed';
          if (want.structure !== h.now.structure || want.sequence !== h.now.sequence || want.position !== h.now.position) {
            changes.push({ p: h.p, atoms: h.atoms, want });
          }
        }
        if (changes.length) await this._applyPieces(changes);
      };
      keepOnly = el('button', {
        type: 'button', class: 'secondary compact', text: 'Keep only',
        title: 'Keep only these residues of their chains; the rest becomes new residues',
        onclick: async () => {
          done();
          await this._runFlow('scaffold', this._spec(), {
            atoms: polymerAtoms, sidechainsOnly: structure.value === 'sidechain', unindexed: position.value === 'anywhere',
          });
        },
      });
    }
    if (ligandAtoms.length) {
      const nFixed = ligandAtoms.filter(hasCrd).length;
      const pose = select('Pose', [['fixed', 'fixed'], ['free', 'free']],
        nFixed === ligandAtoms.length ? 'fixed' : nFixed === 0 ? 'free' : 'mixed');
      rows.push(row(ligNames.length === 1 ? ligNames[0] : 'Ligands', pose));
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
    rows.push(el('div', { class: 'rfd4p-pop-actions' },
      el('button', {
        type: 'button', class: 'danger compact', text: 'Delete',
        onclick: () => { done(); void this._mutate('/delete', { atom_indices: residues }); },
      }),
      keepOnly,
      el('span', { class: 'rfd4p-pop-spacer' }),
      el('button', { type: 'button', class: 'secondary compact', text: 'Cancel', onclick: () => document.getElementById('clear-selection')?.click() }),
      el('button', { type: 'button', class: 'primary compact', text: 'Apply', onclick: () => { this.applying = Promise.resolve(apply()); } })));
    pop.replaceChildren(...rows);
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
          unindexed: Boolean(opts.unindexed),
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
      if (task.recipe && await this._buildRecipe(task)) return;
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
  async _fetchPdb(raw) {
    const [id = '', ...chains] = String(raw ?? '').trim().toUpperCase().split(/[\s,;]+/).filter(Boolean);
    const status = this.els.fetchStatus;
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
      const opened = await window.atomworksStudio?.openFile?.(new File([text], `${id}.cif`, { type: 'chemical/x-mmcif' }));
      if (opened !== false && chains.length) {
        const a = this._inputDoc()?.atoms;
        const present = [...new Set(Array.from(a?.chain_id ?? [], String))];
        const missing = chains.filter((c) => !present.some((p) => p.toUpperCase() === c));
        if (missing.length) throw new Error(`no chain ${missing.join(', ')} (it has ${present.join(', ')})`);
        const drop = [];
        for (let i = 0; i < a.nAtoms; i += 1) if (!chains.includes(String(a.chain_id[i]).toUpperCase())) drop.push(i);
        if (drop.length) await this._mutate('/delete', { atom_indices: drop });
      }
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
   * finished, with no Undo history (an example is a starting point). Resolves false if the entry could not be fetched
   * (nothing is changed then), true once built.
   */
  async _buildRecipe(task) {
    const card = (id) => [...this.els.chainList.querySelectorAll('.rfd4p-chain')]
      .find((c) => c.querySelector('.rfd4p-chain-id')?.textContent === id);
    const subs = (chain) => [...(card(chain)?.querySelectorAll('.rfd4p-sub') ?? [])];
    const isNew = (sub) => sub.querySelector('.rfd4p-sub-status')?.textContent === 'new';
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
    let fetched = false;
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
          const doc = await createStudioDocWithCcd(text, id, { preset: 'rcsb', assembly: '1' });
          if (chains.length) {
            const drop = [];
            for (let i = 0; i < doc.atoms.nAtoms; i += 1) if (!chains.includes(String(doc.atoms.chain_id[i]))) drop.push(i);
            if (drop.length) doc.deleteAtoms(drop);
          }
          this.buildSession = `rfd4p_example_${task.example}_${Date.now().toString(36)}`;
          getSharedStudioStore().sessions.set(this.buildSession, doc);
          this.render();
          fetched = true;
        } else if (step.piece) {
          const s = step.set;
          const sub = subs(step.piece).find((x) => {
            const m = /(\d+)(?:–(\d+))?/.exec(x.querySelector('.rfd4p-sub-name')?.textContent ?? '');
            return !isNew(x) && m && step.at >= Number(m[1]) && step.at <= Number(m[2] ?? m[1]);
          });
          if (!sub) throw new Error(`chain ${step.piece} has no residue ${step.at}`);
          const pop = open(sub);
          if (s.from != null) { fill(pop, 'First residue', s.from); fill(pop, 'Last residue', s.to); }
          if (s.structure) fill(pop, 'Structure', s.structure, 'change');
          if (s.sequence) fill(pop, 'Sequence', s.sequence, 'change');
          if (s.position) fill(pop, 'Position in chain', s.position, 'change');
          if (s.length) { fill(pop, 'Fewest residues', s.length[0]); fill(pop, 'Most residues', s.length[1]); }
          if (s.hotspots) fill(pop, 'Hotspots', s.hotspots);
          if (s.keepOnly) fill(pop, 'Keep only these residues', s.keepOnly);
          await apply(pop);
        } else if (step.removeNew) {
          for (let sub = subs(step.removeNew).find(isNew); sub; sub = subs(step.removeNew).find(isNew)) {
            await click(sub.querySelector('.rfd4p-x'));
          }
        } else if (step.addSeq) {
          await click(this.els.addSeq);
        } else if (step.newLength) {
          const chains = step.chain ? [card(step.chain)] : [...this.els.chainList.querySelectorAll('.rfd4p-chain')].reverse();
          const sub = chains.flatMap((c) => [...(c?.querySelectorAll('.rfd4p-sub') ?? [])]).find(isNew);
          const pop = open(sub);
          fill(pop, 'Fewest new residues', step.newLength[0]);
          fill(pop, 'Most new residues', step.newLength[1]);
          await apply(pop);
        } else if (step.ligand) {
          const sub = [...this.els.chainList.querySelectorAll('.rfd4p-sub')].find((x) => x.querySelector('.rfd4p-sub-name')?.textContent === step.ligand);
          const pop = open(sub);
          fill(pop, 'Pose', step.pose, 'change');
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
      await window.atomworksStudio?.openSession?.(sid);
      return true;
    } catch (err) {
      console.warn('Example set-up stopped:', err);
      this.lastError = `The ${task.title} example could not be set up: ${String(err?.message ?? err)}`;
      return fetched;
    } finally {
      // A build that stopped part way is still shown, as far as it got.
      if (this.buildSession) {
        const sid = this.buildSession;
        this.buildSession = null;
        await window.atomworksStudio?.openSession?.(sid);
      }
      this.building = false;
      document.body.classList.remove('rfd4p-building');
      this._closePop();
      this.showStart = false;
      this.render();
    }
  }
}
