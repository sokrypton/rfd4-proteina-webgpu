/**
 * The RFD4-Proteina card's guided layer over AtomWorks Studio: a "what do you want to design?" start screen, a Design
 * panel that states what the open document will generate (task, chain roles, lengths, contig) and edits it, and the
 * simple/advanced switch that hides Studio's expert tools.
 *
 * Every edit goes through Studio's own mutate path (`window.atomworksStudio?.applyEdit`), so Undo, the viewer and the
 * `atomworks-studio` "changed" event behave exactly as for edits made in the Studio panels.
 */
import { designSpec, formatResidueList, KEEP_MODES, parseResidueList } from './design-spec.js';
import { createStudioDocWithCcd, getSharedStudioStore } from './studio-api.js';
import { isCcdCode, ligandCifFromCcd } from './ligand-ccd.js';
import { fetchCcdComponents } from '../shared/rfd4p/pipeline/ccd.js';

const ADVANCED_KEY = 'rfd4p-advanced-editor';

/** Start-screen tasks, each opening one of the bundled Studio examples (`web/studio/examples`). */
const TASKS = [
  { example: 'unconditional', title: 'Protein from scratch', desc: 'A new fold of the length you choose' },
  { example: 'protein_binder', title: 'Protein binder', desc: 'Bind a target protein (PD-L1)' },
  { example: 'small_molecule_binder', title: 'Small-molecule binder', desc: 'A pocket around a ligand (SAM)' },
  { example: 'motif_scaffolding', title: 'Motif scaffolding', desc: 'Keep key residues, build the rest (1QJG)' },
  { example: 'enzyme_design', title: 'Enzyme', desc: 'Catalytic residues around a ligand' },
  { example: 'antibody_design', title: 'Antibody loop', desc: 'Redesign CDR H3 against HER2' },
  { example: 'nanobody', title: 'Nanobody (VHH)', desc: 'Design CDRs against PD-L1' },
  { example: 'inverse_folding', title: 'Redesign a sequence', desc: 'New sequence for a fixed backbone (2KL8)', then: 'sequence' },
  { example: 'partial_diffusion', title: 'Diversify a structure', desc: 'Variants close to a starting fold (2KL8)' },
];

/** Guided set-ups for a document (shown for an unconstrained structure, or on "Change…"). */
const FLOWS = [
  { key: 'binder', title: 'Design a binder to it', desc: 'Keep it fixed and add a new chain' },
  { key: 'scaffold', title: 'Keep the selected residues', desc: 'Motif scaffolding: fix them, rebuild the rest' },
  { key: 'ligand', title: 'Design a protein around the ligand', desc: 'Small-molecule binder or enzyme pocket' },
  { key: 'sequence', title: 'Redesign the sequence', desc: 'Keep the backbone, new sequence' },
  { key: 'diversify', title: 'Diversify it', desc: 'Partial diffusion from this structure' },
  { key: 'scratch', title: 'Redesign everything', desc: 'Same chain lengths, nothing kept' },
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

const range = (min, max) => (min === max ? `${min}` : `${min}–${max}`);

export class DesignPanel {
  /** @param {object} app the `Rfd4pStudioApp` (input session, partial-diffusion controls, run state) */
  constructor(app) {
    this.app = app;
    this.showFlows = false;
    this.flowForm = null;
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
      changeTask: $('rfd4p-change-task'),
      newDoc: $('rfd4p-new'),
      back: $('rfd4p-back'),
      guided: $('rfd4p-guided'),
      chainList: $('rfd4p-chain-list'),
      addSeq: $('rfd4p-add-seq'),
      addPdb: $('rfd4p-add-pdb'),
      addLig: $('rfd4p-add-lig'),
      undo: $('rfd4p-undo'),
      total: $('rfd4p-total'),
      contig: $('rfd4p-contig'),
      warnings: $('rfd4p-warnings'),
      runSettings: $('rfd4p-run-settings'),
      advanced: $('rfd4p-advanced'),
      editHint: $('rfd4p-edit-hint'),
    };
    // Floating actions next to the pointer for the current selection (viewer or sequence bar).
    this.pop = el('div', { class: 'rfd4p-pop', role: 'dialog', 'aria-label': 'Selection actions', hidden: true });
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
   * "+ pdb": a structure's chains added to the open job as kept chains (a target),
   * optionally in place of the job's current kept chains, so an example's target can be swapped without setting
   * the job up again.
   */
  _buildAddStructure() {
    const actions = this.els.addPdb?.parentElement;
    if (!actions) return;
    const id = el('input', {
      type: 'text', class: 'rfd4p-add-id', spellcheck: 'false', autocomplete: 'off', maxlength: 40,
      placeholder: 'PDB ID, then chains: 5O45 A', 'aria-label': 'PDB ID and chains',
    });
    const replace = el('input', { type: 'checkbox' });
    const replaceLabel = el('label', { class: 'checkbox' }, replace, ' Instead of the current kept chains (swap the target)');
    const fetchBtn = el('button', { type: 'submit', class: 'primary compact', text: 'Add' });
    const fileInput = el('input', { type: 'file', accept: '.cif,.mmcif,.pdb,.ent,.cif.zst', hidden: true });
    const fileBtn = el('button', { type: 'button', class: 'secondary compact', text: 'Open a file…', onclick: () => fileInput.click() });
    const status = el('p', { class: 'rfd4p-add-status muted', 'aria-live': 'polite' });
    const cancel = el('button', { type: 'button', class: 'secondary compact', text: 'Cancel', onclick: () => { form.hidden = true; } });
    const form = el('form', { class: 'rfd4p-add-structure', hidden: true },
      el('strong', { text: 'Add chains from a structure' }),
      el('span', { class: 'muted', text: 'They are kept as they are: a target to design against, for example.' }),
      el('div', { class: 'rfd4p-add-row' }, id, fetchBtn), el('div', { class: 'rfd4p-add-row' }, fileBtn, fileInput, cancel),
      replaceLabel, status);
    this.addStructure = { form, replace, replaceLabel, status, id };
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
          await this._addStructure(await res.text(), pdb, chains, replace.checked);
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
          await this._addStructure(text, file.name.replace(/\.(cif|mmcif|pdb|ent|zst)+$/i, ''), chains, replace.checked);
        } catch (err) {
          status.textContent = `Could not add ${file.name}: ${err?.message ?? err}`;
        }
      })();
    });
    actions.after(form);
  }

  _openAddStructure() {
    const { form, replace, replaceLabel, status, id } = this.addStructure ?? {};
    if (!form) return;
    const hasTarget = Boolean(this._spec()?.chains.some((c) => c.role === 'fixed'));
    replaceLabel.hidden = !hasTarget;
    replace.checked = hasTarget;
    status.textContent = '';
    form.hidden = false;
    id.focus();
  }

  /** Add `chains` (all when empty) of a structure to the job as kept chains; with `replace`, drop the kept chains first. */
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
    if (replace) {
      const spec = this._spec();
      const old = spec?.chains.filter((c) => c.role === 'fixed').flatMap((c) => c.atomIndices) ?? [];
      if (old.length && !await this._mutate('/delete', { atom_indices: old })) throw new Error(this.lastError ?? 'could not remove the old target');
    }
    const added = await this._combine(doc);
    await this._batch([
      { condition: 'coordinate', atom_indices: added, enabled: true },
      { condition: 'sequence', atom_indices: added, enabled: true },
    ]);
    const newChains = [...new Set(added.map((i) => String(this._inputDoc().atoms.chain_id[i])))];
    status.textContent = `Added ${name} as chain${newChains.length === 1 ? '' : 's'} ${newChains.join(', ')}, kept in place.`
      + ' To aim a binder at part of it: Edit ▸ Mark hotspots.';
    this.render();
  }

  /** Merge a document into the job (one undo step); resolves to the added atoms' indices. */
  async _combine(doc) {
    const store = getSharedStudioStore();
    const before = this._inputDoc()?.atoms?.nAtoms ?? 0;
    const tmp = `rfd4p_add_${Date.now().toString(36)}`;
    store.sessions.set(tmp, doc);
    try {
      if (!await this._mutate('/combine', { session_ids: [tmp] })) throw new Error(this.lastError ?? 'could not add it');
    } finally {
      store.sessions.delete(tmp);
    }
    const after = this._inputDoc()?.atoms?.nAtoms ?? before;
    return Array.from({ length: after - before }, (_, k) => before + k);
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
    await this._combine(doc);
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
    els.changeTask?.addEventListener('click', () => {
      this.showFlows = !this.showFlows;
      this.flowForm = null;
      this.render();
    });
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
    window.addEventListener('atomworks-studio', (ev) => {
      if (ev.detail?.type !== 'selection') return;
      this.selectionCount = ev.detail?.count ?? 0;
      queueMicrotask(() => this._renderPop());
      if (!this.els.guided || this.els.guided.hidden) return;
      // An open form keeps what the user typed: refresh only the selection-dependent parts.
      if (this.flowForm && this.formUpdate) this.formUpdate();
      else this._renderFlows(this._spec(), true);
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
    const starting = !hasDoc || Boolean(this.showStart);
    if (els.start) els.start.hidden = !starting;
    if (els.back) els.back.hidden = !hasDoc;
    if (els.design) els.design.hidden = starting;
    if (els.runSettings) els.runSettings.hidden = starting;
    if (starting) return;

    const viewingDesign = this._viewingDesign();
    els.design.classList.toggle('rfd4p-readonly', viewingDesign);
    els.taskLabel.textContent = spec.task.label;
    els.taskDetail.textContent = viewingDesign
      ? 'Showing a generated design. Select Input under View to change the setup.'
      : spec.task.detail;
    els.changeTask.hidden = viewingDesign;
    if (els.newDoc) els.newDoc.hidden = viewingDesign;
    const actions = els.addSeq?.parentElement;
    if (actions) actions.hidden = viewingDesign;
    if (viewingDesign && this.addStructure) this.addStructure.form.hidden = true;
    if (els.editHint) els.editHint.hidden = viewingDesign;
    els.undo.hidden = viewingDesign;
    els.undo.disabled = document.getElementById('undo')?.disabled ?? true;

    els.chainList.replaceChildren(...spec.chains.map((c) => this._chainRow(c, viewingDesign)));
    const designed = range(spec.designedMin, spec.designedMax);
    els.total.textContent = spec.totalMax
      ? [`${designed} residue${spec.designedMax === 1 ? '' : 's'} designed`,
        spec.predicted ? `${spec.predicted} folded from sequence` : '',
        `${range(spec.totalMin, spec.totalMax)} in total`].filter(Boolean).join(' · ')
      : '';
    els.contig.textContent = spec.contig || '—';

    const warnings = this.lastError ? [this.lastError, ...spec.warnings] : [...spec.warnings];
    if (spec.task.key === 'partial' && spec.chains.some((c) => c.segments.length)) {
      warnings.push('Partial diffusion needs every residue defined; this input has new-residue segments.');
    }
    els.warnings.replaceChildren(...warnings.map((w) => el('p', { class: 'rfd4p-warning', text: w })));

    this._renderPartial(spec, viewingDesign);
    this.showHotspots();

    const needsChoice = ['redesign_all', 'nothing'].includes(spec.task.key);
    this._renderFlows(spec, !viewingDesign && (this.showFlows || needsChoice || this.flowForm !== null));

    // Nothing to generate: keep Generate off (a running job keeps its Stop button).
    if (!this.app.activeAbortController && this.app.els.predictBtn) {
      this.app.els.predictBtn.disabled = Boolean(this.app.noWebGpu) || spec.task.key === 'nothing' || spec.task.key === 'empty';
    }
  }

  /**
   * A chain as its pieces, in order (the contig, spelled out). Each piece is one line: what it is (kept, redesigned,
   * new), its residues, its modifiers as chips (× removes one), and one Edit menu for everything else. A bar above
   * draws the pieces to scale.
   */
  _chainRow(c, readonly) {
    const kindName = { dna: ' · DNA', rna: ' · RNA' }[c.kind] ?? '';
    const length = c.lengthMin === c.lengthMax ? String(c.lengthMin) : `${c.lengthMin}–${c.lengthMax}`;
    // Kept: nothing in it is made by the model. Designed: nothing in it is kept. Otherwise, partly designed.
    const kept = c.pieces.filter((p) => p.kind === 'kept').length;
    const what = c.role === 'ligand' ? 'Ligand'
      : `${kept === c.pieces.length ? 'Kept' : kept ? 'Partly designed' : 'Designed'} · ${length} residues${kindName}`;
    const head = el('div', { class: 'rfd4p-chain-head' },
      el('span', { class: 'rfd4p-chain-id', text: c.id }),
      el('span', { class: 'rfd4p-chain-summary', text: `Chain ${c.id} · ${what}` }),
      readonly ? null : el('button', {
        type: 'button', class: 'rfd4p-chain-remove', title: `Remove chain ${c.id}`, 'aria-label': `Remove chain ${c.id}`,
        text: '×', onclick: () => void this._mutate('/delete', { atom_indices: c.atomIndices }),
      }));
    const body = el('div', { class: 'rfd4p-chain-body' });
    const row = el('div', { class: `rfd4p-chain rfd4p-role-${c.role}` }, head, body);
    // The chain's colour, as in the viewer and the sequence bar.
    const color = window.atomworksStudio?.chainColor?.(c.id);
    if (color) row.style.setProperty('--chain-color', color);

    if (c.pieces.length > 1 || c.pieces.some((p) => p.hotspots.length)) body.append(this._chainBar(c));
    for (const p of c.pieces) body.append(this._pieceRow(c, p, readonly));
    for (const l of c.ligands) body.append(this._ligandRow(c, l, readonly));
    return row;
  }

  /** The chain to scale: kept pieces solid, designed ones striped, new residues outlined at their mean length. */
  _chainBar(c) {
    const bar = el('div', { class: 'rfd4p-chain-bar', 'aria-hidden': 'true' });
    for (const p of c.pieces) {
      const len = p.kind === 'new' ? (p.min + p.max) / 2 : p.count;
      const block = el('span', { class: `rfd4p-bar-${p.kind}${p.keep && p.keep !== 'all' ? ' rfd4p-bar-partial' : ''}`, title: this._pieceText(c, p) });
      block.style.flexGrow = String(Math.max(1, len));
      const span = p.kind === 'new' ? 0 : p.end - p.start + 1;
      for (const id of p.hotspots) {
        const tick = el('i', { class: 'rfd4p-bar-hotspot' });
        tick.style.left = `${((id - p.start + 0.5) / span) * 100}%`;
        block.append(tick);
      }
      bar.append(block);
    }
    return bar;
  }

  _range(c, p) {
    return p.start === p.end ? `${c.id}${p.start}` : `${c.id}${p.start}–${p.end}`;
  }

  _pieceText(c, p) {
    if (p.kind === 'new') {
      // "between H22 and H24" -> "after H22": the row is short, and the bar shows where it is.
      const where = p.where.replace(/^between (\S+) and \S+$/, 'after $1').replace('-terminal', '-term');
      return `New${where ? ` · ${where}` : ''}`;
    }
    const n = p.count > 1 ? ` (${p.count})` : '';
    if (p.kind === 'designed') return `Redesigned · ${this._range(c, p)}${n}`;
    return `Kept · ${p.source ? `${p.source} (now ${this._range(c, p)})` : `${this._range(c, p)}${n}`}`;
  }

  _chip(text, title, onRemove) {
    return el('span', { class: 'rfd4p-chip', title },
      el('span', { text }),
      onRemove ? el('button', { type: 'button', class: 'rfd4p-chip-x', 'aria-label': `Remove “${text}”`, text: '×', onclick: onRemove }) : null);
  }

  _pieceKey(c, p) {
    return `${c.id}:${p.kind === 'new' ? `n${p.atomIndex}` : p.start}`;
  }

  _pieceRow(c, p, readonly) {
    const label = el('span', { class: `rfd4p-piece-label rfd4p-piece-${p.kind}`, text: this._pieceText(c, p) });
    label.title = label.textContent;
    const line = el('div', { class: 'rfd4p-piece-line' }, label);
    const piece = el('div', { class: 'rfd4p-piece' }, line);

    if (p.kind === 'new') {
      const min = el('input', { type: 'number', min: 1, max: 1000, value: p.min, 'aria-label': 'Fewest new residues', disabled: readonly });
      const max = el('input', { type: 'number', min: 1, max: 1000, value: p.max, 'aria-label': 'Most new residues', disabled: readonly });
      const commit = () => {
        const lo = Math.max(1, Math.round(Number(min.value)) || p.min);
        const hi = Math.max(lo, Math.round(Number(max.value)) || p.max);
        if (lo === p.min && hi === p.max) return;
        void this._mutate('/segments/range', { atom_index: p.atomIndex, minimum: lo, maximum: hi });
      };
      min.addEventListener('change', commit);
      max.addEventListener('change', commit);
      line.append(el('span', { class: 'rfd4p-seg-inputs', title: 'Length: the model picks one in this range for each design' },
        min, el('span', { text: '–' }), max));
      // A chain made only of this segment is removed with the chain's own ×.
      if (!readonly && this._chainItems(c) > 1) {
        line.append(this._menu(`Edit new residues ${this._pieceText(c, p)}`, [
          ['delete', 'Remove these new residues'],
        ], () => this._mutate('/delete', { atom_indices: p.atomIndices })));
      }
      return piece;
    }

    // Modifiers, as chips: how a kept piece is held, whether it may move along the chain, its hotspots.
    const chips = el('div', { class: 'rfd4p-chips' });
    const can = !readonly;
    if (p.kind === 'kept' && p.keep !== 'all') {
      const m = KEEP_MODES[p.keep];
      chips.append(this._chip(m.chip, m.help, can ? () => void this._setKeep(p, 'all') : null));
    }
    if (p.anywhere) {
      chips.append(this._chip('anywhere in the chain', 'Kept, but the model chooses where in the chain it goes',
        can ? () => void this._batch([{ condition: 'index', atom_indices: p.atomIndices, enabled: false }]) : null));
    }
    const key = this._pieceKey(c, p);
    if (p.hotspots.length || this.editingHotspots === key) chips.append(this._hotspotChip(c, p, key, readonly));
    if (chips.childElementCount) piece.append(chips);

    if (readonly) return piece;
    const options = [];
    if (p.kind === 'kept') {
      options.push(['', '— Keep —']);
      for (const [mode, m] of Object.entries(KEEP_MODES)) {
        if (mode !== p.keep) options.push([`keep:${mode}`, m.label]);
      }
      options.push(['', '— Also —']);
      if (!p.hotspots.length) options.push(['hotspots', 'Mark hotspots (where a binder should touch)…']);
      options.push(p.anywhere ? ['here', 'Keep it where it is in the chain'] : ['anywhere', 'Let it go anywhere in the chain']);
      options.push(['', '— Change —']);
      options.push(['keep:designed', 'Redesign these residues in place']);
    } else {
      options.push(['', '— Change —']);
      options.push(['keep:all', 'Keep these residues instead']);
    }
    options.push(['replace', 'Replace with new residues (any length)']);
    options.push(['before', 'Add new residues before']);
    options.push(['after', 'Add new residues after']);
    // Alone in its chain, a piece is deleted with the chain's own ×.
    if (this._chainItems(c) > 1) options.push(['delete', 'Delete these residues']);
    line.append(this._menu(`Edit ${label.textContent}`, options, (v) => {
      if (v.startsWith('keep:')) return this._setKeep(p, v.slice(5));
      if (v === 'hotspots') { this.editingHotspots = key; this.render(); return null; }
      if (v === 'anywhere' || v === 'here') {
        return this._batch([{ condition: 'index', atom_indices: p.atomIndices, enabled: v === 'anywhere' }]);
      }
      if (v === 'replace') {
        return this._mutate('/segments/replace', {
          atom_indices: p.atomIndices, minimum: Math.max(1, Math.round(p.count * 0.75)), maximum: Math.max(1, Math.round(p.count * 1.25)),
        });
      }
      if (v === 'before' || v === 'after') {
        return this._mutate('/segments', {
          minimum: 5, maximum: 10, atom_index: v === 'before' ? p.atomIndices[0] : p.atomIndices[p.atomIndices.length - 1],
          before: v === 'before', new_chain: false, chain_id: null, chain_type: 'polypeptide(L)',
        });
      }
      if (v === 'delete') return this._mutate('/delete', { atom_indices: p.atomIndices });
      return null;
    }));
    return piece;
  }

  /** Pieces and ligands in a chain: with one, the chain's × is that item's delete. */
  _chainItems(c) {
    return c.pieces.length + c.ligands.length;
  }

  /** One small "Edit" menu: a select whose first option is its label, reset after every choice. */
  _menu(ariaLabel, options, onChoose) {
    const select = el('select', { class: 'rfd4p-piece-menu', 'aria-label': ariaLabel, title: 'Change this piece' });
    select.append(el('option', { value: '', text: 'Edit ▾', selected: true }));
    for (const [value, text] of options) {
      select.append(value ? el('option', { value, text }) : el('option', { value: '', text, disabled: true }));
    }
    select.addEventListener('change', () => {
      const v = select.value;
      select.value = '';
      if (v) void onChoose(v);
    });
    return select;
  }

  /** How a kept piece is held, or 'designed' to redesign it in place. Shape-only holds are cleared first. */
  async _setKeep(p, mode) {
    const all = p.atomIndices;
    const backbone = this._backboneOf(all);
    const side = all.filter((i) => !backbone.includes(i));
    if (p.keep === 'shape' && !await this._mutate('/internal-coords', { atom_indices: all, enabled: false })) return false;
    if (mode === 'shape') {
      return await this._mutate('/internal-coords', { atom_indices: all, enabled: true })
        && this._batch([{ condition: 'sequence', atom_indices: all, enabled: true }]);
    }
    const edits = [{ condition: 'coordinate', atom_indices: all, enabled: false }];
    if (mode === 'all') edits.push({ condition: 'coordinate', atom_indices: all, enabled: true });
    if (mode === 'backbone' && backbone.length) edits.push({ condition: 'coordinate', atom_indices: backbone, enabled: true });
    if (mode === 'sidechain' && side.length) edits.push({ condition: 'coordinate', atom_indices: side, enabled: true });
    edits.push({ condition: 'sequence', atom_indices: all, enabled: mode !== 'backbone' && mode !== 'designed' });
    if (mode === 'designed') edits.push({ condition: 'index', atom_indices: all, enabled: false });
    return this._batch(edits);
  }

  _backboneOf(indices) {
    const a = this._inputDoc()?.atoms;
    if (!a) return [];
    return indices.filter((i) => ['N', 'CA', 'C', 'O'].includes(String(a.atom_name[i])));
  }

  _ligandRow(c, l, readonly) {
    const pose = l.pose === 'fixed' ? ['pose fixed', 'Its position and orientation are kept']
      : l.pose === 'predicted' ? ['pose free', 'The model places it; only what it is is given']
        : [`${l.fixedAtoms} of ${l.atoms} atoms fixed`, 'Only some of its atoms are held in place'];
    const line = el('div', { class: 'rfd4p-piece-line' },
      el('span', { class: 'rfd4p-piece-label rfd4p-piece-ligand', text: `Ligand · ${l.name}` }));
    const piece = el('div', { class: 'rfd4p-piece' }, line, el('div', { class: 'rfd4p-chips' }, this._chip(pose[0], pose[1], null)));
    if (readonly) return piece;
    const atoms = c.ligandAtomIndices;
    const options = [];
    if (l.pose !== 'fixed') options.push(['fix', 'Fix its pose (keep it where it is)']);
    if (l.pose !== 'predicted') options.push(['free', 'Free its pose (the model places it)']);
    options.push(['swap', 'Swap for another ligand…']);
    if (this._chainItems(c) > 1) options.push(['delete', 'Delete this ligand']);
    const key = `${c.id}:${l.name}`;
    line.append(this._menu(`Edit ligand ${l.name}`, options, (v) => {
      if (v === 'swap') { this.swapLigand = key; this.render(); return null; }
      if (v === 'delete') return this._mutate('/delete', { atom_indices: atoms });
      return this._batch([{ condition: 'coordinate', atom_indices: atoms, enabled: v === 'fix' }]);
    }));
    if (this.swapLigand === key) {
      piece.append(this._ligandCodeForm({
        label: `Swap ${l.name} for`,
        button: 'Swap',
        onSubmit: async (code) => {
          await this._addLigand(code, { replace: atoms, chainId: c.id });
          this.swapLigand = null;
          this.render();
        },
        onCancel: () => { this.swapLigand = null; this.render(); },
      }));
    }
    return piece;
  }

  /** A piece's hotspots as an editable chip ("A56, A115–117"); kept in step with clicking residues. */
  _hotspotChip(c, p, key, readonly) {
    const input = el('input', {
      type: 'text', class: 'rfd4p-hotspot-input', spellcheck: 'false', autocomplete: 'off', disabled: readonly,
      value: formatResidueList(c.id, p.hotspots), placeholder: `e.g. ${c.id}${p.start}, ${c.id}${Math.min(p.end, p.start + 4)}`,
      'aria-label': `Hotspots in ${this._range(c, p)}`,
    });
    input.size = Math.max(10, input.value.length + 1);
    const note = this.hotspotNote?.key === key ? this.hotspotNote.text : '';
    input.addEventListener('change', () => {
      const { residues, bad } = parseResidueList(input.value, c.id);
      const a = this._inputDoc()?.atoms;
      if (!a) return;
      const wanted = new Set(residues.filter((r) => r.chain === c.id).map((r) => r.resId));
      const atoms = p.atomIndices.filter((i) => wanted.has(Number(a.res_id[i])));
      const found = new Set(atoms.map((i) => Number(a.res_id[i])));
      const outside = [...wanted].filter((id) => !found.has(id));
      const otherChain = residues.filter((r) => r.chain !== c.id).map((r) => `${r.chain}${r.resId}`);
      const problems = [
        bad.length ? `not understood: ${bad.join(' ')}` : '',
        outside.length ? `not in ${this._range(c, p)}: ${formatResidueList(c.id, outside)}` : '',
        otherChain.length ? `other chains have their own: ${otherChain.slice(0, 3).join(', ')}` : '',
      ].filter(Boolean);
      this.hotspotNote = problems.length ? { key, text: problems.join('; ') } : null;
      this.editingHotspots = atoms.length ? null : key;
      const edits = [{ condition: 'hotspot', atom_indices: p.atomIndices, enabled: false }];
      if (atoms.length) edits.push({ condition: 'hotspot', atom_indices: atoms, enabled: true });
      void this._batch(edits);
    });
    if (this.editingHotspots === key && !p.hotspots.length) queueMicrotask(() => input.focus());
    const chip = el('span', {
      class: 'rfd4p-chip rfd4p-chip-hotspots',
      title: 'Residues a binder should touch: 3–6 at the interface work well. Type them, or select residues in the viewer and press Hotspot.',
    }, el('span', { text: 'hotspots' }), input,
    readonly ? null : el('button', {
      type: 'button', class: 'rfd4p-chip-x', 'aria-label': 'Remove the hotspots', text: '×',
      onclick: () => {
        this.editingHotspots = null;
        this.hotspotNote = null;
        if (p.hotspots.length) void this._batch([{ condition: 'hotspot', atom_indices: p.atomIndices, enabled: false }]);
        else this.render();
      },
    }));
    return note ? el('span', { class: 'rfd4p-chip-wrap' }, chip, el('span', { class: 'rfd4p-hotspot-note muted', text: note })) : chip;
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
        if (hot[i]) residues.set(`${atoms.chain_id[i]}|${atoms.res_id[i]}`, { chain: String(atoms.chain_id[i]), residue: Number(atoms.res_id[i]) });
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
    const seqKept = (i) => { const v = ann.condition_sequence_1_residue?.[i]; return v != null && v !== '<M>' && v !== '.'; };
    const hot = sel.some((i) => ann.condition_hotspot_1_atom?.[i]);
    const state = polymerAtoms.length
      ? (polymerAtoms.every(hasCrd) ? (polymerAtoms.every(seqKept) ? 'fixed' : 'backbone kept')
        : polymerAtoms.every(seqKept) ? 'sequence kept' : polymerAtoms.some(hasCrd) ? 'partly fixed' : 'designed')
      : '';

    const btn = (text, onclick, cls = 'secondary compact') => el('button', { type: 'button', class: cls, text, onclick });
    const act = (fn) => async () => { pop.hidden = true; await fn(); };
    const rows = [];
    const title = [parts.join(', '), ligNames.join(', ')].filter(Boolean).join(' · ');
    rows.push(el('div', { class: 'rfd4p-pop-head' },
      el('strong', { text: title || `${sel.length} atoms` }),
      el('span', { class: 'muted', text: [nRes ? `${nRes} residue${nRes === 1 ? '' : 's'}` : '', state, hot ? 'hotspot' : ''].filter(Boolean).join(' · ') }),
      el('button', {
        type: 'button', class: 'rfd4p-pop-close', 'aria-label': 'Clear selection', text: '×',
        onclick: () => { pop.hidden = true; document.getElementById('clear-selection')?.click(); },
      })));
    if (polymerAtoms.length) {
      const edits = (coord, seq) => [
        { condition: 'coordinate', atom_indices: polymerAtoms, enabled: coord },
        { condition: 'sequence', atom_indices: polymerAtoms, enabled: seq },
      ];
      rows.push(el('div', { class: 'rfd4p-pop-row' },
        btn(hot ? 'Remove hotspot' : 'Hotspot', act(() => this._batch([{ condition: 'hotspot', atom_indices: polymerAtoms, enabled: !hot }])), hot ? 'secondary compact' : 'primary compact'),
        btn('Keep fixed', act(() => this._batch(edits(true, true)))),
        btn('Keep sequence', act(() => this._batch(edits(false, true)))),
        btn('Redesign', act(() => this._batch(edits(false, false))))));
      const n = nRes || 1;
      const min = el('input', { type: 'number', min: 1, max: 500, value: Math.max(1, Math.round(n * 0.75)), 'aria-label': 'New residues minimum' });
      const max = el('input', { type: 'number', min: 1, max: 500, value: Math.max(1, Math.round(n * 1.25)), 'aria-label': 'New residues maximum' });
      rows.push(el('div', { class: 'rfd4p-pop-row' },
        btn('Replace with', act(() => this._mutate('/segments/replace', {
          atom_indices: polymerAtoms, minimum: Number(min.value), maximum: Number(max.value),
        }))),
        min, el('span', { text: '–' }), max, el('span', { class: 'muted', text: 'new' })));
      rows.push(el('div', { class: 'rfd4p-pop-row' },
        btn('Scaffold around these…', () => {
          pop.hidden = true;
          this.showFlows = true;
          this.flowForm = 'scaffold';
          this.render();
          this.els.guided?.scrollIntoView({ block: 'nearest' });
        }),
        btn('Delete', act(() => this._mutate('/delete', { atom_indices: residues })), 'danger compact')));
    }
    if (ligandAtoms.length) {
      const fixed = ligandAtoms.some(hasCrd);
      rows.push(el('div', { class: 'rfd4p-pop-row' },
        btn(fixed ? 'Free pose' : 'Fix pose', act(() => this._batch([{ condition: 'coordinate', atom_indices: ligandAtoms, enabled: !fixed }]))),
        btn('Design a protein around it…', () => {
          pop.hidden = true;
          this.showFlows = true;
          this.flowForm = 'ligand';
          this.render();
        }),
        polymerAtoms.length ? null : btn('Delete', act(() => this._mutate('/delete', { atom_indices: ligandAtoms })), 'danger compact')));
    }
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

  // ---- guided set-ups -----------------------------------------------------------------------------------------

  _renderFlows(spec, visible = !this.els.guided?.hidden) {
    const box = this.els.guided;
    if (!box) return;
    // The open form (if any) installs its own selection updater below.
    this.formUpdate = () => this._updateScaffoldButton();
    box.hidden = !visible || !spec;
    if (box.hidden) return;
    const children = [el('p', { class: 'rfd4p-lead', text: 'What should happen to this structure?' })];
    const hasPolymer = spec.chains.some((c) => c.residues > 0);
    const hasLigand = spec.chains.some((c) => c.ligands.length > 0);
    for (const f of FLOWS) {
      if (f.key === 'ligand' && !hasLigand) continue;
      const needsSelection = f.key === 'scaffold' && this.selectionCount === 0;
      const disabled = (f.key !== 'ligand' && !hasPolymer) || needsSelection;
      children.push(el('button', {
        type: 'button',
        class: `rfd4p-flow${this.flowForm === f.key ? ' active' : ''}`,
        'data-flow': f.key,
        disabled,
        title: needsSelection ? 'Select residues in the sequence bar or viewer first' : null,
        onclick: () => {
          if (f.key === 'binder' || f.key === 'scaffold' || f.key === 'ligand') {
            this.flowForm = this.flowForm === f.key ? null : f.key;
            this._renderFlows(spec, true);
          } else {
            void this._runFlow(f.key, spec);
          }
        },
      }, el('strong', { text: f.title }), el('span', { text: needsSelection ? 'Select residues first' : f.desc })));
      if (f.key === 'binder' && this.flowForm === 'binder') children.push(this._binderForm(spec));
      if (f.key === 'scaffold' && this.flowForm === 'scaffold' && !needsSelection) children.push(this._scaffoldForm(spec));
      if (f.key === 'ligand' && this.flowForm === 'ligand') children.push(this._ligandForm(spec));
    }
    box.replaceChildren(...children);
  }

  /** Selection-dependent state of the scaffold button (enabled only with a selection). */
  _updateScaffoldButton() {
    const b = this.els.guided?.querySelector('[data-flow="scaffold"]');
    if (!b) return;
    const none = this.selectionCount === 0;
    b.disabled = none;
    b.title = none ? 'Select residues in the sequence bar or viewer first' : '';
    const sub = b.querySelector('span');
    if (sub) sub.textContent = none ? 'Select residues first' : FLOWS.find((f) => f.key === 'scaffold').desc;
  }

  _binderForm(spec) {
    const min = el('input', { type: 'number', min: 10, max: 500, value: 60, 'aria-label': 'Binder minimum length' });
    const max = el('input', { type: 'number', min: 10, max: 500, value: 100, 'aria-label': 'Binder maximum length' });
    const hot = el('input', { type: 'checkbox', checked: this.selectionCount > 0, disabled: this.selectionCount === 0 });
    const hotLabel = () => (this.selectionCount
      ? ` Use the ${this._selectedResidueCount()} selected residues as hotspots`
      : ' Hotspots: select 3–6 target residues at the interface first');
    const hotText = document.createTextNode(hotLabel());
    let wasNone = this.selectionCount === 0;
    this.formUpdate = () => {
      hotText.textContent = hotLabel();
      hot.disabled = this.selectionCount === 0;
      // Selecting the first hotspots ticks the box; after that it is the user's choice.
      if (wasNone && this.selectionCount > 0) hot.checked = true;
      wasNone = this.selectionCount === 0;
      this._updateScaffoldButton();
    };
    const apply = el('button', {
      type: 'button', class: 'primary compact', text: 'Set up binder',
      onclick: () => void this._runFlow('binder', spec, {
        min: Number(min.value), max: Number(max.value), hotspots: hot.checked,
      }),
    });
    return el('div', { class: 'rfd4p-flow-form' },
      el('p', { class: 'muted', text: 'Hotspots are recommended: without them the model sometimes returns a copy of the target (a homodimer). Click residues in the sequence bar to select them.' }),
      el('div', { class: 'rfd4p-flow-length' }, el('span', { text: 'Binder length' }), min, el('span', { text: '–' }), max),
      el('label', { class: 'checkbox' }, hot, hotText),
      apply);
  }

  _selectedResidueCount() {
    const a = this._inputDoc()?.atoms;
    const sel = window.atomworksStudio?.getSelection?.() ?? [];
    if (!a) return sel.length;
    return new Set(sel.map((i) => `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code?.[i] ?? ''}`)).size;
  }

  _scaffoldForm(spec) {
    const rebuild = el('input', { type: 'radio', name: 'rfd4p-scaffold-mode', value: 'rebuild', checked: true });
    const inPlace = el('input', { type: 'radio', name: 'rfd4p-scaffold-mode', value: 'in-place' });
    const spread = el('input', { type: 'number', class: 'rfd4p-pct', min: 0, max: 100, step: 5, value: 25, 'aria-label': 'Length variation (%)' });
    const fix = el('select', { 'aria-label': 'What to fix' },
      el('option', { value: 'whole', text: 'Whole residues (backbone + side chain)' }),
      el('option', { value: 'sidechain', text: 'Side chains only (e.g. catalytic groups)' }));
    const where = el('select', { 'aria-label': 'Motif position' },
      el('option', { value: 'indexed', text: 'Same order and spacing as now' }),
      el('option', { value: 'anywhere', text: 'Anywhere in the new chain (unindexed)' }));
    const chainLen = Math.max(...spec.chains.map((c) => c.residues), 1);
    const totMin = el('input', { type: 'number', min: 1, max: 1000, value: Math.round(chainLen * 0.75), 'aria-label': 'Total minimum length' });
    const totMax = el('input', { type: 'number', min: 1, max: 1000, value: Math.round(chainLen * 1.25), 'aria-label': 'Total maximum length' });
    const total = el('div', { class: 'rfd4p-flow-length', hidden: true }, el('span', { text: 'New chain length' }), totMin, el('span', { text: '–' }), totMax);
    const gapLine = el('label', { class: 'checkbox' }, rebuild, ' Rebuild the rest as new segments, each its original length ±', spread, '%');
    const inPlaceLine = el('label', { class: 'checkbox' }, inPlace, ' Keep the chain length; redesign the rest in place');
    where.addEventListener('change', () => {
      const anywhere = where.value === 'anywhere';
      total.hidden = !anywhere;
      gapLine.hidden = anywhere;
      inPlaceLine.hidden = anywhere;
    });
    const apply = el('button', {
      type: 'button', class: 'primary compact', text: 'Set up motif scaffolding',
      onclick: () => void this._runFlow('scaffold', spec, {
        rebuild: rebuild.checked || where.value === 'anywhere',
        spread: Math.max(0, Math.min(100, Number(spread.value) || 0)) / 100,
        sidechainsOnly: fix.value === 'sidechain',
        unindexed: where.value === 'anywhere',
        totalMin: Number(totMin.value),
        totalMax: Number(totMax.value),
      }),
    });
    const count = el('p', { class: 'muted' });
    const showCount = () => {
      const k = this._selectedResidueCount();
      count.textContent = k
        ? `Keeps the ${k} selected residue${k === 1 ? '' : 's'} fixed (sequence, and the structure chosen below).`
        : 'Select the motif residues in the sequence bar or viewer.';
    };
    showCount();
    this.formUpdate = () => {
      showCount();
      apply.disabled = this.selectionCount === 0;
      this._updateScaffoldButton();
    };
    return el('div', { class: 'rfd4p-flow-form' },
      count,
      el('label', {}, 'Fix ', fix),
      el('label', {}, 'Motif position ', where),
      total, gapLine, inPlaceLine, apply);
  }

  _ligandForm(spec) {
    const names = [...new Set(spec.chains.flatMap((c) => c.ligands.map((l) => l.name)))].join(', ');
    const remove = el('input', { type: 'checkbox', checked: spec.chains.some((c) => c.residues > 0) });
    const posed = el('select', { 'aria-label': 'Ligand pose' },
      el('option', { value: 'fixed', text: 'Keep its pose (fixed coordinates)' }),
      el('option', { value: 'predicted', text: 'Predict its pose (only its identity is given)' }));
    const min = el('input', { type: 'number', min: 20, max: 500, value: 100, 'aria-label': 'Protein minimum length' });
    const max = el('input', { type: 'number', min: 20, max: 500, value: 150, 'aria-label': 'Protein maximum length' });
    const apply = el('button', {
      type: 'button', class: 'primary compact', text: 'Set up',
      onclick: () => void this._runFlow('ligand', spec, {
        removeProtein: remove.checked, poseFixed: posed.value === 'fixed', min: Number(min.value), max: Number(max.value),
      }),
    });
    return el('div', { class: 'rfd4p-flow-form' },
      el('p', { class: 'muted', text: `Ligand${names.includes(',') ? 's' : ''}: ${names}.` }),
      el('label', {}, 'Ligand ', posed),
      el('div', { class: 'rfd4p-flow-length' }, el('span', { text: 'Protein length' }), min, el('span', { text: '–' }), max),
      el('label', { class: 'checkbox' }, remove, ' Remove the existing protein chains'),
      apply);
  }

  async _runFlow(key, spec, opts = {}) {
    const polymers = spec.chains.filter((c) => c.residues > 0);
    const ligandAtoms = spec.chains.flatMap((c) => c.ligandAtomIndices);
    const roleEdits = (c, role) => this._roleEdits(c, role);
    const selection = window.atomworksStudio?.getSelection?.() ?? [];
    // Each step must succeed before the next; a failure keeps the form open with its error shown.
    let ok = true;
    if (key === 'binder') {
      const edits = polymers.flatMap((c) => roleEdits(c, 'fixed'));
      if (ligandAtoms.length) edits.push({ condition: 'coordinate', atom_indices: ligandAtoms, enabled: true });
      const lo = Math.max(10, Math.round(opts.min) || 60);
      ok = await this._batch(edits)
        && await this._mutate('/segments', {
          minimum: lo, maximum: Math.max(lo, Math.round(opts.max) || 100), atom_index: null, before: false,
          new_chain: true, chain_id: null, chain_type: 'polypeptide(L)',
        })
        && (!opts.hotspots || !selection.length
          || await this._mutate('/conditions', { condition: 'hotspot', atom_indices: selection, enabled: true, scope: 'atom' }));
    } else if (key === 'ligand') {
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
    } else if (key === 'diversify' || key === 'scratch') {
      ok = await this._batch(polymers.flatMap((c) => roleEdits(c, 'free')));
    }
    if (ok) {
      this.app._setPartialDiffusion?.(key === 'diversify');
      this.showFlows = false;
      this.flowForm = null;
    }
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
    if (typeof window.atomworksStudio?.applyEdit !== 'function' || this.busy) return false;
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
      await window.atomworksStudio?.applyEdit(path, body);
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
    if (typeof window.atomworksStudio?.openExample !== 'function') return;
    const button = this.els.taskGrid?.querySelector(`[data-example="${task.example}"]`);
    button?.classList.add('loading');
    try {
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

  async _fetchPdb(raw) {
    const id = String(raw ?? '').trim().toUpperCase();
    const status = this.els.fetchStatus;
    if (!/^[0-9][A-Z0-9]{3}$/.test(id)) {
      status.textContent = 'Enter a 4-character PDB ID, e.g. 1QYS.';
      return;
    }
    status.textContent = `Downloading ${id} from RCSB…`;
    try {
      const res = await fetch(`https://files.rcsb.org/download/${id}.cif`);
      if (!res.ok) throw new Error(res.status === 404 ? `${id} was not found` : `RCSB returned ${res.status}`);
      const text = await res.text();
      const opened = await window.atomworksStudio?.openFile?.(new File([text], `${id}.cif`, { type: 'chemical/x-mmcif' }));
      status.textContent = opened === false ? `Opening ${id} was cancelled.` : '';
      if (opened !== false) this.showStart = false;
      this.showFlows = true;
    } catch (err) {
      status.textContent = `Could not fetch ${id}: ${err?.message ?? err}`;
    } finally {
      this.render();
    }
  }
}
