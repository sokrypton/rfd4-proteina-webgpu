/**
 * The RFD4-Proteina card's layer over AtomWorks Studio: a "what do you want to design?" start screen, a Design
 * panel that states what the open document will generate (task, chain roles, lengths, contig) and edits it, and the
 * simple/advanced switch that hides Studio's expert tools.
 *
 * Every edit goes through Studio's own mutate path (`window.atomworksStudio?.applyEdit`), so Undo, the viewer and the
 * `atomworks-studio` "changed" event behave exactly as for edits made in the Studio panels.
 */
import { designSpec, formatResidueList, parseResidueList } from './design-spec.js';
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
      total: $('rfd4p-total'),
      contig: $('rfd4p-contig'),
      warnings: $('rfd4p-warnings'),
      runSettings: $('rfd4p-run-settings'),
      advanced: $('rfd4p-advanced'),
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
    const original = chains.length === 1 ? ` chain ${chains[0].toUpperCase()}` : '';
    this._rememberSource(newChains, /^[0-9][A-Za-z0-9]{3}$/.test(name) ? `PDB ${name}${original}` : `${name}${original}`);
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
    const added = await this._combine(doc);
    this._rememberSource([...new Set(added.map((i) => String(this._inputDoc().atoms.chain_id[i])))], `${code} in the PDB's ligand dictionary (pose free to start)`);
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
    window.addEventListener('atomworks-studio', (ev) => {
      if (ev.detail?.type !== 'selection') return;
      this.selectionCount = ev.detail?.count ?? 0;
      queueMicrotask(() => this._renderPop());
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
    els.taskDetail.textContent = viewingDesign ? 'A generated design (read-only). Select Input under View to edit.' : '';
    els.taskDetail.hidden = !viewingDesign;
    if (els.newDoc) els.newDoc.hidden = viewingDesign;
    const actions = els.addSeq?.parentElement;
    if (actions) actions.hidden = viewingDesign;
    if (viewingDesign && this.addStructure) this.addStructure.form.hidden = true;
    els.undo.hidden = viewingDesign;
    els.undo.disabled = document.getElementById('undo')?.disabled ?? true;

    els.chainList.replaceChildren(...spec.chains.map((c) => this._chainRow(c, viewingDesign)));
    const designed = range(spec.designedMin, spec.designedMax);
    els.total.textContent = spec.totalMax ? `${range(spec.totalMin, spec.totalMax)} residues · ${designed} designed` : '';
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

  /**
   * A chain as its pieces, in order (the contig, spelled out). Each piece is one line: what it is (kept, redesigned,
   * new), its residues, its modifiers as chips (× removes one), and one Edit menu for everything else. A bar above
   * draws the pieces to scale.
   */
  _chainRow(c, readonly) {
    const unit = c.kind === 'dna' || c.kind === 'rna' ? 'nt' : 'aa';
    const length = c.lengthMin === c.lengthMax ? String(c.lengthMin) : `${c.lengthMin}–${c.lengthMax}`;
    const remove = readonly ? null : el('button', {
      type: 'button', class: 'rfd4p-chain-remove', title: `Remove chain ${c.id}`, 'aria-label': `Remove chain ${c.id}`,
      text: '×', onclick: () => void this._mutate('/delete', { atom_indices: c.atomIndices }),
    });
    const id = el('span', { class: 'rfd4p-chain-id', text: c.id });
    const items = [...c.pieces.map((p) => this._pieceRow(c, p, readonly)), ...c.ligands.map((l) => this._ligandRow(c, l, readonly))];
    let head;
    const body = el('div', { class: 'rfd4p-chain-body' });
    if (items.length === 1) {
      // One item: the chain is that line (its range, length or ligand says what the header would).
      const line = items[0].querySelector('.rfd4p-piece-line');
      head = el('div', { class: 'rfd4p-chain-head' }, id, line, remove);
      body.append(...[...items[0].children]);
    } else {
      head = el('div', { class: 'rfd4p-chain-head' }, id,
        el('span', { class: 'rfd4p-chain-summary muted', text: `${length} ${unit}` }), this._chainBar(c), remove);
      body.append(...items);
    }
    const row = el('div', { class: `rfd4p-chain rfd4p-role-${c.role}` }, head, body);
    // The chain's colour, as in the viewer and the sequence bar.
    const color = window.atomworksStudio?.chainColor?.(c.id);
    if (color) row.style.setProperty('--chain-color', color);
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

  /** A piece's name: its residues (where a set-up took them from, if renumbered), or "new" and where it goes. */
  _pieceText(c, p) {
    if (p.kind === 'new') {
      // Where it goes, by the piece before it as that piece's row names it ("after A38", not the renumbered "A1").
      const k = c.pieces.indexOf(p);
      const prev = c.pieces.slice(0, k).reverse().find((q) => q.kind !== 'new');
      const where = c.pieces.length === 1 ? '' : k === 0 ? 'N-term' : k === c.pieces.length - 1 ? 'C-term'
        : prev ? `after ${prev.source ?? `${c.id}${prev.end}`}` : '';
      return `new${where ? ` · ${where}` : ''}`;
    }
    // A count only where numbering skips (it would repeat the range otherwise).
    const n = p.count !== p.end - p.start + 1 ? ` (${p.count})` : '';
    return p.source ?? `${this._range(c, p)}${n}`;
  }

  _pieceKey(c, p) {
    return `${c.id}:${p.kind === 'new' ? `n${p.atomIndex}` : p.start}`;
  }

  /** Pieces and ligands in a chain: with one, the chain's × is that item's delete. */
  _chainItems(c) {
    return c.pieces.length + c.ligands.length;
  }

  /** What a piece row says after its residues: what is kept, in a few words. */
  _status(p) {
    if (p.kind === 'new') return `${p.min === p.max ? p.min : `${p.min}–${p.max}`} long`;
    const kept = { all: 'kept', backbone: 'backbone kept', sequence: 'sequence kept', sidechain: 'side chains kept',
      shape: 'kept as a rigid unit' }[p.keep] ?? 'redesigned';
    return [kept, p.anywhere ? 'anywhere' : '', p.hotspots.length ? `${p.hotspots.length} hotspot${p.hotspots.length === 1 ? '' : 's'}` : '']
      .filter(Boolean).join(' · ');
  }

  /** A row that opens its popup: name, status, ⋯. */
  _row(name, status, kindClass, readonly, onOpen) {
    const line = el('button', {
      type: 'button', class: `rfd4p-piece-line rfd4p-row-button ${kindClass}`, disabled: readonly,
      'aria-haspopup': 'dialog', title: readonly ? null : 'Change',
      onclick: (ev) => onOpen(ev.currentTarget),
    },
    el('span', { class: 'rfd4p-piece-label', text: name }),
    el('span', { class: 'rfd4p-piece-status', text: status }),
    readonly ? null : el('span', { class: 'rfd4p-row-more', 'aria-hidden': 'true', text: '⋯' }));
    return el('div', { class: 'rfd4p-piece' }, line);
  }

  _pieceRow(c, p, readonly) {
    const kindClass = p.kind === 'kept' ? 'rfd4p-piece-kept' : 'rfd4p-piece-designed';
    return this._row(this._pieceText(c, p), this._status(p), kindClass, readonly,
      (anchor) => this._openPop(anchor, () => (p.kind === 'new' ? this._newPop(c, p) : this._piecePop(c, p))));
  }

  _ligandRow(c, l, readonly) {
    const status = l.pose === 'fixed' ? 'pose kept' : l.pose === 'predicted' ? 'placed by the model'
      : `${l.fixedAtoms} of ${l.atoms} atoms kept`;
    return this._row(l.name, status, 'rfd4p-piece-ligand', readonly, (anchor) => this._openPop(anchor, () => this._ligandPop(c, l)));
  }

  /** Where a chain's coordinates come from, as the page knows it. */
  _sourceOf(c) {
    const doc = this._inputDoc();
    const added = this.sources?.get(doc)?.get(c.id);
    if (added) return added;
    if (c.role === 'new') return null;
    const name = String(doc?.name ?? '').replace(/\.(cif|mmcif|pdb|zst)+$/i, '');
    const task = TASKS.find((t) => t.example === name);
    if (task) return `the ${task.title} example`;
    return /^[0-9][A-Za-z0-9]{3}$/.test(name) ? `PDB ${name.toUpperCase()}` : name || null;
  }

  _rememberSource(chains, label) {
    const doc = this._inputDoc();
    if (!doc) return;
    this.sources ??= new WeakMap();
    if (!this.sources.has(doc)) this.sources.set(doc, new Map());
    for (const id of chains) this.sources.get(doc).set(id, label);
  }

  /** The one popup, next to the row that opened it. `build` makes its content; closed by ×, Cancel, Escape or outside. */
  _openPop(anchor, build) {
    if (!this.piecePop) {
      this.piecePop = el('div', { class: 'rfd4p-piece-pop', role: 'dialog', hidden: true });
      document.body.append(this.piecePop);
      document.addEventListener('pointerdown', (ev) => {
        if (!this.piecePop.hidden && !this.piecePop.contains(ev.target) && !ev.target.closest?.('.rfd4p-row-button')) this._closePop();
      }, true);
      document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') this._closePop(); });
    }
    const pop = this.piecePop;
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

  _popHead(title, source) {
    return el('div', { class: 'rfd4p-pop-title' },
      el('div', {}, el('strong', { text: title }), source ? el('div', { class: 'muted', text: `from ${source}` }) : null),
      el('button', { type: 'button', class: 'rfd4p-pop-close', 'aria-label': 'Close', text: '×', onclick: () => this._closePop() }));
  }

  _popActions(apply, extra = []) {
    return el('div', { class: 'rfd4p-pop-actions' },
      ...extra,
      el('span', { class: 'rfd4p-pop-spacer' }),
      el('button', { type: 'button', class: 'secondary compact', text: 'Cancel', onclick: () => this._closePop() }),
      el('button', { type: 'button', class: 'primary compact', text: 'Apply', onclick: apply }));
  }

  /** Existing residues: which of them, what to keep of them, and what else is true of them. */
  _piecePop(c, p) {
    const a = this._inputDoc()?.atoms;
    const num = (value, label) => el('input', { type: 'number', min: p.start, max: p.end, value, 'aria-label': label });
    const from = num(p.start, 'First residue');
    const to = num(p.end, 'Last residue');
    const current = p.kind === 'kept' ? p.keep : 'designed';
    const choices = [
      ['all', 'Everything', 'Structure and sequence stay as they are'],
      ['backbone', 'Backbone (new sequence)', 'Same shape; the model writes a new sequence'],
      ['sequence', 'Sequence (new shape)', 'Same amino acids; the model builds the shape'],
      ['sidechain', 'Side chains', 'Side-chain atoms held (e.g. catalytic groups); backbone free'],
      ['shape', 'Rigid unit', 'Held rigid but free to move as a whole (a framework)'],
      ['designed', 'Nothing (redesign)', 'The model makes both structure and sequence'],
      ['replace', 'Replace with new residues', 'Removed; new residues of the length below take their place'],
    ];
    const group = `rfd4p-keep-${Date.now()}`;
    const radios = choices.map(([value, text, hint]) => {
      const input = el('input', { type: 'radio', name: group, value, checked: value === current });
      return el('label', { class: 'rfd4p-pop-choice', title: hint }, input, el('span', { text }));
    });
    const lenMin = el('input', { type: 'number', min: 1, max: 1000, value: Math.max(1, Math.round(p.count * 0.75)), 'aria-label': 'Fewest new residues' });
    const lenMax = el('input', { type: 'number', min: 1, max: 1000, value: Math.max(1, Math.round(p.count * 1.25)), 'aria-label': 'Most new residues' });
    radios[radios.length - 1].append(el('span', { class: 'rfd4p-pop-inline' }, lenMin, el('span', { text: '–' }), lenMax));
    const keepOnly = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      placeholder: `e.g. ${c.id}${p.start}, ${c.id}${Math.min(p.end, p.start + 10)}`, 'aria-label': 'Keep only these residues' });

    const hot = el('input', { type: 'text', class: 'rfd4p-pop-text', spellcheck: 'false', autocomplete: 'off',
      value: formatResidueList(c.id, p.hotspots), placeholder: `none, or e.g. ${c.id}${p.start}, ${c.id}${Math.min(p.end, p.start + 4)}`,
      'aria-label': 'Hotspots' });
    const anywhere = el('input', { type: 'checkbox', checked: Boolean(p.anywhere) });
    const note = el('p', { class: 'rfd4p-pop-note', 'aria-live': 'polite' });

    const apply = async () => {
      const lo = Math.max(p.start, Math.min(p.end, Math.round(Number(from.value)) || p.start));
      const hi = Math.max(lo, Math.min(p.end, Math.round(Number(to.value)) || p.end));
      const atoms = a ? p.atomIndices.filter((i) => Number(a.res_id[i]) >= lo && Number(a.res_id[i]) <= hi) : p.atomIndices;
      const part = { ...p, keep: p.kind === 'kept' ? p.keep : null, atomIndices: atoms, count: hi - lo + 1 };
      const { residues, bad } = parseResidueList(hot.value, c.id);
      const wanted = new Set(residues.filter((r) => r.chain === c.id).map((r) => r.resId));
      const outside = [...wanted].filter((id) => id < p.start || id > p.end);
      if (bad.length || outside.length) {
        note.textContent = bad.length ? `Hotspots not understood: ${bad.join(' ')}` : `Hotspots outside ${this._range(c, p)}: ${formatResidueList(c.id, outside)}`;
        return;
      }
      const mode = radios.map((r) => r.querySelector('input')).find((r) => r.checked)?.value ?? current;
      // "Keep only": those residues kept, the rest of the chain rebuilt as new residues (the motif set-up).
      if (keepOnly.value.trim()) {
        const only = parseResidueList(keepOnly.value, c.id);
        const want = new Set(only.residues.filter((r) => r.chain === c.id).map((r) => r.resId));
        const onlyAtoms = a ? c.polymerAtomIndices.filter((i) => want.has(Number(a.res_id[i]))) : [];
        const found = new Set(onlyAtoms.map((i) => Number(a.res_id[i])));
        const missing = [...want].filter((id) => !found.has(id));
        if (only.bad.length || missing.length || !onlyAtoms.length) {
          note.textContent = only.bad.length ? `Not understood: ${only.bad.join(' ')}` : `Not in chain ${c.id}: ${formatResidueList(c.id, missing)}`;
          return;
        }
        this._closePop();
        await this._runFlow('scaffold', this._spec(), { atoms: onlyAtoms });
        return;
      }
      this._closePop();
      // Hotspots and "anywhere" first: they refer to residues a replacement would remove.
      const hotAtoms = a ? p.atomIndices.filter((i) => wanted.has(Number(a.res_id[i]))) : [];
      const sameHot = formatResidueList(c.id, [...wanted]) === formatResidueList(c.id, p.hotspots);
      if (!sameHot) {
        const edits = [{ condition: 'hotspot', atom_indices: p.atomIndices, enabled: false }];
        if (hotAtoms.length) edits.push({ condition: 'hotspot', atom_indices: hotAtoms, enabled: true });
        if (!await this._batch(edits)) return;
      }
      if (anywhere.checked !== Boolean(p.anywhere) && !await this._batch([{ condition: 'index', atom_indices: atoms, enabled: anywhere.checked }])) return;
      const whole = lo === p.start && hi === p.end;
      if (mode === 'replace') {
        const min = Math.max(1, Math.round(Number(lenMin.value)) || 1);
        await this._mutate('/segments/replace', { atom_indices: atoms, minimum: min, maximum: Math.max(min, Math.round(Number(lenMax.value)) || min) });
      } else if (mode !== current || !whole) {
        await this._setKeep(part, mode);
      }
    };

    const extra = [
      el('button', { type: 'button', class: 'rfd4p-link', text: '+ before', title: 'Add new residues before these',
        onclick: () => { this._closePop(); void this._mutate('/segments', { minimum: 5, maximum: 10, atom_index: p.atomIndices[0], before: true, new_chain: false, chain_id: null, chain_type: 'polypeptide(L)' }); } }),
      el('button', { type: 'button', class: 'rfd4p-link', text: '+ after', title: 'Add new residues after these',
        onclick: () => { this._closePop(); void this._mutate('/segments', { minimum: 5, maximum: 10, atom_index: p.atomIndices[p.atomIndices.length - 1], before: false, new_chain: false, chain_id: null, chain_type: 'polypeptide(L)' }); } }),
      this._chainItems(c) > 1 ? el('button', { type: 'button', class: 'rfd4p-link rfd4p-link-danger', text: 'Delete',
        onclick: () => { this._closePop(); void this._mutate('/delete', { atom_indices: p.atomIndices }); } }) : null,
    ].filter(Boolean);

    return [
      this._popHead(`Chain ${c.id}, ${p.source ?? this._range(c, p)}`, this._sourceOf(c)),
      p.count > 1 ? el('div', { class: 'rfd4p-pop-row', title: 'Narrow the range to change only some of these residues' },
        el('span', { class: 'rfd4p-pop-label', text: 'Residues' }),
        el('span', { class: 'rfd4p-pop-inline' }, el('span', { text: c.id }), from, el('span', { text: '–' }), to)) : null,
      el('div', { class: 'rfd4p-pop-section' }, el('span', { class: 'rfd4p-pop-label', text: 'Keep' }), ...radios),
      el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Hotspots' }), hot),
      el('label', { class: 'rfd4p-pop-row', title: 'The model chooses where in the chain these residues go' },
        el('span', { class: 'rfd4p-pop-label' }), anywhere, el('span', { text: 'Anywhere in the chain' })),
      p.count > 1 ? el('div', { class: 'rfd4p-pop-row', title: 'Keep just these residues; the rest of the chain becomes new residues (a motif)' },
        el('span', { class: 'rfd4p-pop-label', text: 'Keep only' }), keepOnly) : null,
      note,
      this._popActions(() => void apply(), extra),
    ].filter(Boolean);
  }

  /** New residues: how many, or remove them. */
  _newPop(c, p) {
    const min = el('input', { type: 'number', min: 1, max: 1000, value: p.min, 'aria-label': 'Fewest new residues' });
    const max = el('input', { type: 'number', min: 1, max: 1000, value: p.max, 'aria-label': 'Most new residues' });
    const apply = () => {
      const lo = Math.max(1, Math.round(Number(min.value)) || p.min);
      const hi = Math.max(lo, Math.round(Number(max.value)) || p.max);
      this._closePop();
      if (lo !== p.min || hi !== p.max) void this._mutate('/segments/range', { atom_index: p.atomIndex, minimum: lo, maximum: hi });
    };
    const extra = this._chainItems(c) > 1 ? [el('button', { type: 'button', class: 'rfd4p-link rfd4p-link-danger', text: 'Delete',
      onclick: () => { this._closePop(); void this._mutate('/delete', { atom_indices: p.atomIndices }); } })] : [];
    return [
      this._popHead(`Chain ${c.id}, new residues`, null),
      el('div', { class: 'rfd4p-pop-row', title: 'Each design picks one length from this range' },
        el('span', { class: 'rfd4p-pop-label', text: 'Length' }), el('span', { class: 'rfd4p-pop-inline' }, min, el('span', { text: '–' }), max)),
      this._popActions(apply, extra),
    ];
  }

  /** A ligand: where it goes, or another molecule in its place. */
  _ligandPop(c, l) {
    const atoms = c.ligandAtomIndices;
    const group = `rfd4p-pose-${Date.now()}`;
    const keepPose = el('input', { type: 'radio', name: group, checked: l.pose !== 'predicted' });
    const free = el('input', { type: 'radio', name: group, checked: l.pose === 'predicted' });
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
      const fix = keepPose.checked;
      if (fix !== (l.pose === 'fixed') && !(fix && l.pose === 'partly fixed' && false)) {
        void this._batch([{ condition: 'coordinate', atom_indices: atoms, enabled: fix }]);
      }
    };
    const extra = this._chainItems(c) > 1 ? [el('button', { type: 'button', class: 'rfd4p-link rfd4p-link-danger', text: 'Delete',
      onclick: () => { this._closePop(); void this._mutate('/delete', { atom_indices: atoms }); } })] : [];
    return [
      this._popHead(`Ligand ${l.name}`, this._sourceOf(c)),
      el('div', { class: 'rfd4p-pop-section' }, el('span', { class: 'rfd4p-pop-label', text: 'Pose' }),
        el('label', { class: 'rfd4p-pop-choice' }, keepPose, el('span', {}, el('strong', { text: 'Keep it where it is' }))),
        el('label', { class: 'rfd4p-pop-choice' }, free, el('span', {}, el('strong', { text: 'Let the model place it' })))),
      l.pose === 'partly fixed' ? el('p', { class: 'muted', title: 'Set atom by atom in the Advanced editor', text: `${l.fixedAtoms} of ${l.atoms} atoms kept now` }) : null,
      el('div', { class: 'rfd4p-pop-row' }, el('span', { class: 'rfd4p-pop-label', text: 'Swap' }), code),
      note,
      this._popActions(() => void apply(), extra),
    ].filter(Boolean);
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
        btn('Scaffold around these', act(() => this._runFlow('scaffold', this._spec()))),
        btn('Delete', act(() => this._mutate('/delete', { atom_indices: residues })), 'danger compact')));
    }
    if (ligandAtoms.length) {
      const fixed = ligandAtoms.some(hasCrd);
      rows.push(el('div', { class: 'rfd4p-pop-row' },
        btn(fixed ? 'Free pose' : 'Fix pose', act(() => this._batch([{ condition: 'coordinate', atom_indices: ligandAtoms, enabled: !fixed }]))),
        btn('Design a protein around it', act(() => this._runFlow('ligand', this._spec()))),
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
    } catch (err) {
      status.textContent = `Could not fetch ${id}: ${err?.message ?? err}`;
    } finally {
      this.render();
    }
  }
}
