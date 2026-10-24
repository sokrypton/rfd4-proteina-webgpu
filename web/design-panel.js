/**
 * The RFD4-Proteina card's guided layer over AtomWorks Studio: a "what do you want to design?" start screen, a Design
 * panel that states what the open document will generate (task, chain roles, lengths, contig) and edits it, and the
 * simple/advanced switch that hides Studio's expert tools.
 *
 * Every edit goes through Studio's own mutate path (`window.__rfd4pMutate`), so Undo, the viewer and the
 * `atomworks-studio-message` "changed" event behave exactly as for edits made in the Studio panels.
 */
import { designSpec, ROLE_LABELS } from './design-spec.js';
import { getSharedStudioStore } from './studio-api.js';

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
      guided: $('rfd4p-guided'),
      chainList: $('rfd4p-chain-list'),
      addChain: $('rfd4p-add-chain'),
      undo: $('rfd4p-undo'),
      total: $('rfd4p-total'),
      contig: $('rfd4p-contig'),
      warnings: $('rfd4p-warnings'),
      runSettings: $('rfd4p-run-settings'),
      advanced: $('rfd4p-advanced'),
    };
    this._buildTaskGrid();
    this._bind();
    this._applyAdvanced(this._readAdvanced());
    this.render();
  }

  // ---- wiring -------------------------------------------------------------------------------------------------

  _bind() {
    const { els } = this;
    els.fetchForm?.addEventListener('submit', (ev) => {
      ev.preventDefault();
      void this._fetchPdb(els.pdbId.value);
    });
    els.openFile?.addEventListener('click', () => document.getElementById('file-input')?.click());
    els.changeTask?.addEventListener('click', () => {
      this.showFlows = !this.showFlows;
      this.flowForm = null;
      this.render();
    });
    els.addChain?.addEventListener('click', () => void this._mutate('/segments', {
      minimum: 60, maximum: 100, atom_index: null, before: false, new_chain: true, chain_id: null,
      chain_type: 'polypeptide(L)',
    }));
    els.undo?.addEventListener('click', () => document.getElementById('undo')?.click());
    els.advanced?.addEventListener('change', () => this._applyAdvanced(els.advanced.checked, true));
    window.addEventListener('atomworks-studio-message', () => queueMicrotask(() => this.render()));
    window.addEventListener('atomworks-studio-selection', (ev) => {
      this.selectionCount = ev.detail?.count ?? 0;
      if (this.els.guided && !this.els.guided.hidden) this._renderFlows(this._spec(), true);
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
    if (els.start) els.start.hidden = hasDoc;
    if (els.design) els.design.hidden = !hasDoc;
    if (els.runSettings) els.runSettings.hidden = !hasDoc;
    if (!hasDoc) return;

    const viewingDesign = this._viewingDesign();
    els.design.classList.toggle('rfd4p-readonly', viewingDesign);
    els.taskLabel.textContent = spec.task.label;
    els.taskDetail.textContent = viewingDesign
      ? 'Showing a generated design. Select Input under View to change the setup.'
      : spec.task.detail;
    els.changeTask.hidden = viewingDesign;
    els.addChain.hidden = viewingDesign;
    els.undo.hidden = viewingDesign;
    els.undo.disabled = document.getElementById('undo')?.disabled ?? true;

    els.chainList.replaceChildren(...spec.chains.map((c) => this._chainRow(c, viewingDesign)));
    const designed = range(spec.designedMin, spec.designedMax);
    els.total.textContent = spec.totalMax
      ? `${designed} residue${spec.designedMax === 1 ? '' : 's'} designed · ${range(spec.totalMin, spec.totalMax)} in total`
      : '';
    els.contig.textContent = spec.contig || '—';

    const warnings = [...spec.warnings];
    if (spec.task.key === 'partial' && spec.chains.some((c) => c.segments.length)) {
      warnings.push('Partial diffusion needs every residue defined; this input has new-residue segments.');
    }
    els.warnings.replaceChildren(...warnings.map((w) => el('p', { class: 'rfd4p-warning', text: w })));

    const needsChoice = ['redesign_all', 'nothing'].includes(spec.task.key);
    this._renderFlows(spec, !viewingDesign && (this.showFlows || needsChoice || this.flowForm !== null));

    // Nothing to generate: keep Generate off (a running job keeps its Stop button).
    if (!this.app.activeAbortController && this.app.els.predictBtn) {
      this.app.els.predictBtn.disabled = spec.task.key === 'nothing' || spec.task.key === 'empty';
    }
  }

  _chainRow(c, readonly) {
    const head = el('div', { class: 'rfd4p-chain-head' }, el('span', { class: 'rfd4p-chain-id', text: c.id }));
    const body = el('div', { class: 'rfd4p-chain-body' });
    const row = el('div', { class: `rfd4p-chain rfd4p-role-${c.role}` }, head, body);

    if (c.role === 'ligand') {
      for (const l of c.ligands) {
        const label = `${l.name} · ${l.pose === 'fixed' ? 'fixed pose' : l.pose === 'predicted' ? 'pose predicted' : 'pose partly fixed'}`;
        const toggle = el('button', {
          type: 'button',
          class: 'secondary compact',
          disabled: readonly,
          text: l.pose === 'predicted' ? 'Fix pose' : 'Free pose',
          onclick: () => void this._batch([{
            condition: 'coordinate', atom_indices: c.ligandAtomIndices, enabled: l.pose === 'predicted',
          }]),
        });
        body.append(el('div', { class: 'rfd4p-chain-line' }, el('span', { text: label }), toggle));
      }
      return row;
    }

    if (c.residues) {
      const span = c.firstResId != null ? ` (${c.firstResId}–${c.lastResId})` : '';
      const info = el('span', { class: 'rfd4p-chain-info', text: `${c.residues} residue${c.residues === 1 ? '' : 's'}${span}` });
      const select = el('select', { 'aria-label': `Chain ${c.id} role`, disabled: readonly });
      const options = [
        ['fixed', 'Fixed (kept as is)'],
        ['backbone', 'Keep backbone, new sequence'],
        ['free', 'Redesign'],
      ];
      if (!options.some(([k]) => k === c.role)) options.unshift([c.role, `${ROLE_LABELS[c.role]} (custom)`]);
      for (const [value, label] of options) {
        select.append(el('option', { value, text: label, selected: value === c.role }));
      }
      select.addEventListener('change', () => void this._setRole(c, select.value));
      body.append(el('div', { class: 'rfd4p-chain-line' }, info, select));
      const notes = [];
      if (c.role === 'motif') {
        notes.push(c.segments.length && !c.designed
          ? `${c.fixed} motif residue${c.fixed === 1 ? '' : 's'} kept; new residues around them`
          : `${c.fixed} fixed, ${c.designed} designed`);
      }
      if (c.role === 'sequence') notes.push('sequence kept, structure predicted');
      if (c.unindexed) notes.push(`${c.unindexed} motif residue${c.unindexed === 1 ? '' : 's'} placed anywhere`);
      if (c.hotspots) notes.push(`${c.hotspots} hotspot residue${c.hotspots === 1 ? '' : 's'}`);
      if (c.unresolved) notes.push(`${c.unresolved} unmodelled residue${c.unresolved === 1 ? '' : 's'} (built by the model)`);
      if (notes.length) body.append(el('p', { class: 'rfd4p-chain-note muted', text: notes.join(' · ') }));
    } else {
      body.append(el('div', { class: 'rfd4p-chain-line' }, el('span', { class: 'rfd4p-chain-info', text: 'New chain' })));
    }

    for (const seg of c.segments) {
      const min = el('input', { type: 'number', min: 1, max: 1000, value: seg.min, 'aria-label': 'Minimum length', disabled: readonly });
      const max = el('input', { type: 'number', min: 1, max: 1000, value: seg.max, 'aria-label': 'Maximum length', disabled: readonly });
      const commit = () => {
        const lo = Math.max(1, Math.round(Number(min.value)) || seg.min);
        const hi = Math.max(lo, Math.round(Number(max.value)) || seg.max);
        if (lo === seg.min && hi === seg.max) return;
        void this._mutate('/segments/range', { atom_index: seg.atomIndex, minimum: lo, maximum: hi });
      };
      min.addEventListener('change', commit);
      max.addEventListener('change', commit);
      body.append(el('div', { class: 'rfd4p-chain-line rfd4p-seg-line' },
        el('span', { class: 'rfd4p-seg-label', text: c.residues ? `New residues${seg.where ? `, ${seg.where}` : ''}` : 'Length' }),
        el('span', { class: 'rfd4p-seg-inputs' }, min, el('span', { text: '–' }), max)));
    }
    return row;
  }

  // ---- guided set-ups -----------------------------------------------------------------------------------------

  _renderFlows(spec, visible = !this.els.guided?.hidden) {
    const box = this.els.guided;
    if (!box) return;
    box.hidden = !visible || !spec;
    if (box.hidden) return;
    const children = [el('p', { class: 'rfd4p-lead', text: 'What should happen to this structure?' })];
    const hasPolymer = spec.chains.some((c) => c.residues > 0);
    for (const f of FLOWS) {
      const needsSelection = f.key === 'scaffold' && this.selectionCount === 0;
      const disabled = !hasPolymer || needsSelection;
      children.push(el('button', {
        type: 'button',
        class: `rfd4p-flow${this.flowForm === f.key ? ' active' : ''}`,
        disabled,
        title: needsSelection ? 'Select residues in the sequence bar or viewer first' : null,
        onclick: () => {
          if (f.key === 'binder' || f.key === 'scaffold') {
            this.flowForm = this.flowForm === f.key ? null : f.key;
            this._renderFlows(spec, true);
          } else {
            void this._runFlow(f.key, spec);
          }
        },
      }, el('strong', { text: f.title }), el('span', { text: needsSelection ? 'Select residues first' : f.desc })));
      if (f.key === 'binder' && this.flowForm === 'binder') children.push(this._binderForm(spec));
      if (f.key === 'scaffold' && this.flowForm === 'scaffold' && !needsSelection) children.push(this._scaffoldForm(spec));
    }
    box.replaceChildren(...children);
  }

  _binderForm(spec) {
    const min = el('input', { type: 'number', min: 10, max: 500, value: 60, 'aria-label': 'Binder minimum length' });
    const max = el('input', { type: 'number', min: 10, max: 500, value: 100, 'aria-label': 'Binder maximum length' });
    const hot = el('input', { type: 'checkbox', checked: this.selectionCount > 0, disabled: this.selectionCount === 0 });
    const hotText = this.selectionCount
      ? ` Use the ${this.selectionCount} selected atoms as hotspots`
      : ' Hotspots: select 3–6 target residues at the interface first';
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
    const sel = window.__rfd4pSelection?.() ?? [];
    if (!a) return sel.length;
    return new Set(sel.map((i) => `${a.chain_id[i]}:${a.res_id[i]}:${a.ins_code?.[i] ?? ''}`)).size;
  }

  _scaffoldForm(spec) {
    const rebuild = el('input', { type: 'radio', name: 'rfd4p-scaffold-mode', value: 'rebuild', checked: true });
    const inPlace = el('input', { type: 'radio', name: 'rfd4p-scaffold-mode', value: 'in-place' });
    const spread = el('input', { type: 'number', class: 'rfd4p-pct', min: 0, max: 100, step: 5, value: 25, 'aria-label': 'Length variation (%)' });
    const apply = el('button', {
      type: 'button', class: 'primary compact', text: 'Set up motif scaffolding',
      onclick: () => void this._runFlow('scaffold', spec, {
        rebuild: rebuild.checked, spread: Math.max(0, Math.min(100, Number(spread.value) || 0)) / 100,
      }),
    });
    return el('div', { class: 'rfd4p-flow-form' },
      el('p', { class: 'muted', text: `Keeps the ${this._selectedResidueCount()} selected residues fixed (structure and sequence).` }),
      el('label', { class: 'checkbox' }, rebuild, ' Rebuild the rest as new segments, each its original length ±', spread, '%'),
      el('label', { class: 'checkbox' }, inPlace, ' Keep the chain length; redesign the rest in place'),
      apply);
  }

  async _runFlow(key, spec, opts = {}) {
    const polymers = spec.chains.filter((c) => c.residues > 0);
    const ligandAtoms = spec.chains.flatMap((c) => c.ligandAtomIndices);
    const roleEdits = (c, role) => this._roleEdits(c, role);
    const selection = window.__rfd4pSelection?.() ?? [];
    this.app._setPartialDiffusion?.(key === 'diversify');
    try {
      if (key === 'binder') {
        const edits = polymers.flatMap((c) => roleEdits(c, 'fixed'));
        if (ligandAtoms.length) edits.push({ condition: 'coordinate', atom_indices: ligandAtoms, enabled: true });
        await this._batch(edits);
        const lo = Math.max(10, Math.round(opts.min) || 60);
        await this._mutate('/segments', {
          minimum: lo, maximum: Math.max(lo, Math.round(opts.max) || 100), atom_index: null, before: false,
          new_chain: true, chain_id: null, chain_type: 'polypeptide(L)',
        });
        if (opts.hotspots && selection.length) {
          await this._mutate('/conditions', { condition: 'hotspot', atom_indices: selection, enabled: true, scope: 'atom' });
        }
      } else if (key === 'scaffold' && opts.rebuild !== false) {
        if (!selection.length) return;
        await this._mutate('/segments/scaffold', { atom_indices: selection, spread: opts.spread ?? 0.25 });
      } else if (key === 'scaffold') {
        if (!selection.length) return;
        const doc = this._inputDoc();
        const chainsOfSelection = new Set(selection.map((i) => String(doc.atoms.chain_id[i])));
        const edits = polymers.filter((c) => chainsOfSelection.has(c.id)).flatMap((c) => roleEdits(c, 'free'));
        edits.push({ condition: 'coordinate', atom_indices: selection, enabled: true, scope: 'residue' });
        edits.push({ condition: 'sequence', atom_indices: selection, enabled: true, scope: 'residue' });
        await this._batch(edits);
      } else if (key === 'sequence') {
        await this._batch(polymers.flatMap((c) => roleEdits(c, 'backbone')));
      } else if (key === 'diversify' || key === 'scratch') {
        await this._batch(polymers.flatMap((c) => roleEdits(c, 'free')));
      }
      this.showFlows = false;
      this.flowForm = null;
    } finally {
      this.render();
    }
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
    return edits.length ? this._mutate('/conditions/batch', { edits }) : Promise.resolve();
  }

  async _mutate(path, body) {
    if (typeof window.__rfd4pMutate !== 'function' || this.busy) return;
    this.busy = true;
    this.els.design?.classList.add('rfd4p-busy');
    try {
      await window.__rfd4pMutate(path, body);
    } catch (err) {
      console.warn('Design edit failed:', err);
      this.els.warnings?.replaceChildren(el('p', { class: 'rfd4p-warning', text: `Edit failed: ${err?.message ?? err}` }));
    } finally {
      this.busy = false;
      this.els.design?.classList.remove('rfd4p-busy');
    }
  }

  // ---- start screen -------------------------------------------------------------------------------------------

  async _startTask(task) {
    if (typeof window.__rfd4pLoadExample !== 'function') return;
    const button = this.els.taskGrid?.querySelector(`[data-example="${task.example}"]`);
    button?.classList.add('loading');
    try {
      await window.__rfd4pLoadExample(task.example);
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
      const opened = await window.__rfd4pOpenFile?.(new File([text], `${id}.cif`, { type: 'chemical/x-mmcif' }));
      status.textContent = opened === false ? `Opening ${id} was cancelled.` : '';
      this.showFlows = true;
    } catch (err) {
      status.textContent = `Could not fetch ${id}: ${err?.message ?? err}`;
    } finally {
      this.render();
    }
  }
}
