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
      addChain: $('rfd4p-add-chain'),
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
    // "New…" shows the start screen over the open document; opening anything from it replaces the document.
    els.newDoc?.addEventListener('click', () => { this.showStart = true; this.render(); });
    els.back?.addEventListener('click', () => { this.showStart = false; this.render(); });
    document.getElementById('file-input')?.addEventListener('change', () => { this.showStart = false; });
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
    document.addEventListener('pointerdown', (ev) => {
      if (this.pop.contains(ev.target)) return;
      const src = ev.target.closest?.('#viewer, #sequence-view');
      this.lastPointer = src ? { x: ev.clientX, y: ev.clientY } : null;
      if (!src) this.pop.hidden = true;
    }, true);
    window.addEventListener('atomworks-studio-selection', (ev) => {
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
    els.addChain.hidden = viewingDesign;
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

    const needsChoice = ['redesign_all', 'nothing'].includes(spec.task.key);
    this._renderFlows(spec, !viewingDesign && (this.showFlows || needsChoice || this.flowForm !== null));

    // Nothing to generate: keep Generate off (a running job keeps its Stop button).
    if (!this.app.activeAbortController && this.app.els.predictBtn) {
      this.app.els.predictBtn.disabled = Boolean(this.app.noWebGpu) || spec.task.key === 'nothing' || spec.task.key === 'empty';
    }
  }

  _chainRow(c, readonly) {
    const head = el('div', { class: 'rfd4p-chain-head' }, el('span', { class: 'rfd4p-chain-id', text: c.id }),
      readonly ? null : el('button', {
        type: 'button', class: 'rfd4p-chain-remove', title: `Remove chain ${c.id}`, 'aria-label': `Remove chain ${c.id}`,
        text: '×', onclick: () => void this._mutate('/delete', { atom_indices: c.atomIndices }),
      }));
    const body = el('div', { class: 'rfd4p-chain-body' });
    const row = el('div', { class: `rfd4p-chain rfd4p-role-${c.role}` }, head, body);
    // The chain's colour, as in the viewer and the sequence bar.
    const color = window.__rfd4pChainColor?.(c.id);
    if (color) row.style.setProperty('--chain-color', color);

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
      const select = el('select', { class: 'rfd4p-role-select', 'aria-label': `Chain ${c.id} role`, disabled: readonly });
      const options = [
        ['fixed', 'Fixed (kept as is)'],
        ['backbone', 'Keep backbone, new sequence'],
        ['sequence', 'Keep sequence, predict structure'],
        ['free', 'Redesign'],
      ];
      if (!options.some(([k]) => k === c.role)) options.unshift([c.role, `${ROLE_LABELS[c.role]} (custom)`]);
      for (const [value, label] of options) {
        select.append(el('option', { value, text: label, selected: value === c.role }));
      }
      select.addEventListener('change', () => void this._setRole(c, select.value));
      info.title = info.textContent;
      body.append(el('div', { class: 'rfd4p-chain-line' }, info), select);
      const notes = [];
      if (c.role === 'motif') {
        const plural = (k, word) => `${k} ${word}${k === 1 ? '' : 's'}`;
        const seqOnly = c.seqFixed - c.fixed;
        if (c.fixed) notes.push(`${plural(c.fixed, 'residue')} fixed`);
        if (seqOnly > 0) notes.push(`${plural(seqOnly, 'residue')} sequence only`);
        if (c.designed) notes.push(`${plural(c.designed, 'residue')} redesigned`);
        if (c.segments.length) notes.push(plural(c.segments.length, 'new segment'));
      }
      if (c.unindexed) notes.push(`${c.unindexed === c.fixed ? 'motif' : c.unindexed} placed anywhere`);
      if (c.unresolved) notes.push(`${c.unresolved} unmodelled residue${c.unresolved === 1 ? '' : 's'} (built by the model)`);
      if (notes.length) body.append(el('p', { class: 'rfd4p-chain-note muted', text: notes.join(' · ') }));
      if (c.hotspots) {
        body.append(el('div', { class: 'rfd4p-chain-line rfd4p-chain-note' },
          el('span', { class: 'muted', text: `${c.hotspots} hotspot residue${c.hotspots === 1 ? '' : 's'}` }),
          readonly ? null : el('button', {
            type: 'button', class: 'secondary compact', text: 'Clear hotspots',
            onclick: () => void this._batch([{ condition: 'hotspot', atom_indices: c.atomIndices, enabled: false }]),
          })));
      }
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
        el('span', {
          class: 'rfd4p-seg-label',
          text: c.residues ? `New${seg.where ? ` · ${seg.where}` : ' residues'}` : 'Length',
          title: c.residues ? `New residues${seg.where ? `, ${seg.where}` : ''}` : 'Length of the new chain',
        }),
        el('span', { class: 'rfd4p-seg-inputs' }, min, el('span', { text: '–' }), max)));
    }
    return row;
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
    const sel = window.__rfd4pSelection?.() ?? [];
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
    const sel = window.__rfd4pSelection?.() ?? [];
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
    const selection = window.__rfd4pSelection?.() ?? [];
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
    if (typeof window.__rfd4pMutate !== 'function' || this.busy) return false;
    this.busy = true;
    this.els.design?.classList.add('rfd4p-busy');
    try {
      await window.__rfd4pMutate(path, body);
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
    if (typeof window.__rfd4pLoadExample !== 'function') return;
    const button = this.els.taskGrid?.querySelector(`[data-example="${task.example}"]`);
    button?.classList.add('loading');
    try {
      await window.__rfd4pLoadExample(task.example);
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
      const opened = await window.__rfd4pOpenFile?.(new File([text], `${id}.cif`, { type: 'chemical/x-mmcif' }));
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
