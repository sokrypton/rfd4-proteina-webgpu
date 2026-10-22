/**
 * Interactive 3D Molecular & Flow-Matching Trajectory Viewer for RFD4-Proteina.
 * Supports:
 *  - Smooth 3D trackball rotation, pan, zoom, and auto-rotate
 *  - All-atom (`atom37`) + backbone spline/tube rendering with depth-sorted shading
 *  - Conditioning overlays:
 *      * Fixed coordinates (`C_CRD`) vs Unindexed Internal-Coord Motif (`C_CRDGRP > 0`)
 *      * Hotspot halos (`C_HOT`)
 *      * Metal coordination / H-bond / Salt-bridge interaction vectors (`C_MCOORD`, `C_HB`, `C_SALT`)
 *      * Secondary structure (`C_SEC`) and Chain ID coloring
 *  - Hover picking with residue/token/condition callouts
 */

import { UNIFIED_ATOM37_TOKENS } from '../shared/rfd4p/encoding.js';

const CHAIN_PALETTE = [
  '#38bdf8', // sky
  '#f472b6', // pink
  '#a78bfa', // violet
  '#34d399', // emerald
  '#fbbf24', // amber
  '#fb7185', // rose
  '#60a5fa', // blue
  '#c084fc', // purple
];

const ROLE_COLORS = {
  fixed: '#22d3ee',      // cyan: fixed C_CRD motif/target
  intcrd: '#f59e0b',     // amber: unindexed internal-coord motif (C_CRDGRP > 0)
  hotspot: '#f43f5e',    // rose/red: C_HOT hotspot
  denovo: '#818cf8',     // indigo: generated de novo backbone/latents
  ligand: '#10b981',     // emerald: atomized ligand / metal ion
  na: '#e879f9',         // fuchsia: RNA / DNA
};

const SEC_COLORS = {
  0: '#94a3b8', // Unspecified
  1: '#f43f5e', // Helix
  2: '#38bdf8', // Sheet
  3: '#a3e635', // Coil/Loop
};

const ELEMENT_COLORS = {
  C: '#cbd5e1',
  N: '#3b82f6',
  O: '#ef4444',
  S: '#eab308',
  P: '#f97316',
  ZN: '#10b981',
  MG: '#22c55e',
  FE: '#d97706',
};

// Intra-residue covalent bond pairs among canonical atom37 slots for clean ball-and-stick rendering
const INTRA_RESIDUE_BOND_MAX_DIST = 1.95;

export class Rfd4pViewer3D {
  constructor(canvas, tooltipEl = null) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.tooltipEl = tooltipEl;

    this.colorMode = 'role'; // 'role' | 'chain' | 'sec' | 'rainbow'
    this.reprMode = 'hybrid'; // 'hybrid' | 'backbone' | 'allatom'
    this.showInteractions = true;
    this.autoRotate = false;

    // Camera state (3x3 rotation matrix + zoom + pan)
    this.rot = [
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ];
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;

    // Scene data
    this.scene = null;
    this.projectedAtoms = [];
    this.hoveredAtom = null;

    this._dragging = false;
    this._dragMode = 'rotate';
    this._lastX = 0;
    this._lastY = 0;
    this._rafId = null;

    this._bindEvents();
    this.rotateView(-0.35, 0.45);
  }

  _bindEvents() {
    this.canvas.addEventListener('mousedown', (e) => {
      this._dragging = true;
      this._dragMode = e.shiftKey || e.button === 2 ? 'pan' : 'rotate';
      this._lastX = e.clientX;
      this._lastY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      this._dragging = false;
    });

    window.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      if (this._dragging) {
        const dx = e.clientX - this._lastX;
        const dy = e.clientY - this._lastY;
        this._lastX = e.clientX;
        this._lastY = e.clientY;

        if (this._dragMode === 'pan') {
          this.panX += dx;
          this.panY += dy;
        } else {
          this.rotateView(dy * 0.008, dx * 0.008);
        }
        this.render();
        return;
      }

      if (mx >= 0 && my >= 0 && mx <= rect.width && my <= rect.height) {
        this._updateHover(mx, my);
      } else if (this.hoveredAtom) {
        this.hoveredAtom = null;
        if (this.tooltipEl) this.tooltipEl.hidden = true;
        this.render();
      }
    });

    this.canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const factor = Math.exp(-e.deltaY * 0.0012);
        this.zoom = Math.max(0.25, Math.min(5.0, this.zoom * factor));
        this.render();
      },
      { passive: false }
    );

    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    window.addEventListener('resize', () => this.render());
  }

  rotateView(rx, ry) {
    const cx = Math.cos(rx);
    const sx = Math.sin(rx);
    const cy = Math.cos(ry);
    const sy = Math.sin(ry);

    // Multiply R_new = R_x * R_y * R_old
    const r00 = cy;
    const r01 = 0;
    const r02 = sy;
    const r10 = sx * sy;
    const r11 = cx;
    const r12 = -sx * cy;
    const r20 = -cx * sy;
    const r21 = sx;
    const r22 = cx * cy;

    const m = this.rot;
    const n00 = r00 * m[0] + r01 * m[3] + r02 * m[6];
    const n01 = r00 * m[1] + r01 * m[4] + r02 * m[7];
    const n02 = r00 * m[2] + r01 * m[5] + r02 * m[8];

    const n10 = r10 * m[0] + r11 * m[3] + r12 * m[6];
    const n11 = r10 * m[1] + r11 * m[4] + r12 * m[7];
    const n12 = r10 * m[2] + r11 * m[5] + r12 * m[8];

    const n20 = r20 * m[0] + r21 * m[3] + r22 * m[6];
    const n21 = r20 * m[1] + r21 * m[4] + r22 * m[7];
    const n22 = r20 * m[2] + r21 * m[5] + r22 * m[8];

    this.rot = [n00, n01, n02, n10, n11, n12, n20, n21, n22];
  }

  resetCamera() {
    this.rot = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    this.rotateView(-0.35, 0.45);
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.render();
  }

  setAutoRotate(enabled) {
    this.autoRotate = Boolean(enabled);
    if (this.autoRotate && !this._rafId) {
      const tick = () => {
        if (!this.autoRotate) {
          this._rafId = null;
          return;
        }
        if (!this._dragging) {
          this.rotateView(0.0, 0.012);
          this.render();
        }
        this._rafId = requestAnimationFrame(tick);
      };
      this._rafId = requestAnimationFrame(tick);
    }
  }

  /**
   * Load either a decoded all-atom result (`decoded` + `batch`) or an intermediate
   * flow-matching CA trajectory frame (`caCoordsAngstrom` + `batch`).
   */
  setStructure({ batch, decoded = null, caCoordsAngstrom = null, stepInfo = null }) {
    if (!batch) return;
    const n = batch.n;
    const atoms = [];
    const bonds = [];
    const caTrace = [];

    // Compute token roles & metadata
    const chainMap = new Map();
    const tokenInfos = [];

    for (let i = 0; i < n; i += 1) {
      const tok = batch.tokens[i];
      if (!chainMap.has(tok.chainId)) {
        chainMap.set(tok.chainId, chainMap.size);
      }
      const chainIdx = chainMap.get(tok.chainId);

      let hasFixedCrd = false;
      let crdGrp = -1;
      let isHotspot = false;
      let isMcoord = false;
      let isHbond = false;
      let isSalt = false;

      for (let a = 0; a < 37; a += 1) {
        const slot = i * 37 + a;
        if (batch.cond_CRDGRP[slot] === 0 && Number.isFinite(batch.cond_CRD[slot * 3])) {
          hasFixedCrd = true;
        }
        if (batch.cond_CRDGRP[slot] > 0) {
          crdGrp = batch.cond_CRDGRP[slot];
        }
        if (batch.cond_HOT[slot]) isHotspot = true;
        if (batch.cond_MCOORD[slot]) isMcoord = true;
        if (batch.cond_HB[slot]) isHbond = true;
        if (batch.cond_SALT[slot]) isSalt = true;
      }

      const resIdx = decoded ? decoded.aatype_max[i] : batch.cond_SEQ[i];
      const resName = resIdx > 0 ? UNIFIED_ATOM37_TOKENS[resIdx] || tok.resName : tok.resName;
      const isAtomized = batch.token_type[i] === 5;
      const isNucleic = batch.token_type[i] === 2 || batch.token_type[i] === 3;

      let role = 'denovo';
      if (isAtomized) role = 'ligand';
      else if (isHotspot) role = 'hotspot';
      else if (hasFixedCrd) role = 'fixed';
      else if (crdGrp > 0) role = 'intcrd';
      else if (isNucleic) role = 'na';

      tokenInfos.push({
        tokenIdx: i,
        chainId: tok.chainId,
        chainIdx,
        resId: tok.resId,
        resName,
        role,
        crdGrp,
        hasFixedCrd,
        isHotspot,
        isMcoord,
        isHbond,
        isSalt,
        sec: batch.cond_SEC[i],
        isAtomized,
      });
    }

    if (decoded && decoded.coors_angstrom && decoded.atom_mask) {
      const coords = decoded.coors_angstrom;
      const mask = decoded.atom_mask;

      for (let i = 0; i < n; i += 1) {
        const info = tokenInfos[i];
        const resAtomIndices = [];

        for (let a = 0; a < 37; a += 1) {
          const slot = i * 37 + a;
          if (!mask[slot]) continue;
          const x = coords[slot * 3 + 0];
          const y = coords[slot * 3 + 1];
          const z = coords[slot * 3 + 2];
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

          const isBackbone = a <= 3;
          const isCa = a === 1;
          const elem = info.isAtomized
            ? (info.resName.replace(/[^A-Za-z]/g, '').toUpperCase().slice(0, 2) || 'ZN')
            : a === 0 ? 'N' : a === 1 || a === 2 || a === 4 ? 'C' : a === 3 ? 'O' : 'C';

          const atomIdx = atoms.length;
          atoms.push({
            atomIdx,
            tokenIdx: i,
            atomSlot: a,
            x,
            y,
            z,
            elem,
            isCa,
            isBackbone,
            info,
          });
          resAtomIndices.push(atomIdx);

          if (isCa) {
            caTrace.push({
              atomIdx,
              tokenIdx: i,
              x,
              y,
              z,
              info,
            });
          }
        }

        // Intra-residue covalent bonds
        for (let u = 0; u < resAtomIndices.length; u += 1) {
          const a1 = atoms[resAtomIndices[u]];
          for (let v = u + 1; v < resAtomIndices.length; v += 1) {
            const a2 = atoms[resAtomIndices[v]];
            const d = Math.hypot(a1.x - a2.x, a1.y - a2.y, a1.z - a2.z);
            if (d > 0.4 && d <= INTRA_RESIDUE_BOND_MAX_DIST) {
              bonds.push({
                a1: a1.atomIdx,
                a2: a2.atomIdx,
                kind: a1.isBackbone && a2.isBackbone ? 'backbone' : 'sidechain',
              });
            }
          }
        }
      }
    } else if (caCoordsAngstrom) {
      for (let i = 0; i < n; i += 1) {
        const info = tokenInfos[i];
        const x = caCoordsAngstrom[i * 3 + 0];
        const y = caCoordsAngstrom[i * 3 + 1];
        const z = caCoordsAngstrom[i * 3 + 2];
        const atomIdx = atoms.length;
        atoms.push({
          atomIdx,
          tokenIdx: i,
          atomSlot: 1,
          x,
          y,
          z,
          elem: info.isAtomized ? 'ZN' : 'C',
          isCa: true,
          isBackbone: true,
          info,
        });
        caTrace.push({ atomIdx, tokenIdx: i, x, y, z, info });
      }
    }

    // Center & radius of gyration
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (const a of atoms) {
      cx += a.x;
      cy += a.y;
      cz += a.z;
    }
    if (atoms.length > 0) {
      cx /= atoms.length;
      cy /= atoms.length;
      cz /= atoms.length;
    }
    let maxR = 8.0;
    for (const a of atoms) {
      const r = Math.hypot(a.x - cx, a.y - cy, a.z - cz);
      if (r > maxR) maxR = r;
    }

    // Build interaction dashed links (e.g. metal coordination to MCOORD residues)
    const interactions = [];
    const ligandNodes = caTrace.filter((c) => c.info.isAtomized);
    const mcoordNodes = caTrace.filter((c) => c.info.isMcoord && !c.info.isAtomized);
    for (const lig of ligandNodes) {
      for (const mc of mcoordNodes) {
        interactions.push({
          from: lig,
          to: mc,
          color: '#10b981',
          label: 'Metal Coordination (C_MCOORD)',
        });
      }
    }

    this.scene = {
      n,
      atoms,
      bonds,
      caTrace,
      interactions,
      center: [cx, cy, cz],
      maxRadius: maxR,
      stepInfo,
    };
    this.render();
  }

  _colorForToken(info) {
    if (this.colorMode === 'chain') {
      return CHAIN_PALETTE[info.chainIdx % CHAIN_PALETTE.length];
    }
    if (this.colorMode === 'sec') {
      return SEC_COLORS[info.sec] || SEC_COLORS[0];
    }
    if (this.colorMode === 'rainbow') {
      const total = Math.max(1, (this.scene?.n ?? 1) - 1);
      const hue = Math.round((info.tokenIdx / total) * 260);
      return `hsl(${hue}, 82%, 62%)`;
    }
    return ROLE_COLORS[info.role] || ROLE_COLORS.denovo;
  }

  _project(x, y, z, width, height, scale) {
    const [cx, cy, cz] = this.scene.center;
    const dx = x - cx;
    const dy = y - cy;
    const dz = z - cz;
    const m = this.rot;
    const rx = m[0] * dx + m[1] * dy + m[2] * dz;
    const ry = m[3] * dx + m[4] * dy + m[5] * dz;
    const rz = m[6] * dx + m[7] * dy + m[8] * dz;

    const fov = 4.2 * this.scene.maxRadius;
    const persp = fov / Math.max(8.0, fov - rz * 0.45);
    const sx = width * 0.5 + this.panX + rx * scale * persp;
    const sy = height * 0.5 + this.panY + ry * scale * persp;
    return { sx, sy, rz, persp };
  }

  _updateHover(mx, my) {
    let best = null;
    let bestDist = 14.0;
    for (const p of this.projectedAtoms) {
      const d = Math.hypot(p.sx - mx, p.sy - my);
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
    if (best !== this.hoveredAtom) {
      this.hoveredAtom = best;
      if (this.tooltipEl) {
        if (best) {
          const info = best.atom.info;
          const badges = [];
          if (info.hasFixedCrd) badges.push('Fixed C_CRD');
          if (info.crdGrp > 0) badges.push(`Internal Motif #${info.crdGrp}`);
          if (info.isHotspot) badges.push('Hotspot C_HOT');
          if (info.isMcoord) badges.push('Metal Coord C_MCOORD');
          if (info.isHbond) badges.push('H-Bond C_HB');
          if (info.isSalt) badges.push('Salt Bridge C_SALT');
          if (badges.length === 0) badges.push('De Novo Generated');

          this.tooltipEl.innerHTML = `
            <div class="tooltip-title">Chain ${info.chainId} · ${info.resName} ${info.resId} <span class="tooltip-token">#${info.tokenIdx + 1}</span></div>
            <div class="tooltip-sub">${ badges.join(' · ') }</div>
          `;
          this.tooltipEl.style.left = `${Math.min(this.canvas.clientWidth - 210, Math.max(10, best.sx + 14))}px`;
          this.tooltipEl.style.top = `${Math.max(10, best.sy - 36)}px`;
          this.tooltipEl.hidden = false;
        } else {
          this.tooltipEl.hidden = true;
        }
      }
      this.render();
    }
  }

  render() {
    const dpr = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth || 720;
    const height = this.canvas.clientHeight || 520;
    if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(width * dpr);
      this.canvas.height = Math.round(height * dpr);
    }

    const ctx = this.ctx;
    ctx.save();
    ctx.scale(dpr, dpr);

    // Subtle dark radial studio backdrop
    const bgGrad = ctx.createRadialGradient(
      width * 0.5,
      height * 0.45,
      20,
      width * 0.5,
      height * 0.5,
      Math.max(width, height) * 0.72
    );
    bgGrad.addColorStop(0, '#131c31');
    bgGrad.addColorStop(1, '#090d16');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    if (!this.scene || this.scene.atoms.length === 0) {
      ctx.restore();
      return;
    }

    const scale = (Math.min(width, height) * 0.37 * this.zoom) / Math.max(6.0, this.scene.maxRadius);

    // Project all atoms
    const projByAtomIdx = new Array(this.scene.atoms.length);
    for (let i = 0; i < this.scene.atoms.length; i += 1) {
      const a = this.scene.atoms[i];
      const p = this._project(a.x, a.y, a.z, width, height, scale);
      projByAtomIdx[i] = { ...p, atom: a };
    }

    // 1. Draw dashed interaction vectors (metal coordination / hotspots)
    if (this.showInteractions && this.scene.interactions.length > 0) {
      ctx.save();
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1.8;
      for (const inter of this.scene.interactions) {
        const p1 = projByAtomIdx[inter.from.atomIdx];
        const p2 = projByAtomIdx[inter.to.atomIdx];
        if (!p1 || !p2) continue;
        ctx.strokeStyle = inter.color;
        ctx.beginPath();
        ctx.moveTo(p1.sx, p1.sy);
        ctx.lineTo(p2.sx, p2.sy);
        ctx.stroke();
      }
      ctx.restore();
    }

    // 2. Build depth-sorted primitives (CA backbone segments, sidechain bonds, atoms)
    const primitives = [];

    // Consecutive CA backbone tube segments within the same chain
    if (this.reprMode === 'hybrid' || this.reprMode === 'backbone') {
      const trace = this.scene.caTrace;
      for (let i = 0; i < trace.length - 1; i += 1) {
        const c1 = trace[i];
        const c2 = trace[i + 1];
        if (c1.info.chainId !== c2.info.chainId) continue;
        if (c1.info.isAtomized || c2.info.isAtomized) continue;
        const d3d = Math.hypot(c1.x - c2.x, c1.y - c2.y, c1.z - c2.z);
        if (d3d > 9.5) continue; // chain break / unindexed jump

        const p1 = projByAtomIdx[c1.atomIdx];
        const p2 = projByAtomIdx[c2.atomIdx];
        primitives.push({
          type: 'tube',
          rz: 0.5 * (p1.rz + p2.rz),
          p1,
          p2,
          color: this._colorForToken(c1.info),
          sec: c1.info.sec,
        });
      }
    }

    // All-atom covalent bonds
    if (this.reprMode === 'hybrid' || this.reprMode === 'allatom') {
      for (const b of this.scene.bonds) {
        const p1 = projByAtomIdx[b.a1];
        const p2 = projByAtomIdx[b.a2];
        if (!p1 || !p2) continue;
        if (this.reprMode === 'hybrid' && b.kind === 'backbone') continue;
        primitives.push({
          type: 'bond',
          rz: 0.5 * (p1.rz + p2.rz),
          p1,
          p2,
          color: this._colorForToken(p1.atom.info),
        });
      }
    }

    // Atom spheres
    const pickable = [];
    for (const p of projByAtomIdx) {
      const a = p.atom;
      if (this.reprMode === 'backbone' && !a.isCa) continue;
      primitives.push({
        type: 'atom',
        rz: p.rz,
        p,
      });
      if (a.isCa || this.reprMode === 'allatom') {
        pickable.push(p);
      }
    }
    this.projectedAtoms = pickable;

    // Sort back-to-front by rz
    primitives.sort((a, b) => a.rz - b.rz);

    const maxR = this.scene.maxRadius;
    for (const prim of primitives) {
      const depthAlpha = Math.max(0.32, Math.min(1.0, 0.72 + (prim.rz / (maxR * 1.4)) * 0.35));

      if (prim.type === 'tube') {
        const widthPx = (prim.sec === 1 ? 6.2 : prim.sec === 2 ? 5.4 : 4.0) * ((prim.p1.persp + prim.p2.persp) * 0.5);
        ctx.save();
        ctx.globalAlpha = depthAlpha;
        ctx.lineCap = 'round';
        // Outer dark rim for 3D depth separation
        ctx.strokeStyle = 'rgba(6, 10, 20, 0.75)';
        ctx.lineWidth = widthPx + 2.0;
        ctx.beginPath();
        ctx.moveTo(prim.p1.sx, prim.p1.sy);
        ctx.lineTo(prim.p2.sx, prim.p2.sy);
        ctx.stroke();

        // Main tube
        ctx.strokeStyle = prim.color;
        ctx.lineWidth = widthPx;
        ctx.beginPath();
        ctx.moveTo(prim.p1.sx, prim.p1.sy);
        ctx.lineTo(prim.p2.sx, prim.p2.sy);
        ctx.stroke();
        ctx.restore();
      } else if (prim.type === 'bond') {
        ctx.save();
        ctx.globalAlpha = depthAlpha * 0.85;
        ctx.strokeStyle = prim.color;
        ctx.lineWidth = 1.9 * ((prim.p1.persp + prim.p2.persp) * 0.5);
        ctx.beginPath();
        ctx.moveTo(prim.p1.sx, prim.p1.sy);
        ctx.lineTo(prim.p2.sx, prim.p2.sy);
        ctx.stroke();
        ctx.restore();
      } else if (prim.type === 'atom') {
        const { sx, sy, persp, atom } = prim.p;
        const info = atom.info;
        const isLig = info.isAtomized;
        const baseR = isLig
          ? 7.5
          : atom.isCa
          ? (this.reprMode === 'backbone' ? 4.8 : 4.2)
          : 2.3;
        const radius = Math.max(1.4, baseR * persp * Math.min(1.3, Math.sqrt(this.zoom)));

        ctx.save();
        ctx.globalAlpha = depthAlpha;

        // Hotspot halo ring
        if (atom.isCa && info.isHotspot) {
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.85)';
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.arc(sx, sy, radius + 4.2, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Unindexed internal coord motif ring
        if (atom.isCa && info.crdGrp > 0) {
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.arc(sx, sy, radius + 3.2, 0, Math.PI * 2);
          ctx.stroke();
        }

        const fillColor =
          atom.isCa || isLig
            ? this._colorForToken(info)
            : ELEMENT_COLORS[atom.elem] || this._colorForToken(info);

        const sphereGrad = ctx.createRadialGradient(
          sx - radius * 0.32,
          sy - radius * 0.32,
          radius * 0.1,
          sx,
          sy,
          radius
        );
        sphereGrad.addColorStop(0, '#ffffff');
        sphereGrad.addColorStop(0.35, fillColor);
        sphereGrad.addColorStop(1, '#090d16');

        ctx.fillStyle = sphereGrad;
        ctx.beginPath();
        ctx.arc(sx, sy, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }

    // Highlight hovered atom
    if (this.hoveredAtom) {
      const { sx, sy } = this.hoveredAtom;
      ctx.save();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(sx, sy, 8.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore();
  }
}
