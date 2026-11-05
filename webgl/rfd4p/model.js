/**
 * RFD4-Proteina on WebGL2: the trunk (`LocalLatentsTransformer`), the autoencoder's decoder and encoder, as the float32
 * CPU reference (`cpu/rfd4p/reference.js`, `decoder.js`, `encoder.js`) computes them, function for function, on the
 * kernels of `gl.js`. Features and postprocessing stay on the CPU and are shared with the reference.
 *
 * The fallback for browsers without WebGPU: every tensor lives on the GPU through a forward pass and only the outputs
 * are read back. `forwardTrunkGL`, `decodeLatentsGL` and `encodeLatentsGL` take and return what their CPU twins do,
 * so `generateRfd4pCpu` runs on them unchanged (`kernels`).
 */

import { GLContext, TW } from './gl.js';
import { aeMasks, decoderPostprocess, decoderSeqFeatures } from '../../cpu/rfd4p/decoder.js';
import { encoderPostprocess, encoderSeqFeatures } from '../../cpu/rfd4p/encoder.js';

const ACT = { relu: 1, sigmoid: 2 };
// (Block kernels' output planes in narrow textures, 64 texels wide, tried: the matmul alone ~10% faster in
// tools/gpu/bench-webgl.js --layout=1, the whole pass ~4% slower, M2.)

export class Rfd4pModelGL {
  /** @param {object} weights tensors by name (`.data` decodes on read), as the CPU reference takes them */
  constructor(weights, { ctx = null } = {}) {
    this.ctx = ctx ?? new GLContext();
    this.weights = weights;
    this.cache = new Map();
    // Normalised pair activations feeding the big matmuls and the triangle contractions held in float16.
    this.halfPair = true;
  }

  get backendName() {
    return 'WebGL2';
  }

  /** A weight on the GPU: large matrices in float16 (as the WebGPU path holds them), the rest in float32. */
  W(name, optional = false) {
    if (this.cache.has(name)) return this.cache.get(name);
    const entry = this.weights.tensors ? this.weights.tensors[name] : this.weights[name];
    if (!entry) {
      if (optional) return null;
      throw new Error(`Missing weight tensor: ${name}`);
    }
    const data = entry.data ?? entry;
    const t = this.ctx.upload(data, { half: data.length > 65536, weight: true });
    this.cache.set(name, t);
    return t;
  }

  /** Weight `name` (`[outD, inD]`) with its rows zero-padded to `width`. */
  padded(name, outD, inD, width) {
    const key = `${name}#${width}`;
    if (this.cache.has(key)) return this.cache.get(key);
    const entry = this.weights.tensors ? this.weights.tensors[name] : this.weights[name];
    const data = entry.data ?? entry;
    const out = new Float32Array(outD * width);
    for (let o = 0; o < outD; o += 1) out.set(data.subarray(o * inD, (o + 1) * inD), o * width);
    const t = this.ctx.upload(out, { half: out.length > 65536, weight: true });
    this.cache.set(key, t);
    return t;
  }

  destroy() {
    this.ctx.destroy();
  }

  // ---- kernels ------------------------------------------------------------------------------------------------

  /** `Y = X W^T (+ bias)(act)`, X `[rows, inD]`, W `[outD, inD]` (a weight, or any tensor). */
  /**
   * `resid`: { z, mask, gate? } makes the result `z + mask[r] * Y * sigmoid(gate)` (a residual update, gated or not),
   * fused into the packed matmul's epilogue.
   */
  linear(x, rows, inD, w, outD, bias = null, act = null, half = false, resid = null) {
    const c = this.ctx;
    const out = c.tensor(rows * outD);
    const outHalf = half === 'packed' ? 'packed' : half && this.halfPair;
    const packedOk = rows >= 64 && inD % 8 === 0 && outD % 4 === 0 && TW % (inD / 8) === 0 && !this.noPacked;
    if (resid && !packedOk) {
      out.free();
      const y = this.linear(x, rows, inD, w, outD, bias, act);
      const r = c.elementwise(`resid${resid.gate ? '-g' : ''}`, `el(Z, e) + (el(PM, e / uC) != 0.0 ? el(Y, e)${resid.gate ? ' * sigm(el(G, e))' : ''} : 0.0)`,
        rows * outD, { inputs: { Z: resid.z, Y: y, PM: resid.mask, ...(resid.gate ? { G: resid.gate } : {}) }, ints: { uC: outD } });
      y.free();
      return r;
    }
    if (x.half === 'packed' && !packedOk) {
      const xu = this._unpack(x);
      out.free();
      const y = this.linear(xu, rows, inD, w, outD, bias, act, half);
      xu.free();
      return y;
    }
    const inputs = { X: x, Wt: w, B: bias ?? x };
    const ints = { uIn: inD, uOut: outD, uBias: bias ? 1 : 0, uAct: ACT[act] ?? 0, uRows: rows };
    const epi = `
float epi(float v) { return uAct == 1 ? max(v, 0.0) : uAct == 2 ? sigm(v) : v; }`;
    const decl = 'uniform sampler2D X; uniform sampler2D Wt; uniform sampler2D B; uniform int uIn; uniform int uOut; uniform int uBias; uniform int uAct; uniform int uRows;';
    if (packedOk && (outHalf || resid) && rows >= 4096 && outD % 8 === 0 && !this.no48) {
      // An output that may be float16 (packed or half): 4 rows x 8 columns a fragment, written as packed halves (four
      // RGBA32UI targets, 32 values a pixel within the 64-byte budget): 12 texel reads per 256 multiply-adds.
      out.free();
      const xp = this._pack(x, rows * inD);
      const wp = w.weight ? this._packWeight(w) : this._pack(w, outD * inD);
      const rq = Math.ceil(rows / 4);
      const qs = [0, 1, 2, 3].map(() => c.tensor(rq * outD, 'packed'));
      c.kernel('linear4x8pk', `
uniform highp usampler2D X; uniform highp usampler2D Wt; uniform sampler2D B;
uniform int uIn; uniform int uOut; uniform int uBias; uniform int uAct; uniform int uRows;${epi}
vec4 epi4(vec4 a) { return vec4(epi(a.x), epi(a.y), epi(a.z), epi(a.w)); }
void unpack8(uvec4 u, out vec4 lo, out vec4 hi) { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
uvec4 pk(vec4 a, vec4 b) { return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw)); }
void compute4(int t, out uvec4 r0, out uvec4 r1, out uvec4 r2, out uvec4 r3) {
  int e0 = t * 8;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int ib = uIn / 8;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wt, 0).x;
  ivec2 xc[4];
  bool live[4];
  for (int i = 0; i < 4; i++) { int r = q * 4 + i; live[i] = r < uRows; int xt = (live[i] ? r : 0) * ib; xc[i] = ivec2(xt % xw, xt / xw); }
  ivec2 wc[8];
  for (int j = 0; j < 8; j++) { int wt = (o0 + j) * ib; wc[j] = ivec2(wt % ww, wt / ww); }
  vec4 a[4], b[4];
  for (int i = 0; i < 4; i++) { a[i] = vec4(0.0); b[i] = vec4(0.0); }
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    vec4 l[8], h[8];
    for (int j = 0; j < 8; j++) unpack8(texelFetch(Wt, wc[j] + dk, 0), l[j], h[j]);
    mat4 wla = mat4(l[0], l[1], l[2], l[3]), wha = mat4(h[0], h[1], h[2], h[3]);
    mat4 wlb = mat4(l[4], l[5], l[6], l[7]), whb = mat4(h[4], h[5], h[6], h[7]);
    for (int i = 0; i < 4; i++) {
      vec4 xl, xh;
      unpack8(texelFetch(X, xc[i] + dk, 0), xl, xh);
      a[i] += xl * wla + xh * wha;
      b[i] += xl * wlb + xh * whb;
    }
  }
  vec4 ba = uBias == 1 ? tx(B, o0 / 4) : vec4(0.0), bbv = uBias == 1 ? tx(B, o0 / 4 + 1) : vec4(0.0);
  r0 = live[0] ? pk(epi4(a[0] + ba), epi4(b[0] + bbv)) : uvec4(0u);
  r1 = live[1] ? pk(epi4(a[1] + ba), epi4(b[1] + bbv)) : uvec4(0u);
  r2 = live[2] ? pk(epi4(a[2] + ba), epi4(b[2] + bbv)) : uvec4(0u);
  r3 = live[3] ? pk(epi4(a[3] + ba), epi4(b[3] + bbv)) : uvec4(0u);
}`, qs, { inputs: { X: xp, Wt: wp, B: bias ?? w }, ints, work: inD * 3 });
      if (xp !== x) xp.free();
      if (wp !== w && !wp.weight) wp.free();
      // A residual update: the float16 result is the delta; the repack adds it to z in float32.
      return resid ? this._repackResid(qs, rows, outD, resid) : this._repackPacked(qs, rows, outD, outHalf);
    }
    if (packedOk) {
      // Operands as packed halves (8 a texel): a 4 x 4 block a fragment reads half the texels of the float path (2x on
      // an M2, tools/gpu/bench-webgl.js --packed=1), into four render targets, then repacked.
      out.free();
      const xp = this._pack(x, rows * inD);
      const wp = w.weight ? this._packWeight(w) : this._pack(w, outD * inD);
      const rq = Math.ceil(rows / 4);
      const qs = [0, 1, 2, 3].map(() => c.tensor(rq * outD));
      c.kernel('linear4x4pk', `
uniform highp usampler2D X; uniform highp usampler2D Wt; uniform sampler2D B;
uniform sampler2D Z; uniform sampler2D G; uniform sampler2D PM; uniform int uRes; uniform int uGate;
uniform int uIn; uniform int uOut; uniform int uBias; uniform int uAct; uniform int uRows;${epi}
vec4 fin(vec4 v, int r, int o0) {
  if (uRes == 0) return v;
  int zt = (r * uOut + o0) / 4;
  vec4 g = uGate == 1 ? vec4(1.0) / (vec4(1.0) + exp(-tx(G, zt))) : vec4(1.0);
  return tx(Z, zt) + (el(PM, r) != 0.0 ? v * g : vec4(0.0));
}
vec4 epi4(vec4 a) { return vec4(epi(a.x), epi(a.y), epi(a.z), epi(a.w)); }
void unpack8(uvec4 u, out vec4 lo, out vec4 hi) { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
void compute4(int t, out vec4 r0, out vec4 r1, out vec4 r2, out vec4 r3) {
  int e0 = t * 4;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int ib = uIn / 8;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wt, 0).x;
  ivec2 xc[4];
  bool live[4];
  for (int i = 0; i < 4; i++) { int r = q * 4 + i; live[i] = r < uRows; int xt = (live[i] ? r : 0) * ib; xc[i] = ivec2(xt % xw, xt / xw); }
  ivec2 wc[4];
  for (int i = 0; i < 4; i++) { int wt = (o0 + i) * ib; wc[i] = ivec2(wt % ww, wt / ww); }
  vec4 a[4];
  for (int i = 0; i < 4; i++) a[i] = vec4(0.0);
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    vec4 l0, h0, l1, h1, l2, h2, l3, h3;
    unpack8(texelFetch(Wt, wc[0] + dk, 0), l0, h0);
    unpack8(texelFetch(Wt, wc[1] + dk, 0), l1, h1);
    unpack8(texelFetch(Wt, wc[2] + dk, 0), l2, h2);
    unpack8(texelFetch(Wt, wc[3] + dk, 0), l3, h3);
    mat4 wl = mat4(l0, l1, l2, l3), wh = mat4(h0, h1, h2, h3);
    for (int i = 0; i < 4; i++) {
      vec4 xl, xh;
      unpack8(texelFetch(X, xc[i] + dk, 0), xl, xh);
      a[i] += xl * wl + xh * wh;
    }
  }
  vec4 bv = uBias == 1 ? tx(B, o0 / 4) : vec4(0.0);
  r0 = live[0] ? fin(epi4(a[0] + bv), q * 4, o0) : vec4(0.0);
  r1 = live[1] ? fin(epi4(a[1] + bv), q * 4 + 1, o0) : vec4(0.0);
  r2 = live[2] ? fin(epi4(a[2] + bv), q * 4 + 2, o0) : vec4(0.0);
  r3 = live[3] ? fin(epi4(a[3] + bv), q * 4 + 3, o0) : vec4(0.0);
}`, qs, { inputs: { X: xp, Wt: wp, B: bias ?? w, Z: resid?.z ?? w, G: resid?.gate ?? w, PM: resid?.mask ?? w },
        ints: { ...ints, uRes: resid ? 1 : 0, uGate: resid?.gate ? 1 : 0 }, work: inD * 2 });
      if (xp !== x) xp.free();
      if (wp !== w && !wp.weight) wp.free();
      return this._repack(qs, rows, outD, outHalf);
    }
    if (rows >= 64 && inD % 4 === 0 && outD % 4 === 0 && TW % (inD / 4) === 0 && !this.no4x4) {
      // A block of R = 4 rows x 4 columns a fragment (R render targets, a 1/R of the rows each), then repacked: 8 texel
      // reads per 64 multiply-adds rather than 5 per 16. Measured slower or no better (M2, tools/gpu/bench-webgl.js):
      // 8 rows over 8 targets, float16 targets, the k loop unrolled by two.
      out.free();
      const R = 4;
      const rq = Math.ceil(rows / R);
      const qs = Array.from({ length: R }, () => c.tensor(rq * outD));
      if (outHalf === 'packed') throw new Error('linear: packed output from the float path');
      const idx = [...Array(R).keys()];
      c.kernel(`linear${R}x4`, `${decl}${epi}
vec4 epi4(vec4 a) { return vec4(epi(a.x), epi(a.y), epi(a.z), epi(a.w)); }
void compute4(int t, ${idx.map((i) => `out vec4 r${i}`).join(', ')}) {
  int e0 = t * 4;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int ib = uIn / 4;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wt, 0).x;
  ${idx.map((i) => `int rr${i} = q * ${R} + ${i}; bool live${i} = rr${i} < uRows; int xt${i} = (live${i} ? rr${i} : 0) * ib; ivec2 xc${i} = ivec2(xt${i} % xw, xt${i} / xw); vec4 a${i} = vec4(0.0);`).join('\n  ')}
  int wt = o0 * ib;
  ivec2 w0 = ivec2(wt % ww, wt / ww);
  wt += ib;
  ivec2 w1 = ivec2(wt % ww, wt / ww);
  wt += ib;
  ivec2 w2 = ivec2(wt % ww, wt / ww);
  wt += ib;
  ivec2 w3 = ivec2(wt % ww, wt / ww);
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    mat4 wm = mat4(texelFetch(Wt, w0 + dk, 0), texelFetch(Wt, w1 + dk, 0), texelFetch(Wt, w2 + dk, 0), texelFetch(Wt, w3 + dk, 0));
    ${idx.map((i) => `a${i} += texelFetch(X, xc${i} + dk, 0) * wm;`).join('\n    ')}
  }
  vec4 bv = uBias == 1 ? tx(B, o0 / 4) : vec4(0.0);
  ${idx.map((i) => `r${i} = live${i} ? epi4(a${i} + bv) : vec4(0.0);`).join('\n  ')}
}`, qs, { inputs, ints, work: inD * R });
      const qin = Object.fromEntries(qs.map((t, i) => [`Q${i}`, t]));
      const res = c.kernel(`repack${R}`, `
${idx.map((i) => `uniform sampler2D Q${i};`).join(' ')} uniform int uOut;
vec4 compute(int t) {
  int e0 = t * 4;
  int r = e0 / uOut;
  int qt = ((r / ${R}) * uOut + e0 - r * uOut) / 4;
  int i = r - (r / ${R}) * ${R};
  ${idx.slice(0, -1).map((i) => `if (i == ${i}) return tx(Q${i}, qt);`).join(' ')}
  return tx(Q${R - 1}, qt);
}`, c.tensor(rows * outD, outHalf), { inputs: qin, ints: { uOut: outD } });
      qs.forEach((t) => t.free());
      return res;
    }
    if (inD % 4 === 0 && outD % 4 === 0 && TW % (inD / 4) === 0) {
      // Rows never wrap (TW): each operand row's texture coordinate computed once, then walked.
      return c.kernel('linear4r', `${decl}${epi}
vec4 compute(int t) {
  int e0 = t * 4;
  int r = e0 / uOut;
  int o0 = e0 - r * uOut;
  int ib = uIn / 4;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wt, 0).x;
  int xt = r * ib;
  ivec2 xc = ivec2(xt % xw, xt / xw);
  int wt = o0 * ib;
  ivec2 w0 = ivec2(wt % ww, wt / ww);
  wt += ib;
  ivec2 w1 = ivec2(wt % ww, wt / ww);
  wt += ib;
  ivec2 w2 = ivec2(wt % ww, wt / ww);
  wt += ib;
  ivec2 w3 = ivec2(wt % ww, wt / ww);
  vec4 acc = vec4(0.0);
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    vec4 xv = texelFetch(X, xc + dk, 0);
    acc.x += dot(xv, texelFetch(Wt, w0 + dk, 0));
    acc.y += dot(xv, texelFetch(Wt, w1 + dk, 0));
    acc.z += dot(xv, texelFetch(Wt, w2 + dk, 0));
    acc.w += dot(xv, texelFetch(Wt, w3 + dk, 0));
  }
  if (uBias == 1) acc += tx(B, o0 / 4);
  return vec4(epi(acc.x), epi(acc.y), epi(acc.z), epi(acc.w));
}`, out, { inputs, ints, work: inD });
    }
    if (inD % 4 === 0 && outD % 4 === 0) {
      return c.kernel('linear4', `${decl}${epi}
vec4 compute(int t) {
  int e0 = t * 4;
  int r = e0 / uOut;
  int o0 = e0 - r * uOut;
  int ib = uIn / 4;
  int xb = r * ib;
  int w0 = o0 * ib;
  vec4 acc = vec4(0.0);
  for (int k = 0; k < ib; k++) {
    vec4 xv = tx(X, xb + k);
    acc.x += dot(xv, tx(Wt, w0 + k));
    acc.y += dot(xv, tx(Wt, w0 + ib + k));
    acc.z += dot(xv, tx(Wt, w0 + 2 * ib + k));
    acc.w += dot(xv, tx(Wt, w0 + 3 * ib + k));
  }
  if (uBias == 1) acc += tx(B, o0 / 4);
  return vec4(epi(acc.x), epi(acc.y), epi(acc.z), epi(acc.w));
}`, out, { inputs, ints, work: inD });
    }
    // Odd widths (feature inputs, heads): one value at a time, by texels when the input width allows.
    return c.elementwise(`linear1-${inD % 4 === 0 ? 'v' : 's'}`, 'lin(e)', rows * outD, {
      inputs, ints, work: inD * 4, out,
      pre: `${epi}
float lin(int e) {
  int r = e / uOut;
  int o = e - r * uOut;
  float acc = 0.0;
  ${inD % 4 === 0 ? `int ib = uIn / 4;
  for (int k = 0; k < ib; k++) acc += dot(tx(X, r * ib + k), tx(Wt, o * ib + k));` : `for (int k = 0; k < uIn; k++) acc += el(X, r * uIn + k) * el(Wt, o * uIn + k);`}
  if (uBias == 1) acc += el(B, o);
  return epi(acc);
}`,
    });
  }

  /**
   * `mask[r] * (X Wp^T + bp) * sigmoid(X Wg^T + bg)`, both projections of one input in one pass (the triangle update's
   * a and b): 4 rows x 4 columns of each a fragment, four render targets, then repacked.
   */
  gatedLinear(x, rows, inD, wp, bp, wg, bg, outD, mask, half = false) {
    const c = this.ctx;
    const ib = inD / 4;
    if (inD % 4 || outD % 4 || TW % ib) {
      // (linear unpacks a packed x itself)
      const p = this.linear(x, rows, inD, wp, outD, bp);
      const g = this.linear(x, rows, inD, wg, outD, bg);
      const out = c.elementwise('trigate', 'el(PM, e / uC) != 0.0 ? el(P, e) * sigm(el(G, e)) : 0.0', rows * outD,
        { inputs: { P: p, G: g, PM: mask }, ints: { uC: outD }, half: half && this.halfPair });
      p.free();
      g.free();
      return out;
    }
    const rq = Math.ceil(rows / 4);
    // Also measured and slower or no better (M2, one warm pass at n = 208): the projection's and gate's weights
    // interleaved into one texture (2.38 s vs 2.33), the two as separate 4 x 8 matmuls plus a combine pass (2.54 s).
    // (4 rows x 8 columns of both projections, packed outputs, measured 1.2x slower: register pressure, M2.)
    if (inD % 8 === 0 && TW % (inD / 8) === 0 && !this.noPacked) {
      const xp = this._pack(x, rows * inD);
      const wpp = this._packWeight(wp);
      const wgp = this._packWeight(wg);
      const qs = [0, 1, 2, 3].map(() => c.tensor(rq * outD));
      c.kernel('gatedlinear4x4pk', `
uniform highp usampler2D X; uniform highp usampler2D Wp; uniform highp usampler2D Wg; uniform sampler2D Bp; uniform sampler2D Bg; uniform sampler2D M;
uniform int uIn; uniform int uOut; uniform int uRows; uniform int uBp; uniform int uBg;
void unpack8(uvec4 u, out vec4 lo, out vec4 hi) { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
void compute4(int t, out vec4 r0, out vec4 r1, out vec4 r2, out vec4 r3) {
  int e0 = t * 4;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int ib = uIn / 8;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wp, 0).x;
  ivec2 xc[4];
  bool live[4];
  for (int i = 0; i < 4; i++) { int r = q * 4 + i; live[i] = r < uRows && el(M, r) != 0.0; int xt = (r < uRows ? r : 0) * ib; xc[i] = ivec2(xt % xw, xt / xw); }
  ivec2 wc[4];
  for (int i = 0; i < 4; i++) { int wt = (o0 + i) * ib; wc[i] = ivec2(wt % ww, wt / ww); }
  vec4 p[4], g[4];
  for (int i = 0; i < 4; i++) { p[i] = vec4(0.0); g[i] = vec4(0.0); }
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    vec4 l0, h0, l1, h1, l2, h2, l3, h3;
    unpack8(texelFetch(Wp, wc[0] + dk, 0), l0, h0);
    unpack8(texelFetch(Wp, wc[1] + dk, 0), l1, h1);
    unpack8(texelFetch(Wp, wc[2] + dk, 0), l2, h2);
    unpack8(texelFetch(Wp, wc[3] + dk, 0), l3, h3);
    mat4 pl = mat4(l0, l1, l2, l3), ph = mat4(h0, h1, h2, h3);
    unpack8(texelFetch(Wg, wc[0] + dk, 0), l0, h0);
    unpack8(texelFetch(Wg, wc[1] + dk, 0), l1, h1);
    unpack8(texelFetch(Wg, wc[2] + dk, 0), l2, h2);
    unpack8(texelFetch(Wg, wc[3] + dk, 0), l3, h3);
    mat4 gm = mat4(l0, l1, l2, l3), gh = mat4(h0, h1, h2, h3);
    for (int i = 0; i < 4; i++) {
      vec4 xl, xh;
      unpack8(texelFetch(X, xc[i] + dk, 0), xl, xh);
      p[i] += xl * pl + xh * ph;
      g[i] += xl * gm + xh * gh;
    }
  }
  vec4 bpv = uBp == 1 ? tx(Bp, o0 / 4) : vec4(0.0), bgv = uBg == 1 ? tx(Bg, o0 / 4) : vec4(0.0);
  r0 = live[0] ? (p[0] + bpv) / (vec4(1.0) + exp(-(g[0] + bgv))) : vec4(0.0);
  r1 = live[1] ? (p[1] + bpv) / (vec4(1.0) + exp(-(g[1] + bgv))) : vec4(0.0);
  r2 = live[2] ? (p[2] + bpv) / (vec4(1.0) + exp(-(g[2] + bgv))) : vec4(0.0);
  r3 = live[3] ? (p[3] + bpv) / (vec4(1.0) + exp(-(g[3] + bgv))) : vec4(0.0);
}`, qs, { inputs: { X: xp, Wp: wpp, Wg: wgp, Bp: bp ?? mask, Bg: bg ?? mask, M: mask },
        ints: { uIn: inD, uOut: outD, uRows: rows, uBp: bp ? 1 : 0, uBg: bg ? 1 : 0 }, work: inD * 4 });
      if (xp !== x) xp.free();
      return this._repack(qs, rows, outD, half && this.halfPair);
    }
    if (x.half === 'packed') {
      const xu = this._unpack(x);
      const y = this.gatedLinear(xu, rows, inD, wp, bp, wg, bg, outD, mask, half);
      xu.free();
      return y;
    }
    const qs = [0, 1, 2, 3].map(() => c.tensor(rq * outD, half && this.halfPair));
    c.kernel('gatedlinear4x4', `
uniform sampler2D X; uniform sampler2D Wp; uniform sampler2D Wg; uniform sampler2D Bp; uniform sampler2D Bg; uniform sampler2D M;
uniform int uIn; uniform int uOut; uniform int uRows; uniform int uBp; uniform int uBg;
void compute4(int t, out vec4 r0, out vec4 r1, out vec4 r2, out vec4 r3) {
  int e0 = t * 4;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int ib = uIn / 4;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wp, 0).x;
  ivec2 xc[4];
  bool live[4];
  for (int i = 0; i < 4; i++) {
    int r = q * 4 + i;
    live[i] = r < uRows && el(M, r) != 0.0;
    int xt = (r < uRows ? r : 0) * ib;
    xc[i] = ivec2(xt % xw, xt / xw);
  }
  ivec2 wc[4];
  for (int i = 0; i < 4; i++) { int wt = (o0 + i) * ib; wc[i] = ivec2(wt % ww, wt / ww); }
  vec4 p0 = vec4(0.0), p1 = vec4(0.0), p2 = vec4(0.0), p3 = vec4(0.0);
  vec4 g0 = vec4(0.0), g1 = vec4(0.0), g2 = vec4(0.0), g3 = vec4(0.0);
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    mat4 wpm = mat4(texelFetch(Wp, wc[0] + dk, 0), texelFetch(Wp, wc[1] + dk, 0), texelFetch(Wp, wc[2] + dk, 0), texelFetch(Wp, wc[3] + dk, 0));
    mat4 wgm = mat4(texelFetch(Wg, wc[0] + dk, 0), texelFetch(Wg, wc[1] + dk, 0), texelFetch(Wg, wc[2] + dk, 0), texelFetch(Wg, wc[3] + dk, 0));
    vec4 x0 = texelFetch(X, xc[0] + dk, 0), x1 = texelFetch(X, xc[1] + dk, 0), x2 = texelFetch(X, xc[2] + dk, 0), x3 = texelFetch(X, xc[3] + dk, 0);
    p0 += x0 * wpm; g0 += x0 * wgm;
    p1 += x1 * wpm; g1 += x1 * wgm;
    p2 += x2 * wpm; g2 += x2 * wgm;
    p3 += x3 * wpm; g3 += x3 * wgm;
  }
  vec4 bpv = uBp == 1 ? tx(Bp, o0 / 4) : vec4(0.0), bgv = uBg == 1 ? tx(Bg, o0 / 4) : vec4(0.0);
  r0 = live[0] ? (p0 + bpv) / (vec4(1.0) + exp(-(g0 + bgv))) : vec4(0.0);
  r1 = live[1] ? (p1 + bpv) / (vec4(1.0) + exp(-(g1 + bgv))) : vec4(0.0);
  r2 = live[2] ? (p2 + bpv) / (vec4(1.0) + exp(-(g2 + bgv))) : vec4(0.0);
  r3 = live[3] ? (p3 + bpv) / (vec4(1.0) + exp(-(g3 + bgv))) : vec4(0.0);
}`, qs, { inputs: { X: x, Wp: wp, Wg: wg, Bp: bp ?? mask, Bg: bg ?? mask, M: mask },
      ints: { uIn: inD, uOut: outD, uRows: rows, uBp: bp ? 1 : 0, uBg: bg ? 1 : 0 }, work: inD * 8 });
    const res = c.kernel('repack4', `
uniform sampler2D Q0; uniform sampler2D Q1; uniform sampler2D Q2; uniform sampler2D Q3; uniform int uOut;
vec4 compute(int t) {
  int e0 = t * 4;
  int r = e0 / uOut;
  int qt = ((r / 4) * uOut + e0 - r * uOut) / 4;
  int i = r - (r / 4) * 4;
  if (i == 0) return tx(Q0, qt); if (i == 1) return tx(Q1, qt); if (i == 2) return tx(Q2, qt);
  return tx(Q3, qt);
}`, c.tensor(rows * outD, half && this.halfPair), { inputs: { Q0: qs[0], Q1: qs[1], Q2: qs[2], Q3: qs[3] }, ints: { uOut: outD } });
    qs.forEach((t) => t.free());
    return res;
  }

  /** `x` as packed halves (8 a texel), for the packed matmuls; `x` itself when it already is. */
  _pack(x, size) {
    if (x.half === 'packed') return x;
    return this.ctx.kernel('pack8', `
uniform sampler2D S;
uvec4 compute(int t) {
  vec4 a = tx(S, 2 * t), b = tx(S, 2 * t + 1);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}`, this.ctx.tensor(size, 'packed'), { inputs: { S: x } });
  }

  /** How a site's output is held: packed halves, unless the site is listed in `noPackedSites` (bisecting). */
  pk(site) {
    return this.noPackedSites?.has(site) ? true : 'packed';
  }

  /** A packed tensor back to floats (for a kernel that reads floats). */
  _unpack(x) {
    if (x.half !== 'packed') return x;
    return this.ctx.kernel('unpack8', `
uniform highp usampler2D S;
vec4 compute(int t) {
  int w = textureSize(S, 0).x;
  int pt = t >> 1;
  uvec4 u = texelFetch(S, ivec2(pt % w, pt / w), 0);
  return (t & 1) == 0 ? vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)) : vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w));
}`, this.ctx.tensor(x.size), { inputs: { S: x } });
  }

  /** A weight packed once, on first use, and kept. */
  _packWeight(w) {
    this.packedW ??= new Map();
    let p = this.packedW.get(w);
    if (!p) {
      p = this._pack(w, w.size);
      p.weight = true;
      this.packedW.set(w, p);
    }
    return p;
  }

  /** Four planes of a 4-row block kernel (plane i holds rows 4q + i) into one row-major tensor. */
  _repack(qs, rows, outD, half) {
    if (half === 'packed') {
      const res = this.ctx.elementwise('repack4', 'rp(e)', rows * outD, {
        inputs: { Q0: qs[0], Q1: qs[1], Q2: qs[2], Q3: qs[3] }, ints: { uOut: outD }, half: 'packed',
        pre: `float rp(int e) {
  int r = e / uOut;
  int qe = (r / 4) * uOut + e - r * uOut;
  int i = r - (r / 4) * 4;
  return i == 0 ? el(Q0, qe) : i == 1 ? el(Q1, qe) : i == 2 ? el(Q2, qe) : el(Q3, qe);
}`,
      });
      qs.forEach((t) => t.free());
      return res;
    }
    const res = this.ctx.kernel('repack4', `
uniform sampler2D Q0; uniform sampler2D Q1; uniform sampler2D Q2; uniform sampler2D Q3; uniform int uOut;
vec4 compute(int t) {
  int e0 = t * 4;
  int r = e0 / uOut;
  int qt = ((r / 4) * uOut + e0 - r * uOut) / 4;
  int i = r - (r / 4) * 4;
  if (i == 0) return tx(Q0, qt); if (i == 1) return tx(Q1, qt); if (i == 2) return tx(Q2, qt);
  return tx(Q3, qt);
}`, this.ctx.tensor(rows * outD, half), { inputs: { Q0: qs[0], Q1: qs[1], Q2: qs[2], Q3: qs[3] }, ints: { uOut: outD } });
    qs.forEach((t) => t.free());
    return res;
  }

  /**
   * `_repack` for packed planes (plane i holds rows 4q + i), into packed halves or float16, a texel at a time (outD a
   * multiple of 8, so a texel never straddles rows).
   */
  _repackPacked(qs, rows, outD, half) {
    const c = this.ctx;
    const packed = half === 'packed';
    const body = `
uniform highp usampler2D Q0; uniform highp usampler2D Q1; uniform highp usampler2D Q2; uniform highp usampler2D Q3; uniform int uOut;
uvec4 plane(int i, int t) {
  int w = textureSize(Q0, 0).x;
  ivec2 p = ivec2(t % w, t / w);
  return i == 0 ? texelFetch(Q0, p, 0) : i == 1 ? texelFetch(Q1, p, 0) : i == 2 ? texelFetch(Q2, p, 0) : texelFetch(Q3, p, 0);
}
${packed ? `uvec4 compute(int t) {
  int e0 = t * 8;
  int r = e0 / uOut;
  return plane(r & 3, ((r >> 2) * uOut + e0 - r * uOut) >> 3);
}` : `vec4 compute(int t) {
  int e0 = t * 4;
  int r = e0 / uOut;
  int qe = (r >> 2) * uOut + e0 - r * uOut;
  uvec4 u = plane(r & 3, qe >> 3);
  return (qe & 4) == 0 ? vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)) : vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w));
}`}`;
    const res = c.kernel(`repack4p-${packed ? 'pk' : 'h'}`, body, c.tensor(rows * outD, packed ? 'packed' : true),
      { inputs: { Q0: qs[0], Q1: qs[1], Q2: qs[2], Q3: qs[3] }, ints: { uOut: outD } });
    qs.forEach((t) => t.free());
    return res;
  }

  /** Packed planes of a delta into `z + mask[r] * delta * sigmoid(gate)` (gate optional), float32, a texel at a time. */
  _repackResid(qs, rows, outD, { z, mask, gate }) {
    const c = this.ctx;
    const res = c.kernel(`repack4p-res${gate ? '-g' : ''}`, `
uniform highp usampler2D Q0; uniform highp usampler2D Q1; uniform highp usampler2D Q2; uniform highp usampler2D Q3;
uniform sampler2D Z; uniform sampler2D PM; uniform sampler2D G; uniform int uOut;
uvec4 plane(int i, int t) {
  int w = textureSize(Q0, 0).x;
  ivec2 p = ivec2(t % w, t / w);
  return i == 0 ? texelFetch(Q0, p, 0) : i == 1 ? texelFetch(Q1, p, 0) : i == 2 ? texelFetch(Q2, p, 0) : texelFetch(Q3, p, 0);
}
vec4 compute(int t) {
  int e0 = t * 4;
  int r = e0 / uOut;
  int qe = (r >> 2) * uOut + e0 - r * uOut;
  uvec4 u = plane(r & 3, qe >> 3);
  vec4 d = (qe & 4) == 0 ? vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)) : vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w));
  ${gate ? 'd /= vec4(1.0) + exp(-tx(G, t));' : ''}
  return tx(Z, t) + (el(PM, r) != 0.0 ? d : vec4(0.0));
}`, c.tensor(rows * outD), { inputs: { Q0: qs[0], Q1: qs[1], Q2: qs[2], Q3: qs[3], Z: z, PM: mask, G: gate ?? z }, ints: { uOut: outD } });
    qs.forEach((t) => t.free());
    return res;
  }

  /** RMSNorm (`type` other than 'layer_norm') or LayerNorm over the last `dim` of `[rows, dim]`. */
  norm(x, rows, dim, type, w = null, b = null, eps = 1e-5, half = false) {
    const c = this.ctx;
    const ln = type === 'layer_norm';
    const vec = dim % 4 === 0;
    if (vec && TW % (dim / 4) === 0 && (half !== 'packed' || dim % 8 === 0) && !this.noFastNorm) return this._normFast(x, rows, dim, ln, w, b, eps, half);
    const stats = c.kernel(`normstats-${ln ? 'ln' : 'rms'}-${vec ? 'v' : 's'}`, `
uniform sampler2D X; uniform int uDim; uniform float uEps;
vec4 compute(int r) {
  float s = 0.0;
  ${vec ? 'int nb = uDim / 4; int base = r * nb;' : 'int base = r * uDim;'}
  ${ln ? `${vec ? 'for (int k = 0; k < nb; k++) { vec4 v = tx(X, base + k); s += v.x + v.y + v.z + v.w; }' : 'for (int k = 0; k < uDim; k++) s += el(X, base + k);'}
  float mean = s / float(uDim);
  float q = 0.0;
  ${vec ? 'for (int k = 0; k < nb; k++) { vec4 v = tx(X, base + k) - vec4(mean); q += dot(v, v); }' : 'for (int k = 0; k < uDim; k++) { float v = el(X, base + k) - mean; q += v * v; }'}
  return vec4(mean, 1.0 / sqrt(q / float(uDim) + uEps), 0.0, 0.0);` : `${vec ? 'for (int k = 0; k < nb; k++) { vec4 v = tx(X, base + k); s += dot(v, v); }' : 'for (int k = 0; k < uDim; k++) { float v = el(X, base + k); s += v * v; }'}
  return vec4(0.0, 1.0 / sqrt(s / float(uDim) + uEps), 0.0, 0.0);`}
}`, c.tensor(rows * 4), { inputs: { X: x }, ints: { uDim: dim }, floats: { uEps: eps }, work: dim * 2 });
    const out = c.elementwise(`normapply-${w ? 'w' : ''}${b ? 'b' : ''}`,
      `(el(X, e) - tx(S, e / uDim).x) * tx(S, e / uDim).y${w ? ' * el(Wn, e - (e / uDim) * uDim)' : ''}${b ? ' + el(Bn, e - (e / uDim) * uDim)' : ''}`,
      rows * dim, { inputs: { X: x, S: stats, ...(w ? { Wn: w } : {}), ...(b ? { Bn: b } : {}) }, ints: { uDim: dim }, half: half === 'packed' ? 'packed' : half && this.halfPair });
    stats.free();
    return out;
  }

  /**
   * `norm` for rows that never wrap (TW): statistics in one walk of the row (sum and sum of squares about the row's
   * first value, four accumulators), then applied a texel at a time (8 values when packed) rather than a value at a time.
   */
  _normFast(x, rows, dim, ln, w, b, eps, half) {
    const c = this.ctx;
    const stats = c.kernel(`normstats1-${ln ? 'ln' : 'rms'}`, `
uniform sampler2D X; uniform int uDim; uniform float uEps;
vec4 compute(int r) {
  int nb = uDim / 4;
  int w = textureSize(X, 0).x;
  int base = r * nb;
  ivec2 rc = ivec2(base % w, base / w);
  ${ln ? 'float k0 = texelFetch(X, rc, 0).x;' : 'float k0 = 0.0;'}
  vec4 s1 = vec4(0.0), s2 = vec4(0.0);
  for (int k = 0; k < nb; k++) {
    vec4 v = texelFetch(X, rc + ivec2(k, 0), 0) - vec4(k0);
    s1 += v;
    s2 += v * v;
  }
  float n = float(uDim);
  float m1 = (s1.x + s1.y + s1.z + s1.w) / n;
  float m2 = (s2.x + s2.y + s2.z + s2.w) / n;
  ${ln ? 'return vec4(k0 + m1, 1.0 / sqrt(max(m2 - m1 * m1, 0.0) + uEps), 0.0, 0.0);' : 'return vec4(0.0, 1.0 / sqrt(m2 + uEps), 0.0, 0.0);'}
}`, c.tensor(rows * 4), { inputs: { X: x }, ints: { uDim: dim }, floats: { uEps: eps }, work: dim / 4 });
    const packed = half === 'packed';
    const outHalf = packed ? 'packed' : half && this.halfPair;
    const inputs = { X: x, S: stats, Wn: w ?? x, Bn: b ?? x };
    const ints = { uDim: dim, uW: w ? 1 : 0, uB: b ? 1 : 0 };
    const head = `
uniform sampler2D X; uniform sampler2D S; uniform sampler2D Wn; uniform sampler2D Bn; uniform int uDim; uniform int uW; uniform int uB;
vec4 ap(int t) {
  int nb = uDim / 4;
  int r = t / nb;
  int d4 = t - r * nb;
  vec4 st = tx(S, r);
  vec4 v = (tx(X, t) - vec4(st.x)) * st.y;
  if (uW == 1) v *= tx(Wn, d4);
  if (uB == 1) v += tx(Bn, d4);
  return v;
}`;
    const out = packed
      ? c.kernel('normapply1-pk', `${head}
uvec4 compute(int t) {
  vec4 a = ap(2 * t), bb = ap(2 * t + 1);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(bb.xy), packHalf2x16(bb.zw));
}`, c.tensor(rows * dim, 'packed'), { inputs, ints })
      : c.kernel('normapply1', `${head}
vec4 compute(int t) { return ap(t); }`, c.tensor(rows * dim, outHalf), { inputs, ints });
    stats.free();
    return out;
  }

  /** Rows of `[rows, dim]` whose mask is 0 set to 0. */
  maskRows(x, rows, dim, mask) {
    return this.ctx.elementwise('maskrows', 'el(M, e / uDim) != 0.0 ? el(X, e) : 0.0', rows * dim,
      { inputs: { X: x, M: mask }, ints: { uDim: dim } });
  }

  zeros(size) {
    return this.ctx.elementwise('zeros', '0.0', size);
  }

  /** AdaptiveLayerNorm (`adaptive_ln_scale.py`): per-row (`condRows === rows`) or broadcast (`condRows === 1`) cond. */
  adaLN(x, cond, mask, rows, dim, condRows, dimCond, type, prefix, eps = 1e-5) {
    const nx = this.norm(x, rows, dim, type, null, null, eps);
    const nc = this.norm(cond, condRows, dimCond, type, this.W(`${prefix}.norm_cond.weight`), this.W(`${prefix}.norm_cond.bias`, true), eps);
    const g = this.linear(nc, condRows, dimCond, this.W(`${prefix}.to_gamma.0.weight`), dim, this.W(`${prefix}.to_gamma.0.bias`));
    const be = this.linear(nc, condRows, dimCond, this.W(`${prefix}.to_beta.weight`), dim);
    const out = this.ctx.elementwise('adaln', 'ada(e)', rows * dim, {
      inputs: { NX: nx, G: g, BE: be, M: mask }, ints: { uDim: dim, uBroadcast: condRows === 1 ? 1 : 0 },
      pre: `float ada(int e) {
  int r = e / uDim;
  if (el(M, r) == 0.0) return 0.0;
  int ce = uBroadcast == 1 ? e - r * uDim : e;
  return el(NX, e) * sigm(el(G, ce)) + el(BE, ce);
}`,
    });
    [nx, nc, g, be].forEach((t) => t.free());
    return out;
  }

  /** AdaptiveOutputScale. */
  outScale(x, cond, mask, rows, dim, dimCond, prefix) {
    const g = this.linear(cond, rows, dimCond, this.W(`${prefix}.to_adaln_zero_gamma.0.weight`), dim, this.W(`${prefix}.to_adaln_zero_gamma.0.bias`));
    const out = this.ctx.elementwise('outscale', 'el(M, e / uDim) != 0.0 ? el(X, e) * sigm(el(G, e)) : 0.0', rows * dim,
      { inputs: { X: x, G: g, M: mask }, ints: { uDim: dim } });
    g.free();
    return out;
  }

  /** SwiGLU transition (`seq_transition_af3.py`). */
  swiglu(x, mask, rows, dim, factor, prefix) {
    const inner = dim * factor;
    const h = this.linear(x, rows, dim, this.W(`${prefix}.swish_linear.0.weight`), inner * 2);
    const mid = this.ctx.elementwise('swiglu', 'sw(e)', rows * inner, {
      inputs: { H: h }, ints: { uI: inner },
      pre: `float sw(int e) {
  int r = e / uI;
  int d = e - r * uI;
  float g = el(H, r * 2 * uI + uI + d);
  return g * sigm(g) * el(H, r * 2 * uI + d);
}`,
    });
    h.free();
    const y = this.linear(mid, rows, inner, this.W(`${prefix}.linear_out.weight`), dim);
    mid.free();
    const out = this.maskRows(y, rows, dim, mask);
    y.free();
    return out;
  }

  /** PairBiasAttention (`pair_bias_attn.py`); `maskAny`: the token mask has some valid token. */
  attention(nodes, pair, mask, n, D, P, H, usePairBias, useQkln, type, prefix, eps = 1e-5) {
    const c = this.ctx;
    const dh = Math.floor(D / H);
    const W = (k, opt = false) => this.W(`${prefix}.${k}`, opt);
    const normed = this.norm(nodes, n, D, type, W('node_norm.weight'), W('node_norm.bias', true), eps);
    const qkv = this.linear(normed, n, D, W('to_qkv.weight'), 3 * D, W('to_qkv.bias', true));
    const part = (k) => c.elementwise('qkvpart', 'el(QKV, (e / uD) * 3 * uD + uOff + e - (e / uD) * uD)', n * D,
      { inputs: { QKV: qkv }, ints: { uD: D, uOff: k * D } });
    let q = part(0);
    let k = part(1);
    const v = part(2);
    qkv.free();
    if (useQkln) {
      const qn = this.norm(q, n, D, type, W('q_layer_norm.weight'), W('q_layer_norm.bias', true), eps);
      const kn = this.norm(k, n, D, type, W('k_layer_norm.weight'), W('k_layer_norm.bias', true), eps);
      q.free();
      k.free();
      q = qn;
      k = kn;
    }
    const g = this.linear(normed, n, D, W('to_g.weight'), D, W('to_g.bias', true));
    normed.free();
    let bias = null;
    if (usePairBias && pair) {
      const np = this.norm(pair, n * n, P, type, W('pair_norm.weight'), W('pair_norm.bias', true), eps, this.pk('np'));
      bias = this.linear(np, n * n, P, W('to_bias.weight'), H);
      np.free();
    }
    // Scores [H, n, n]: scaled q.k, masked pairs -1e4 when the row has a valid key, the pair bias otherwise added.
    const logits = c.elementwise(`attlogits${bias ? '-b' : ''}`, 'score(e)', H * n * n, {
      inputs: { Q: q, K: k, M: mask, ...(bias ? { AB: bias } : {}) },
      ints: { uN: n, uD: D, uDh: dh, uH: H },
      floats: { uScale: 1 / Math.sqrt(dh) },
      work: dh,
      pre: `float score(int e) {
  int h = e / (uN * uN);
  int rem = e - h * uN * uN;
  int i = rem / uN;
  int j = rem - i * uN;
  int qb = (i * uD + h * uDh) / 4;
  int kb = (j * uD + h * uDh) / 4;
  float s = 0.0;
  for (int d = 0; d < uDh / 4; d++) s += dot(tx(Q, qb + d), tx(K, kb + d));
  s *= uScale;
  if (el(M, i) != 0.0) {
    if (el(M, j) == 0.0) s = -1e4;
    ${bias ? 'else s += el(AB, (i * uN + j) * uH + h);' : ''}
  }
  return s;
}`,
    });
    q.free();
    k.free();
    bias?.free();
    // Row max and 1/sum of exp, per (h, i).
    const stats = c.kernel('attstats', `
uniform sampler2D L; uniform int uN;
vec4 compute(int row) {
  int base = row * uN;
  float m = -3.4e38;
  for (int j = 0; j < uN; j++) m = max(m, el(L, base + j));
  float s = 0.0;
  for (int j = 0; j < uN; j++) s += exp(el(L, base + j) - m);
  return vec4(m, s > 0.0 ? 1.0 / s : 0.0, 0.0, 0.0);
}`, c.tensor(H * n * 4), { inputs: { L: logits }, ints: { uN: n }, work: 2 * n });
    // out[i, h*dh + d] = sigmoid(g) * sum_j p[h, i, j] v[j, h*dh + d], four channels of one head a texel.
    const gated = c.kernel('attpv', `
uniform sampler2D L; uniform sampler2D S; uniform sampler2D V; uniform sampler2D G; uniform int uN; uniform int uD; uniform int uDh;
vec4 compute(int t) {
  int e0 = t * 4;
  int i = e0 / uD;
  int c0 = e0 - i * uD;
  int h = c0 / uDh;
  int row = h * uN + i;
  vec4 st = tx(S, row);
  vec4 acc = vec4(0.0);
  for (int j = 0; j < uN; j++) acc += exp(el(L, row * uN + j) - st.x) * st.y * tx(V, (j * uD + c0) / 4);
  vec4 gv = tx(G, t);
  return acc / (vec4(1.0) + exp(-gv));
}`, c.tensor(n * D), { inputs: { L: logits, S: stats, V: v, G: g }, ints: { uN: n, uD: D, uDh: dh }, work: n });
    [logits, stats, v, g].forEach((t) => t.free());
    const out = this.linear(gated, n, D, W('to_out_node.weight'), D, W('to_out_node.bias', true));
    gated.free();
    return out;
  }

  /** One `MultiheadAttnAndTransition` block (`attn_n_transition.py`). */
  attnBlock(x, pair, cond, mask, n, D, P, H, C, usePairBias, useQkln, type, prefix, eps = 1e-5) {
    const add = (a, b) => this.ctx.elementwise('addmasked', 'el(M, e / uDim) != 0.0 ? el(A, e) + el(B, e) : 0.0', n * D,
      { inputs: { A: a, B: b, M: mask }, ints: { uDim: D } });
    const xa = this.adaLN(x, cond, mask, n, D, n, C, type, `${prefix}.mhba.adaln`, eps);
    const att = this.attention(xa, usePairBias ? pair : null, mask, n, D, P, H, usePairBias, useQkln, type, `${prefix}.mhba.mha`, eps);
    xa.free();
    const att2 = this.outScale(att, cond, mask, n, D, C, `${prefix}.mhba.scale_output`);
    att.free();
    const after = add(x, att2);
    att2.free();
    const xt = this.adaLN(after, cond, mask, n, D, n, C, type, `${prefix}.transition.adaln`, eps);
    const tr = this.swiglu(xt, mask, n, D, 4, `${prefix}.transition.transition`);
    xt.free();
    const tr2 = this.outScale(tr, cond, mask, n, D, C, `${prefix}.transition.scale_output`);
    tr.free();
    const out = add(after, tr2);
    after.free();
    tr2.free();
    return out;
  }

  /** TriangleMultiplicativeUpdate, `outgoing` or `incoming` (`triangular_multiplicative_update.py`): `z` plus it. */
  triMul(z, pairMask, n, cZ, cH, direction, prefix, eps = 1e-5) {
    const c = this.ctx;
    const rows = n * n;
    const W = (k, opt = false) => this.W(`${prefix}.${k}`, opt);
    const zn = this.norm(z, rows, cZ, 'layer_norm', W('layer_norm_in.weight'), W('layer_norm_in.bias', true), eps, this.pk('zn'));
    const gated = (side) => this.gatedLinear(zn, rows, cZ, W(`linear_${side}_p.weight`), W(`linear_${side}_p.bias`, true),
      W(`linear_${side}_g.weight`), W(`linear_${side}_g.bias`, true), cH, pairMask, true);
    let a = gated('a');
    let b = gated('b');
    // x[i, j, c] = sum_k a[i, k, c] b[j, k, c] (outgoing) or a[k, i, c] b[k, j, c] (incoming), never n^3 in memory.
    const out = direction === 'outgoing';
    // Per channel the contraction is a matmul, x_c = A_c B_c^T: a and b go channel-major as packed halves (k padded to
    // 8, i and j to 4), one batched packed matmul over the channels, and the result gathered back to [n^2, c].
    if (!this.noTriMM) {
      const x = this._triMatmul(a, b, n, cH, out);
      a.free();
      b.free();
      return this._triTail(x, zn, rows, cZ, cH, W, eps, z, pairMask);
    }
    // A 2 x 2 block of (i, j) a fragment, into four render targets (one per parity of i and j), then repacked: 4 texel
    // reads per 16 multiply-adds rather than 2 per 4. Each operand advances by a fixed stride per k (c4 texels
    // outgoing, n * c4 incoming): its texture coordinate is stepped rather than divided out on every read.
    if (a.half === 'packed') { const t = this._unpack(a); a.free(); a = t; }
    if (b.half === 'packed') { const t = this._unpack(b); b.free(); b = t; }
    const n2 = Math.ceil(n / 2);
    const planes = [0, 1, 2, 3].map(() => c.tensor(n2 * n2 * cH));
    c.kernel(`tri2-${out ? 'out' : 'in'}`, `
uniform sampler2D A; uniform sampler2D B; uniform int uN; uniform int uN2; uniform int uC;
void compute4(int t, out vec4 r00, out vec4 r01, out vec4 r10, out vec4 r11) {
  int c4 = uC / 4;
  int r = t / c4;
  int cb = t - r * c4;
  int bi = r / uN2;
  int bj = r - bi * uN2;
  int i0 = 2 * bi, j0 = 2 * bj;
  bool i1ok = i0 + 1 < uN, j1ok = j0 + 1 < uN;
  int i1 = i1ok ? i0 + 1 : i0, j1 = j1ok ? j0 + 1 : j0;
  int w = textureSize(A, 0).x;
  ${out
    ? 'int a0 = i0 * uN * c4 + cb, a1 = i1 * uN * c4 + cb, b0 = j0 * uN * c4 + cb, b1 = j1 * uN * c4 + cb; int st = c4;'
    : 'int a0 = i0 * c4 + cb, a1 = i1 * c4 + cb, b0 = j0 * c4 + cb, b1 = j1 * c4 + cb; int st = uN * c4;'}
  ivec2 ac0 = ivec2(a0 % w, a0 / w), ac1 = ivec2(a1 % w, a1 / w);
  ivec2 bc0 = ivec2(b0 % w, b0 / w), bc1 = ivec2(b1 % w, b1 / w);
  ivec2 sd = ivec2(st % w, st / w);
  vec4 s00 = vec4(0.0), s01 = vec4(0.0), s10 = vec4(0.0), s11 = vec4(0.0);
  for (int k = 0; k < uN; k++) {
    vec4 x0 = texelFetch(A, ac0, 0), x1 = texelFetch(A, ac1, 0);
    vec4 y0 = texelFetch(B, bc0, 0), y1 = texelFetch(B, bc1, 0);
    s00 += x0 * y0; s01 += x0 * y1; s10 += x1 * y0; s11 += x1 * y1;
    ac0 += sd; if (ac0.x >= w) ac0 += ivec2(-w, 1);
    ac1 += sd; if (ac1.x >= w) ac1 += ivec2(-w, 1);
    bc0 += sd; if (bc0.x >= w) bc0 += ivec2(-w, 1);
    bc1 += sd; if (bc1.x >= w) bc1 += ivec2(-w, 1);
  }
  r00 = s00;
  r01 = j1ok ? s01 : vec4(0.0);
  r10 = i1ok ? s10 : vec4(0.0);
  r11 = i1ok && j1ok ? s11 : vec4(0.0);
}`, planes, { inputs: { A: a, B: b }, ints: { uN: n, uN2: n2, uC: cH }, work: n * 2 });
    const x = c.kernel('tri2-repack', `
uniform sampler2D P0; uniform sampler2D P1; uniform sampler2D P2; uniform sampler2D P3; uniform int uN; uniform int uN2; uniform int uC;
vec4 compute(int t) {
  int c4 = uC / 4;
  int r = t / c4;
  int cb = t - r * c4;
  int i = r / uN;
  int j = r - i * uN;
  int qt = ((i >> 1) * uN2 + (j >> 1)) * c4 + cb;
  int p = (i & 1) * 2 + (j & 1);
  return p == 0 ? tx(P0, qt) : p == 1 ? tx(P1, qt) : p == 2 ? tx(P2, qt) : tx(P3, qt);
}`, c.tensor(rows * cH), { inputs: { P0: planes[0], P1: planes[1], P2: planes[2], P3: planes[3] }, ints: { uN: n, uN2: n2, uC: cH } });
    planes.forEach((t) => t.free());
    a.free();
    b.free();
    return this._triTail(x, zn, rows, cZ, cH, W, eps, z, pairMask);
  }

  /** The triangle update after the contraction: norm, the output projection, gated by the input's own projection. */
  _triTail(x, zn, rows, cZ, cH, W, eps, z, pairMask) {
    const xn = this.norm(x, rows, cH, 'layer_norm', W('layer_norm_out.weight'), W('layer_norm_out.bias', true), eps, this.pk('xn'));
    x.free();
    const gp = this.linear(zn, rows, cZ, W('linear_g.weight'), cZ, W('linear_g.bias', true), null, true);
    zn.free();
    // z + pairMask * linear_z(xn) * sigmoid(gp), in the matmul's epilogue.
    const res = this.linear(xn, rows, cH, W('linear_z.weight'), cZ, W('linear_z.bias', true), null, false, { z, mask: pairMask, gate: gp });
    xn.free();
    gp.free();
    return res;
  }

  /** x[i, j, c] = sum_k a[i, k, c] b[j, k, c] (outgoing) or a[k, i, c] b[k, j, c] (incoming), as one batched matmul. */
  _triMatmul(a, b, n, cH, outgoing) {
    const c = this.ctx;
    const Ip = Math.ceil(n / 4) * 4;
    const Kp = Math.ceil(n / 8) * 8;
    const ib = Kp / 8;
    const rd = (t) => (t.half === 'packed' ? 'elp' : 'el');
    const cm = (src, rowsPer) => c.elementwise(`tri-cm-${outgoing ? 'out' : 'in'}${src.half === 'packed' ? '-p' : ''}`, 'cmv(e)', cH * rowsPer * Kp, {
      inputs: { S: src }, ints: { uN: n, uC: cH, uR: rowsPer, uK: Kp }, half: 'packed', rowTexels: ib,
      pre: `float cmv(int e) {
  int r = e / uK;
  int k = e - r * uK;
  int ch = r / uR;
  int i = r - ch * uR;
  if (i >= uN || k >= uN) return 0.0;
  return ${outgoing ? `${rd(src)}(S, (i * uN + k) * uC + ch)` : `${rd(src)}(S, (k * uN + i) * uC + ch)`};
}`,
    });
    // j padded to 8: a fragment writes 4 rows (i) x 8 columns (j) as packed halves (the 4x8 block, 64 bytes a pixel).
    const Jp = Math.ceil(n / 8) * 8;
    // (4 rows x 8 columns as packed halves, j padded to 8, measured slower with the packed gather: M2.)
    const A = cm(a, Ip);
    const B = cm(b, Ip);
    const rowsAll = cH * Ip;
    const rq = rowsAll / 4;
    const qs = [0, 1, 2, 3].map(() => c.tensor(rq * Ip));
    c.kernel('bmm4x4pk', `
uniform highp usampler2D X; uniform highp usampler2D Wt; uniform int uIn; uniform int uOut; uniform int uIp;
void unpack8(uvec4 u, out vec4 lo, out vec4 hi) { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
void compute4(int t, out vec4 r0, out vec4 r1, out vec4 r2, out vec4 r3) {
  int e0 = t * 4;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int ib = uIn / 8;
  int ch = (q * 4) / uIp;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wt, 0).x;
  ivec2 xc[4];
  ivec2 wc[4];
  for (int i = 0; i < 4; i++) {
    int xt = (q * 4 + i) * ib;
    xc[i] = ivec2(xt % xw, xt / xw);
    int wt = (ch * uOut + o0 + i) * ib;
    wc[i] = ivec2(wt % ww, wt / ww);
  }
  vec4 a[4];
  for (int i = 0; i < 4; i++) a[i] = vec4(0.0);
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    vec4 l0, h0, l1, h1, l2, h2, l3, h3;
    unpack8(texelFetch(Wt, wc[0] + dk, 0), l0, h0);
    unpack8(texelFetch(Wt, wc[1] + dk, 0), l1, h1);
    unpack8(texelFetch(Wt, wc[2] + dk, 0), l2, h2);
    unpack8(texelFetch(Wt, wc[3] + dk, 0), l3, h3);
    mat4 wl = mat4(l0, l1, l2, l3), wh = mat4(h0, h1, h2, h3);
    for (int i = 0; i < 4; i++) {
      vec4 xl, xh;
      unpack8(texelFetch(X, xc[i] + dk, 0), xl, xh);
      a[i] += xl * wl + xh * wh;
    }
  }
  r0 = a[0]; r1 = a[1]; r2 = a[2]; r3 = a[3];
}`, qs, { inputs: { X: A, Wt: B }, ints: { uIn: Kp, uOut: Ip, uIp: Ip }, work: Kp * 2 });
    A.free();
    B.free();
    // Back to [n^2, c]: element (i, j, ch) is row ch * Ip + i, column j of the batched result (rows in planes of 4).
    const x = c.elementwise('tri-gather', 'gv(e)', n * n * cH, {
      inputs: { Q0: qs[0], Q1: qs[1], Q2: qs[2], Q3: qs[3] }, ints: { uN: n, uC: cH, uIp: Ip },
      pre: `float gv(int e) {
  int r = e / uC;
  int ch = e - r * uC;
  int i = r / uN;
  int j = r - i * uN;
  int R = ch * uIp + i;
  int qe = (R >> 2) * uIp + j;
  int p = R & 3;
  return p == 0 ? el(Q0, qe) : p == 1 ? el(Q1, qe) : p == 2 ? el(Q2, qe) : el(Q3, qe);
}`,
    });
    qs.forEach((t) => t.free());
    return x;
  }

  /** PairReprUpdate (`pair_update.py`), hadamard pair write. */
  pairUpdate(x, z, mask, pairMask, n, D, P, TRI, prefix, eps = 1e-5) {
    const c = this.ctx;
    const rows = n * n;
    const W = (k) => this.W(`${prefix}.${k}`);
    const xm = this.maskRows(x, n, D, mask);
    const xn = this.norm(xm, n, D, 'layer_norm', W('layer_norm_in.weight'), W('layer_norm_in.bias'), eps);
    xm.free();
    const pij = this.linear(xn, n, D, W('linear_x.weight'), 2 * P);
    xn.free();
    // pi from j, pj from i (pair_update.py's broadcast).
    let zz = c.elementwise('pairwrite', 'pw(e)', rows * P, {
      inputs: { Z: z, PIJ: pij, PM: pairMask }, ints: { uN: n, uP: P },
      pre: `float pw(int e) {
  int r = e / uP;
  if (el(PM, r) == 0.0) return 0.0;
  int d = e - r * uP;
  int i = r / uN;
  int j = r - i * uN;
  return el(Z, e) + el(PIJ, j * 2 * uP + d) * el(PIJ, i * 2 * uP + uP + d);
}`,
    });
    pij.free();
    for (const dir of ['outgoing', 'incoming']) {
      const z2 = this.triMul(zz, pairMask, n, P, TRI, dir, `${prefix}.tri_mult_${dir === 'outgoing' ? 'out' : 'in'}`, eps);
      zz.free();
      zz = z2;
    }
    const tp = `${prefix}.transition_out`;
    const zt = this.norm(zz, rows, P, 'layer_norm', this.W(`${tp}.layer_norm.weight`), this.W(`${tp}.layer_norm.bias`), eps, this.pk('zt'));
    const l1B = this.W(`${tp}.linear_1.bias`);
    const hidden = l1B.size;
    const h = this.linear(zt, rows, P, this.W(`${tp}.linear_1.weight`), hidden, l1B, 'relu', this.pk('h'));
    zt.free();
    const out = this.linear(h, rows, hidden, this.W(`${tp}.linear_2.weight`), P, this.W(`${tp}.linear_2.bias`), null, false, { z: zz, mask: pairMask });
    h.free();
    zz.free();
    return out;
  }

  _masks(mask, n) {
    const m = this.ctx.upload(Float32Array.from(mask, (v) => (v ? 1 : 0)));
    const pm = this.ctx.elementwise('pairmask', 'el(M, e / uN) * el(M, e - (e / uN) * uN)', n * n, { inputs: { M: m }, ints: { uN: n } });
    return { m, pm };
  }

  // ---- networks -----------------------------------------------------------------------------------------------

  /** As `forwardTrunkCpu(n, f, mask, weights, cfg, nRecycle, opts)`; returns `{bb_ca, local_latents, logits, insertion}`. */
  async forwardTrunk(n, f, mask, cfg, nRecycle = 1, opts = {}) {
    const c = this.ctx;
    const D = cfg.token_dim ?? 768;
    const P = cfg.pair_repr_dim ?? 384;
    const C = cfg.dim_cond ?? 256;
    const PC = cfg.pair_cond_factory_dim ?? 256;
    const TRI = cfg.tri_mult_c ?? 256;
    const H = cfg.nheads ?? 12;
    const norm = 'rmsnorm_torch';
    const { m, pm } = this._masks(mask, n);
    const condC = (feats) => {
      const ft = c.upload(feats);
      let x = this.linear(ft, n, feats.length / n, this.W('nn.cond_factory.linear_out.weight'), C);
      ft.free();
      for (const k of [1, 2]) {
        const y = this.swiglu(x, m, n, C, 2, `nn.transition_c_${k}`);
        x.free();
        x = y;
      }
      return x;
    };
    const c0 = condC(f.cond0);
    const c1 = nRecycle > 1 ? condC(f.cond1 ?? f.cond0) : c0;

    const seqT = c.upload(f.seq);
    const sLin = this.linear(seqT, n, f.seq.length / n, this.W('nn.init_repr_factory.linear_out.weight'), D);
    seqT.free();
    const sInit = this.maskRows(sLin, n, D, m);
    sLin.free();

    // The pair features (107 wide) padded with zeros to 128, feature and weight alike, take the blocked matmul.
    const pf = f.pair.length / (n * n);
    const pw = Math.ceil(pf / 32) * 32;
    const padded = new Float32Array(n * n * pw);
    for (let r = 0; r < n * n; r += 1) padded.set(f.pair.subarray(r * pf, (r + 1) * pf), r * pw);
    const pairT = c.upload(padded);
    const zLin = this.linear(pairT, n * n, pw, this.padded('nn.pair_repr_builder.init_repr_factory.linear_out.weight', P, pf, pw), P);
    pairT.free();
    const zN = this.norm(zLin, n * n, P, norm, this.W('nn.pair_repr_builder.init_repr_factory.ln_out.weight'));
    zLin.free();
    const zMasked = this.maskRows(zN, n * n, P, pm);
    zN.free();
    const pcT = c.upload(f.pairCond);
    const zcLin = this.linear(pcT, 1, f.pairCond.length, this.W('nn.pair_repr_builder.cond_factory.linear_out.weight'), PC);
    pcT.free();
    const zCond = this.norm(zcLin, 1, PC, norm, this.W('nn.pair_repr_builder.cond_factory.ln_out.weight'));
    zcLin.free();
    const zInit = this.adaLN(zMasked, zCond, pm, n * n, P, 1, PC, norm, 'nn.pair_repr_builder.adaln');
    zMasked.free();
    zCond.free();

    let s = this.zeros(n * D);
    let z = this.zeros(n * n * P);
    for (let cycle = 0; cycle < Math.max(1, nRecycle); cycle += 1) {
      const sn = this.norm(s, n, D, norm, this.W('nn.s_cycle_norm.weight'));
      const sd = this.linear(sn, n, D, this.W('nn.s_cycle_proj.weight'), D);
      sn.free();
      const zn = this.norm(z, n * n, P, norm, this.W('nn.z_cycle_norm.weight'), null, 1e-5, this.pk('zc'));
      const zd = this.linear(zn, n * n, P, this.W('nn.z_cycle_proj.weight'), P, null, null, true);
      zn.free();
      s.free();
      z.free();
      s = c.elementwise('add', 'el(A, e) + el(B, e)', n * D, { inputs: { A: sInit, B: sd } });
      z = c.elementwise('add', 'el(A, e) + el(B, e)', n * n * P, { inputs: { A: zInit, B: zd } });
      sd.free();
      zd.free();
      const cc = cycle === 0 ? c0 : c1;
      for (let i = 0; i < cfg.nlayers; i += 1) {
        const s2 = this.attnBlock(s, z, cc, m, n, D, P, H, C, true, true, norm, `nn.transformer_layers.${i}`);
        s.free();
        s = s2;
        if (cfg.update_pair_repr && i < cfg.nlayers - 1 && i % (cfg.update_pair_repr_every_n ?? 2) === 0) {
          const z2 = this.pairUpdate(s, z, m, pm, n, D, P, TRI, `nn.pair_update_layers.${i}`);
          z.free();
          z = z2;
        }
      }
    }
    const head = async (name, out) => {
      const h = this.norm(s, n, D, norm, this.W(`nn.${name}.0.weight`));
      const y = this.linear(h, n, D, this.W(`nn.${name}.1.weight`), out);
      h.free();
      const v = Float32Array.from(await c.readAsync(y));
      y.free();
      return v;
    };
    const res = {
      bb_ca: await head('ca_linear', 3),
      local_latents: await head('local_latents_linear', 32),
      logits: await head('sequence_readout', 147),
      insertion: null,
    };
    if (opts.insertion && this.W('nn.insertion_q_proj.weight', true)) {
      const q = this.linear(s, n, D, this.W('nn.insertion_q_proj.weight'), D);
      const k = this.linear(s, n, D, this.W('nn.insertion_k_proj.weight'), D);
      const qk = this.linear(q, n, D, k, n);
      const scaled = c.elementwise('scale', 'el(X, e) * uS', n * n, { inputs: { X: qk }, floats: { uS: 1 / Math.sqrt(D) } });
      res.insertion = Float32Array.from(await c.readAsync(scaled));
      [q, k, qk, scaled].forEach((t) => t.free());
    }
    [s, z, sInit, zInit, m, pm, c0].forEach((t) => t.free());
    if (c1 !== c0) c1.free();
    return res;
  }

  /** The autoencoder transformer stack shared by decoder and encoder: `s` after its layers. */
  _aeStack(feats, fdim, attn, n, cfg, prefix) {
    const c = this.ctx;
    const D = cfg.token_dim;
    const C = cfg.dim_cond;
    const m = c.upload(Float32Array.from(attn, (v) => (v ? 1 : 0)));
    // c: an empty cond factory, zeros, through two bias-free SwiGLU transitions.
    let cond = this.zeros(n * C);
    for (const k of [1, 2]) {
      const y = this.swiglu(cond, m, n, C, 2, `${prefix}.transition_c_${k}`);
      cond.free();
      cond = y;
    }
    const ft = c.upload(feats);
    const sLin = this.linear(ft, n, fdim, this.W(`${prefix}.init_repr_factory.linear_out.weight`), D);
    ft.free();
    let s = this.maskRows(sLin, n, D, m);
    sLin.free();
    for (let l = 0; l < cfg.nlayers; l += 1) {
      const s2 = this.attnBlock(s, null, cond, m, n, D, cfg.pair_repr_dim, cfg.nheads, C, false, cfg.use_qkln ?? true,
        'layer_norm', `${prefix}.transformer_layers.${l}`);
      s.free();
      s = s2;
    }
    cond.free();
    m.free();
    return s;
  }

  async _lnHead(s, n, D, prefix, out) {
    const h = this.norm(s, n, D, 'layer_norm', this.W(`${prefix}.0.weight`), this.W(`${prefix}.0.bias`));
    const y = this.linear(h, n, D, this.W(`${prefix}.1.weight`), out);
    h.free();
    const v = Float32Array.from(await this.ctx.readAsync(y));
    y.free();
    return v;
  }

  /** As `decodeLatentsCpu(batch, {n, caNm, z, seq, weights, cfg})`. */
  async decode(batch, { n, caNm, z, seq, cfg }) {
    const PREFIX = 'autoencoder.decoder';
    const nTypes = cfg.n_residue_types ?? 147;
    const { attn } = aeMasks(batch, n);
    const { data: feats, dim: fdim } = decoderSeqFeatures(batch, { n, caNm, z, seq, weights: this.weights, cfg });
    const s = this._aeStack(feats, fdim, attn, n, cfg, PREFIX);
    const D = cfg.token_dim;
    const logits = await this._lnHead(s, n, D, `${PREFIX}.logit_linear`, nTypes);
    const coorsFlat = await this._lnHead(s, n, D, `${PREFIX}.struct_linear`, 111);
    const presence = await this._lnHead(s, n, D, `${PREFIX}.atom_presence_linear`, 37);
    s.free();
    return decoderPostprocess(batch, { n, caNm, seq, logits, coorsFlat, presence, nTypes });
  }

  /** As `encodeLatentsCpu(batch, {n, weights, cfg, rng, deterministic})`. */
  async encode(batch, { n, cfg, rng, deterministic = false }) {
    const PREFIX = 'autoencoder.encoder';
    const latentDim = cfg.latent_z_dim ?? 32;
    const { attn } = aeMasks(batch, n);
    const { data: feats, dim: fdim } = encoderSeqFeatures(batch, { n, weights: this.weights, cfg });
    const s = this._aeStack(feats, fdim, attn, n, cfg, PREFIX);
    const flatOut = await this._lnHead(s, n, cfg.token_dim, `${PREFIX}.latent_decoder_mean_n_log_scale`, 2 * latentDim);
    s.free();
    return encoderPostprocess(batch, { n, flatOut, rng, latentDim, deterministic });
  }

  /** The `kernels` that make `generateRfd4pCpu` run on this model. */
  kernels() {
    return {
      forwardTrunk: (n, f, mask, weights, cfg, nRecycle, opts) => this.forwardTrunk(n, f, mask, cfg, nRecycle, opts),
      decode: (batch, o) => this.decode(batch, o),
      encode: (batch, o) => this.encode(batch, o),
    };
  }
}
