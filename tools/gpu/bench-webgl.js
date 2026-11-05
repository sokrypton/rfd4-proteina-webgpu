/**
 * WebGL2 matmul microbenchmark: `Rfd4pModelGL.linear` on one shape, GFLOP/s.
 *
 *   node tools/gpu-chrome.mjs tools/gpu/bench-webgl.js [--rows=43264 --in=384 --out=256 --half=1]
 */

import { Rfd4pModelGL } from '../../webgl/rfd4p/model.js';

export async function main(_device, args) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const rows = Number(opt.rows ?? 43264);
  const inD = Number(opt.in ?? 384);
  const outD = Number(opt.out ?? 256);
  const xHalf = opt.half !== '0';
  const rand = (n) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);
  const weights = { w: { data: rand(outD * inD) }, b: { data: rand(outD) } };
  const model = new Rfd4pModelGL(weights);
  const c = model.ctx;
  if (opt.band) c.bandWork = Number(opt.band);
  const x32 = c.upload(rand(rows * inD));
  const x = xHalf ? c.elementwise('copyh', 'el(X, e)', rows * inD, { inputs: { X: x32 }, half: true }) : x32;
  const w = model.W('w');
  const b = model.W('b');
  const results = {};
  for (const variant of (opt.variants ?? 'default').split(',')) {
    model.variant = variant; // a hook for trying kernel variants
    const run = () => model.linear(x, rows, inD, w, outD, b);
    (await c.readAsync(run(), 4));
    const reps = Number(opt.reps ?? 5);
    const t0 = performance.now();
    let y = null;
    for (let i = 0; i < reps; i += 1) { y?.free(); y = run(); }
    await c.readAsync(y, 4);
    const ms = (performance.now() - t0) / reps;
    c.profile = new Map();
    c.lastT = performance.now();
    run().free();
    const prof = Object.fromEntries([...c.profile].map(([k, v]) => [k, Number(v.toFixed(1))]));
    c.profile = null;
    results[variant] = { ms: Number(ms.toFixed(1)), gflops: Number(((2 * rows * inD * outD) / ms / 1e6).toFixed(0)), prof };
  }
  if (opt.packed === '1') results.packed = benchPacked(c, x32, w, b, rows, inD, outD, Number(opt.reps ?? 5));
  if (opt.packed === '1') results.packed4x8 = benchPacked48(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5));
  if (opt.probe === '1') Object.assign(results, benchProbe(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5)));
  model.destroy();
  return { rows, inD, outD, xHalf, renderer: c.renderer, results };
}

/** Packed-half operands (RGBA32UI, 8 values a texel): a 4 x 4 block matmul reading half as many texels. */
function benchPacked(c, x32, w32, b, rows, inD, outD, reps) {
  const pack = (src, size) => c.kernel('pack8', `
uniform sampler2D S;
uvec4 compute(int t) {
  vec4 a = tx(S, 2 * t), b = tx(S, 2 * t + 1);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}`, c.tensor(size, 'packed'), { inputs: { S: src } });
  const wf = c.elementwise('copyf', 'el(X, e)', outD * inD, { inputs: { X: w32 } });
  const xp = pack(x32, rows * inD);
  const wp = pack(wf, outD * inD);
  const rq = Math.ceil(rows / 4);
  const run = () => {
    const qs = [0, 1, 2, 3].map(() => c.tensor(rq * outD));
    c.kernel('linear4x4pk', `
uniform highp usampler2D X; uniform highp usampler2D Wt; uniform int uIn; uniform int uOut; uniform int uRows;
void unpack8(uvec4 u, out vec4 lo, out vec4 hi) { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
void compute4(int t, out vec4 r0, out vec4 r1, out vec4 r2, out vec4 r3) {
  int e0 = t * 4;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int ib = uIn / 8;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wt, 0).x;
  ivec2 xc[4];
  for (int i = 0; i < 4; i++) { int r = min(q * 4 + i, uRows - 1); int xt = r * ib; xc[i] = ivec2(xt % xw, xt / xw); }
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
  r0 = a[0]; r1 = a[1]; r2 = a[2]; r3 = a[3];
}`, qs, { inputs: { X: xp, Wt: wp }, ints: { uIn: inD, uOut: outD, uRows: rows }, work: inD * 4 });
    return qs;
  };
  const ms = timeIt(c, run, reps);
  return { ms: Number(ms.toFixed(1)), gflops: Number(((2 * rows * inD * outD) / ms / 1e6).toFixed(0)), note: 'kernel only, no repack' };
}

/** 4 rows x 8 columns a fragment, outputs packed halves (one RGBA32UI target a row, 8 values): 32 outputs a pixel. */
function benchPacked48(c, x32, w32, rows, inD, outD, reps) {
  const pack = (src, size) => c.kernel('pack8', `
uniform sampler2D S;
uvec4 compute(int t) {
  vec4 a = tx(S, 2 * t), b = tx(S, 2 * t + 1);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}`, c.tensor(size, 'packed'), { inputs: { S: src } });
  const wf = c.elementwise('copyf', 'el(X, e)', outD * inD, { inputs: { X: w32 } });
  const xp = pack(x32, rows * inD);
  const wp = pack(wf, outD * inD);
  const rq = Math.ceil(rows / 4);
  const run = () => {
    const qs = [0, 1, 2, 3].map(() => c.tensor(rq * outD, 'packed'));
    c.kernel('linear4x8pk', `
uniform highp usampler2D X; uniform highp usampler2D Wt; uniform int uIn; uniform int uOut; uniform int uRows;
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
  for (int i = 0; i < 4; i++) { int r = min(q * 4 + i, uRows - 1); int xt = r * ib; xc[i] = ivec2(xt % xw, xt / xw); }
  ivec2 wc[8];
  for (int i = 0; i < 8; i++) { int wt = (o0 + i) * ib; wc[i] = ivec2(wt % ww, wt / ww); }
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
  r0 = pk(a[0], b[0]); r1 = pk(a[1], b[1]); r2 = pk(a[2], b[2]); r3 = pk(a[3], b[3]);
}`, qs, { inputs: { X: xp, Wt: wp }, ints: { uIn: inD, uOut: outD, uRows: rows }, work: inD * 4 });
    return qs;
  };
  const ms = timeIt(c, run, reps);
  return { ms: Number(ms.toFixed(1)), gflops: Number(((2 * rows * inD * outD) / ms / 1e6).toFixed(0)), note: 'kernel only, packed outputs' };
}

/** ms per call of `run` (which returns tensors to free): warmed, then synced by a tiny draw read back (GL is in order). */
function timeIt(c, run, reps) {
  const sync = () => { const t = c.elementwise('sync', '1.0', 4); c.read(t, 4); t.free(); };
  run().forEach((t) => t.free());
  sync();
  const t0 = performance.now();
  for (let i = 0; i < reps; i += 1) run().forEach((t) => t.free());
  sync();
  return (performance.now() - t0) / reps;
}

/** Where the packed 4x4 matmul's time goes: the same loop with fetches only (no math) and math only (no fetches). */
function benchProbe(c, x32, w32, rows, inD, outD, reps) {
  const pack = (src, size) => c.kernel('probe-pack8', `
uniform sampler2D S;
uvec4 compute(int t) {
  vec4 a = tx(S, 2 * t), b = tx(S, 2 * t + 1);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}`, c.tensor(size, 'packed'), { inputs: { S: src } });
  const wf = c.elementwise('probe-copyf', 'el(X, e)', outD * inD, { inputs: { X: w32 } });
  const xp = pack(x32, rows * inD);
  const wp = pack(wf, outD * inD);
  const rq = Math.ceil(rows / 4);
  const body = (mode) => `
uniform highp usampler2D X; uniform highp usampler2D Wt; uniform int uIn; uniform int uOut; uniform int uRows;
void unpack8(uvec4 u, out vec4 lo, out vec4 hi) { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
void compute4(int t, out vec4 r0, out vec4 r1, out vec4 r2, out vec4 r3) {
  int e0 = t * 4;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int ib = uIn / 8;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wt, 0).x;
  ivec2 xc[4];
  for (int i = 0; i < 4; i++) { int r = min(q * 4 + i, uRows - 1); int xt = r * ib; xc[i] = ivec2(xt % xw, xt / xw); }
  ivec2 wc[4];
  for (int i = 0; i < 4; i++) { int wt = (o0 + i) * ib; wc[i] = ivec2(wt % ww, wt / ww); }
  vec4 a[4];
  for (int i = 0; i < 4; i++) a[i] = vec4(0.0);
  uvec4 acc = uvec4(0u);
  ${mode === 'fetch' ? `for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    acc ^= texelFetch(Wt, wc[0] + dk, 0) ^ texelFetch(Wt, wc[1] + dk, 0) ^ texelFetch(Wt, wc[2] + dk, 0) ^ texelFetch(Wt, wc[3] + dk, 0);
    for (int i = 0; i < 4; i++) acc ^= texelFetch(X, xc[i] + dk, 0);
  }
  a[0] = vec4(acc);` : `uvec4 u0 = texelFetch(Wt, wc[0], 0), u1 = texelFetch(Wt, wc[1], 0), u2 = texelFetch(Wt, wc[2], 0), u3 = texelFetch(Wt, wc[3], 0);
  uvec4 x0 = texelFetch(X, xc[0], 0);
  for (int k = 0; k < ib; k++) {
    vec4 l0, h0, l1, h1, l2, h2, l3, h3;
    unpack8(u0 + uint(k), l0, h0); unpack8(u1 + uint(k), l1, h1); unpack8(u2 + uint(k), l2, h2); unpack8(u3 + uint(k), l3, h3);
    mat4 wl = mat4(l0, l1, l2, l3), wh = mat4(h0, h1, h2, h3);
    for (int i = 0; i < 4; i++) {
      vec4 xl, xh;
      unpack8(x0 + uint(k + i), xl, xh);
      a[i] += xl * wl + xh * wh;
    }
  }`}
  r0 = a[0]; r1 = a[1]; r2 = a[2]; r3 = a[3];
}`;
  const out = {};
  for (const mode of ['fetch', 'math']) {
    const run = () => {
      const qs = [0, 1, 2, 3].map(() => c.tensor(rq * outD));
      c.kernel(`probe-${mode}`, body(mode), qs, { inputs: { X: xp, Wt: wp }, ints: { uIn: inD, uOut: outD, uRows: rows }, work: inD * 2 });
      return qs;
    };
    out[`probe-${mode}`] = { ms: Number(timeIt(c, run, reps).toFixed(1)) };
  }
  return out;
}
