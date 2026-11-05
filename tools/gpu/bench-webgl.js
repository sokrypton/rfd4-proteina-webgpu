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
  if (opt.packed === '1' || opt.p88 === '1') results.packed8x8 = benchPacked88(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5));
  if (opt.p88 === '1') results.packed4x8 = benchPacked48(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5));
  if (opt.p88 === '1') results.packed4x8small = benchPacked48(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5), true);
  if (opt.outw) for (const ow of opt.outw.split(',').map(Number)) results[`packed4x8-w${ow}`] = benchPacked48(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5), false, ow);
  if (opt.quad === '1') results.quad4x8 = benchQuad48(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5));
  if (opt.quad === '1') results.packed4x8 = benchPacked48(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5));
  if (opt.p88 === '1') results.packed4x8named = benchPacked48(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5), 'named');
  if (opt.probe === '1') Object.assign(results, benchProbe(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5)));
  if (opt.layout === '1') Object.assign(results, benchLayout(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5)));
  if (opt.int8 === '1') results.int8x4x4 = benchInt8(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5));
  if (opt.tf === '1') for (const [rb, cb] of [[4, 8], [8, 4], [8, 8], [4, 16]]) results[`tf${rb}x${cb}`] = benchTF(c, x32, w, rows, inD, outD, Number(opt.reps ?? 5), rb, cb);
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
function benchPacked48(c, x32, w32, rows, inD, outD, reps, small = false, outW = 0) {
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
    // `small`: the same loop, half the output bytes (a probe of the output's cost; not a matmul).
    const qs = [0, 1, 2, 3].map(() => (small === true ? c.tensor(rq * outD / 2, true) : c.tensor(rq * outD, 'packed', 0, outW)));
    c.kernel(small === 'named' ? 'linear4x8pk-named' : small ? 'linear4x8pk-probe-small' : 'linear4x8pk', `
uniform highp usampler2D X; uniform highp usampler2D Wt; uniform int uIn; uniform int uOut; uniform int uRows;
void unpack8(uvec4 u, out vec4 lo, out vec4 hi) { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
uvec4 pk(vec4 a, vec4 b) { return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw)); }
void compute4(int t, ${small === true ? 'out vec4 r0, out vec4 r1, out vec4 r2, out vec4 r3' : 'out uvec4 r0, out uvec4 r1, out uvec4 r2, out uvec4 r3'}) {
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
    ${small === 'named' ? `vec4 l0, h0, l1, h1, l2, h2, l3, h3;
    unpack8(texelFetch(Wt, wc[0] + dk, 0), l0, h0); unpack8(texelFetch(Wt, wc[1] + dk, 0), l1, h1);
    unpack8(texelFetch(Wt, wc[2] + dk, 0), l2, h2); unpack8(texelFetch(Wt, wc[3] + dk, 0), l3, h3);
    mat4 wla = mat4(l0, l1, l2, l3), wha = mat4(h0, h1, h2, h3);
    unpack8(texelFetch(Wt, wc[4] + dk, 0), l0, h0); unpack8(texelFetch(Wt, wc[5] + dk, 0), l1, h1);
    unpack8(texelFetch(Wt, wc[6] + dk, 0), l2, h2); unpack8(texelFetch(Wt, wc[7] + dk, 0), l3, h3);
    mat4 wlb = mat4(l0, l1, l2, l3), whb = mat4(h0, h1, h2, h3);` : `vec4 l[8], h[8];
    for (int j = 0; j < 8; j++) unpack8(texelFetch(Wt, wc[j] + dk, 0), l[j], h[j]);
    mat4 wla = mat4(l[0], l[1], l[2], l[3]), wha = mat4(h[0], h[1], h[2], h[3]);
    mat4 wlb = mat4(l[4], l[5], l[6], l[7]), whb = mat4(h[4], h[5], h[6], h[7]);`}
    for (int i = 0; i < 4; i++) {
      vec4 xl, xh;
      unpack8(texelFetch(X, xc[i] + dk, 0), xl, xh);
      a[i] += xl * wla + xh * wha;
      b[i] += xl * wlb + xh * whb;
    }
  }
  ${small === true ? 'r0 = a[0] * b[0]; r1 = a[1] * b[1]; r2 = a[2] * b[2]; r3 = a[3] * b[3];' : 'r0 = pk(a[0], b[0]); r1 = pk(a[1], b[1]); r2 = pk(a[2], b[2]); r3 = pk(a[3], b[3]);'}
}`, qs, { inputs: { X: xp, Wt: wp }, ints: { uIn: inD, uOut: outD, uRows: rows }, work: inD * 4 });
    return qs;
  };
  const ms = timeIt(c, run, reps);
  return { ms: Number(ms.toFixed(1)), gflops: Number(((2 * rows * inD * outD) / ms / 1e6).toFixed(0)), note: 'kernel only, packed outputs' };
}

/**
 * ms per call of `run` (which returns tensors to free): warmed, then synced by a tiny draw read back (GL is in order).
 * Each call's outputs are freed after the next call, so consecutive calls draw into different textures: redrawn into
 * the same ones, the tiler's hidden-surface removal can cull all but the last draw of a render pass.
 */
function timeIt(c, run, reps) {
  const sync = () => { const t = c.elementwise('sync', '1.0', 4); c.read(t, 4); t.free(); };
  let prev = run();
  sync();
  const t0 = performance.now();
  for (let i = 0; i < reps; i += 1) {
    const cur = run();
    prev.forEach((t) => t.free());
    prev = cur;
  }
  sync();
  const ms = (performance.now() - t0) / reps;
  prev.forEach((t) => t.free());
  return ms;
}

/**
 * The 4 x 8 packed matmul with each 2 x 2 quad of fragments sharing its fetches: horizontal neighbours (the same 4 rows,
 * column blocks 2c and 2c + 1) fetch 2 rows of x each and swap them by dFdx, vertical neighbours (the same columns)
 * fetch 4 of the 8 weight rows each and swap them by dFdy: 6 texel reads per 256 multiply-adds instead of 12.
 */
function benchQuad48(c, x32, w32, rows, inD, outD, reps) {
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
  let check = null;
  const run = () => {
    const qs = [0, 1, 2, 3].map(() => c.tensor(rq * outD, 'packed'));
    c.kernel('linear4x8pk-quad', `
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
  bool right = (int(gl_FragCoord.x) & 1) == 1, bottom = (int(gl_FragCoord.y) & 1) == 1;
  float sx = right ? -1.0 : 1.0, sy = bottom ? -1.0 : 1.0;
  // This fragment's 2 rows (right: rows 2, 3) and 4 weight rows (bottom: columns 4 to 7).
  ivec2 xc0, xc1;
  { int r = min(q * 4 + (right ? 2 : 0), uRows - 1); int xt = r * ib; xc0 = ivec2(xt % xw, xt / xw);
    r = min(q * 4 + (right ? 3 : 1), uRows - 1); xt = r * ib; xc1 = ivec2(xt % xw, xt / xw); }
  ivec2 wc[4];
  for (int j = 0; j < 4; j++) { int wt = min(o0 + (bottom ? 4 : 0) + j, uOut - 1) * ib; wc[j] = ivec2(wt % ww, wt / ww); }
  vec4 a[4], b[4];
  for (int i = 0; i < 4; i++) { a[i] = vec4(0.0); b[i] = vec4(0.0); }
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    vec4 m0l, m0h, m1l, m1h;
    unpack8(texelFetch(X, xc0 + dk, 0), m0l, m0h);
    unpack8(texelFetch(X, xc1 + dk, 0), m1l, m1h);
    vec4 o0l = m0l + sx * dFdx(m0l), o0h = m0h + sx * dFdx(m0h), o1l = m1l + sx * dFdx(m1l), o1h = m1h + sx * dFdx(m1h);
    vec4 xl[4], xh[4];
    xl[0] = right ? o0l : m0l; xh[0] = right ? o0h : m0h; xl[1] = right ? o1l : m1l; xh[1] = right ? o1h : m1h;
    xl[2] = right ? m0l : o0l; xh[2] = right ? m0h : o0h; xl[3] = right ? m1l : o1l; xh[3] = right ? m1h : o1h;
    vec4 l0, h0, l1, h1, l2, h2, l3, h3;
    unpack8(texelFetch(Wt, wc[0] + dk, 0), l0, h0); unpack8(texelFetch(Wt, wc[1] + dk, 0), l1, h1);
    unpack8(texelFetch(Wt, wc[2] + dk, 0), l2, h2); unpack8(texelFetch(Wt, wc[3] + dk, 0), l3, h3);
    mat4 ml = mat4(l0, l1, l2, l3), mh = mat4(h0, h1, h2, h3);
    mat4 ol = mat4(l0 + sy * dFdy(l0), l1 + sy * dFdy(l1), l2 + sy * dFdy(l2), l3 + sy * dFdy(l3));
    mat4 oh = mat4(h0 + sy * dFdy(h0), h1 + sy * dFdy(h1), h2 + sy * dFdy(h2), h3 + sy * dFdy(h3));
    mat4 wla = bottom ? ol : ml, wha = bottom ? oh : mh, wlb = bottom ? ml : ol, whb = bottom ? mh : oh;
    for (int i = 0; i < 4; i++) {
      a[i] += xl[i] * wla + xh[i] * wha;
      b[i] += xl[i] * wlb + xh[i] * whb;
    }
  }
  r0 = pk(a[0], b[0]); r1 = pk(a[1], b[1]); r2 = pk(a[2], b[2]); r3 = pk(a[3], b[3]);
}`, qs, { inputs: { X: xp, Wt: wp }, ints: { uIn: inD, uOut: outD, uRows: rows }, work: inD * 4, quad: true });
    return qs;
  };
  const ms = timeIt(c, run, reps);
  // Against the plain 4 x 8 kernel's planes, on a few texels.
  const ref = benchPacked48Planes(c, xp, wp, rows, inD, outD);
  const got = run();
  let maxd = 0;
  for (let i = 0; i < 4; i += 1) {
    const g = c.readU32 ? null : null;
    const a = readPacked(c, got[i]);
    const b = readPacked(c, ref[i]);
    for (let k = 0; k < Math.min(a.length, 200000); k += 1) maxd = Math.max(maxd, Math.abs(a[k] - b[k]));
  }
  got.forEach((t) => t.free());
  ref.forEach((t) => t.free());
  return { ms: Number(ms.toFixed(1)), gflops: Number(((2 * rows * inD * outD) / ms / 1e6).toFixed(0)), maxDiffVsPlain: maxd };
}

/** A packed tensor's values (unpacked to floats on the GPU and read). */
function readPacked(c, t) {
  const f = c.kernel('unpack-read', `
uniform highp usampler2D S;
vec4 compute(int t) {
  int w = textureSize(S, 0).x;
  int pt = t >> 1;
  uvec4 u = texelFetch(S, ivec2(pt % w, pt / w), 0);
  return (t & 1) == 0 ? vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)) : vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w));
}`, c.tensor(t.size), { inputs: { S: t } });
  const d = Float32Array.from(c.read(f));
  f.free();
  return d;
}

/** The plain 4 x 8 kernel's planes, for checking. */
function benchPacked48Planes(c, xp, wp, rows, inD, outD) {
  const rq = Math.ceil(rows / 4);
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
}

/** 8 rows x 8 columns a fragment as packed halves over eight render targets (128 bytes a pixel). Kernel only. */
function benchPacked88(c, x32, w32, rows, inD, outD, reps) {
  const pack = (src, size) => c.kernel('pack8', `
uniform sampler2D S;
uvec4 compute(int t) {
  vec4 a = tx(S, 2 * t), b = tx(S, 2 * t + 1);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}`, c.tensor(size, 'packed'), { inputs: { S: src } });
  const wf = c.elementwise('copyf', 'el(X, e)', outD * inD, { inputs: { X: w32 } });
  const xp = pack(x32, rows * inD);
  const wp = pack(wf, outD * inD);
  const rq = Math.ceil(rows / 8);
  const run = () => {
    const qs = [0, 1, 2, 3, 4, 5, 6, 7].map(() => c.tensor(rq * outD, 'packed'));
    c.kernel('linear8x8pk', `
uniform highp usampler2D X; uniform highp usampler2D Wt; uniform int uIn; uniform int uOut; uniform int uRows;
void unpack8(uvec4 u, out vec4 lo, out vec4 hi) { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
uvec4 pk(vec4 a, vec4 b) { return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw)); }
void compute4(int t, out uvec4 r0, out uvec4 r1, out uvec4 r2, out uvec4 r3, out uvec4 r4, out uvec4 r5, out uvec4 r6, out uvec4 r7) {
  int e0 = t * 8;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int ib = uIn / 8;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wt, 0).x;
  ivec2 xc[8];
  for (int i = 0; i < 8; i++) { int r = min(q * 8 + i, uRows - 1); int xt = r * ib; xc[i] = ivec2(xt % xw, xt / xw); }
  ivec2 wc[8];
  for (int i = 0; i < 8; i++) { int wt = (o0 + i) * ib; wc[i] = ivec2(wt % ww, wt / ww); }
  vec4 a[8], b[8];
  for (int i = 0; i < 8; i++) { a[i] = vec4(0.0); b[i] = vec4(0.0); }
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    vec4 l[8], h[8];
    for (int j = 0; j < 8; j++) unpack8(texelFetch(Wt, wc[j] + dk, 0), l[j], h[j]);
    mat4 wla = mat4(l[0], l[1], l[2], l[3]), wha = mat4(h[0], h[1], h[2], h[3]);
    mat4 wlb = mat4(l[4], l[5], l[6], l[7]), whb = mat4(h[4], h[5], h[6], h[7]);
    for (int i = 0; i < 8; i++) {
      vec4 xl, xh;
      unpack8(texelFetch(X, xc[i] + dk, 0), xl, xh);
      a[i] += xl * wla + xh * wha;
      b[i] += xl * wlb + xh * whb;
    }
  }
  r0 = pk(a[0], b[0]); r1 = pk(a[1], b[1]); r2 = pk(a[2], b[2]); r3 = pk(a[3], b[3]);
  r4 = pk(a[4], b[4]); r5 = pk(a[5], b[5]); r6 = pk(a[6], b[6]); r7 = pk(a[7], b[7]);
}`, qs, { inputs: { X: xp, Wt: wp }, ints: { uIn: inD, uOut: outD, uRows: rows }, work: inD * 8 });
    return qs;
  };
  const ms = timeIt(c, run, reps);
  return { ms: Number(ms.toFixed(1)), gflops: Number(((2 * rows * inD * outD) / ms / 1e6).toFixed(0)), note: 'kernel only, packed outputs, 8 targets' };
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

/**
 * Transform feedback: a vertex shader per 8 x 8 output block (64 floats out, the guaranteed interleaved minimum),
 * operands as packed halves, results into a buffer (no fragment output budget). Kernel only.
 */
function benchTF(c, x32, w32, rows, inD, outD, reps, RB = 8, CB = 8) {
  const gl = c.gl;
  const pack = (src, size) => c.kernel('tf-pack8', `
uniform sampler2D S;
uvec4 compute(int t) {
  vec4 a = tx(S, 2 * t), b = tx(S, 2 * t + 1);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}`, c.tensor(size, 'packed'), { inputs: { S: src } });
  const wf = c.elementwise('tf-copyf', 'el(X, e)', outD * inD, { inputs: { X: w32 } });
  const xp = pack(x32, rows * inD);
  const wp = pack(wf, outD * inD);
  const nOut = (RB * CB) / 4;
  const outs = [...Array(nOut).keys()].map((i) => `tfo${i}`);
  const vs = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform usampler2D X; uniform usampler2D Wt; uniform int uIn; uniform int uOut; uniform int uRows;
${outs.map((o) => `flat out vec4 ${o};`).join('\n')}
void unpack8(uvec4 u, out vec4 lo, out vec4 hi) { lo = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); hi = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w)); }
void main() {
  int nb = uOut / ${CB};
  int blk = gl_VertexID;
  int qr = blk / nb;
  int o0 = (blk - qr * nb) * ${CB};
  int ib = uIn / 8;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wt, 0).x;
  ivec2 xc[${RB}];
  for (int i = 0; i < ${RB}; i++) { int r = min(qr * ${RB} + i, uRows - 1); int xt = r * ib; xc[i] = ivec2(xt % xw, xt / xw); }
  ivec2 wc[${CB}];
  for (int j = 0; j < ${CB}; j++) { int wt = (o0 + j) * ib; wc[j] = ivec2(wt % ww, wt / ww); }
  vec4 acc[${RB * CB / 4}];
  for (int i = 0; i < ${RB * CB / 4}; i++) acc[i] = vec4(0.0);
  for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    vec4 l[${CB}], h[${CB}];
    for (int j = 0; j < ${CB}; j++) unpack8(texelFetch(Wt, wc[j] + dk, 0), l[j], h[j]);
    for (int i = 0; i < ${RB}; i++) {
      vec4 xl, xh;
      unpack8(texelFetch(X, xc[i] + dk, 0), xl, xh);
      for (int g = 0; g < ${CB / 4}; g++) acc[i * ${CB / 4} + g] += xl * mat4(l[4 * g], l[4 * g + 1], l[4 * g + 2], l[4 * g + 3]) + xh * mat4(h[4 * g], h[4 * g + 1], h[4 * g + 2], h[4 * g + 3]);
    }
  }
  ${[...Array(nOut).keys()].map((i) => `tfo${i} = acc[${i}];`).join(' ')}
  gl_Position = vec4(0.0);
  gl_PointSize = 1.0;
}`;
  const fs = `#version 300 es
precision highp float; out vec4 f; void main() { f = vec4(0.0); }`;
  const compile = (type, src) => { const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh); if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)); return sh; };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
  gl.transformFeedbackVaryings(prog, outs, gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const blocks = Math.ceil(rows / RB) * (outD / CB);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, buf);
  gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, blocks * RB * CB * 4, gl.DYNAMIC_COPY);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
  const tfo = gl.createTransformFeedback();
  const vao = gl.createVertexArray();
  const run = () => {
    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, xp.tex); gl.uniform1i(gl.getUniformLocation(prog, 'X'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, wp.tex); gl.uniform1i(gl.getUniformLocation(prog, 'Wt'), 1);
    gl.uniform1i(gl.getUniformLocation(prog, 'uIn'), inD);
    gl.uniform1i(gl.getUniformLocation(prog, 'uOut'), outD);
    gl.uniform1i(gl.getUniformLocation(prog, 'uRows'), rows);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tfo);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, blocks);
    gl.endTransformFeedback();
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindVertexArray(null);
    return [];
  };
  const ms = timeIt(c, run, reps);
  // Correctness of one block against a CPU dot product.
  const got = new Float32Array(RB * CB);
  gl.bindBuffer(gl.COPY_READ_BUFFER, buf);
  gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, got);
  gl.bindBuffer(gl.COPY_READ_BUFFER, null);
  const xs = c.read(x32, RB * inD);
  const ws = c.read(w32, CB * inD);
  let err = 0;
  for (let i = 0; i < RB; i += 1) for (let j = 0; j < CB; j += 1) {
    let d = 0; for (let k = 0; k < inD; k += 1) d += xs[i * inD + k] * ws[j * inD + k];
    const g = got[i * CB + j];
    err = Math.max(err, Math.abs(g - d) / (Math.abs(d) + 1));
  }
  return { ms: Number(ms.toFixed(1)), gflops: Number(((2 * rows * inD * outD) / ms / 1e6).toFixed(0)), maxRelErr: Number(err.toExponential(2)),
    maxComponents: gl.getParameter(gl.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS), note: `kernel only, ${RB}x${CB} via transform feedback` };
}

/** Block-64 int8 weights (16 codes a texel, a float scale a block) against packed-half activations, 4 x 4 blocks. */
function benchInt8(c, x32, w32, rows, inD, outD, reps) {
  const wv = c.read(w32, outD * inD);
  const nblk = inD / 64;
  const codes = new Int8Array(outD * inD);
  const scales = new Float32Array(outD * nblk);
  for (let b = 0; b < outD * nblk; b += 1) {
    let m = 0;
    for (let i = 0; i < 64; i += 1) m = Math.max(m, Math.abs(wv[b * 64 + i]));
    const sc = m / 127 || 1;
    scales[b] = sc;
    for (let i = 0; i < 64; i += 1) codes[b * 64 + i] = Math.round(wv[b * 64 + i] / sc);
  }
  const wq = c.tensor((outD * inD) / 2, 'packed');
  c.writeU32(wq, new Uint32Array(codes.buffer));
  const st = c.upload(scales);
  const xp = c.kernel('i8-pack8', `
uniform sampler2D S;
uvec4 compute(int t) {
  vec4 a = tx(S, 2 * t), b = tx(S, 2 * t + 1);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}`, c.tensor(rows * inD, 'packed'), { inputs: { S: x32 } });
  const rq = Math.ceil(rows / 4);
  const run = () => {
    const qs = [0, 1, 2, 3].map(() => c.tensor(rq * outD));
    c.kernel('linear4x4i8', `
uniform highp usampler2D X; uniform highp usampler2D Wq; uniform sampler2D Sc; uniform int uIn; uniform int uOut; uniform int uRows;
vec4 sb(uint u) { return vec4(float(int(u << 24u) >> 24), float(int(u << 16u) >> 24), float(int(u << 8u) >> 24), float(int(u) >> 24)); }
void compute4(int t, out vec4 r0, out vec4 r1, out vec4 r2, out vec4 r3) {
  int e0 = t * 4;
  int q = e0 / uOut;
  int o0 = e0 - q * uOut;
  int xb = uIn / 8;
  int wb = uIn / 16;
  int nblk = uIn / 64;
  int xw = textureSize(X, 0).x;
  int ww = textureSize(Wq, 0).x;
  ivec2 xc[4];
  for (int i = 0; i < 4; i++) { int r = min(q * 4 + i, uRows - 1); int xt = r * xb; xc[i] = ivec2(xt % xw, xt / xw); }
  ivec2 wc[4];
  for (int j = 0; j < 4; j++) { int wt = (o0 + j) * wb; wc[j] = ivec2(wt % ww, wt / ww); }
  vec4 a[4];
  for (int i = 0; i < 4; i++) a[i] = vec4(0.0);
  for (int blk = 0; blk < nblk; blk++) {
    vec4 sc = vec4(el(Sc, o0 * nblk + blk), el(Sc, (o0 + 1) * nblk + blk), el(Sc, (o0 + 2) * nblk + blk), el(Sc, (o0 + 3) * nblk + blk));
    vec4 acc[4];
    for (int i = 0; i < 4; i++) acc[i] = vec4(0.0);
    for (int s = 0; s < 4; s++) {
      int k16 = blk * 4 + s;
      uvec4 u0 = texelFetch(Wq, wc[0] + ivec2(k16, 0), 0), u1 = texelFetch(Wq, wc[1] + ivec2(k16, 0), 0);
      uvec4 u2 = texelFetch(Wq, wc[2] + ivec2(k16, 0), 0), u3 = texelFetch(Wq, wc[3] + ivec2(k16, 0), 0);
      mat4 m0 = mat4(sb(u0.x), sb(u1.x), sb(u2.x), sb(u3.x));
      mat4 m1 = mat4(sb(u0.y), sb(u1.y), sb(u2.y), sb(u3.y));
      mat4 m2 = mat4(sb(u0.z), sb(u1.z), sb(u2.z), sb(u3.z));
      mat4 m3 = mat4(sb(u0.w), sb(u1.w), sb(u2.w), sb(u3.w));
      for (int i = 0; i < 4; i++) {
        uvec4 x0 = texelFetch(X, xc[i] + ivec2(2 * k16, 0), 0), x1 = texelFetch(X, xc[i] + ivec2(2 * k16 + 1, 0), 0);
        acc[i] += vec4(unpackHalf2x16(x0.x), unpackHalf2x16(x0.y)) * m0 + vec4(unpackHalf2x16(x0.z), unpackHalf2x16(x0.w)) * m1
          + vec4(unpackHalf2x16(x1.x), unpackHalf2x16(x1.y)) * m2 + vec4(unpackHalf2x16(x1.z), unpackHalf2x16(x1.w)) * m3;
      }
    }
    for (int i = 0; i < 4; i++) a[i] += acc[i] * sc;
  }
  r0 = a[0]; r1 = a[1]; r2 = a[2]; r3 = a[3];
}`, qs, { inputs: { X: xp, Wq: wq, Sc: st }, ints: { uIn: inD, uOut: outD, uRows: rows }, work: inD * 2 });
    return qs;
  };
  const ms = timeIt(c, run, reps);
  // One block against the quantized weights on the CPU.
  const qs = run();
  const got = c.read(qs[0], 4);
  const xs = c.read(x32, inD);
  let err = 0;
  for (let j = 0; j < 4; j += 1) {
    let d = 0; for (let k = 0; k < inD; k += 1) d += xs[k] * codes[j * inD + k] * scales[j * nblk + (k >> 6)];
    err = Math.max(err, Math.abs(got[j] - d) / (Math.abs(d) + 1));
  }
  qs.forEach((t) => t.free());
  return { ms: Number(ms.toFixed(1)), gflops: Number(((2 * rows * inD * outD) / ms / 1e6).toFixed(0)), maxRelErr: Number(err.toExponential(2)), note: 'kernel only, int8 weights' };
}

/** The 4 x 8 packed matmul with its output planes at several texture widths, and with a prefetching loop. */
function benchLayout(c, x32, w32, rows, inD, outD, reps) {
  const pack = (src, size) => c.kernel('lay-pack8', `
uniform sampler2D S;
uvec4 compute(int t) {
  vec4 a = tx(S, 2 * t), b = tx(S, 2 * t + 1);
  return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}`, c.tensor(size, 'packed'), { inputs: { S: src } });
  const wf = c.elementwise('lay-copyf', 'el(X, e)', outD * inD, { inputs: { X: w32 } });
  const xp = pack(x32, rows * inD);
  const wp = pack(wf, outD * inD);
  const rq = Math.ceil(rows / 4);
  const texels = (rq * outD) / 8;
  const plane = (width) => {
    const h = Math.ceil(texels / width);
    const tex = c._texture(width, h, 'packed');
    return { tex, w: width, h, half: 'packed', texels, size: texels * 8, free() { c.gl.deleteTexture(tex); } };
  };
  const body = (prefetch) => `
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
  for (int j = 0; j < 8; j++) { int wt = (o0 + j) * ib; wc[j] = ivec2(wt % ww, wt / ww); }
  vec4 a[4], b[4];
  for (int i = 0; i < 4; i++) { a[i] = vec4(0.0); b[i] = vec4(0.0); }
  ${prefetch ? `uvec4 wn[8], xn[4];
  for (int j = 0; j < 8; j++) wn[j] = texelFetch(Wt, wc[j], 0);
  for (int i = 0; i < 4; i++) xn[i] = texelFetch(X, xc[i], 0);
  for (int k = 0; k < ib; k++) {
    uvec4 wcur[8], xcur[4];
    for (int j = 0; j < 8; j++) wcur[j] = wn[j];
    for (int i = 0; i < 4; i++) xcur[i] = xn[i];
    if (k + 1 < ib) {
      ivec2 dn = ivec2(k + 1, 0);
      for (int j = 0; j < 8; j++) wn[j] = texelFetch(Wt, wc[j] + dn, 0);
      for (int i = 0; i < 4; i++) xn[i] = texelFetch(X, xc[i] + dn, 0);
    }
    vec4 l[8], h[8];
    for (int j = 0; j < 8; j++) unpack8(wcur[j], l[j], h[j]);` : `for (int k = 0; k < ib; k++) {
    ivec2 dk = ivec2(k, 0);
    vec4 l[8], h[8];
    for (int j = 0; j < 8; j++) unpack8(texelFetch(Wt, wc[j] + dk, 0), l[j], h[j]);`}
    mat4 wla = mat4(l[0], l[1], l[2], l[3]), wha = mat4(h[0], h[1], h[2], h[3]);
    mat4 wlb = mat4(l[4], l[5], l[6], l[7]), whb = mat4(h[4], h[5], h[6], h[7]);
    for (int i = 0; i < 4; i++) {
      vec4 xl, xh;
      unpack8(${prefetch ? 'xcur[i]' : 'texelFetch(X, xc[i] + dk, 0)'}, xl, xh);
      a[i] += xl * wla + xh * wha;
      b[i] += xl * wlb + xh * whb;
    }
  }
  r0 = pk(a[0], b[0]); r1 = pk(a[1], b[1]); r2 = pk(a[2], b[2]); r3 = pk(a[3], b[3]);
}`;
  const out = {};
  for (const [label, width, prefetch] of [['w3072', 3072, false], ['w256', 256, false], ['w64', 64, false], ['w32', 32, false], ['w3072-prefetch', 3072, true]]) {
    const run = () => {
      const qs = [0, 1, 2, 3].map(() => plane(width));
      c.kernel(`lay-4x8${prefetch ? '-pf' : ''}`, body(prefetch), qs, { inputs: { X: xp, Wt: wp }, ints: { uIn: inD, uOut: outD, uRows: rows }, work: inD * 3 });
      return qs;
    };
    out[label] = { ms: Number(timeIt(c, run, reps).toFixed(2)) };
  }
  return out;
}
