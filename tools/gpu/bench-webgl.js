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
  run().forEach((t) => t.free());
  c.read(xp.half === 'packed' ? c.tensor(4) : xp, 4);
  const t0 = performance.now();
  let last = null;
  for (let i = 0; i < reps; i += 1) { last?.forEach((t) => t.free()); last = run(); }
  c.read(last[0], 4);
  const ms = (performance.now() - t0) / reps;
  return { ms: Number(ms.toFixed(1)), gflops: Number(((2 * rows * inD * outD) / ms / 1e6).toFixed(0)), note: 'kernel only, no repack' };
}
