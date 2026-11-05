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
  model.destroy();
  return { rows, inD, outD, xHalf, renderer: c.renderer, results };
}
