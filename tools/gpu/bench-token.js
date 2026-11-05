/**
 * The trunk's token-level linears on the released weights (`Rfd4pModelGL.linear`, pack and repack included), alone:
 * ms per call and TFLOP/s, at n rows.
 *
 *   node tools/gpu-chrome.mjs tools/gpu/bench-token.js [--n=100 --reps=20 --flags=noCM,noSplitK]
 */

import { Rfd4pModelGL } from '../../webgl/rfd4p/model.js';
import { loadReleased } from './check-trunk.js';

export async function main(_device, args) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const n = Number(opt.n ?? 100);
  const reps = Number(opt.reps ?? 20);
  const { weights } = await loadReleased();
  const model = new Rfd4pModelGL(weights);
  for (const k of (opt.flags ?? '').split(',').filter(Boolean)) model[k] = true;
  if (opt.splitFrags) model.splitFrags = Number(opt.splitFrags);
  const c = model.ctx;
  const p = 'nn.transformer_layers.0';
  const shapes = [
    ['qkv', `${p}.mhba.mha.to_qkv.weight`, 768, 2304],
    ['g', `${p}.mhba.mha.to_g.weight`, 768, 768],
    ['swiglu-in', `${p}.transition.transition.swish_linear.0.weight`, 768, 6144],
    ['swiglu-out', `${p}.transition.transition.linear_out.weight`, 3072, 768],
  ];
  const sync = () => { const t = c.elementwise('sync', '1.0', 4); c.read(t, 4); t.free(); };
  const results = {};
  for (const [name, wn, inD, outD] of shapes) {
    const x = c.upload(Float32Array.from({ length: n * inD }, () => Math.random() - 0.5));
    const w = model.W(wn);
    let prev = model.linear(x, n, inD, w, outD);
    sync();
    const t0 = performance.now();
    for (let i = 0; i < reps; i += 1) {
      const cur = model.linear(x, n, inD, w, outD);
      prev.free();
      prev = cur;
    }
    sync();
    const ms = (performance.now() - t0) / reps;
    prev.free();
    x.free();
    results[name] = { ms: Number(ms.toFixed(3)), tflops: Number(((2 * n * inD * outD) / ms / 1e9).toFixed(2)) };
  }
  model.destroy();
  return { n, results };
}
