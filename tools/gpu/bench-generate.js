/**
 * Whole designs, end to end on the GPU generator: a 60-step design with one trunk pass a step (Thinking "none"), per
 * engine variant (`gemmVariant`), interleaved over rounds. Wall time and the final Cα of each variant against the first.
 *
 *   node tools/gpu-chrome.mjs tools/gpu/bench-generate.js --cases=uncond_short,ppi_pdl1 --rounds=2 \
 *     --variants='[{},{"direct":false}]'
 */

import { Rfd4pGeneratorGpu } from '../../webgpu/rfd4p/generate.js';
import { loadOracleCase, loadReleased } from './check-trunk.js';

export async function main(device, args) {
  const opt = Object.fromEntries(args.map((a) => { const i = a.indexOf('='); return [a.slice(2, i), a.slice(i + 1)]; }));
  const variants = JSON.parse(opt.variants ?? '[{},{"direct":false}]');
  const rounds = Number(opt.rounds ?? 2);
  const nsteps = Number(opt.nsteps ?? 60);
  const { cfg, weights } = await loadReleased();
  const gen = new Rfd4pGeneratorGpu(device, weights, cfg, { half: true });
  const out = [];
  for (const caseName of (opt.cases ?? 'uncond_short,ppi_pdl1').split(',')) {
    const { t } = await loadOracleCase(caseName);
    const batch = {};
    for (const [k, v] of Object.entries(t)) if (k.startsWith('batch.')) batch[k.slice(6)] = v;
    const best = variants.map(() => Infinity);
    const ca = [];
    for (let r = 0; r < rounds + 1; r += 1) {
      for (let v = 0; v < variants.length; v += 1) {
        gen.trunk.engine.gemmVariant = variants[v];
        gen.trunk.n = -1;
        const t0 = performance.now();
        const res = await gen.generate(batch, { seed: 7, nsteps, recycleSchedule: 'none', earlyStop: false });
        const ms = performance.now() - t0;
        if (r > 0) best[v] = Math.min(best[v], ms); // round 0 warms up (pipelines, weights)
        if (r === 0) ca[v] = Float32Array.from(res.x.bb_ca);
      }
    }
    const rmsd = ca.map((c) => {
      let s = 0;
      for (let i = 0; i < c.length; i += 1) s += (c[i] - ca[0][i]) ** 2;
      return Number((Math.sqrt(s / (c.length / 3)) * 10).toFixed(3));
    });
    out.push({ case: caseName, n: batch.padding_mask.shape[1], nsteps,
      variants: variants.map((v, i) => ({ variant: v, s: Number((best[i] / 1000).toFixed(2)), speedup: Number((best[0] / best[i]).toFixed(3)), caRmsdVsFirstA: rmsd[i] })) });
  }
  gen.destroy();
  return out;
}
