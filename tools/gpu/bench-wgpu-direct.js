/**
 * A "direct" WebGPU GEMM, shaped like the WebGL2 fallback's fragment-shader matmul: each thread computes 4 rows x 8
 * columns from packed-f16 operands read straight from memory (no workgroup staging, no barriers), against the trunk's
 * own pair GEMM rates. Operands as storage buffers (`buf`) or as rgba32uint textures 384 texels wide (`tex`, the
 * WebGL2 layout, read through the texture cache). C = A W^T, A [M, K], W [N, K], C packed f16.
 *
 *   node tools/gpu-chrome.mjs tools/gpu/bench-wgpu-direct.js [--m=43264 --k=384 --n=1408 --reps=20]
 */

const TW = 384;

function rndHalves(count) {
  const h = new Uint16Array(count);
  for (let i = 0; i < count; i += 1) h[i] = (Math.random() < 0.5 ? 0x8000 : 0) | ((Math.floor(Math.random() * 4) + 11) << 10) | Math.floor(Math.random() * 1024);
  return new Uint32Array(h.buffer);
}

function h2f(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 31;
  const m = h & 1023;
  return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15);
}

function kernelSource(mode, wg, cols, tileR = 0) {
  const tex = mode === 'tex';
  const decl = tex
    ? `@group(0) @binding(0) var A: texture_2d<u32>;
@group(0) @binding(1) var W: texture_2d<u32>;`
    : `@group(0) @binding(0) var<storage, read> A: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> W: array<vec4<u32>>;`;
  const ld = (name, idx) => (tex ? `textureLoad(${name}, vec2<i32>(i32((${idx}) % ${TW}u), i32((${idx}) / ${TW}u)), 0)` : `${name}[${idx}]`);
  return `
${decl}
@group(0) @binding(2) var<storage, read_write> C: array<vec4<u32>>;
struct P { m: u32, k: u32, n: u32, groupsX: u32 }
@group(0) @binding(3) var<uniform> p: P;
fn lo(u: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
fn hi(u: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.z), unpack2x16float(u.w)); }
fn pk(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> { return vec4<u32>(pack2x16float(a.xy), pack2x16float(a.zw), pack2x16float(b.xy), pack2x16float(b.zw)); }
@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let g = wid.x + wid.y * p.groupsX;
  let nb = p.n / ${cols}u;
  ${tileR ? `// A workgroup is a ${tileR} x ${wg / tileR} block of (row quad, column block): neighbours share rows and columns.
  let cbs = nb / ${wg / tileR}u;
  let gq = g / cbs;
  let gc = g - gq * cbs;
  let q = gq * ${tileR}u + li / ${wg / tileR}u;
  let ob = gc * ${wg / tileR}u + li % ${wg / tileR}u;
  if (q * 4u >= p.m) { return; }
  let o0 = ob * ${cols}u;` : `let t = g * ${wg}u + li;
  let q = t / nb;
  if (q * 4u >= p.m) { return; }
  let o0 = (t - q * nb) * ${cols}u;`}
  let ib = p.k / 8u;
  var r: array<u32, 4>;
  for (var i = 0u; i < 4u; i++) { r[i] = min(q * 4u + i, p.m - 1u) * ib; }
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
  var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
  for (var k = 0u; k < ib; k++) {
    let w0 = ${ld('W', '(o0 + 0u) * ib + k')}; let w1 = ${ld('W', '(o0 + 1u) * ib + k')};
    let w2 = ${ld('W', '(o0 + 2u) * ib + k')}; let w3 = ${ld('W', '(o0 + 3u) * ib + k')};
    let wla = mat4x4<f32>(lo(w0), lo(w1), lo(w2), lo(w3));
    let wha = mat4x4<f32>(hi(w0), hi(w1), hi(w2), hi(w3));
    ${cols === 8 ? `let w4 = ${ld('W', '(o0 + 4u) * ib + k')}; let w5 = ${ld('W', '(o0 + 5u) * ib + k')};
    let w6 = ${ld('W', '(o0 + 6u) * ib + k')}; let w7 = ${ld('W', '(o0 + 7u) * ib + k')};
    let wlb = mat4x4<f32>(lo(w4), lo(w5), lo(w6), lo(w7));
    let whb = mat4x4<f32>(hi(w4), hi(w5), hi(w6), hi(w7));` : ''}
    let x0 = ${ld('A', 'r[0] + k')}; let x1 = ${ld('A', 'r[1] + k')}; let x2 = ${ld('A', 'r[2] + k')}; let x3 = ${ld('A', 'r[3] + k')};
    a0 += lo(x0) * wla + hi(x0) * wha; a1 += lo(x1) * wla + hi(x1) * wha;
    a2 += lo(x2) * wla + hi(x2) * wha; a3 += lo(x3) * wla + hi(x3) * wha;
    ${cols === 8 ? `b0 += lo(x0) * wlb + hi(x0) * whb; b1 += lo(x1) * wlb + hi(x1) * whb;
    b2 += lo(x2) * wlb + hi(x2) * whb; b3 += lo(x3) * wlb + hi(x3) * whb;` : ''}
  }
  // Row-major output, packed: row r's 8 columns from o0 (cols 8), or 4 columns as half a texel (cols 4: a's only).
  let rowT = p.n / 8u;
  let base = q * 4u;
  ${cols === 8
    ? `if (base < p.m) { C[(base) * rowT + o0 / 8u] = pk(a0, b0); }
  if (base + 1u < p.m) { C[(base + 1u) * rowT + o0 / 8u] = pk(a1, b1); }
  if (base + 2u < p.m) { C[(base + 2u) * rowT + o0 / 8u] = pk(a2, b2); }
  if (base + 3u < p.m) { C[(base + 3u) * rowT + o0 / 8u] = pk(a3, b3); }`
    : `// (4 columns: two threads per packed texel; written as a pair of u32 for the check only)
  if (base < p.m) { C[(base) * rowT + o0 / 8u] = pk(a0, a0); }`}
}`;
}

/** R rows x C columns a thread (C a multiple of 4), storage buffers, `tr` row groups x `cg` column groups a workgroup. */
function blockSource(R, C, tr, cg) {
  const wg = tr * cg;
  const rs = [...Array(R).keys()];
  const cs = [...Array(C).keys()];
  const c4 = C / 4;
  return `
@group(0) @binding(0) var<storage, read> A: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Cc: array<vec4<u32>>;
struct P { m: u32, k: u32, n: u32, groupsX: u32 }
@group(0) @binding(3) var<uniform> p: P;
fn lo(u: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
fn hi(u: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.z), unpack2x16float(u.w)); }
@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let g = wid.x + wid.y * p.groupsX;
  let nb = p.n / ${C}u;
  let cbs = nb / ${cg}u;
  let gq = g / cbs;
  let gc = g - gq * cbs;
  let q = gq * ${tr}u + li / ${cg}u;
  let ob = gc * ${cg}u + li % ${cg}u;
  if (q * ${R}u >= p.m) { return; }
  let o0 = ob * ${C}u;
  let ib = p.k / 8u;
  ${rs.map((i) => `let r${i} = min(q * ${R}u + ${i}u, p.m - 1u) * ib;`).join(' ')}
  ${cs.map((j) => `let wr${j} = (o0 + ${j}u) * ib;`).join(' ')}
  ${rs.map((i) => [...Array(c4).keys()].map((b) => `var a${i}_${b} = vec4<f32>(0.0);`).join(' ')).join('\n  ')}
  for (var k = 0u; k < ib; k++) {
    ${[...Array(c4).keys()].map((b) => `let wu${b}_0 = W[wr${b * 4} + k]; let wu${b}_1 = W[wr${b * 4 + 1} + k]; let wu${b}_2 = W[wr${b * 4 + 2} + k]; let wu${b}_3 = W[wr${b * 4 + 3} + k];
    let wl${b} = mat4x4<f32>(lo(wu${b}_0), lo(wu${b}_1), lo(wu${b}_2), lo(wu${b}_3));
    let wh${b} = mat4x4<f32>(hi(wu${b}_0), hi(wu${b}_1), hi(wu${b}_2), hi(wu${b}_3));`).join('\n    ')}
    ${rs.map((i) => `{ let x = A[r${i} + k]; let xl = lo(x); let xh = hi(x); ${[...Array(c4).keys()].map((b) => `a${i}_${b} += xl * wl${b} + xh * wh${b};`).join(' ')} }`).join('\n    ')}
  }
  let rowT = p.n / 8u;
  ${c4 === 1 ? rs.map((i) => `if (q * ${R}u + ${i}u < p.m) { Cc[(q * ${R}u + ${i}u) * rowT + o0 / 8u] = vec4<u32>(pack2x16float(a${i}_0.xy), pack2x16float(a${i}_0.zw), 0u, 0u); }`).join('\n  ') : ''}
  ${c4 === 1 ? '' : rs.map((i) => `if (q * ${R}u + ${i}u < p.m) { ${[...Array(Math.floor(c4 / 2)).keys()].map((b) => `Cc[(q * ${R}u + ${i}u) * rowT + o0 / 8u + ${b}u] = vec4<u32>(pack2x16float(a${i}_${2 * b}.xy), pack2x16float(a${i}_${2 * b}.zw), pack2x16float(a${i}_${2 * b + 1}.xy), pack2x16float(a${i}_${2 * b + 1}.zw));`).join(' ')} }`).join('\n  ')}
}`;
}

export async function main(device, args) {
  const opt = Object.fromEntries(args.filter((a) => a.includes('=')).map((a) => { const i = a.indexOf('='); return [a.slice(2, i), a.slice(i + 1)]; }));
  const ui = (x) => Math.max(16, Math.ceil(x / 16) * 16);
  const M = Number(opt.m ?? 43264);
  const K = Number(opt.k ?? 384);
  const N = Number(opt.n ?? 1408);
  const reps = Number(opt.reps ?? 20);
  const aWords = rndHalves(M * K);
  const wWords = rndHalves(N * K);
  const mk = (words, usage) => {
    const b = device.createBuffer({ size: Math.max(16, words.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, words);
    return b;
  };
  const texOf = (words) => {
    const texels = words.length / 4;
    const h = Math.ceil(texels / TW);
    const data = new Uint32Array(TW * h * 4);
    data.set(words);
    const t = device.createTexture({ size: [TW, h], format: 'rgba32uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: t }, data, { bytesPerRow: TW * 16 }, [TW, h]);
    return t;
  };
  const aBuf = mk(aWords, GPUBufferUsage.STORAGE);
  const wBuf = mk(wWords, GPUBufferUsage.STORAGE);
  const aTex = texOf(aWords);
  const wTex = texOf(wWords);
  const results = { M, K, N, args };
  if (!Number.isFinite(M * N)) return results;
  const cBufs = [0, 1].map(() => device.createBuffer({ size: ui(M * N * 2), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
  const configs = opt.configs ? JSON.parse(opt.configs) : [['tex', 256, 8, 16], ['tex', 256, 8, 8], ['tex', 256, 8, 32], ['buf', 256, 8, 16], ['buf', 256, 8, 8], ['buf', 256, 8, 32], ['buf', 128, 8, 8], ['tex', 128, 8, 4]];
  for (const cfgItem of configs) {
    if (cfgItem[0] === 'blk') {
      const [, R, C, tr, cg] = cfgItem;
      const nbk = N / C;
      if (nbk % cg) continue;
      const groups = Math.ceil(Math.ceil(M / R) / tr) * (nbk / cg);
      const gx = Math.min(groups, 65535);
      const gy = Math.ceil(groups / gx);
      const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(uni, 0, new Uint32Array([M, K, N, gx]));
      const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: blockSource(R, C, tr, cg) }), entryPoint: 'main' } });
      const bgs = cBufs.map((cb) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: aBuf } }, { binding: 1, resource: { buffer: wBuf } }, { binding: 2, resource: { buffer: cb } }, { binding: 3, resource: { buffer: uni } }] }));
      const run = (count) => {
        const enc = device.createCommandEncoder();
        for (let i = 0; i < count; i += 1) { const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bgs[i & 1]); pass.dispatchWorkgroups(gx, gy); pass.end(); }
        device.queue.submit([enc.finish()]);
        return device.queue.onSubmittedWorkDone();
      };
      await run(2);
      let best = Infinity;
      for (let r = 0; r < 3; r += 1) { const t0 = performance.now(); await run(reps); best = Math.min(best, (performance.now() - t0) / reps); }
      results[`blk-r${R}-c${C}-${tr}x${cg}`] = { ms: Number(best.toFixed(2)), tflops: Number(((2 * M * K * N) / best / 1e9).toFixed(2)) };
      continue;
    }
    const [mode, wg, cols, tileR] = cfgItem;
    const nbk = N / cols;
    if (tileR && nbk % (wg / tileR)) continue;
    const groups = tileR ? Math.ceil(Math.ceil(M / 4) / tileR) * (nbk / (wg / tileR)) : Math.ceil((Math.ceil(M / 4) * nbk) / wg);
    const gx = Math.min(groups, 65535);
    const gy = Math.ceil(groups / gx);
    const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uni, 0, new Uint32Array([M, K, N, gx]));
    const module = device.createShaderModule({ code: kernelSource(mode, wg, cols, tileR) });
    const pipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bgs = cBufs.map((cb) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: mode === 'tex' ? aTex.createView() : { buffer: aBuf } },
      { binding: 1, resource: mode === 'tex' ? wTex.createView() : { buffer: wBuf } },
      { binding: 2, resource: { buffer: cb } },
      { binding: 3, resource: { buffer: uni } },
    ] }));
    const run = (count) => {
      const enc = device.createCommandEncoder();
      for (let i = 0; i < count; i += 1) {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bgs[i & 1]);
        pass.dispatchWorkgroups(gx, gy);
        pass.end();
      }
      device.queue.submit([enc.finish()]);
      return device.queue.onSubmittedWorkDone();
    };
    await run(2);
    let best = Infinity;
    for (let r = 0; r < 3; r += 1) {
      const t0 = performance.now();
      await run(reps);
      best = Math.min(best, (performance.now() - t0) / reps);
    }
    // Check one output texel against the CPU.
    const rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(cBufs[1], 0, rb, 0, 16);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const got = Array.from(new Uint16Array(rb.getMappedRange().slice(0))).map(h2f);
    rb.unmap();
    const ah = new Uint16Array(aWords.buffer);
    const wh = new Uint16Array(wWords.buffer);
    const want = [0, 1, 2, 3, 4, 5, 6, 7].map((o) => { let s = 0; for (let k = 0; k < K; k += 1) s += h2f(ah[k]) * h2f(wh[o * K + k]); return s; });
    const maxRel = Math.max(...want.map((w, i) => Math.abs(w - got[i]) / Math.max(1, Math.abs(w))));
    results[`${mode}-wg${wg}-t${tileR}`] = { ms: Number(best.toFixed(2)), tflops: Number(((2 * M * K * N) / best / 1e9).toFixed(2)), maxRel: Number(maxRel.toExponential(1)) };
  }
  return results;
}
