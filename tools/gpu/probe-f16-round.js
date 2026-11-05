/**
 * How WebGPU here rounds float32 -> float16, by each conversion the engine uses (`f16(x)` into an `array<f16>`,
 * `bitcast<u32>(vec2<f16>(...))`, `pack2x16float`): 1 + 0.75 ulp(f16) rounds to 1 + 1 ulp to nearest, to 1 if truncated
 * (the WebGL2 path found render-target writes truncating on an M2; truncation's bias adds up over a trunk pass).
 *
 *   node tools/gpu-chrome.mjs tools/gpu/probe-f16-round.js
 */

export async function main(device) {
  const f16 = device.features.has('shader-f16');
  const code = `${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
${f16 ? '@group(0) @binding(1) var<storage, read_write> h: array<f16>;' : ''}
@compute @workgroup_size(1)
fn main() {
  let u = 1.0 / 1024.0;
  let a = 1.0 + 0.75 * u;
  let b = -(1.0 + 0.75 * u);
  o[0] = pack2x16float(vec2<f32>(a, b));
  ${f16 ? `o[1] = bitcast<u32>(vec2<f16>(vec2<f32>(a, b)));
  h[0] = f16(a);
  h[1] = f16(b);` : ''}
}`;
  const ob = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const hb = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ob } }, ...(f16 ? [{ binding: 1, resource: { buffer: hb } }] : [])] });
  const rb = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(1);
  pass.end();
  enc.copyBufferToBuffer(ob, 0, rb, 0, 16);
  enc.copyBufferToBuffer(hb, 0, rb, 16, 16);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const h = new Uint16Array(rb.getMappedRange().slice(0));
  rb.unmap();
  // 0x3c01 / 0xbc01 = nearest (1 + 1 ulp); 0x3c00 / 0xbc00 = truncated.
  const verdict = (x) => (x === 0x3c01 || x === 0xbc01 ? 'nearest' : x === 0x3c00 || x === 0xbc00 ? 'truncated' : x.toString(16));
  return {
    pack2x16float: [verdict(h[0]), verdict(h[1])],
    ...(f16 ? { vec2f16Bitcast: [verdict(h[2]), verdict(h[3])], f16Store: [verdict(h[8]), verdict(h[9])] } : {}),
  };
}
