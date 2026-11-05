/**
 * A small WebGL2 tensor library: the GPU fallback for browsers without WebGPU (Safari before 26, older Chrome).
 *
 * A tensor is a flat float array held in an RGBA32F texture, 4 values to a texel, row-major and wrapped at `TW` texels
 * a row (a pair tensor of n^2 rows is far taller than a texture may be). Weights are the same layout in RGBA16F (large
 * matrices, as the WebGPU path holds them) or RGBA32F (vectors and small matrices). Every kernel is a fragment shader
 * that writes one texel - 4 outputs - per fragment; `kernel` runs one, `elementwise` wraps the common case of an output
 * computed one value at a time.
 *
 * Needs WebGL2 and EXT_color_buffer_float (rendering to float textures): Safari 15+, Chrome 56+, Firefox 51+.
 */

// 384 texels (1536 floats) a row: the row widths the fast paths walk (24, 48, 64, 96, 128, 192, 384 texels) divide it,
// so no such row wraps onto the next texture row and a kernel walks one with a fixed texture row (a kernel checks, and
// takes its general path otherwise). Narrow rather than wide: a GPU tile of fragments then covers nearby rows of its
// operands, which stay in the texture cache (a trunk pass at n = 208 on an M2: 2.35 s at 3072 texels, 1.64 s at 768,
// 1.47 s at 384, 1.59 s at 192). A tensor too tall at this width takes a multiple of it, so its rows stay whole.
export const TW = 384;
const VERT = `#version 300 es
in vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

/** GLSL shared by every kernel: texel and element reads of any tensor, and the output texel's index. */
const PRELUDE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform int uOutW;
uniform int uOutTexels;
vec4 tx(sampler2D s, int t) { int w = textureSize(s, 0).x; return texelFetch(s, ivec2(t % w, t / w), 0); }
float el(sampler2D s, int e) { vec4 v = tx(s, e >> 2); int c = e & 3; return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }
float sigm(float v) { return 1.0 / (1.0 + exp(-v)); }
// Element e of a packed tensor (eight float16 values a texel, two a channel).
float elp(highp usampler2D s, int e) {
  int w = textureSize(s, 0).x;
  int t = e >> 3;
  uvec4 u = texelFetch(s, ivec2(t % w, t / w), 0);
  int c = (e >> 1) & 3;
  vec2 h = unpackHalf2x16(c == 0 ? u.x : c == 1 ? u.y : c == 2 ? u.z : u.w);
  return (e & 1) == 0 ? h.x : h.y;
}
`;

/** To the nearest float16 value, ties to even, as a float (normal range; written to a float16 target, it is exact). */
const R16 = `vec4 r16(vec4 x) {
  uvec4 u = floatBitsToUint(x);
  u = (u + uvec4(0xFFFu) + ((u >> 13u) & uvec4(1u))) & uvec4(0xFFFFE000u);
  return uintBitsToFloat(u);
}`;

export function webgl2Available() {
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2');
    return Boolean(gl && gl.getExtension('EXT_color_buffer_float'));
  } catch {
    return false;
  }
}

export class GLTensor {
  constructor(ctx, size, tex, w, h, half = false) {
    Object.assign(this, { ctx, size, tex, w, h, half, texels: Math.ceil(size / (half === 'packed' ? 8 : 4)) });
  }

  /** Back to the pool (the texture is reused by the next tensor of its shape). */
  free() {
    if (!this.tex || this.weight) return;
    this.ctx._release(this);
    this.tex = null;
  }
}

export class GLContext {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = 1;
    this.canvas.height = 1;
    const gl = this.canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: true,
      premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL2 is not available');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('WebGL2 here cannot render to float textures');
    this.gl = gl;
    this.maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.maxDrawBuffers = Math.min(gl.getParameter(gl.MAX_DRAW_BUFFERS), gl.getParameter(gl.MAX_COLOR_ATTACHMENTS));
    this.programs = new Map();
    this.pool = new Map();
    this.fb = gl.createFramebuffer();
    this.vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    this.bytes = 0;
    this.renderer = (() => {
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    })();
  }

  /** `per`: values a texel holds (4 floats, or 8 packed halves). */
  _shape(size, per = 4, rowTexels = 0) {
    const texels = Math.max(1, Math.ceil(size / per));
    // `rowTexels` not dividing TW: a multiple of it instead, so rows of that length stay whole.
    const unit = rowTexels > 0 && TW % rowTexels ? rowTexels * Math.max(1, Math.floor(TW / rowTexels)) : TW;
    let w = Math.min(unit, texels);
    // Too tall: wider by whole units (still a multiple of every row length that divides the unit).
    for (let m = 2; Math.ceil(texels / w) > this.maxSize && unit * m <= this.maxSize; m *= 2) w = Math.min(unit * m, texels);
    const h = Math.ceil(texels / w);
    if (h > this.maxSize) throw new Error(`tensor of ${size} floats is too large for this GPU's textures`);
    return { w, h };
  }

  /** `half`: false (RGBA32F), true (RGBA16F) or 'packed' (RGBA32UI, eight float16 values a texel, two a channel). */
  _texture(w, h, half) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, half === 'packed' ? gl.RGBA32UI : half ? gl.RGBA16F : gl.RGBA32F, w, h);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    this.bytes += w * h * (half ? 8 : 16);
    return tex;
  }

  /** An activation tensor of `size` floats (contents undefined until written); `half`: stored as float16. */
  tensor(size, half = false, rowTexels = 0) {
    const { w, h } = this._shape(size, half === 'packed' ? 8 : 4, rowTexels);
    const key = `${w}x${h}:${half === 'packed' ? 'u32' : half ? 'f16' : 'f32'}`;
    const free = this.pool.get(key);
    const tex = free?.length ? free.pop() : this._texture(w, h, half);
    return new GLTensor(this, size, tex, w, h, half);
  }

  _release(t) {
    // By format too: a texture comes back only as the format it was made in.
    const key = `${t.w}x${t.h}:${t.half === 'packed' ? 'u32' : t.half ? 'f16' : 'f32'}`;
    if (!this.pool.has(key)) this.pool.set(key, []);
    this.pool.get(key).push(t.tex);
  }

  /** A tensor holding `data` (Float32Array or array). `half` stores it as float16 (weights). */
  upload(data, { half = false, weight = false } = {}) {
    const src = data instanceof Float32Array ? data : Float32Array.from(data);
    const { w, h } = this._shape(src.length);
    const t = weight
      ? new GLTensor(this, src.length, this._texture(w, h, half), w, h, half)
      : this.tensor(src.length);
    t.weight = weight;
    this.write(t, src);
    return t;
  }

  /** Raw 32-bit words into a packed (RGBA32UI) tensor's texture (int8 codes, sixteen a texel, or packed halves). */
  writeU32(t, words) {
    const gl = this.gl;
    const full = t.w * t.h * 4;
    let data = words;
    if (words.length !== full) {
      data = new Uint32Array(full);
      data.set(words.subarray(0, Math.min(words.length, full)));
    }
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, t.w, t.h, gl.RGBA_INTEGER, gl.UNSIGNED_INT, data);
  }

  write(t, src) {
    const gl = this.gl;
    const full = t.w * t.h * 4;
    let data = src;
    if (src.length !== full) {
      data = new Float32Array(full);
      data.set(src.subarray(0, Math.min(src.length, full)));
    }
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, t.w, t.h, gl.RGBA, gl.FLOAT, data);
  }

  /** The tensor's values (blocks until the GPU is done). */
  read(t, size = t.size) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
    const rows = Math.ceil(Math.ceil(size / 4) / t.w);
    const out = new Float32Array(t.w * rows * 4);
    gl.readPixels(0, 0, t.w, rows, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out.subarray(0, size);
  }

  /** As `read`, after the GPU has finished, polled, so the page stays responsive while a long pass runs. */
  async readAsync(t, size = t.size) {
    await this.idle();
    return this.read(t, size);
  }

  /**
   * Resolves once the GPU has run everything issued so far (a fence, polled, so the page stays responsive). (Polled by
   * message-channel yields, unclamped, near the expected end: no faster in a trunk pass, M2.)
   */
  async idle() {
    const gl = this.gl;
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    for (;;) {
      const st = gl.clientWaitSync(sync, 0, 0);
      if (st === gl.ALREADY_SIGNALED || st === gl.CONDITION_SATISFIED || st === gl.WAIT_FAILED) break;
      await new Promise((r) => setTimeout(r, 4));
    }
    gl.deleteSync(sync);
  }

  _program(key, body) {
    let p = this.programs.get(key);
    if (p) return p;
    const gl = this.gl;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        throw new Error(`WebGL kernel ${key}: ${gl.getShaderInfoLog(s)}\n${src}`);
      }
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, PRELUDE + body));
    gl.bindAttribLocation(prog, 0, 'p');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`WebGL kernel ${key}: ${gl.getProgramInfoLog(prog)}`);
    p = { prog, loc: new Map() };
    this.programs.set(key, p);
    return p;
  }

  _loc(p, name) {
    if (!p.loc.has(name)) p.loc.set(name, this.gl.getUniformLocation(p.prog, name));
    return p.loc.get(name);
  }

  /**
   * Run kernel `key` (GLSL `body`, which defines `vec4 compute(int t)` for output texel `t`) into `out`.
   * `inputs`: { samplerName: tensor }; `ints` / `floats`: { uniformName: value }. `work` is the loop length of one
   * fragment, so a long draw is split into bands (and flushed between them) before a slow GPU's watchdog notices.
   */
  kernel(key, body, out, { inputs = {}, ints = {}, floats = {}, work = 1, quad = false } = {}) {
    const gl = this.gl;
    // Four outputs (an array of four tensors of one shape): `body` defines compute4(t, r0..r3) writing all of them.
    const outs = Array.isArray(out) ? out : [out];
    const mrt = outs.length > 1;
    const ot = outs[0].half === 'packed' ? 'uvec4' : 'vec4';
    // A float16 target is written rounded to the nearest float16 value, ties to even, first: the conversion on writing
    // truncates (Apple M2, ANGLE Metal), and truncation's bias added up over a trunk pass (z in float16 then lost 6e-3
    // relative on motif_idx_1prw, 4e-4 rounded).
    const h16 = outs[0].half === true && !this.noR16;
    const p = this._program(`${mrt ? `${key}#mrt` : key}${h16 ? ':h16' : ''}${quad ? ':quad' : ''}`, mrt ? `${outs.map((_, i) => `layout(location = ${i}) out ${ot} ${h16 ? 'g' : 'f'}${i};`).join('\n')}
${h16 ? `${R16}
vec4 ${outs.map((_, i) => `f${i}`).join(', ')};` : ''}
${body}
void main() {
  int t = int(gl_FragCoord.y) * uOutW + int(gl_FragCoord.x);
  ${quad ? '// (no early return: the whole 2 x 2 quad runs, its fragments exchanging values by derivatives)' : `if (t >= uOutTexels) { ${outs.map((_, i) => `${h16 ? 'g' : 'f'}${i} = ${ot}(0);`).join(' ')} return; }`}
  compute4(t, ${outs.map((_, i) => `f${i}`).join(', ')});${h16 ? `\n  ${outs.map((_, i) => `g${i} = r16(f${i});`).join(' ')}` : ''}
}` : outs[0].half === 'packed' ? `out uvec4 fragOut;
${body}
void main() {
  int t = int(gl_FragCoord.y) * uOutW + int(gl_FragCoord.x);
  if (t >= uOutTexels) { fragOut = uvec4(0u); return; }
  fragOut = compute(t);
}` : `out vec4 fragOut;
${h16 ? R16 : ''}
${body}
void main() {
  int t = int(gl_FragCoord.y) * uOutW + int(gl_FragCoord.x);
  if (t >= uOutTexels) { fragOut = vec4(0.0); return; }
  fragOut = ${h16 ? 'r16(compute(t))' : 'compute(t)'};
}`);
    out = outs[0];
    gl.useProgram(p.prog);
    if (this.counts) this.counts.set(key, (this.counts.get(key) ?? 0) + 1);
    let unit = 0;
    for (const [name, t] of Object.entries(inputs)) {
      if (!t?.tex) throw new Error(`kernel ${key}: input ${name} is missing`);
      if (outs.includes(t)) throw new Error(`kernel ${key}: output aliases input ${name}`);
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t.tex);
      gl.uniform1i(this._loc(p, name), unit);
      unit += 1;
    }
    gl.uniform1i(this._loc(p, 'uOutW'), out.w);
    gl.uniform1i(this._loc(p, 'uOutTexels'), out.texels);
    for (const [name, v] of Object.entries(ints)) gl.uniform1i(this._loc(p, name), v);
    for (const [name, v] of Object.entries(floats)) gl.uniform1f(this._loc(p, name), v);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
    outs.forEach((o, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, o.tex, 0));
    if (mrt) gl.drawBuffers(outs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    gl.viewport(0, 0, out.w, out.h);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vb);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    // About 2^31 inner-loop steps per band: only a very long draw is split (bands cost ~25% on a tiled GPU, each a
    // render pass of its own; a 43k x 384 x 256 matmul is 18 ms whole on an M2).
    const rowsPerBand = Math.max(1, Math.floor((this.bandWork ?? 2 ** 31) / Math.max(1, out.w * work)));
    if (this.skipDraw || this.skipKeys?.test(key)) {
      // (timing probe: everything but the draw)
    } else if (rowsPerBand >= out.h) {
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    } else {
      gl.enable(gl.SCISSOR_TEST);
      for (let y = 0; y < out.h; y += rowsPerBand) {
        gl.scissor(0, y, out.w, Math.min(rowsPerBand, out.h - y));
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.flush();
      }
      gl.disable(gl.SCISSOR_TEST);
    }
    if (mrt) {
      // Back to one attachment: a stale one could be sampled by the next kernel (a feedback loop).
      for (let i = 1; i < outs.length; i += 1) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, null, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // Profiling (`this.profile = new Map()`): each kernel waited for and timed, by key.
    if (this.profile) {
      // A one-texel read waits for the draw (Chrome's finish() does not); a float texture's (an integer one cannot be
      // read as floats), drawn after it, since GL runs in order.
      if (out.half === 'packed') {
        this.syncT ??= this.tensor(4);
        const prof = this.profile;
        this.profile = null;
        this.kernel('profile-sync', 'vec4 compute(int t) { return vec4(1.0); }', this.syncT);
        this.profile = prof;
        this.read(this.syncT, 4);
      } else {
        this.read(out, 4);
      }
      const now = performance.now();
      const k = `${key} ${work}${this.profileSizes ? ` @${out.size ?? outs?.[0]?.size}` : ""}`;
      this.profile.set(k, (this.profile.get(k) ?? 0) + now - (this.lastT ?? now));
      this.lastT = now;
    }
    return out;
  }

  /**
   * `out[e] = expr` for every element `e < size`, `expr` GLSL over `e` (an int) that may read the inputs with
   * `el(name, index)`; `pre` declares helpers. A new tensor unless `out` is given.
   */
  elementwise(key, expr, size, { inputs = {}, ints = {}, floats = {}, pre = '', work = 1, out = null, half = false, rowTexels = 0 } = {}) {
    const decl = [
      ...Object.entries(inputs).map(([n, t]) => `uniform ${t.half === 'packed' ? 'highp usampler2D' : 'sampler2D'} ${n};`),
      ...Object.keys(ints).map((n) => `uniform int ${n};`),
      ...Object.keys(floats).map((n) => `uniform float ${n};`),
      'uniform int uSize;',
    ].join('\n');
    const body = `${decl}
${pre}
float value(int e) { return ${expr}; }
${half === 'packed' ? `float v8(int e) { return e < uSize ? value(e) : 0.0; }
uvec4 compute(int t) {
  int e = t * 8;
  return uvec4(packHalf2x16(vec2(v8(e), v8(e + 1))), packHalf2x16(vec2(v8(e + 2), v8(e + 3))),
    packHalf2x16(vec2(v8(e + 4), v8(e + 5))), packHalf2x16(vec2(v8(e + 6), v8(e + 7))));
}` : `vec4 compute(int t) {
  int e = t * 4;
  return vec4(value(e), e + 1 < uSize ? value(e + 1) : 0.0, e + 2 < uSize ? value(e + 2) : 0.0, e + 3 < uSize ? value(e + 3) : 0.0);
}`}`;
    const target = out ?? this.tensor(size, half, rowTexels);
    return this.kernel(`ew:${key}${half === 'packed' ? ':pk' : ''}`, body, target, { inputs, ints: { ...ints, uSize: size }, floats, work });
  }

  destroy() {
    const gl = this.gl;
    for (const list of this.pool.values()) for (const tex of list) gl.deleteTexture(tex);
    this.pool.clear();
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
