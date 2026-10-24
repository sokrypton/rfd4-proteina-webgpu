/**
 * Minimal safetensors reader for the browser and Node: typed-array views over one ArrayBuffer (zero-copy when the
 * tensor is aligned, copied otherwise). I64 is narrowed to Int32Array; BF16/F16 are widened to Float32Array.
 */

const CTORS = { F32: Float32Array, F64: Float64Array, I32: Int32Array, I16: Int16Array, I8: Int8Array, U8: Uint8Array, BOOL: Uint8Array };

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

/**
 * @param {ArrayBuffer} buffer
 * @returns {{ tensors: Record<string, {shape: number[], dtype: string, data: ArrayLike<number>}>, metadata: object }}
 */
export function parseSafetensors(buffer) {
  const dv = new DataView(buffer);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen)));
  const base = 8 + headerLen;
  const tensors = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [s, e] = info.data_offsets;
    const at = base + s;
    const bytes = e - s;
    let data;
    if (info.dtype === 'I64') {
      const src = new DataView(buffer, at, bytes);
      data = new Int32Array(bytes / 8);
      for (let i = 0; i < data.length; i += 1) data[i] = Number(src.getBigInt64(i * 8, true));
    } else if (info.dtype === 'F16' || info.dtype === 'BF16') {
      const src = new DataView(buffer, at, bytes);
      data = new Float32Array(bytes / 2);
      for (let i = 0; i < data.length; i += 1) {
        const h = src.getUint16(i * 2, true);
        data[i] = info.dtype === 'F16' ? halfToFloat(h) : new Float32Array(new Uint32Array([h << 16]).buffer)[0];
      }
    } else {
      const Ctor = CTORS[info.dtype];
      if (!Ctor) throw new Error(`${name}: unsupported safetensors dtype ${info.dtype}`);
      data = at % Ctor.BYTES_PER_ELEMENT === 0
        ? new Ctor(buffer, at, bytes / Ctor.BYTES_PER_ELEMENT)
        : new Ctor(buffer.slice(at, at + bytes));
    }
    tensors[name] = { shape: info.shape, dtype: info.dtype, data };
  }
  return { tensors, metadata: header.__metadata__ ?? {} };
}

export const DEFAULT_LOCAL_WEIGHTS_URL = './weights/model.safetensors';
export const DEFAULT_LOCAL_SHARDED_DIR = './weights/int8';
export const DEFAULT_LOCAL_CONFIG_URL = './weights/config.json';
export const DEFAULT_HF_REPO = 'nvidia/RFD4-Proteina';
export const DEFAULT_HF_WEIGHTS_URL = `https://huggingface.co/${DEFAULT_HF_REPO}/resolve/main/model.safetensors`;
export const DEFAULT_HF_CONFIG_URL = `https://huggingface.co/${DEFAULT_HF_REPO}/resolve/main/config.json`;
const WEIGHT_CACHE_NAME = 'rfd4p-weights-v1';
const MAX_CONCURRENT_DOWNLOADS = 8;
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);
const CACHE_TIMEOUT_MS = 2000;

/**
 * Whether `url` is served by another machine (worth keeping in the Cache API). Relative URLs are resolved against the
 * page, so the deployed site's `./weights/...` counts as remote; a local dev server does not.
 */
function isRemoteUrl(url) {
  let resolved;
  try {
    resolved = new URL(url, globalThis.location?.href);
  } catch {
    return false;
  }
  return /^https?:$/.test(resolved.protocol) && !['localhost', '127.0.0.1', '[::1]'].includes(resolved.hostname);
}

function withTimeout(promise, fallback) {
  return Promise.race([
    promise,
    new Promise((resolve) => setTimeout(() => resolve(fallback), CACHE_TIMEOUT_MS)),
  ]);
}

async function fetchWithRetry(url, signal = null) {
  let lastStatus = 0;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const res = await fetch(url, { signal });
    if (res.ok) return res;
    lastStatus = res.status;
    if (!RETRYABLE_STATUS.has(res.status) || attempt === 4) break;
    await res.body?.cancel();
    await new Promise((r) => setTimeout(r, 250 * 2 ** attempt));
  }
  throw new Error(`Failed to fetch ${url}: HTTP ${lastStatus}`);
}

function viewAligned(Kind, buffer, byteOffset, length) {
  const width = Kind.BYTES_PER_ELEMENT;
  if (byteOffset % width === 0) return new Kind(buffer, byteOffset, length);
  return new Kind(buffer.slice(byteOffset, byteOffset + length * width));
}

/**
 * Decode a single tensor record from a shard `ArrayBuffer` into a `Float32Array`,
 * matching `../localfold/shared/weights/dtype.js` (`float32`, `float16`, and symmetric block-64 `int8`).
 *
 * @param {{dtype: string, shape: number[], byteOffset: number, block?: number, scaleOffset?: number}} record
 * @param {ArrayBuffer} buffer
 * @param {boolean} [copy=false]
 * @returns {Float32Array}
 */
export function readShardedTensor(record, buffer, copy = false) {
  const elements = record.shape.reduce((p, v) => p * v, 1);
  const { dtype, byteOffset = 0 } = record;

  if (dtype === 'float32') {
    if (!copy && byteOffset % 4 === 0) {
      return new Float32Array(buffer, byteOffset, elements);
    }
    return new Float32Array(buffer.slice(byteOffset, byteOffset + elements * 4));
  }

  if (dtype === 'float16') {
    if (typeof Float16Array === 'function') {
      const f16 = viewAligned(Float16Array, buffer, byteOffset, elements);
      return Float32Array.from(f16);
    }
    const u16 = viewAligned(Uint16Array, buffer, byteOffset, elements);
    const out = new Float32Array(elements);
    for (let i = 0; i < elements; i += 1) out[i] = halfToFloat(u16[i]);
    return out;
  }

  if (dtype === 'int8') {
    const block = record.block ?? 64;
    const scaleAt = record.scaleOffset;
    const codes = new Int8Array(buffer, byteOffset, elements);
    const blocks = Math.ceil(elements / block);
    const out = new Float32Array(elements);
    if (typeof Float16Array === 'function') {
      const scales = viewAligned(Float16Array, buffer, scaleAt, blocks);
      for (let b = 0; b < blocks; b += 1) {
        const scale = scales[b];
        const start = b * block;
        const end = Math.min(start + block, elements);
        for (let i = start; i < end; i += 1) out[i] = codes[i] * scale;
      }
    } else {
      const scaleBits = viewAligned(Uint16Array, buffer, scaleAt, blocks);
      for (let b = 0; b < blocks; b += 1) {
        const scale = halfToFloat(scaleBits[b]);
        const start = b * block;
        const end = Math.min(start + block, elements);
        for (let i = start; i < end; i += 1) out[i] = codes[i] * scale;
      }
    }
    return out;
  }

  throw new Error(`Unsupported sharded tensor dtype: ${dtype}`);
}

/**
 * Load a `../localfold`-style sharded bundle (`manifest.json` + `shard_XXX.bin`) over HTTP
 * with 8-way concurrent shard downloads, Cache API persistence, and optional lazy Encoder loading.
 *
 * @param {string} [baseDir] Directory URL containing `manifest.json` and `.bin` shards
 * @param {{
 *   includeEncoder?: boolean,
 *   onProgress?: (p: {bytesLoaded: number, loadedBytes: number, totalBytes: number, loadedShards: number, totalShards: number, fraction: number}) => void,
 *   signal?: AbortSignal,
 *   cache?: boolean,
 * }} [opts]
 */
export async function fetchShardedWeights(baseDir = DEFAULT_LOCAL_SHARDED_DIR, opts = {}) {
  const cleanBase = baseDir.replace(/\/+$/, '');
  const isRemote = isRemoteUrl(cleanBase);
  const {
    includeEncoder = false,
    onProgress = null,
    signal = null,
    cache: useCache = isRemote,
  } = opts;

  const manifestRes = await fetchWithRetry(`${cleanBase}/manifest.json`, signal);
  const manifest = await manifestRes.json();
  const tensorsMeta = manifest.tensors ?? {};

  // Group tensor names by shard file
  const byShard = new Map();
  for (const [name, rec] of Object.entries(tensorsMeta)) {
    let list = byShard.get(rec.file);
    if (!list) {
      list = [];
      byShard.set(rec.file, list);
    }
    list.push([name, rec]);
  }

  const allShards = manifest.shards ?? [...byShard.keys()].map((file) => ({ file, bytes: 0, role: 'core' }));
  let cacheStore = null;
  if (useCache && typeof caches !== 'undefined') {
    const cacheName = `rfd4p-sharded-${manifest.bundle?.format ?? 'int8'}-${manifest.bundle?.bytes ?? 0}`;
    cacheStore = await withTimeout(caches.open(cacheName).catch(() => null), null);
    if (cacheStore) {
      // Ask the browser not to evict the cached shards under storage pressure (granted silently in Chrome for
      // engaged sites; a no-op elsewhere). Persisted storage is never evicted, so drop shard caches of older bundles.
      navigator.storage?.persist?.().catch(() => {});
      caches.keys()
        .then((names) => Promise.all(names
          .filter((name) => name.startsWith('rfd4p-sharded-') && name !== cacheName)
          .map((name) => caches.delete(name))))
        .catch(() => {});
    }
  }

  const tensors = {};

  async function downloadShardList(shardList, progressCb) {
    const totalBytes = shardList.reduce((acc, s) => acc + (s.bytes || 0), 0);
    const totalShards = shardList.length;
    let bytesLoaded = 0;
    let loadedShards = 0;
    let cursor = 0;

    async function worker() {
      while (cursor < shardList.length) {
        const idx = cursor++;
        const shard = shardList[idx];
        const shardUrl = `${cleanBase}/${shard.file}`;
        let buf = null;

        if (cacheStore) {
          const hit = await withTimeout(cacheStore.match(shardUrl).catch(() => null), null);
          if (hit && hit.ok) {
            buf = await hit.arrayBuffer();
          }
        }

        if (!buf) {
          const res = await fetchWithRetry(shardUrl, signal);
          buf = await res.arrayBuffer();
          if (cacheStore) {
            await withTimeout(
              cacheStore
                .put(
                  shardUrl,
                  new Response(buf.slice(0), {
                    headers: {
                      'content-type': 'application/octet-stream',
                      'content-length': String(buf.byteLength),
                    },
                  }),
                )
                .catch(() => null),
              null,
            );
          }
        }

        const entries = byShard.get(shard.file) ?? [];
        // Keep Float32Array views when dtype === 'float32', otherwise decode into owned Float32Array
        for (const [name, rec] of entries) {
          const data = readShardedTensor(rec, buf, rec.dtype === 'float32');
          tensors[name] = { shape: rec.shape, dtype: 'F32', data };
        }

        bytesLoaded += shard.bytes || buf.byteLength;
        loadedShards += 1;
        if (progressCb) {
          progressCb({
            bytesLoaded,
            loadedBytes: bytesLoaded,
            totalBytes: totalBytes || bytesLoaded,
            loadedShards,
            totalShards,
            fraction: totalBytes > 0 ? bytesLoaded / totalBytes : loadedShards / totalShards,
          });
        }
      }
    }

    const nWorkers = Math.min(MAX_CONCURRENT_DOWNLOADS, shardList.length);
    await Promise.all(Array.from({ length: nWorkers }, () => worker()));
  }

  const coreShards = includeEncoder ? allShards : allShards.filter((s) => s.role !== 'encoder');
  const encoderShards = allShards.filter((s) => s.role === 'encoder');
  let encoderLoaded = includeEncoder || encoderShards.length === 0;

  await downloadShardList(coreShards, onProgress);

  let config = null;
  try {
    config = await fetchRfd4pConfig(`${cleanBase}/config.json`);
  } catch {
    try {
      config = await fetchRfd4pConfig(DEFAULT_LOCAL_CONFIG_URL);
    } catch {
      /* optional */
    }
  }

  return {
    tensors,
    config,
    metadata: manifest.metadata ?? {},
    manifest,
    get encoderLoaded() {
      return encoderLoaded;
    },
    async loadEncoder(encProgress = onProgress) {
      if (encoderLoaded) return tensors;
      await downloadShardList(encoderShards, encProgress);
      encoderLoaded = true;
      return tensors;
    },
  };
}

/**
 * Fetch and parse RFD4-Proteina `config.json` (tries local `./weights/config.json`, then HuggingFace).
 * @param {string} [url]
 * @returns {Promise<object>} Unwrapped model configuration (`{ nn, autoencoder_config, ... }`)
 */
export async function fetchRfd4pConfig(url = DEFAULT_LOCAL_CONFIG_URL) {
  let res = await fetch(url).catch(() => null);
  if (!res || !res.ok) {
    if (url === DEFAULT_LOCAL_CONFIG_URL) {
      res = await fetch(DEFAULT_HF_CONFIG_URL);
    }
  }
  if (!res || !res.ok) {
    throw new Error(`Failed to fetch RFD4-Proteina config from ${url} (${res?.status ?? 'network error'})`);
  }
  const raw = await res.json();
  return raw.config ?? raw;
}

/**
 * Fetch and parse a `.safetensors` checkpoint from `url`, using the browser Cache API when available
 * and streaming into a single zero-copy `ArrayBuffer` with progress callbacks.
 *
 * @param {string} [url]
 * @param {{onProgress?: (p: {bytesLoaded: number, loadedBytes: number, totalBytes: number, fraction: number, cached?: boolean}) => void, signal?: AbortSignal, cache?: boolean}} [opts]
 */
export async function fetchSafetensors(url = DEFAULT_LOCAL_WEIGHTS_URL, opts = {}) {
  const isRemote = isRemoteUrl(url);
  const { onProgress = null, signal = null, cache: useCache = isRemote } = opts;
  let cacheStore = null;
  if (useCache && typeof caches !== 'undefined') {
    try {
      cacheStore = await caches.open(WEIGHT_CACHE_NAME);
      const hit = await cacheStore.match(url);
      if (hit && hit.ok) {
        const buf = await hit.arrayBuffer();
        if (onProgress) {
          onProgress({
            bytesLoaded: buf.byteLength,
            loadedBytes: buf.byteLength,
            totalBytes: buf.byteLength,
            fraction: 1,
            cached: true,
          });
        }
        return parseSafetensors(buf);
      }
    } catch {
      cacheStore = null;
    }
  }

  const res = await fetch(url, { signal });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
  }

  const totalBytes = Number(res.headers.get('content-length')) || 0;
  let buffer;
  if (onProgress && res.body && totalBytes > 0) {
    const u8 = new Uint8Array(totalBytes);
    const reader = res.body.getReader();
    let offset = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      u8.set(value, offset);
      offset += value.byteLength;
      onProgress({
        bytesLoaded: offset,
        loadedBytes: offset,
        totalBytes,
        fraction: offset / totalBytes,
        cached: false,
      });
    }
    buffer = u8.buffer;
  } else {
    buffer = await res.arrayBuffer();
    if (onProgress) {
      onProgress({
        bytesLoaded: buffer.byteLength,
        loadedBytes: buffer.byteLength,
        totalBytes: buffer.byteLength,
        fraction: 1,
        cached: false,
      });
    }
  }

  if (cacheStore) {
    try {
      await cacheStore.put(
        url,
        new Response(buffer, {
          headers: {
            'content-type': 'application/octet-stream',
            'content-length': String(buffer.byteLength),
          },
        }),
      );
    } catch {
      // Quota exceeded or opaque origin — ignore cache write failure
    }
  }

  return parseSafetensors(buffer);
}
