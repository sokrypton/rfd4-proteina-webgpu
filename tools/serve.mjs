/**
 * Static dev server for the studio (`npm run serve`, http://localhost:4173/).
 *
 * `python3 -m http.server` sends Last-Modified without Cache-Control, so browsers cache by heuristic and a normal
 * reload can keep an old app.js, stylesheet or Mol* bundle after an edit. This sends `Cache-Control: no-cache`, so
 * every load revalidates (an unchanged file is a cheap 304).
 *
 *   node tools/serve.mjs [port]
 */
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const PORT = Number(process.argv[2] ?? process.env.PORT ?? 4173);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.webmanifest': 'application/manifest+json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.wasm': 'application/wasm',
  '.cif': 'chemical/x-mmcif', '.zst': 'application/zstd', '.bin': 'application/octet-stream',
  '.safetensors': 'application/octet-stream', '.txt': 'text/plain; charset=utf-8',
};

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    let path = resolve(join(ROOT, decodeURIComponent(url.pathname)));
    if (path !== ROOT && !path.startsWith(ROOT + sep)) return res.writeHead(403).end();
    let info = await stat(path).catch(() => null);
    if (info?.isDirectory()) {
      path = join(path, 'index.html');
      info = await stat(path).catch(() => null);
    }
    if (!info?.isFile()) return res.writeHead(404).end('Not found');
    const etag = `"${info.size.toString(16)}-${Math.floor(info.mtimeMs).toString(16)}"`;
    const headers = { 'Cache-Control': 'no-cache', ETag: etag, 'Content-Type': TYPES[extname(path)] ?? 'application/octet-stream' };
    if (req.headers['if-none-match'] === etag) return res.writeHead(304, headers).end();
    res.writeHead(200, { ...headers, 'Content-Length': info.size });
    if (req.method === 'HEAD') return res.end();
    createReadStream(path).pipe(res);
  } catch (err) {
    res.writeHead(500).end(String(err));
  }
}).listen(PORT, () => console.log(`Serving ${ROOT} at http://localhost:${PORT}/`));
