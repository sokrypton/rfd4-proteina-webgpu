import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { buildBatch } from '../shared/rfd4p/pipeline/index.js';
import { designSpec } from '../web/design-spec.js';
import { StudioDoc } from '../web/studio-api.js';

const binder = () => new StudioDoc(readFileSync(new URL('../web/studio/examples/protein_binder.cif', import.meta.url), 'utf8'), 'binder');
const resIdsOf = (doc, chain) => {
  const a = doc.atoms;
  const out = [];
  for (let i = 0; i < a.nAtoms; i += 1) {
    if (String(a.chain_id[i]) === chain && !/^\d+-\d+$/.test(String(a.res_name[i])) && out.at(-1) !== a.res_id[i]) out.push(a.res_id[i]);
  }
  return out;
};
const atomOf = (doc, chain, resId) => doc.atoms.chain_id.findIndex((c, i) => String(c) === chain && doc.atoms.res_id[i] === resId);

test('new residues inserted inside a chain leave its numbering alone', () => {
  const doc = binder();
  const before = resIdsOf(doc, 'A');
  doc.insertSegment({ minimum: 5, maximum: 10, atomIndex: atomOf(doc, 'A', 41), before: true });
  doc.insertSegment({ minimum: 3, maximum: 3, atomIndex: atomOf(doc, 'A', 41), before: true });
  assert.deepEqual(resIdsOf(doc, 'A'), before);
  const sentinels = doc.atoms.res_name.map((r, i) => [r, doc.atoms.res_id[i], doc.atoms.ins_code[i]]).filter(([r]) => /^\d+-\d+$/.test(String(r)));
  assert.deepEqual(sentinels.filter(([, , c]) => c).map(([r, id, c]) => `${r}@${id}${c}`), ['5-10@40A', '3-3@40B']);
  const pieces = designSpec(doc.atoms).chains.find((c) => c.id === 'A').pieces
    .map((p) => (p.kind === 'new' ? `new ${p.min}-${p.max}` : `${p.start}-${p.end}`));
  assert.deepEqual(pieces, ['17-40', 'new 5-10', 'new 3-3', '41-131']);
});

test('the model sees the same input as when the chain was renumbered', () => {
  // The old way: the sentinel takes 41 and the rest of the chain moves up by one.
  const old = binder();
  old.insertSegment({ minimum: 6, maximum: 6, atomIndex: atomOf(old, 'A', 41), before: true });
  const a = old.atoms;
  const s = a.res_name.findIndex((r) => r === '6-6');
  a.ins_code[s] = '';
  for (let i = 0; i < a.nAtoms; i += 1) {
    if (String(a.chain_id[i]) === 'A' && i > s) { a.res_id[i] += 1; a.auth_seq_id[i] += 1; }
  }
  a.res_id[s] = 41;
  a.auth_seq_id[s] = 41;
  const now = binder();
  now.insertSegment({ minimum: 6, maximum: 6, atomIndex: atomOf(now, 'A', 41), before: true });

  const lengths = { segmentLengths: [6, 50] };
  const rx = buildBatch(old.toCif(), { seed: 3, ...lengths });
  const ry = buildBatch(now.toCif(), { seed: 3, ...lengths });
  // Expanded, the chain is numbered straight through either way (no "40A"s reaching a design).
  const numbering = (atoms) => {
    const out = [];
    for (let i = 0; i < atoms.nAtoms; i += 1) {
      const k = `${atoms.res_id[i]}${atoms.ins_code?.[i] ?? ''}`;
      if (String(atoms.chain_id[i]) === 'A' && out.at(-1) !== k) out.push(k);
    }
    return out;
  };
  assert.deepEqual(numbering(ry.atoms), numbering(rx.atoms));
  assert.deepEqual(numbering(ry.atoms).slice(20, 30), ['37', '38', '39', '40', '41', '42', '43', '44', '45', '46']);
  const x = rx.batch;
  const y = ry.batch;
  assert.deepEqual(Object.keys(y).sort(), Object.keys(x).sort());
  const flat = (v) => (v?.data ? Array.from(v.data) : ArrayBuffer.isView(v) ? Array.from(v) : v);
  for (const k of Object.keys(x)) assert.deepEqual(flat(y[k]), flat(x[k]), k);
});
