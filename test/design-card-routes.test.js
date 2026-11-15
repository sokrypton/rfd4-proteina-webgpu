/**
 * The read routes the design card (AtomWorks Studio) uses on a document: a chain's plan, a preview of a rebuild, the chain as
 * loaded, and where new-residue segments came from. They answer what `chainPlan`, `previewRebuild`, `chainSources` and
 * `segmentOrigin` give, so a REST-only card sees the same as the page's own did.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { StudioDoc, getSharedStudioStore, handleStudioApiRequest } from '../web/studio-api.js';

const atom = (k, name, el, seq, x) =>
  `ATOM ${k} ${el} ${name} ALA A 1 ${seq} ${seq} ALA A ${name} ${x}.000 0.000 0.000 1.00 0.00 0 1`;
const header = ['data_t', 'loop_', ...['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_comp_id', 'label_asym_id',
  'label_entity_id', 'label_seq_id', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id', 'Cartn_x', 'Cartn_y',
  'Cartn_z', 'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge', 'pdbx_PDB_model_num'].map((c) => `_atom_site.${c}`)];
const lines = [];
let n = 0;
for (let r = 1; r <= 20; r += 1) for (const [name, el] of [['N', 'N'], ['CA', 'C'], ['C', 'C'], ['O', 'O']]) lines.push(atom(++n, name, el, r, r * 4 + (n % 4)));
const sid = 'design_card_routes_test';
const open = () => {
  const doc = new StudioDoc([...header, ...lines, ''].join('\n'), 't', { preset: 'rcsb' });
  getSharedStudioStore().sessions.set(sid, doc);
  return doc;
};
const call = async (path, method = 'GET', body = null) => {
  const res = await handleStudioApiRequest(new URL(`/api/v1/sessions/${sid}${path}`, 'http://localhost/'), { method, body: body && JSON.stringify(body) });
  return { status: res.status, json: await res.json() };
};

test('plan: the chain as loaded is one kept part', async () => {
  const doc = open();
  const { status, json } = await call('/chains/A/plan');
  assert.equal(status, 200);
  assert.deepEqual(json.parts.map((p) => [p.kind, p.lo, p.hi]), [['kept', 1, 20]]);
  assert.deepEqual(json, JSON.parse(JSON.stringify(doc.chainPlan('A'))));
});

test('preview: the rebuilt chain as a CIF, the session unchanged', async () => {
  const doc = open();
  const before = doc.atoms.nAtoms;
  const { status, json } = await call('/chains/preview', 'POST', { chain_id: 'A', parts: [{ kind: 'kept', lo: 1, hi: 5 }, { kind: 'new', min: 3, max: 4 }, { kind: 'kept', lo: 10, hi: 20 }] });
  assert.equal(status, 200);
  assert.match(json.cif, /^data_/);
  assert.equal(doc.atoms.nAtoms, before);
});

test('source: the chain as loaded, or 404', async () => {
  open();
  const { status, json } = await call('/chains/A/source');
  assert.equal(status, 200);
  assert.equal(json.chain, 'A');
  assert.equal((json.cif.match(/^ATOM /gm) ?? []).length, 80);
  assert.equal((await call('/chains/Q/source')).status, 404);
});

test('origins: a segment that replaced residues remembers them', async () => {
  const doc = open();
  assert.deepEqual((await call('/segments/origins')).json, { origins: {} });
  const res = await call('/segments/replace', 'POST', { atom_indices: Array.from({ length: 24 }, (_, i) => 20 + i), minimum: 6, maximum: 6 });
  assert.equal(res.status, 200);
  const { origins } = (await call('/segments/origins')).json;
  const seg = Object.keys(origins);
  assert.equal(seg.length >= 1, true);
  assert.equal(origins[seg[0]].count, 6);
  assert.deepEqual(origins[seg[0]].keys.length, 6);
  assert.equal(Boolean(doc.segmentOrigin(Number(seg[0]))), true);
});
