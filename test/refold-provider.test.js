import test from 'node:test';
import assert from 'node:assert/strict';

import { refoldRecordForZip, refoldTermsStorage } from '../web/refold-provider.js';

test('a refold kept by Studio\'s Refold panel is handed to the card\'s zip as the page always saved one', () => {
  // What the panel hands back for a design (`panel.records(id)`): its settings and result, the page's extras in `data`.
  const record = {
    model: 'openbind0', modelLabel: 'OpenBind-0', seed: 3, recycles: 3, steps: 25, cif: 'data_overlay', pass: true,
    metrics: { rmsd: 1.1, placedRmsd: NaN, plddt: 91, minIpae: 1.4 }, text: 'OpenBind-0 refold: 1.10 Å',
    data: { pdb: 'ATOM', cif: 'data_refold', rmsd: 1.1, confidence: { meanPlddt: 91, ptm: 0.9 }, templated: ['A'], partial: [] },
  };
  const zip = refoldRecordForZip(record);
  // The refold itself (its own mmCIF, B-factors the pLDDT), not the overlay the viewer draws.
  assert.equal(zip.cif, 'data_refold');
  assert.deepEqual([zip.model, zip.seed, zip.recycles, zip.steps, zip.pass, zip.text], ['openbind0', 3, 3, 25, true, 'OpenBind-0 refold: 1.10 Å']);
  // The designed chain fitted on itself, and the refold model's confidence: the zip's own columns.
  assert.equal(zip.rmsd, 1.1);
  assert.deepEqual(zip.confidence, { meanPlddt: 91, ptm: 0.9 });
  assert.deepEqual([zip.templated, zip.partial], [['A'], []]);
  assert.equal(zip.metrics.minIpae, 1.4);
});

test('AlphaFold 3\'s terms keep the key this page stored them under, which Studio now asks for by its own', () => {
  const store = new Map();
  globalThis.localStorage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) };
  try {
    refoldTermsStorage.set('atomworks-studio.refoldTerms.af3', '1');
    assert.equal(store.get('rfd4p.refoldTerms.af3'), '1');
    assert.equal(refoldTermsStorage.get('atomworks-studio.refoldTerms.af3'), '1');
    refoldTermsStorage.set('other', 'x');
    assert.equal(store.get('other'), 'x');
  } finally { delete globalThis.localStorage; }
});
