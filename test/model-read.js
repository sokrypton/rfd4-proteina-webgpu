/**
 * A document as the model reads it (`buildBatch` of its annotated CIF), for tests that compare two ways of making the
 * same input: per chain, how many tokens, how many of them have their sequence masked (designed), how many polymer
 * tokens are held at their coordinates, and how many hotspot residues. No numbering, no card: what the model is given
 * (a ligand's atoms are held or not by their pose, which the examples do not all set the same way, so are left out). (The card's own reading of a document, `designSpec`, is Studio's and tested there.)
 */
import { buildBatch } from '../shared/rfd4p/pipeline/index.js';

const finite = (c) => Array.isArray(c) && Number.isFinite(c[0]) && Number.isFinite(c[1]) && Number.isFinite(c[2]);

export function modelRead(doc, seed = 1) {
  const a = buildBatch(doc.toCif(), { seed }).atoms;
  const sequence = a.condition_sequence_1_residue ?? a.annotations?.condition_sequence_1_residue;
  const coordinate = a.condition_coordinate_1_atom ?? a.annotations?.condition_coordinate_1_atom;
  const hotspot = a.condition_hotspot_1_atom ?? a.annotations?.condition_hotspot_1_atom;
  const tokens = new Map();
  for (let i = 0; i < a.token_id.length; i += 1) {
    const chain = String(a.chain_id[i]);
    const t = tokens.get(a.token_id[i]) ?? { chain, masked: false, held: false, hotspot: false, ligand: Boolean(a.hetero?.[i]) };
    if (a.res_name[i] === '<M>' || sequence?.[i] === '<M>') t.masked = true;
    if (finite(coordinate?.[i])) t.held = true;
    if (hotspot?.[i]) t.hotspot = true;
    tokens.set(a.token_id[i], t);
  }
  const chains = {};
  for (const t of tokens.values()) {
    const c = (chains[t.chain] ??= { tokens: 0, designed: 0, held: 0, hotspots: 0, ligand: t.ligand });
    c.tokens += 1;
    if (t.masked) c.designed += 1;
    if (t.held && !t.ligand) c.held += 1;
    if (t.hotspot) c.hotspots += 1;
  }
  return chains;
}

/** The blocks of new residues of a document, as { atomIndex, min, max } (the first atom of each), in order. */
export function segmentsOf(doc) {
  const a = doc.atoms;
  const out = [];
  for (let i = 0; i < a.nAtoms; i += 1) {
    const m = /^(\d+)-(\d+)$/.exec(String(a.res_name[i]));
    if (m && (i === 0 || a.res_name[i - 1] !== a.res_name[i] || a.res_id[i - 1] !== a.res_id[i] || String(a.chain_id[i - 1]) !== String(a.chain_id[i]))) {
      out.push({ atomIndex: i, min: Number(m[1]), max: Number(m[2]) });
    }
  }
  return out;
}
