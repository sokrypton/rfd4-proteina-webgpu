/**
 * A ligand from its Chemical Component Dictionary code, as a one-residue mmCIF the Studio can merge into a job:
 * the dictionary's ideal coordinates (its model coordinates where an ideal one is missing), heavy atoms only,
 * moved so its centre sits at `center`. Pure, so it is unit-tested without a browser; the design panel fetches the
 * entry (shared/rfd4p/pipeline/ccd.js) and merges the result through Studio's `/combine`.
 */
import { catRows, parseCifBlock } from '../shared/rfd4p/pipeline/cif.js';

/** CCD codes are 1–5 letters or digits ("SAM", "ZN", "HEM", "A1AAA"). */
export const isCcdCode = (code) => /^[A-Z0-9]{1,5}$/.test(String(code ?? ''));

const num = (v) => {
  const x = Number.parseFloat(v);
  return Number.isFinite(x) ? x : null;
};

/**
 * @param {string|object} ccd the CCD entry's text, or its parsed block
 * @param {{chainId?: string, center?: number[]|null}} [opts]
 * @returns {{cif: string, atoms: number, code: string}}
 */
export function ligandCifFromCcd(ccd, { chainId = 'L', center = null } = {}) {
  const block = typeof ccd === 'string' ? parseCifBlock(ccd) : ccd;
  const t = catRows(block?.chem_comp_atom);
  if (!t.n) throw new Error('the dictionary entry has no atoms');
  const atoms = [];
  for (let r = 0; r < t.n; r += 1) {
    const element = String(t.get('type_symbol', r) ?? '').toUpperCase();
    if (element === 'H' || element === 'D') continue;
    const ideal = ['pdbx_model_Cartn_x_ideal', 'pdbx_model_Cartn_y_ideal', 'pdbx_model_Cartn_z_ideal'].map((c) => num(t.get(c, r)));
    const model = ['model_Cartn_x', 'model_Cartn_y', 'model_Cartn_z'].map((c) => num(t.get(c, r)));
    const xyz = ideal.every((v) => v !== null) ? ideal : model.every((v) => v !== null) ? model : null;
    if (!xyz) continue;
    atoms.push({
      code: String(t.get('comp_id', r)),
      name: String(t.get('atom_id', r)),
      element,
      charge: Number.parseInt(t.get('charge', r), 10) || 0,
      xyz,
    });
  }
  if (!atoms.length) throw new Error('the dictionary entry has no coordinates');
  const code = atoms[0].code;
  const shift = [0, 1, 2].map((k) => (center ? center[k] : 0) - atoms.reduce((s, a) => s + a.xyz[k], 0) / atoms.length);
  // Atom names may hold quotes or spaces in the dictionary ("C1'"): quote them the way mmCIF does.
  const q = (s) => (/[\s'"]/.test(s) ? (s.includes('"') ? `'${s}'` : `"${s}"`) : s);
  const lines = [
    `data_${code}`,
    'loop_',
    ...['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_comp_id', 'label_asym_id', 'label_entity_id',
      'label_seq_id', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'auth_atom_id', 'Cartn_x', 'Cartn_y', 'Cartn_z',
      'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge', 'pdbx_PDB_model_num'].map((c) => `_atom_site.${c}`),
    ...atoms.map((a, k) => {
      const [x, y, z] = a.xyz.map((v, j) => (v + shift[j]).toFixed(3));
      return ['HETATM', k + 1, a.element, q(a.name), code, chainId, 1, '.', 1, code, chainId, q(a.name), x, y, z,
        '1.00', '0.00', a.charge, 1].join(' ');
    }),
    '',
  ];
  return { cif: lines.join('\n'), atoms: atoms.length, code };
}
