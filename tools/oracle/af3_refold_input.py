#!/usr/bin/env python
"""A design as the one file the page's refold reads: RFD4's own AF3-oracle input, as a multi-block mmCIF.

    python tools/oracle/af3_refold_input.py design.cif refold-input.cif [--template-query "sa_mask('C_CRD') & ~is_nucleic_acid"]

Needs RFD4-Proteina and its AtomWorks (rfd4-proteina on PYTHONPATH, atomworks==3.0.4: the version its pin resolves to
and the one that has `to_cif_string`). Nothing is re-derived here. The design is parsed the way `evaluate.py` parses
one and handed to `AF3Oracle._build_manifest`, which is what builds the input AF3 folds in the oracle: the design's CIF
with its chains disambiguated and its covalent modifiers renamed, a *user CCD* defining every non-standard residue and
ligand from the design's own atoms and bonds, and templates where the oracle's `template_query` selects atoms. This
file is those parts in one text, so a refold needs nothing else:

    data_<name>                  the manifest's `cif`: the design (chains, atoms, _chem_comp types, _struct_conn)
    data_<code> ...              the manifest's `user_ccd`, one dictionary block per non-standard residue / ligand
    data_template_chain_<X> ...  the manifest's `template[X].mmcif`, plus a `_refold_template` category carrying
                                 its `query_indices` and `template_indices` (the manifest's JSON fields)

web/refold-input.js reads it (`foldInputFromRefoldCif`); web/refold-cif.js writes the same thing in the browser for a
design the page has just made, and test/refold-cif.test.js holds that writer to this one on a design with a modified
residue and ligands the PDB does not have.
"""
from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path


def blocks_of(text: str) -> list[str]:
    return [b for b in re.split(r"(?m)^(?=data_)", text) if b.strip()]


def refold_input_text(entry: dict) -> str:
    """The manifest entry `AF3Oracle._build_manifest` made, as one multi-block CIF."""
    parts = [entry["cif"].rstrip() + "\n"]
    for block in blocks_of(entry.get("user_ccd") or ""):
        parts.append(block.rstrip() + "\n")
    for chain_id, template in (entry.get("template") or {}).items():
        block = template["mmcif"].rstrip() + "\n#\n"
        block += f"_refold_template.chain_id {chain_id}\n"
        block += f'_refold_template.query_indices "{" ".join(map(str, template["query_indices"]))}"\n'
        block += f'_refold_template.template_indices "{" ".join(map(str, template["template_indices"]))}"\n'
        parts.append(block)
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", type=Path, help="the design as a CIF (what the page's Save, or RFD4, writes)")
    ap.add_argument("out", type=Path, help="where to write the refold input")
    ap.add_argument("--template-query", default=None, help="AtomWorks query for the atoms to template (the oracle's)")
    ap.add_argument("--name", default=None, help="the job name (default: the design's file stem)")
    args = ap.parse_args()

    from atomworks.io.parser import parse
    from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
    from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
    from rfproteina.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY

    atom_array = parse(args.design, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
    oracle = AF3Oracle(template_query=args.template_query)
    name = args.name or args.design.stem
    with tempfile.TemporaryDirectory() as tmp:
        job = OracleJobInput(atom_array=atom_array, example_id=name, out_dir=Path(tmp))
        entries, _ = oracle._build_manifest([job], Path(tmp))
    args.out.write_text(refold_input_text(entries[0]))
    print(f"{args.out}: {args.out.stat().st_size} bytes, {len(blocks_of(args.out.read_text()))} blocks")


if __name__ == "__main__":
    main()
