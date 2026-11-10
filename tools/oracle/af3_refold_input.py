#!/usr/bin/env python
"""A design as the one file the page's refold reads: RFD4's own AF3-oracle input, as a multi-block mmCIF.

    python tools/oracle/af3_refold_input.py design.cif refold-input.cif [--template-query "sa_mask('C_CRD') & ~is_nucleic_acid"]

Needs RFD4-Proteina (`pip install rfd4-proteina`) and its AtomWorks (atomworks==3.0.4: the version its pin resolves to
and the one that has `to_cif_string`). Nothing is re-derived here. The design is parsed the way `evaluate.py` parses
one and handed to RFD4's own construction of the input AF3 folds: `rfproteina.af3.af3_input.build_af3_input` where the
installed RFD4-Proteina has it (RosettaCommons/RFD4-Proteina#96 made it public), else `AF3Oracle._build_manifest`,
which it was before. Either way it is the oracle's code, and builds: the design's CIF
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
    from rfproteina.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY

    atom_array = parse(args.design, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
    name = args.name or args.design.stem
    try:
        from rfproteina.af3.af3_input import build_af3_input
    except ImportError:  # an RFD4-Proteina from before the construction was public: the oracle's private method
        from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
        from rfproteina.metrics.oracles.oracle_metric import OracleJobInput

        oracle = AF3Oracle(template_query=args.template_query)
        with tempfile.TemporaryDirectory() as tmp:
            job = OracleJobInput(atom_array=atom_array, example_id=name, out_dir=Path(tmp))
            entries, _ = oracle._build_manifest([job], Path(tmp))
        entry = entries[0]
    else:
        af3_input = build_af3_input(atom_array, name, template_query=args.template_query)
        entry = {"cif": af3_input.cif, "user_ccd": af3_input.user_ccd, "template": af3_input.template}
    args.out.write_text(refold_input_text(entry))
    print(f"{args.out}: {args.out.stat().st_size} bytes, {len(blocks_of(args.out.read_text()))} blocks")


if __name__ == "__main__":
    main()
