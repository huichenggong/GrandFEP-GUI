#!/usr/bin/env python3
"""
check_constraint.py — Remove H atoms from atom_map whose constraint lengths differ
between the two ligand states.

Reads ligand_preparation/summary.json and all edge_*/mapping.json files inside the
given system directory, runs hybird_constraint_check on each edge, then writes
mapping_constraint_checked.json alongside the original mapping.json.

Usage:
    python check_constraint.py <system_dir>

Example:
    python check_constraint.py \
        tests/.../structure_inputs/waterset/hsp90_woodhead
"""

import argparse
import json
import sys
from pathlib import Path

import openmm
from openmm import app

_REPO_ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from grandfep.hybrid_topology.hybrid_factory import hybrid_constraint_check


def load_amber_system(prmtop_path: Path, inpcrd_path: Path):
    """Load AMBER prmtop/inpcrd, return (system, topology) with HBond constraints."""
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )
    return system, prmtop.topology


def process_edge(
    edge_dir: Path,
    prep_dir: Path,
    idx_to_dir: dict[int, str],
) -> None:
    parts = edge_dir.name.split("_")  # ["edge", "i", "j"]
    if len(parts) != 3:
        return
    i, j = int(parts[1]), int(parts[2])

    mapping_path = edge_dir / "mapping.json"
    if not mapping_path.exists():
        print(f"  SKIP {edge_dir.name}: mapping.json not found")
        return

    with open(mapping_path) as fh:
        mapping = json.load(fh)

    atom_map = mapping["atom_map"]  # [[local_i, local_j], ...]

    for idx, label in ((i, "i"), (j, "j")):
        if idx not in idx_to_dir:
            print(f"  SKIP {edge_dir.name}: ligand index {idx} not in summary")
            return

    dir_i = idx_to_dir[i]
    dir_j = idx_to_dir[j]

    prmtop_i = prep_dir / dir_i / "01_dry.prmtop"
    inpcrd_i = prep_dir / dir_i / "01_dry.inpcrd"
    prmtop_j = prep_dir / dir_j / "01_dry.prmtop"
    inpcrd_j = prep_dir / dir_j / "01_dry.inpcrd"

    for p in (prmtop_i, inpcrd_i, prmtop_j, inpcrd_j):
        if not p.exists():
            print(f"  SKIP {edge_dir.name}: missing {p.name}")
            return

    system_i, top_i = load_amber_system(prmtop_i, inpcrd_i)
    system_j, top_j = load_amber_system(prmtop_j, inpcrd_j)

    # Dry systems contain only the ligand, so local indices == global indices.
    new_map, removed = hybrid_constraint_check(
        atom_map, system_i, top_i, system_j, top_j
    )

    if removed:
        print(f"  {edge_dir.name}: removed {len(removed)} pair(s): {removed}")
    else:
        print(f"  {edge_dir.name}: OK")

    # Build the set of A-side indices that were removed to filter parallel lists.
    removed_a = {a for a, _ in removed}
    kept_pos = [pos for pos, (a, _) in enumerate(atom_map) if a not in removed_a]

    updated = dict(mapping)
    updated["atom_map"] = new_map

    out_path = edge_dir / "mapping_constraint_checked.json"
    with open(out_path, "w") as fh:
        json.dump(updated, fh, indent=2)
    print(f"    -> {out_path.name}")


def build_idx_to_dir(summary_molecules: list) -> dict[int, str]:
    """Map 0-based CSV ligand index to its preparation directory.

    summary.json uses 1-based molecule_index; CSV uses 0-based index.
    Molecules are listed in CSV order, so molecules[k] -> CSV index k.
    """
    return {k: entry["directory"] for k, entry in enumerate(summary_molecules)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "system_dir",
        type=Path,
        help="System directory containing ligand_index.csv, ligand_preparation/, edge_*/",
    )
    args = parser.parse_args()

    system_dir: Path = args.system_dir.resolve()
    if not system_dir.is_dir():
        print(f"ERROR: {system_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    prep_dir = system_dir / "ligand_preparation"
    summary_path = prep_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(summary_path) as fh:
        summary = json.load(fh)
    idx_to_dir = build_idx_to_dir(summary["molecules"])

    edge_dirs = sorted(
        p for p in system_dir.iterdir()
        if p.is_dir() and p.name.startswith("edge_")
    )
    if not edge_dirs:
        print(f"No edge_* directories found in {system_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"System : {system_dir.name}")
    print(f"Edges  : {len(edge_dirs)}")
    for edge_dir in edge_dirs:
        process_edge(edge_dir, prep_dir, idx_to_dir)


if __name__ == "__main__":
    main()
