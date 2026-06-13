#!/usr/bin/env python3
"""
view_edge.py — PyMOL script to visualize one LOMAP edge mapping.

Usage (command line):
    pymol -r view_edge.py -- mol_a.sdf mol_b.sdf mapping.json

Usage (inside PyMOL):
    run /path/to/view_edge.py
    view_edge mol_a.sdf, mol_b.sdf, mapping.json

Visualization key
-----------------
- Paired atoms (mapped)      : matching colored spheres in both molecules
- Unmapped atoms              : gray sticks
- Broken-bond atoms           : red spheres (larger), red sticks on the broken bond
- Grid mode                   : mol_a (left) and mol_b (right) in separate panels
"""

import colorsys
import json
import random
import sys
from pathlib import Path

from pymol import cmd  # noqa: E402


def _rank_sel(obj: str, idx: int) -> str:
    return f"({obj} and rank {idx})"


def _multi_rank_sel(obj: str, indices) -> str:
    parts = " or ".join(f"rank {i}" for i in indices)
    return f"({obj} and ({parts}))"


def view_edge(mol_a_sdf: str | Path, mol_b_sdf: str | Path, mapping_json: str | Path):
    """Visualize a LOMAP edge mapping in PyMOL.

    Parameters
    ----------
    mol_a_sdf :
        Path to mol_a SDF file.
    mol_b_sdf :
        Path to mol_b SDF file.
    mapping_json :
        Path to mapping JSON file (mapping_constraint_checked.json).
    """
    mol_a_path = Path(mol_a_sdf)
    mol_b_path = Path(mol_b_sdf)
    mapping_path = Path(mapping_json)

    for p in (mol_a_path, mol_b_path, mapping_path):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return

    with open(mapping_path) as fh:
        mapping = json.load(fh)

    atom_map = mapping["atom_map"]           # [[ia, ib], ...]
    broken_a = mapping["broken_bonds_moli"]  # [[u, v], ...]
    broken_b = mapping["broken_bonds_molj"]

    # ── Load ─────────────────────────────────────────────────────────────────
    cmd.reinitialize()
    cmd.set("retain_order", 1)
    cmd.load(str(mol_a_path), "mol_a")
    cmd.load(str(mol_b_path), "mol_b")

    # ── Base representation ───────────────────────────────────────────────────
    cmd.hide("everything", "all")
    cmd.show("sticks", "all")
    cmd.set("stick_radius", 0.12)
    cmd.set("sphere_scale", 0.30)

    # Unmapped atoms: gray
    cmd.color("gray50", "mol_a")
    cmd.color("gray50", "mol_b")

    # ── Mapped pairs: unique hue, spheres ────────────────────────────────────
    n = len(atom_map)
    _GOLDEN = 0.618033988749895   # 1/φ — maximises hue distance between neighbours
    _hue_offset = random.random()  # shift the whole color cycle each run
    for i, (ia, ib) in enumerate(atom_map):
        r, g, b = colorsys.hsv_to_rgb((_hue_offset + i * _GOLDEN) % 1.0, 0.80, 0.95)
        cname = f"pair_{i}"
        cmd.set_color(cname, [r, g, b])
        cmd.color(cname, _rank_sel("mol_a", ia))
        cmd.color(cname, _rank_sel("mol_b", ib))
        cmd.show("spheres", _rank_sel("mol_a", ia))
        cmd.show("spheres", _rank_sel("mol_b", ib))

    # ── Broken bonds: red, larger spheres, thicker sticks on the bond ────────
    def highlight_broken(obj, bonds):
        if not bonds:
            return
        all_atoms = sorted({idx for pair in bonds for idx in pair})
        sel_atoms = _multi_rank_sel(obj, all_atoms)
        combined = f"brk_{obj}"
        cmd.select(combined, sel_atoms)

        for k, (u, v) in enumerate(bonds):
            bname = f"brk_{obj}_{k}"
            cmd.select(bname, f"{obj} and (rank {u} or rank {v})")
            cmd.set("stick_radius", 0.35, bname)

    highlight_broken("mol_a", broken_a)
    highlight_broken("mol_b", broken_b)

    cmd.label("mol_a", "index - 1")
    cmd.label("mol_b", "index - 1")

    # ── Grid mode ─────────────────────────────────────────────────────────────
    cmd.set("grid_mode", 1)
    cmd.set("grid_slot", 1, "mol_a")
    cmd.set("grid_slot", 2, "mol_b")
    cmd.zoom("all", buffer=3)
    cmd.deselect()

    # ── Load element-colored copies ───────────────────────────────────────────
    cmd.load(str(mol_a_path), "mol_a_element")
    cmd.load(str(mol_b_path), "mol_b_element")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== {mol_a_path.absolute().parent.name} ===")
    print(f"  mol_a    : {mol_a_path.name}")
    print(f"  mol_b    : {mol_b_path.name}")
    print(f"  mapping  : {mapping_path.name}")
    print(f"  Mapped pairs : {n}")
    if broken_a:
        print(f"  Broken bonds mol_a : {broken_a}")
    if broken_b:
        print(f"  Broken bonds mol_b : {broken_b}")
    print()


# ── Register as PyMOL command ──────────────────────────────────────────────────
cmd.extend("view_edge", view_edge)

# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__" or "--" in sys.argv or "-r" in sys.argv:
    # pymol -r view_edge.py -- mol_a.sdf mol_b.sdf mapping.json
    args = [a for a in sys.argv[1:] if not a.startswith("-") and a != "--"]
    if len(args) >= 3:
        view_edge(args[-3], args[-2], args[-1])
