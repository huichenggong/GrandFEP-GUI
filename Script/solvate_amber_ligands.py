#!/usr/bin/env python3
"""
Solvate AMBER ligand systems from a parameterization summary.json.

Workflow
--------
1. Load ALL ligands in tleap and combine them into one unit before solvating.
   This ensures no water molecule is placed where any ligand would sit.
2. Run solvateoct on the combined unit; extract the water-only PDB (02_water.pdb)
   and record the OCT box dimensions from the CRYST1 record.
3. For each ligand: combine mol2 + shared water in tleap with ``setBOX SYS vdw``
   (required to stamp the periodic-boundary flag in the prmtop).
4. Use ParmEd to overwrite the rectangular placeholder box with the exact OCT
   vectors parsed from the reference CRYST1 record (a = b = c = L,
   alpha = beta = gamma = 109.471 deg).

Example
-------
python solvate_amber_ligands.py summary.json --buffer 12.0 --concentration 0.15
python solvate_amber_ligands.py summary.json --concentration 0  # neutralize only

Requirements
------------
AmberTools (tleap) and ParmEd (parmed Python package) in PATH / environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path


WATER_MOLARITY = 55.5  # mol/L at 298 K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"'{name}' not found in PATH. Activate an AmberTools environment first."
        )


def require_parmed() -> None:
    try:
        import parmed  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "ParmEd is required to fix OCT box vectors.\n"
            "Install with: conda install -c conda-forge parmed"
        )


def run_tleap(script: str, work_dir: Path, log_path: Path) -> None:
    """Write tleap input, run from work_dir, append stdout/stderr to log."""
    log_path.write_text(script)
    result = subprocess.run(
        ["tleap", "-f", log_path.name],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
    )
    with log_path.open("a") as fh:
        fh.write("\n--- stdout ---\n")
        fh.write(result.stdout)
        fh.write("\n--- stderr ---\n")
        fh.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"tleap failed (return code {result.returncode}). "
            f"Check log: {log_path}"
        )


def fix_box_parmed(
    prmtop: Path,
    inpcrd: Path,
    box: tuple[float, float, float, float, float, float],
) -> None:
    """Use ParmEd to write the exact OCT box vectors into prmtop and inpcrd.

    Parameters
    ----------
    box : (a, b, c, alpha, beta, gamma)  in Å and degrees.
          For a truncated octahedron: a = b = c and alpha = beta = gamma = 109.471.
    """
    import parmed as pmd

    a, b, c, alpha, beta, gamma = box
    structure = pmd.load_file(str(prmtop), str(inpcrd))
    structure.box = [a, b, c, alpha, beta, gamma]
    structure.save(str(prmtop), overwrite=True)
    structure.save(str(inpcrd), overwrite=True)


def extract_water_pdb(
    solvated_pdb: Path,
    water_pdb: Path,
) -> tuple[int, tuple[float, float, float, float, float, float]]:
    """Extract water ATOM lines from a solvated PDB; also parse CRYST1 box.

    Returns
    -------
    (n_water_residues, (a, b, c, alpha, beta, gamma))
    """
    box_params: tuple | None = None
    water_residues: set[tuple[str, str]] = set()
    lines_out: list[str] = []
    last_was_water = False

    with solvated_pdb.open() as fh:
        for raw in fh:
            if raw.startswith("CRYST1"):
                parts = raw.split()
                box_params = (
                    float(parts[1]), float(parts[2]), float(parts[3]),
                    float(parts[4]), float(parts[5]), float(parts[6]),
                )
            elif raw.startswith(("ATOM  ", "HETATM")):
                resname = raw[17:20].strip()
                if resname in ("WAT", "HOH", "TP3"):
                    chain  = raw[21]
                    resnum = raw[22:26].strip()
                    water_residues.add((chain, resnum))
                    lines_out.append(raw)
                    last_was_water = True
                else:
                    last_was_water = False
            elif raw.startswith("TER"):
                if last_was_water:
                    lines_out.append(raw)
                last_was_water = False

    if not lines_out:
        raise RuntimeError(
            f"No water atoms (WAT/HOH/TP3) found in {solvated_pdb}. "
            "Did solvateoct succeed?"
        )
    if box_params is None:
        raise RuntimeError(f"No CRYST1 record found in {solvated_pdb}.")

    lines_out.append("END\n")
    water_pdb.write_text("".join(lines_out))
    return len(water_residues), box_params


def extract_ligand_pdb(solvated_pdb: Path, resname: str, output_pdb: Path) -> int:
    """Extract atoms of one residue from the combined solvated PDB.

    Returns the number of atoms extracted.  The atom ordering in the output
    matches the order in ``solvated_pdb``, which is the same as the original
    mol2 ordering (tleap preserves it).
    """
    lines_out: list[str] = []
    with solvated_pdb.open() as fh:
        for raw in fh:
            if raw.startswith(("ATOM  ", "HETATM")):
                if raw[17:20].strip() == resname:
                    lines_out.append(raw)
    if not lines_out:
        raise RuntimeError(
            f"No atoms with residue name '{resname}' found in {solvated_pdb}."
        )
    lines_out.append("END\n")
    output_pdb.write_text("".join(lines_out))
    return len(lines_out) - 1  # subtract END line


def make_centered_mol2(original_mol2: Path, centered_pdb: Path, output_mol2: Path) -> None:
    """Write a mol2 identical to original_mol2 but with coordinates from centered_pdb.

    ``solvateoct`` re-centers the solute before placing water, so the water PDB
    uses the centered coordinate frame.  Each per-ligand mol2 must also be in
    that frame so that the combined (ligand + water) system has no clashes.

    Coordinates are updated by direct text substitution so that all GAFF2 atom
    types, bond orders, and charges in the original mol2 are preserved exactly.
    Atom ordering in the PDB must match the mol2 (tleap preserves this when
    writing savepdb from a mol2-loaded unit).
    """
    # Read centered coordinates from PDB (fixed columns 30-37, 38-45, 46-53 Å)
    pdb_coords: list[tuple[float, float, float]] = []
    with centered_pdb.open() as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                pdb_coords.append((
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ))

    # Rewrite mol2, replacing x y z in the @<TRIPOS>ATOM section
    result_lines: list[str] = []
    in_atom = False
    atom_idx = 0

    for line in original_mol2.read_text().splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "@<TRIPOS>ATOM":
            in_atom = True
            result_lines.append(line)
            continue
        if stripped.startswith("@<TRIPOS>"):
            in_atom = False

        if in_atom and stripped:
            parts = stripped.split()
            if len(parts) >= 5:
                x, y, z = pdb_coords[atom_idx]
                # antechamber mol2: id name x y z type [subst_id subst_name charge]
                tail = " ".join(parts[5:])
                result_lines.append(
                    f"{int(parts[0]):>7} {parts[1]:<12}"
                    f" {x:>10.4f} {y:>10.4f} {z:>10.4f}"
                    f" {tail}\n"
                )
                atom_idx += 1
                continue

        result_lines.append(line)

    if atom_idx != len(pdb_coords):
        raise RuntimeError(
            f"Atom count mismatch: mol2 has {atom_idx} ATOM lines, "
            f"PDB has {len(pdb_coords)} atoms ({centered_pdb})."
        )

    output_mol2.write_text("".join(result_lines))


def calc_n_extra_ions(n_water: int, concentration: float) -> int:
    """Number of cation+anion pairs for a given NaCl concentration."""
    return round(n_water * concentration / WATER_MOLARITY)


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def build_shared_water_box(
    molecules: list[dict],
    summary_dir: Path,
    work_dir: Path,
    buffer: float,
    water_ff: str,
    ligand_ff: str,
) -> tuple[Path, int, tuple[float, float, float, float, float, float], dict[str, Path]]:
    """Solvate ALL ligands combined so no water clashes with any ligand.

    Combining all molecules before calling solvateoct ensures that the water
    positions avoid the union of all ligand volumes, then each per-ligand step
    reuses those water coordinates.

    ``solvateoct`` re-centers the solute before placing water, so the ligand
    coordinates in the original mol2 files differ from the water coordinate
    frame.  After solvation we extract each ligand's new (centered) coordinates
    from the combined PDB and write a centered mol2 — preserving all GAFF2
    atom types — so that each per-ligand tleap step loads the ligand at the
    correct position.

    Returns
    -------
    (water_pdb, n_water, (a, b, c, alpha, beta, gamma), centered_mol2s)
        centered_mol2s maps residue_name → path of the centered mol2 file.
    """
    solvated_pdb = work_dir / "00_ref_solvated.pdb"
    water_pdb    = work_dir / "02_water.pdb"
    log_path     = work_dir / "00_build_water_box_leap.in"

    # Build per-molecule load lines and collect names
    load_lines: list[str] = []
    mol_names:  list[str] = []
    for i, mol_info in enumerate(molecules):
        mol_name   = f"MOL{i + 1}"
        mol2       = (summary_dir / mol_info["mol2"]).resolve()
        frcmod     = (summary_dir / mol_info["frcmod"]).resolve()
        mol2_rel   = os.path.relpath(mol2,   work_dir)
        frcmod_rel = os.path.relpath(frcmod, work_dir)
        load_lines.append(f"{mol_name} = loadmol2 {mol2_rel}")
        load_lines.append(f"loadamberparams {frcmod_rel}")
        mol_names.append(mol_name)

    combine_expr = "{ " + " ".join(mol_names) + " }"
    load_block   = "\n".join(load_lines)
    out_rel      = os.path.relpath(solvated_pdb, work_dir)

    script = textwrap.dedent(f"""\
        source {water_ff}
        source {ligand_ff}

        {load_block}

        COMBINED = combine {combine_expr}

        solvateoct COMBINED TIP3PBOX {buffer}

        savepdb COMBINED {out_rel}

        quit
    """)

    run_tleap(script, work_dir, log_path)

    n_water, box_params = extract_water_pdb(solvated_pdb, water_pdb)

    # Build a centered mol2 for each ligand: extract its atoms from the
    # combined solvated PDB (at re-centered coordinates) then use ParmEd to
    # transplant those coordinates into the original mol2 (keeping GAFF2 types).
    centered_mol2s: dict[str, Path] = {}
    for mol_info in molecules:
        resname      = mol_info["residue_name"]
        original_mol2 = (summary_dir / mol_info["mol2"]).resolve()
        centered_pdb  = work_dir / f"00_{resname}_centered.pdb"
        centered_mol2 = work_dir / f"00_{resname}_centered.mol2"
        extract_ligand_pdb(solvated_pdb, resname, centered_pdb)
        make_centered_mol2(original_mol2, centered_pdb, centered_mol2)
        centered_mol2s[resname] = centered_mol2

    return water_pdb, n_water, box_params, centered_mol2s


def solvate_one_ligand(
    mol_info: dict,
    summary_dir: Path,
    centered_mol2: Path,
    water_pdb: Path,
    n_water: int,
    box_params: tuple[float, float, float, float, float, float],
    concentration: float,
    cation: str,
    anion: str,
    water_ff: str,
    ligand_ff: str,
    output_prefix: str,
) -> None:
    """Build solvated prmtop/inpcrd for one ligand, then fix OCT box with ParmEd.

    ``centered_mol2`` must contain the ligand at the same coordinate frame as
    the shared water box (i.e., after solvateoct re-centering).
    """
    resname    = mol_info["residue_name"]
    mol2       = (summary_dir / mol_info["mol2"]).resolve()
    frcmod     = (summary_dir / mol_info["frcmod"]).resolve()
    net_charge = mol_info["net_charge_used_by_antechamber"]
    lig_dir    = mol2.parent

    n_extra  = calc_n_extra_ions(n_water, concentration)
    n_cation = max(0, -net_charge) + n_extra
    n_anion  = max(0,  net_charge) + n_extra

    mol2_rel   = os.path.relpath(centered_mol2, lig_dir)  # centered coords
    frcmod_rel = os.path.relpath(frcmod,         lig_dir)
    water_rel  = os.path.relpath(water_pdb,      lig_dir)

    out_pdb    = lig_dir / f"{output_prefix}.pdb"
    out_prmtop = lig_dir / f"{output_prefix}.prmtop"
    out_inpcrd = lig_dir / f"{output_prefix}.inpcrd"
    log_path   = lig_dir / f"{output_prefix}_leap.in"

    # addIonsRand replaces water molecules with ions, keeping total particle
    # count (and box density) constant — correct when reusing a pre-built box.
    if n_cation > 0 and n_anion > 0:
        ion_block = f"addIonsRand SYS {cation} {n_cation} {anion} {n_anion}"
    elif n_cation > 0:
        ion_block = f"addIonsRand SYS {cation} {n_cation}"
    elif n_anion > 0:
        ion_block = f"addIonsRand SYS {anion} {n_anion}"
    else:
        ion_block = ""

    # setBOX SYS vdw: sets the IFBOX (periodic-boundary) flag in the prmtop.
    # The box is rectangular here — ParmEd fixes it to OCT afterwards.
    script = textwrap.dedent(f"""\
        source {water_ff}
        source {ligand_ff}

        MOL = loadmol2 {mol2_rel}
        loadamberparams {frcmod_rel}

        water = loadpdb {water_rel}
        SYS = combine {{ MOL water }}

        setBOX SYS vdw

        {ion_block}

        savepdb       SYS {out_pdb.name}
        saveamberparm SYS {out_prmtop.name} {out_inpcrd.name}

        quit
    """)

    run_tleap(script, lig_dir, log_path)

    # Fix box: overwrite the rectangular placeholder with the reference OCT vectors.
    fix_box_parmed(out_prmtop, out_inpcrd, box_params)

    a, b, c, alpha, beta, gamma = box_params
    print(
        f"  {resname}: charge={net_charge:+d}  "
        f"{n_cation} {cation} + {n_anion} {anion}  "
        f"box={a:.2f} Å / {alpha:.2f}°  →  {out_prmtop.name}"
    )


def verify_box(prmtop_path: Path, inpcrd_path: Path) -> None:
    """Load with OpenMM and print the box vectors as a quick sanity check."""
    try:
        from openmm import app
    except ImportError:
        print("  (OpenMM not available — skipping box verification)")
        return

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    inpcrd = app.AmberInpcrdFile(str(inpcrd_path))
    vecs   = inpcrd.boxVectors
    if vecs is None:
        print(f"  WARNING: {prmtop_path.name} has no box vectors!")
        return
    from openmm import unit as _unit
    def _ang(v):
        return tuple(round(float(x.value_in_unit(_unit.angstrom)), 3) for x in v)
    print(f"  box vectors (Å):  v1={_ang(vecs[0])}  v2={_ang(vecs[1])}  v3={_ang(vecs[2])}")
    # OpenMM's reduced-cell OCT representation: v2 has a non-zero x component.
    v2x = float(vecs[1][0].value_in_unit(_unit.angstrom))
    if abs(v2x) < 1e-3:
        print("  WARNING: box looks rectangular (v2 x-component is zero).")
    else:
        print(f"  Box is non-orthogonal (v2.x = {v2x:.3f} Å) → OCT ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solvate AMBER ligands from a parameterization summary.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "summary_json", type=Path,
        help="Path to summary.json produced by parameterize_sdf_antechamber.py",
    )
    parser.add_argument("--buffer",        type=float, default=12.0,
                        help="OCT box buffer distance in Å")
    parser.add_argument("--concentration", type=float, default=0.15,
                        help="Target NaCl concentration in mol/L (0 = neutralize only)")
    parser.add_argument("--cation",        default="Na+",
                        help="Cation ion name for tleap")
    parser.add_argument("--anion",         default="Cl-",
                        help="Anion ion name for tleap")
    parser.add_argument("--water-ff",      default="leaprc.water.tip3p",
                        help="tleap water force-field leaprc")
    parser.add_argument("--ligand-ff",     default="leaprc.gaff2",
                        help="tleap ligand force-field leaprc")
    parser.add_argument("--output-prefix", default="02_solv",
                        help="Filename prefix for per-ligand output files")
    args = parser.parse_args()

    require_program("tleap")
    require_parmed()

    summary_json = args.summary_json.resolve()
    summary_dir  = summary_json.parent
    summary      = json.loads(summary_json.read_text())
    molecules    = summary["molecules"]

    if not molecules:
        raise ValueError("No molecules found in summary.json.")

    print(
        f"Building shared OCT water box from {len(molecules)} combined ligand(s) "
        f"(buffer={args.buffer} Å) ..."
    )
    water_pdb, n_water, box_params, centered_mol2s = build_shared_water_box(
        molecules=molecules,
        summary_dir=summary_dir,
        work_dir=summary_dir,
        buffer=args.buffer,
        water_ff=args.water_ff,
        ligand_ff=args.ligand_ff,
    )
    a, b, c, alpha, beta, gamma = box_params
    n_extra_display = calc_n_extra_ions(n_water, args.concentration)
    print(f"  {n_water} water molecules")
    print(f"  CRYST1 box: {a:.3f} {b:.3f} {c:.3f}  {alpha:.3f}° {beta:.3f}° {gamma:.3f}°")
    print(f"  salt ions per neutral ligand: {n_extra_display} {args.cation} + {n_extra_display} {args.anion}")

    print(f"\nSolvating {len(molecules)} ligand(s) ...")
    for mol_info in molecules:
        solvate_one_ligand(
            mol_info=mol_info,
            summary_dir=summary_dir,
            centered_mol2=centered_mol2s[mol_info["residue_name"]],
            water_pdb=water_pdb,
            n_water=n_water,
            box_params=box_params,
            concentration=args.concentration,
            cation=args.cation,
            anion=args.anion,
            water_ff=args.water_ff,
            ligand_ff=args.ligand_ff,
            output_prefix=args.output_prefix,
        )
        prmtop = (summary_dir / mol_info["directory"] / f"{args.output_prefix}.prmtop").resolve()
        inpcrd = prmtop.with_suffix(".inpcrd")
        verify_box(prmtop, inpcrd)

    print(f"\nDone. Output files use prefix '{args.output_prefix}' in each ligand directory.")


if __name__ == "__main__":
    main()
