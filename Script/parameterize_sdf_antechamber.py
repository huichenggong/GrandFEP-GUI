#!/usr/bin/env python3

"""
Parameterize a multi-molecule SDF using AmberTools antechamber and parmchk2.

Example
-------
python parameterize_sdf_antechamber.py ligands.sdf -o parameterized_ligands

Requirements
------------
- RDKit
- AmberTools with antechamber and parmchk2 available in PATH
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from rdkit import Chem


def residue_name_from_index(index: int) -> str:
    """Return a 3-character Amber-compatible residue name.

    Naming scheme:
    1-99     : A01, A02, ..., A99
    100-2600 : B01, ..., Z99

    Parameters
    ----------
    index : int
        One-based molecule index.

    Returns
    -------
    str
        Three-character residue name.

    Raises
    ------
    ValueError
        If index is outside the supported range.
    """
    if index < 1:
        raise ValueError("Index should be one-based and positive.")
    
    elif index > 2600:
        raise ValueError("This naming scheme supports up to 2600 molecules.")
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter= alphabet[(index - 1) // 99]
    number = (index - 1) % 99 + 1
    return f"{letter}{number:02d}"




def require_program(program: str) -> None:
    """Check that an external program is available."""
    if shutil.which(program) is None:
        raise RuntimeError(
            f"Could not find '{program}' in PATH. "
            "Please activate an AmberTools environment first."
        )

def molecule_has_explicit_hydrogens(mol: Chem.Mol) -> bool:
    """Return True if molecule contains at least one explicit hydrogen."""
    return any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())


def find_atoms_with_implicit_hydrogens(mol: Chem.Mol) -> list[int]:
    """Find atoms that still have implicit hydrogens.

    If the SDF already contains all hydrogens explicitly, this should normally
    be empty after sanitization.
    """
    bad_atoms: list[int] = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue

        if atom.GetNumImplicitHs() > 0:
            bad_atoms.append(atom.GetIdx())

    return bad_atoms

def get_total_formal_charge(mol: Chem.Mol) -> int:
    """Return total formal charge from RDKit."""
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))

def assign_atom_names(mol: Chem.Mol) -> list[str]:
    """Assign simple unique atom names based on element.

    Example:
    C1, C2, O1, N1, H1, H2, ...

    The names are stored in the atom property ``atomName``.
    """
    element_counts: dict[str, int] = {}
    atom_names: list[str] = []

    for atom in mol.GetAtoms():
        element = atom.GetSymbol().upper()
        element_counts[element] = element_counts.get(element, 0) + 1
        atom_name = f"{element}{element_counts[element]}"

        # MOL2/PDB atom names are usually easier if not too long.
        # This keeps e.g. CL1 instead of Cl1.
        atom.SetProp("atomName", atom_name)
        atom_names.append(atom_name)

    return atom_names


def write_single_sdf(mol: Chem.Mol, path: Path, residue_name: str) -> None:
    """Write one molecule to an SDF file."""
    mol = Chem.Mol(mol)

    mol.SetProp("_Name", residue_name)
    mol.SetProp("RESNAME", residue_name)

    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def write_single_pdb(mol: Chem.Mol, path: Path, residue_name: str) -> None:
    """Write one molecule to a PDB file with a given residue name.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule. Should already contain explicit hydrogens and 3D coordinates.
    path : Path
        Output PDB path.
    residue_name : str
        Three-character residue name, for example ``"MO1"``, ``"M10"``, or ``"MOA"``.
    """
    mol = Chem.Mol(mol)

    if len(residue_name) > 3:
        raise ValueError(
            f"PDB residue names should be at most 3 characters, got {residue_name!r}"
        )

    residue_name = residue_name.upper()

    # Count atom names by element: C1, C2, O1, H1, ...
    element_counts: dict[str, int] = {}

    for atom in mol.GetAtoms():
        element = atom.GetSymbol().upper()
        element_counts[element] = element_counts.get(element, 0) + 1

        atom_name = f"{element}{element_counts[element]}"

        # PDB atom name field is 4 characters.
        atom_name = atom_name[:4]

        pdb_info = Chem.AtomPDBResidueInfo()
        pdb_info.SetName(f"{atom_name:>4s}")
        pdb_info.SetResidueName(f"{residue_name:>3s}")
        pdb_info.SetResidueNumber(1)
        pdb_info.SetChainId("A")
        pdb_info.SetIsHeteroAtom(True)

        atom.SetMonomerInfo(pdb_info)

    mol.SetProp("_Name", residue_name)

    pdb_block = Chem.MolToPDBBlock(mol, flavor=2 | 8) # no CONECT records

    with open(path, "w") as handle:
        handle.write(pdb_block)


def run_command(command: list[str], cwd: Path, log_file: Path) -> None:
    """Run external command and write stdout/stderr to a log file."""
    with log_file.open("w") as handle:
        handle.write("Command:\n")
        handle.write(" ".join(command))
        handle.write("\n\n")

        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}: "
            f"{' '.join(command)}\n"
            f"See log file: {log_file}"
        )


def parameterize_one_molecule(
    mol: Chem.Mol,
    molecule_index: int,
    output_dir: Path,
    charge_method: str,
    atom_type: str,
    net_charge: int | None,
) -> dict[str, Any]:
    """Parameterize one molecule with antechamber and parmchk2."""
    residue_name = residue_name_from_index(molecule_index)
    molecule_dir = output_dir / residue_name
    molecule_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.Mol(mol)

    assign_atom_names(mol)

    formal_charge = get_total_formal_charge(mol)
    used_charge = formal_charge if net_charge is None else net_charge

    input_sdf = molecule_dir / f"{residue_name}.sdf"
    input_pdb = molecule_dir / f"{residue_name}.pdb"
    output_mol2 = molecule_dir / f"{residue_name}.mol2"
    output_frcmod = molecule_dir / f"{residue_name}.frcmod"

    write_single_sdf(mol, input_sdf, residue_name)
    write_single_pdb(mol, input_pdb, residue_name)

    antechamber_log = molecule_dir / "antechamber.log"
    parmchk2_log = molecule_dir / "parmchk2.log"

    antechamber_cmd = [
        "antechamber",
        "-i",
        input_pdb.name,
        "-fi",
        "pdb",
        "-o",
        output_mol2.name,
        "-fo",
        "mol2",
        "-c",
        charge_method,
        "-s",
        "2",
        "-at",
        atom_type,
        "-nc",
        str(used_charge),
        "-rn",
        residue_name,
    ]

    run_command(antechamber_cmd, cwd=molecule_dir, log_file=antechamber_log)

    parmchk2_cmd = [
        "parmchk2",
        "-i",
        output_mol2.name,
        "-f",
        "mol2",
        "-o",
        output_frcmod.name,
        "-s",
        atom_type,
    ]

    run_command(parmchk2_cmd, cwd=molecule_dir, log_file=parmchk2_log)

    atom_info: list[dict[str, Any]] = []

    for atom in mol.GetAtoms():
        atom_info.append(
            {
                "rdkit_index": atom.GetIdx(),
                "atom_name": atom.GetProp("atomName"),
                "element": atom.GetSymbol(),
                "formal_charge": atom.GetFormalCharge(),
                "hybridization": str(atom.GetHybridization()),
                "is_aromatic": atom.GetIsAromatic(),
                "degree": atom.GetDegree(),
                "explicit_valence": atom.GetValence(Chem.ValenceType.EXPLICIT),
                "implicit_hydrogens": atom.GetNumImplicitHs(),
            }
        )

    return {
        "molecule_index": molecule_index,
        "residue_name": residue_name,
        "directory": str(molecule_dir.resolve()),
        "input_sdf": str(input_sdf.resolve()),
        "mol2": str(output_mol2.resolve()),
        "frcmod": str(output_frcmod.resolve()),
        "antechamber_log": str(antechamber_log.resolve()),
        "parmchk2_log": str(parmchk2_log.resolve()),
        "formal_charge_from_rdkit": formal_charge,
        "net_charge_used_by_antechamber": used_charge,
        "charge_method": charge_method,
        "atom_type": atom_type,
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1),
        "num_hydrogens": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1),
        "atoms": atom_info,
    }


def load_sdf(input_sdf: Path) -> list[Chem.Mol]:
    """Load molecules from SDF while preserving explicit hydrogens."""
    supplier = Chem.SDMolSupplier(str(input_sdf), removeHs=False, sanitize=True)

    molecules: list[Chem.Mol] = []

    for i, mol in enumerate(supplier, start=1):
        if mol is None:
            raise ValueError(f"Failed to parse molecule {i} from {input_sdf}")

        molecules.append(mol)

    if not molecules:
        raise ValueError(f"No molecules found in {input_sdf}")

    return molecules


def safety_check_molecule(mol: Chem.Mol, molecule_index: int) -> None:
    """Check that molecule has explicit hydrogens and no implicit hydrogens."""
    if not molecule_has_explicit_hydrogens(mol):
        raise ValueError(
            f"Molecule {molecule_index} does not contain explicit hydrogens."
        )

    atoms_with_implicit_h = find_atoms_with_implicit_hydrogens(mol)

    if atoms_with_implicit_h:
        details = []

        for atom_idx in atoms_with_implicit_h:
            atom = mol.GetAtomWithIdx(atom_idx)
            details.append(
                {
                    "atom_index": atom_idx,
                    "element": atom.GetSymbol(),
                    "implicit_hydrogens": atom.GetNumImplicitHs(),
                }
            )

        raise ValueError(
            f"Molecule {molecule_index} appears to have missing explicit hydrogens. "
            f"Atoms with implicit hydrogens: {details}"
        )


def _rel(path_str: str, base: Path) -> str:
    return str(Path(os.path.relpath(path_str, base)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameterize a multi-molecule SDF with AmberTools antechamber."
    )

    parser.add_argument(
        "input_sdf",
        type=Path,
        help="Input multi-molecule SDF. Hydrogens should already be explicit.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("antechamber_output"),
        help="Output directory.",
    )

    parser.add_argument(
        "--charge-method",
        default="bcc",
        choices=["bcc", "gas", "mul", "resp", "rc"],
        help="Charge method for antechamber. Default: bcc.",
    )

    parser.add_argument(
        "--atom-type",
        default="gaff2",
        choices=["gaff", "gaff2"],
        help="GAFF atom type version. Default: gaff2.",
    )

    parser.add_argument(
        "--net-charge",
        type=int,
        default=None,
        help=(
            "Override net charge for all molecules. "
            "By default, the RDKit formal charge is used separately for each molecule."
        ),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to output JSON summary. Default: <output-dir>/summary.json",
    )

    args = parser.parse_args()

    require_program("antechamber")
    require_program("parmchk2")

    input_sdf = args.input_sdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else output_dir / "summary.json"
    )

    molecules = load_sdf(input_sdf)

    if len(molecules) > 125:
        raise ValueError(
            "This residue naming scheme supports up to 125 molecules: "
            "MO1-MO9, M10-M99, MOA-MOZ."
        )

    # First do all safety checks before running antechamber.
    for i, mol in enumerate(molecules, start=1):
        safety_check_molecule(mol, i)

    summary: dict[str, Any] = {
        "input_sdf": str(input_sdf),
        "output_dir": str(output_dir),
        "num_molecules": len(molecules),
        "charge_method": args.charge_method,
        "atom_type": args.atom_type,
        "molecules": [],
    }

    for i, mol in enumerate(molecules, start=1):
        info = parameterize_one_molecule(
            mol=mol,
            molecule_index=i,
            output_dir=output_dir,
            charge_method=args.charge_method,
            atom_type=args.atom_type,
            net_charge=args.net_charge,
        )
        summary["molecules"].append(info)

    base = summary_json.parent
    summary["input_sdf"] = _rel(summary["input_sdf"], base)
    summary["output_dir"] = _rel(summary["output_dir"], base)
    for mol_info in summary["molecules"]:
        for key in ("directory", "input_sdf", "mol2", "frcmod", "antechamber_log", "parmchk2_log"):
            mol_info[key] = _rel(mol_info[key], base)

    with summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Parameterized {len(molecules)} molecules.")
    print(f"Output directory: {output_dir}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
