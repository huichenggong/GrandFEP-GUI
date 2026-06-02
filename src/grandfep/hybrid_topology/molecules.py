from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, NamedTuple

from openmm import app, unit, openmm


@dataclass
class Atom:
    id: int
    name: str | None = None
    element: str | None = None
    atom_type: str | None = None
    charge: float = 0.0
    mass: float = 0.0
    residue_id: int | None = None
    molecule_id: int | None = None
    sigma: float | None = None
    epsilon: float | None = None
    is_virtual_site: bool = False
    is_rest2: bool = False
    is_env_core_uniq: str = "env"
    hybridization: str | None = None

@dataclass
class Residue:
    id: int
    name: str
    atom_ids: list[int] = field(default_factory=list)
    chain_id: str | None = None
    resid: int | None = None

@dataclass(frozen=True)
class BondPotential:
    length0: float
    k: float
    functional_form: str = "harmonic"
    is_constraint: bool = False

@dataclass(frozen=True)
class AnglePotential:
    theta0: float
    k: float
    functional_form: str = "harmonic"

@dataclass(frozen=True)
class DihedralPotential:
    functional_form: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class BondTerm:
    atoms: tuple[int, int]
    potential: BondPotential

@dataclass(frozen=True)
class AngleTerm:
    atoms: tuple[int, int, int]
    potential: AnglePotential

@dataclass(frozen=True)
class DihedralTerm:
    atoms: tuple[int, int, int, int]
    potential: DihedralPotential


@dataclass(frozen=True)
class NonbondedExceptionPotential:
    """Modified nonbonded interaction for a specific atom pair.

    In AMBER/GAFF force fields:

    * 1-2 and 1-3 pairs → **exclusions**: ``chargeProd == 0`` and
      ``epsilon == 0`` (``is_exclusion=True``).
    * 1-4 pairs → **exceptions**: charge product and LJ epsilon are
      scaled (typically 1/1.2 and 0.5 respectively) and
      ``is_exclusion=False``.

    All values in OpenMM native units (nm, kJ/mol, elementary_charge²).
    """
    chargeProd: float
    sigma: float
    epsilon: float
    is_exclusion: bool = False


@dataclass(frozen=True)
class NonbondedExceptionTerm:
    atoms: tuple[int, int]
    potential: NonbondedExceptionPotential


class TermTable:
    expected_n_atoms: int | None = None

    def __init__(self):
        self._terms = []
        self._by_atom = defaultdict(list)

    def add(self, term):
        if self.expected_n_atoms is not None:
            if len(term.atoms) != self.expected_n_atoms:
                raise ValueError(
                    f"Expected {self.expected_n_atoms} atoms, got {len(term.atoms)}"
                )

        idx = len(self._terms)
        self._terms.append(term)

        for atom_id in term.atoms:
            self._by_atom[atom_id].append(idx)

        return idx

    def terms_for_atom(self, atom_id: int):
        return [self._terms[i] for i in self._by_atom.get(atom_id, [])]

    def indices_for_atom(self, atom_id: int):
        return list(self._by_atom.get(atom_id, []))

    def __getitem__(self, idx):
        return self._terms[idx]

    def __iter__(self):
        return iter(self._terms)

    def __len__(self):
        return len(self._terms)

class BondTable(TermTable):
    expected_n_atoms = 2

class AngleTable(TermTable):
    expected_n_atoms = 3

class DihedralTable(TermTable):
    expected_n_atoms = 4

class NonbondedExceptionTable(TermTable):
    expected_n_atoms = 2

class DihedralPartition(NamedTuple):
    """Three-way partition of all dihedral terms, used to build a REST2 system.

    Attributes
    ----------
    proper_rest2 : list[DihedralTerm]
        Proper dihedrals whose central bond is in ``rotatable_bonds``.
        These terms have their force constant scaled in REST2.
    proper_not_rest2 : list[DihedralTerm]
        Proper dihedrals whose central bond is NOT in ``rotatable_bonds``
        (e.g. ring bonds, double bonds).  Kept at full strength.
    improper : list[DihedralTerm]
        Improper (out-of-plane) torsions.  Never scaled in REST2.
    """
    proper_rest2: list
    proper_not_rest2: list
    improper: list


class MolecularSystem:
    """Flat, ID-keyed store of atoms and bonded interactions for one simulation system.

    Provides fast bidirectional lookups between atoms and their bonded terms.
    This is the foundation for constructing hybrid topologies for alchemical
    FEP transformations and REST2 enhanced sampling.

    Atom and residue IDs match the 0-based particle indices used by OpenMM,
    so the object can be built directly from—and compared back to—an OpenMM
    ``System``.  All numerical values are in OpenMM native units (nm, kJ/mol,
    radians, elementary charge, Da).

    Attributes
    ----------
    atoms : dict[int, Atom]
        Map from atom index to :class:`Atom`.  Keys match OpenMM particle indices.
    residues : dict[int, Residue]
        Map from residue index to :class:`Residue`.
    bonds : BondTable
        Harmonic bond terms (from ``HarmonicBondForce``).  Constraints are
        stored separately in ``constraints_list``.
    angles : AngleTable
        Harmonic angle terms (from ``HarmonicAngleForce``).
    proper_dihedrals : DihedralTable
        Proper (chain) torsion terms: i–j–k–l where i–j, j–k, and k–l are
        all bonded pairs.  These are the terms scaled in REST2.  Multi-term
        dihedrals are stored as separate rows sharing the same four atom
        indices.
    improper_dihedrals : DihedralTable
        Improper (out-of-plane) torsion terms: any torsion that does not
        form a bond chain.  Used for planarity enforcement; not scaled in
        REST2.
    rotatable_bonds : set[tuple[int, int]]
        Central-bond pairs ``(min_idx, max_idx)`` of rotatable bonds,
        populated after construction (e.g. from RDKit).  Only proper
        dihedrals whose central bond ``(atoms[1], atoms[2])`` appears here
        are scaled in REST2.  Empty by default.
    box_vectors : tuple[tuple[float,float,float], ...] | None
        Periodic box vectors as a 3×3 nested tuple ``((ax,ay,az),(bx,by,bz),(cx,cy,cz))``
        in **nm**, or ``None`` for non-periodic systems.  Extracted from
        ``topology.getPeriodicBoxVectors()``.
    nonbonded_exceptions : NonbondedExceptionTable
        Modified (or zeroed) nonbonded interactions for specific atom pairs,
        extracted from ``NonbondedForce.getExceptionParameters``.  Includes
        both *exclusions* (1-2 and 1-3 bonded pairs, where
        ``chargeProd == 0`` and ``epsilon == 0``) and *exceptions* (1-4
        pairs with scaled charge/LJ).  ``NonbondedExceptionPotential.is_exclusion``
        distinguishes the two cases.  These must be faithfully reproduced
        in any hybrid topology to avoid spurious self-interactions.
    constraints_list : list[BondTerm]
        Bond-length constraints (from ``System.getConstraintParameters``).
        Kept separate from ``bonds`` because they carry no force constant
        and require different treatment when building a hybrid topology.

    Examples
    --------
    .. code-block:: python

        from grandfep import hybrid_topology, utils
        from pathlib import Path

        base = Path("tests")
        inpcrd, prmtop, system = utils.load_amber_sys(
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.inpcrd",
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.prmtop",
        )
        mol_sys = hybrid_topology.MolecularSystem().gen_from_openmm_system(
            system, prmtop.topology
        )

        # All bonded terms that involve atom 0
        print(mol_sys.terms_for_atom(0))

        # Atom objects that participate in the first stored bond
        first_bond = mol_sys.bonds[0]
        print(mol_sys.atoms_for_term(first_bond))

    """
    def __init__(self):
        self.atoms: dict[int, Atom] = {}
        self.residues: dict[int, Residue] = {}

        self.bonds = BondTable()
        self.angles = AngleTable()
        self.proper_dihedrals = DihedralTable()
        self.improper_dihedrals = DihedralTable()
        self.rotatable_bonds: set[tuple[int, int]] = set()
        self.nonbonded_exceptions = NonbondedExceptionTable()
        self.constraints_list = []
        self.box_vectors: tuple | list | None = None

        self._next_atom_id = 0
        self._next_residue_id = 0
        self._next_molecule_id = 0

    def terms_for_atom(self, atom_id: int) -> dict[str, list]:
        """Return all bonded terms that involve *atom_id*.

        Parameters
        ----------
        atom_id : int
            OpenMM particle index.

        Returns
        -------
        dict[str, list]
            Keys ``"bonds"``, ``"angles"``, ``"proper_dihedrals"``,
            ``"improper_dihedrals"``, ``"nonbonded_exceptions"``; each
            value is a list of the corresponding term objects.
        """
        return {
            "bonds": self.bonds.terms_for_atom(atom_id),
            "angles": self.angles.terms_for_atom(atom_id),
            "proper_dihedrals": self.proper_dihedrals.terms_for_atom(atom_id),
            "improper_dihedrals": self.improper_dihedrals.terms_for_atom(atom_id),
            "nonbonded_exceptions": self.nonbonded_exceptions.terms_for_atom(atom_id),
        }

    def atoms_for_term(self, term) -> list:
        """Return the :class:`Atom` objects that participate in *term*.

        Parameters
        ----------
        term : BondTerm | AngleTerm | DihedralTerm
            Any term whose ``.atoms`` attribute holds a tuple of atom indices.

        Returns
        -------
        list[Atom]
            Atoms in the same order as ``term.atoms``.
        """
        return [self.atoms[i] for i in term.atoms]

    def rest2_scalable_dihedrals(self) -> DihedralPartition:
        """Partition all dihedral terms into the three groups needed for REST2.

        Populate :attr:`rotatable_bonds` from RDKit before calling this method.

        Returns
        -------
        DihedralPartition
            Named tuple with fields:

            ``proper_rest2``
                Proper dihedrals whose central bond is in
                :attr:`rotatable_bonds` — scale the force constant in REST2.
            ``proper_not_rest2``
                Proper dihedrals whose central bond is NOT in
                :attr:`rotatable_bonds` (ring / double bonds) — keep at
                full strength.
            ``improper``
                Improper (out-of-plane) torsions — never scaled.
        """
        proper_rest2, proper_not_rest2 = [], []
        for t in self.proper_dihedrals:
            key = (min(t.atoms[1], t.atoms[2]), max(t.atoms[1], t.atoms[2]))
            if key in self.rotatable_bonds:
                proper_rest2.append(t)
            else:
                proper_not_rest2.append(t)
        return DihedralPartition(
            proper_rest2=proper_rest2,
            proper_not_rest2=proper_not_rest2,
            improper=list(self.improper_dihedrals),
        )

    def gen_from_openmm_system(self, system: openmm.System, topology: app.topology.Topology):
        """Populate this MolecularSystem from an OpenMM System and Topology.

        All values are stored in OpenMM native units:
          lengths   → nm
          energies  → kJ/mol
          angles    → radians
          charges   → elementary charge
          masses    → Da
        """
        force_dict = {
            "HarmonicBondForce"    : None,
            "HarmonicAngleForce"   : None,
            "PeriodicTorsionForce" : None,
            "NonbondedForce"       : None,
            "CMMotionRemover"      : None,
        }
        for force in system.getForces():
            name = type(force).__name__
            if name in force_dict:
                force_dict[name] = force

        nb_force = force_dict["NonbondedForce"]

        # --- Periodic box ---
        self.box_vectors = system.getDefaultPeriodicBoxVectors()

        # --- Residues ---
        for res in topology.residues():
            try:
                resid = int(res.id)
            except (ValueError, TypeError):
                resid = None
            self.residues[res.index] = Residue(
                id=res.index,
                name=res.name,
                chain_id=res.chain.id,
                resid=resid,
            )

        # --- Atoms ---
        for atom in topology.atoms():
            idx = atom.index
            element_symbol = atom.element.symbol if atom.element is not None else None
            mass = system.getParticleMass(idx).value_in_unit(unit.dalton)
            is_vs = system.isVirtualSite(idx)

            if nb_force is not None:
                charge, sigma, epsilon = nb_force.getParticleParameters(idx)
                charge_val  = charge.value_in_unit(unit.elementary_charge)
                sigma_val   = sigma.value_in_unit(unit.nanometer)
                epsilon_val = epsilon.value_in_unit(unit.kilojoule_per_mole)
            else:
                charge_val = sigma_val = epsilon_val = 0.0

            a = Atom(
                id=idx,
                name=atom.name,
                element=element_symbol,
                mass=mass,
                residue_id=atom.residue.index,
                charge=charge_val,
                sigma=sigma_val,
                epsilon=epsilon_val,
                is_virtual_site=is_vs,
            )
            self.atoms[idx] = a
            self.residues[atom.residue.index].atom_ids.append(idx)

        # --- Nonbonded exceptions / exclusions ---
        if nb_force is not None:
            for i in range(nb_force.getNumExceptions()):
                a1, a2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
                cp_val  = chargeProd.value_in_unit(unit.elementary_charge**2)
                sig_val = sigma.value_in_unit(unit.nanometer)
                eps_val = epsilon.value_in_unit(unit.kilojoule_per_mole)
                self.nonbonded_exceptions.add(NonbondedExceptionTerm(
                    atoms=(a1, a2),
                    potential=NonbondedExceptionPotential(
                        chargeProd=cp_val,
                        sigma=sig_val,
                        epsilon=eps_val,
                        is_exclusion=(cp_val == 0.0 and eps_val == 0.0),
                    ),
                ))

        # --- Bonds (HarmonicBondForce) ---
        # bonded_set is used below to classify proper vs improper dihedrals.
        bonded_set: set[tuple[int, int]] = set()
        hbf = force_dict["HarmonicBondForce"]
        if hbf is not None:
            for i in range(hbf.getNumBonds()):
                a1, a2, length, k = hbf.getBondParameters(i)
                bonded_set.add((min(a1, a2), max(a1, a2)))
                self.bonds.add(BondTerm(
                    atoms=(a1, a2),
                    potential=BondPotential(
                        length0=length.value_in_unit(unit.nanometer),
                        k=k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2),
                    ),
                ))

        # --- Constraints ---
        for i in range(system.getNumConstraints()):
            a1, a2, dist = system.getConstraintParameters(i)
            bonded_set.add((min(a1, a2), max(a1, a2)))
            self.constraints_list.append(BondTerm(
                atoms=(a1, a2),
                potential=BondPotential(
                    length0=dist.value_in_unit(unit.nanometer),
                    k=0.0,
                    is_constraint=True,
                ),
            ))

        # --- Angles (HarmonicAngleForce) ---
        haf = force_dict["HarmonicAngleForce"]
        if haf is not None:
            for i in range(haf.getNumAngles()):
                a1, a2, a3, theta0, k = haf.getAngleParameters(i)
                self.angles.add(AngleTerm(
                    atoms=(a1, a2, a3),
                    potential=AnglePotential(
                        theta0=theta0.value_in_unit(unit.radian),
                        k=k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2),
                    ),
                ))

        # --- Dihedrals (PeriodicTorsionForce) ---
        # Proper: i-j, j-k, k-l all in bonded_set.  Improper: anything else.
        # Which proper dihedrals are REST2-scalable is determined later via
        # rotatable_bonds (populated from RDKit, not from OpenMM data).
        ptf = force_dict["PeriodicTorsionForce"]
        if ptf is not None:
            for i in range(ptf.getNumTorsions()):
                a1, a2, a3, a4, periodicity, phase, k = ptf.getTorsionParameters(i)
                term = DihedralTerm(
                    atoms=(a1, a2, a3, a4),
                    potential=DihedralPotential(
                        functional_form="periodic",
                        parameters={
                            "periodicity": int(periodicity),
                            "phase": phase.value_in_unit(unit.radian),
                            "k": k.value_in_unit(unit.kilojoule_per_mole),
                        },
                    ),
                )
                is_proper = (
                    (min(a1, a2), max(a1, a2)) in bonded_set and
                    (min(a2, a3), max(a2, a3)) in bonded_set and
                    (min(a3, a4), max(a3, a4)) in bonded_set
                )
                if is_proper:
                    self.proper_dihedrals.add(term)
                else:
                    self.improper_dihedrals.add(term)

        self._next_atom_id = system.getNumParticles()
        self._next_residue_id = sum(1 for _ in topology.residues())

        return self

    def set_rotatable_bonds(self, rotatable_bonds):
        """
        Set internal set of rotatable bonds
        """
        self.rotatable_bonds = {(min(a, b), max(a, b)) for a, b in rotatable_bonds}
