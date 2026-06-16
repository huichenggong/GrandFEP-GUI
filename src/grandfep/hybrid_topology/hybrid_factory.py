import warnings
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from MDAnalysis.lib.formats.libmdaxdr import namedtuple

from openmm import unit, app, openmm
from grandfep import hybrid_topology


# Normal REST2
class Rest2TopologyFactory:
    """
    This class generate a topology for REST2 simulation, set 2 global parameters k_rest2 and k_rest2_sqrt to control
    the scaling. The caller is responsible for keeping k_rest2 = k_rest2_sqrt^2.

    Scaling convention (n_hot = number of hot atoms in a term):
      n_hot = 2 → scale by k_rest2      (= k_rest2_sqrt^2)
      n_hot = 1 → scale by k_rest2_sqrt (= sqrt(k_rest2))
      n_hot = 0 → unscaled

    Attributes
    ----------
    system : openmm.System

    topology : app.topology.Topology

    Examples
    --------
    .. code-block:: python

        from grandfep import hybrid_topology, utils
        from pathlib import Path

        base = Path("tests")

        # find rotatable bonds
        sdf_path = base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/A01.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = supplier[0]
        rot_bonds = mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)
        hot_atoms = list(range(mol.GetNumAtoms()))

        # construct new system/topology
        inpcrd, prmtop, system = utils.load_amber_sys(
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.inpcrd",
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.prmtop",
        )
        factory = Rest2TopologyFactory(system, prmtop.topology, hot_atoms, rot_bonds)

    """
    def __init__(self, system:openmm.System, topology, nb_hot_atoms, rotatable_bonds):
        self.basic_check(system)
        self.molecule_system = hybrid_topology.MolecularSystem().gen_from_openmm_system(system, topology)
        self.hot_set = frozenset(nb_hot_atoms)
        self.topology = topology

        for idx in self.hot_set:
            self.molecule_system.atoms[idx].is_rest2 = True

        self.molecule_system.set_rotatable_bonds(rotatable_bonds)

        self._orig_nb_force = None
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                self._orig_nb_force = force

        # self._orig_system reserved for future virtual-site copying (skipped for now)

        self.system = openmm.System()
        self._prepare_system()
        self._prepare_bond()
        self._prepare_angle()
        self._prepare_dihe()
        self._prepare_nonbonded_force()

    def basic_check(self, system:openmm.System):
        """
        Check if the input system is a proper openmm system that this code can customize.

        The input system should have 4 forces `HarmonicBondForce`, `HarmonicAngleForce`, `PeriodicTorsionForce`,
        `NonbondedForce`. `CMMotionRemover` will be ignored

        Parameters
        ----------
        system:
            We can't customize Charmm system for now

        Raises
        ------
        ValueError
            If required forces are missing or unsupported force types are present.
        """
        required = {
            openmm.HarmonicBondForce,
            openmm.HarmonicAngleForce,
            openmm.PeriodicTorsionForce,
            openmm.NonbondedForce,
        }
        found = set()
        for force in system.getForces():
            if isinstance(force, openmm.CMMotionRemover):
                continue
            force_type = type(force)
            if force_type not in required:
                raise ValueError(
                    f"Unsupported force type '{force_type.__name__}'. "
                    "Only AMBER/GAFF systems with HarmonicBondForce, HarmonicAngleForce, "
                    "PeriodicTorsionForce, and NonbondedForce are supported."
                )
            found.add(force_type)
        missing = required - found
        if missing:
            raise ValueError(
                f"Missing required forces: {sorted(f.__name__ for f in missing)}"
            )

    def _prepare_system(self):
        """
        prepare basic property of system, including: atom, mass, constraint, default box, center of mass motion remove
        """
        ms = self.molecule_system
        for idx in sorted(ms.atoms):
            self.system.addParticle(ms.atoms[idx].mass)

        for idx, vs_info in ms.virtual_sites.items():
            self.system.setVirtualSite(idx, vs_info.to_openmm())

        for c in ms.constraints_list:
            a1, a2 = c.atoms
            self.system.addConstraint(a1, a2, c.potential.length0)

        if ms.box_vectors is not None:
            self.system.setDefaultPeriodicBoxVectors(*ms.box_vectors)

    def _prepare_bond(self):
        """
        HarmonicBondForce — no REST2 scaling for bonds.
        """
        hbf = openmm.HarmonicBondForce()
        for t in self.molecule_system.bonds:
            a1, a2 = t.atoms
            hbf.addBond(a1, a2, t.potential.length0, t.potential.k)
        self.system.addForce(hbf)

    def _prepare_angle(self):
        """
        No REST2 scaling for angle
        """
        haf = openmm.HarmonicAngleForce()
        for t in self.molecule_system.angles:
            a1, a2, a3 = t.atoms
            haf.addAngle(a1, a2, a3, t.potential.theta0, t.potential.k)
        self.system.addForce(haf)

    def _prepare_dihe(self):
        """
        Build two torsion forces according to REST2 scaling rules.

        Only **proper** dihedrals on **rotatable bonds** are eligible for REST2 scaling
        (``rest2_scalable_dihedrals()`` returns these as ``proper_rest2``).
        Improper dihedrals and proper dihedrals on non-rotatable bonds are always unscaled.

        **Routing logic**

        For each term in ``proper_rest2``, ``n_hot`` is the count of hot atoms among the
        two *central* atoms of the dihedral (``atoms[1]`` and ``atoms[2]``):

        +--------+--------------------------------------------+----------------------------+
        | n_hot  | Physical meaning                           | Scale factor               |
        +========+============================================+============================+
        |   2    | Both central atoms are hot — torsion is    | ``k_rest2_sqrt^2``         |
        |        | entirely within the hot region             | (= ``k_rest2``)            |
        +--------+--------------------------------------------+----------------------------+
        |   1    | One central atom hot, one cold — torsion   | ``k_rest2_sqrt^1``         |
        |        | crosses the hot/cold boundary              | (= ``sqrt(k_rest2)``)      |
        +--------+--------------------------------------------+----------------------------+
        |   0    | Both central atoms cold — no REST2 effect, | 1 (unscaled)               |
        |        | even if the bond is formally rotatable     |                            |
        +--------+--------------------------------------------+----------------------------+

        Terms with ``n_hot >= 1`` are added to a ``CustomTorsionForce`` with the expression::

            k_rest2_sqrt^n_hot * k * (1 + cos(n * theta - phase))

        where ``n_hot`` is stored as a per-torsion parameter so a single global parameter
        ``k_rest2_sqrt`` handles both the n_hot=1 and n_hot=2 cases.
        Terms with ``n_hot == 0`` fall through to the unscaled ``PeriodicTorsionForce``.
        """
        ms = self.molecule_system
        part = ms.rest2_scalable_dihedrals()

        # PeriodicTorsionForce: all unscaled torsions (cold + non-rotatable + zero-hot rotatable)
        # CustomTorsionForce:   hot rotatable torsions, fully scaled by k_rest2_sqrt^n_hot
        ptf = openmm.PeriodicTorsionForce()
        ctf = openmm.CustomTorsionForce(
            "k_rest2_sqrt^n_hot * k * (1 + cos(n * theta - phase))"
        )
        ctf.addGlobalParameter("k_rest2_sqrt", 1.0)
        ctf.addPerTorsionParameter("n_hot")
        ctf.addPerTorsionParameter("k")
        ctf.addPerTorsionParameter("n")
        ctf.addPerTorsionParameter("phase")

        for t in part.proper_not_rest2:
            p = t.potential.parameters
            ptf.addTorsion(*t.atoms, p["periodicity"], p["phase"], p["k"])

        for t in part.improper:
            p = t.potential.parameters
            ptf.addTorsion(*t.atoms, p["periodicity"], p["phase"], p["k"])

        for t in part.proper_rest2:
            p = t.potential.parameters
            n_hot = len({t.atoms[1], t.atoms[2]} & self.hot_set)
            if n_hot >= 1:
                ctf.addTorsion(*t.atoms, [float(n_hot), p["k"], float(p["periodicity"]), p["phase"]])
            else:
                ptf.addTorsion(*t.atoms, p["periodicity"], p["phase"], p["k"])

        self.system.addForce(ptf)
        self.system.addForce(ctf)

    def _prepare_nonbonded_force(self):
        """
        Build a single REST2-ready NonbondedForce using addParticleParameterOffset
        and addExceptionParameterOffset.

        Hot-atom charges and epsilons are stored as base=0 and recovered via
        parameter offsets, so the NonbondedForce natively scales all interactions
        — including PME reciprocal space — when k_rest2_sqrt / k_rest2 change.

        Scaling applied to each offset parameter:
          charge:    multiply hot-atom charge    by k_rest2_sqrt  (linear offset)
          epsilon:   multiply hot-atom epsilon   by k_rest2       (linear offset)
          Mixed pairs follow from the Lorentz-Berthelot combining rules automatically:
            hot-hot charge:    k_rest2_sqrt * q_i * k_rest2_sqrt * q_j = k_rest2 * q_i*q_j
            hot-cold charge:   k_rest2_sqrt * q_i * q_j
            hot-hot epsilon_ij:  sqrt(k_rest2*eps_i * k_rest2*eps_j) = k_rest2 * sqrt(eps_i*eps_j)
            hot-cold epsilon_ij: sqrt(k_rest2*eps_i * eps_j)         = k_rest2_sqrt * sqrt(eps_i*eps_j)

        1-4 exceptions involving hot atoms are handled with addExceptionParameterOffset:
          n_hot=1: param = k_rest2_sqrt
          n_hot=2: param = k_rest2
        """
        if self._orig_nb_force is None:
            return

        ms = self.molecule_system
        hot_set = self.hot_set
        orig = self._orig_nb_force

        nbf = openmm.NonbondedForce()
        # Parameters must be declared on the NonbondedForce itself before addParticleParameterOffset
        nbf.addGlobalParameter("k_rest2_sqrt", 1.0)
        nbf.addGlobalParameter("k_rest2", 1.0)
        nbf.setNonbondedMethod(orig.getNonbondedMethod())
        nbf.setPMEParameters(*orig.getPMEParameters())
        nbf.setCutoffDistance(orig.getCutoffDistance())
        nbf.setEwaldErrorTolerance(orig.getEwaldErrorTolerance())
        nbf.setUseSwitchingFunction(orig.getUseSwitchingFunction())
        if orig.getUseSwitchingFunction():
            nbf.setSwitchingDistance(orig.getSwitchingDistance())
        nbf.setUseDispersionCorrection(orig.getUseDispersionCorrection())
        nbf.setExceptionsUsePeriodicBoundaryConditions(
            orig.getExceptionsUsePeriodicBoundaryConditions()
        )

        # Particles: cold atoms use original params; hot atoms use base=0 + offset
        for idx in sorted(ms.atoms):
            atom = ms.atoms[idx]
            if idx in hot_set:
                nbf.addParticle(0.0, atom.sigma, 0.0)
            else:
                nbf.addParticle(atom.charge, atom.sigma, atom.epsilon)

        for idx in hot_set:
            atom = ms.atoms[idx]
            nbf.addParticleParameterOffset("k_rest2_sqrt", idx, atom.charge, 0.0, 0.0)
            nbf.addParticleParameterOffset("k_rest2",      idx, 0.0, 0.0, atom.epsilon)

        # Exceptions: exclusions and cold 1-4 are copied as-is; hot 1-4 use offsets
        for term in ms.nonbonded_exceptions:
            a1, a2 = term.atoms
            p = term.potential
            if p.is_exclusion or not (a1 in hot_set or a2 in hot_set):
                nbf.addException(a1, a2, p.chargeProd, p.sigma, p.epsilon)
            else:
                exc_idx = nbf.addException(a1, a2, 0.0, p.sigma, 0.0)
                n_hot = (1 if a1 in hot_set else 0) + (1 if a2 in hot_set else 0)
                param_name = "k_rest2" if n_hot == 2 else "k_rest2_sqrt"
                nbf.addExceptionParameterOffset(param_name, exc_idx, p.chargeProd, 0.0, p.epsilon)

        self.system.addForce(nbf)


class HybridIndexMapping:
    """Map atom indices from two end-state topologies onto a single hybrid topology.

    Each pair of topologies (A and B) shares the same residue count.  Residues
    listed in ``index_a2b`` are *perturbed* — they change between A and B.  All
    other residues are *environment* residues that are identical in both states.

    Hybrid atom ordering within a perturbed residue:
      1. **Core** atoms — mapped between A and B; parameters change during the
         alchemical transformation.  Ordered by A's local index.
      2. **Unique-A** atoms — only in A (dummy in state B).
      3. **Unique-B** atoms — only in B (dummy in state A).

    The four disjoint atom classes cover all ``n_atoms`` hybrid indices:
    ``core_atoms ∪ unique_A_atoms ∪ unique_B_atoms ∪ env_atoms == {0 … n_atoms-1}``

    Attributes
    ----------
    map_A_to_hybrid : dict[int, int]
        Global atom index in topologyA → hybrid atom index.
        Covers all atoms of topologyA (core + unique-A + env).
    map_B_to_hybrid : dict[int, int]
        Global atom index in topologyB → hybrid atom index.
        Covers all atoms of topologyB (core + unique-B + env).
    map_hybrid_to_A : dict[int, int]
        Reverse of ``map_A_to_hybrid``.
    map_hybrid_to_B : dict[int, int]
        Reverse of ``map_B_to_hybrid``.
    n_atoms : int
        Total number of atoms in the hybrid topology.
    core_atoms : set[int]
        Hybrid indices of mapped atoms whose parameters change between A and B.
        If ``"core_A"`` is absent from the residue entry, all mapped atoms are core.
    unique_A_atoms : set[int]
        Hybrid indices present only in A (dummy in state B).
    unique_B_atoms : set[int]
        Hybrid indices present only in B (dummy in state A).
    env_atoms : set[int]
        Hybrid indices that are identical in A and B (unchanged environment).
    hybrid_top : openmm.app.Topology
        Hybrid topology: topologyA atoms plus unique-B atoms appended to their
        perturbed residue.  Bonds from both topologies are included.
    broken_bonds_A : list[tuple[int, int]]
        Hybrid index pairs for bonds present in A but absent in B
        (from ``"broken_bonds_moli"`` in the mapping entry).
    broken_bonds_B : list[tuple[int, int]]
        Hybrid index pairs for bonds present in B but absent in A
        (from ``"broken_bonds_molj"`` in the mapping entry).

    Parameters
    ----------
    topologyA : openmm.app.Topology
    topologyB : openmm.app.Topology
    index_a2b : dict[int, dict]
        Maps residue index (0-based) to a mapping entry with keys:

        - ``"atom_map"``           : list of ``[local_A, local_B]`` pairs (required)
        - ``"core_A"``             : list of local-A indices that are core (optional;
                                     absent → all mapped atoms are core)
        - ``"broken_bonds_moli"``  : list of ``[i, j]`` local-A pairs (optional)
        - ``"broken_bonds_molj"``  : list of ``[i, j]`` local-B pairs (optional)
        - ``"hybridization_moli"`` : list of hybridization string for atoms in state A
        - ``"hybridization_moli"`` : list of hybridization string for atoms in state B

    Examples
    --------
    .. code-block:: python

        from grandfep import utils
        from grandfep.hybrid_topology.hybrid_factory import HybridIndexMapping
        import json

        ligand_path = Path("tests/public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/macrocycles/2B8V_lig24and25_alpha05")
        inpcrdA, prmtopA, systemA = utils.load_amber_sys(
            ligand_path / "ligand_preparation/A01/01_dry.inpcrd",
            ligand_path / "ligand_preparation/A01/01_dry.prmtop")
        inpcrdB, prmtopB, systemB = utils.load_amber_sys(
            ligand_path / "ligand_preparation/A02/01_dry.inpcrd",
            ligand_path / "ligand_preparation/A02/01_dry.prmtop")

        with open(ligand_path / "edge_0_1/mapping.json") as f:
            mapping = json.load(f)

        m = HybridIndexMapping(prmtopA.topology, prmtopB.topology, {0: mapping})
        # m.n_atoms == 58
        # m.core_atoms     == {0, …, 55}   (56 mapped atoms, all core — no core_A key)
        # m.unique_A_atoms == set()
        # m.unique_B_atoms == {56, 57}
        # m.env_atoms      == set()         (single-residue system, no environment)
        # m.broken_bonds_A == [(11, 12)]

    """
    def __init__(self, topologyA: app.topology.Topology, topologyB: app.topology.Topology, index_a2b: dict):
        self.hybrid_top = app.Topology()
        self.topologyA = topologyA
        self.topologyB = topologyB
        self.map_A_to_hybrid = {}
        self.map_B_to_hybrid = {}

        topA_residues = list(topologyA.residues())
        topB_residues = list(topologyB.residues())

        if len(topA_residues) != len(topB_residues):
            raise ValueError("The number of residues does not match in A and B")

        self._perturbed_res_indices = set(index_a2b.keys())
        self.broken_bonds_A: list[tuple[int, int]] = []  # hybrid pairs: in A, absent in B
        self.broken_bonds_B: list[tuple[int, int]] = []  # hybrid pairs: in B, absent in A
        self.core_atoms: set[int] = set()  # hybrid indices whose parameters change between A and B
        self.hybridization = {
            "A":{idx:res_map_info["hybridization_moli"] for idx,  res_map_info in index_a2b.items()},
            "B":{idx:res_map_info["hybridization_molj"] for idx,  res_map_info in index_a2b.items()},
        }

        hybrid_atom_count = 0
        for res_idx, (resA, resB) in enumerate(zip(topA_residues, topB_residues)):
            if res_idx in index_a2b:
                hybrid_atom_count = self._map_perturbed_residue(
                    resA, resB, index_a2b[res_idx], hybrid_atom_count
                )
            else:
                hybrid_atom_count = self._map_env_residue(
                    resA, resB, hybrid_atom_count
                )

        self.n_atoms = hybrid_atom_count

        self.map_hybrid_to_A: dict[int, int] = {h: a for a, h in self.map_A_to_hybrid.items()}
        self.map_hybrid_to_B: dict[int, int] = {h: b for b, h in self.map_B_to_hybrid.items()}
        a_set = set(self.map_hybrid_to_A)
        b_set = set(self.map_hybrid_to_B)
        self.unique_A_atoms: set[int] = a_set - b_set
        self.unique_B_atoms: set[int] = b_set - a_set
        self.env_atoms: set[int] = set(range(self.n_atoms)) - self.core_atoms - self.unique_A_atoms - self.unique_B_atoms
        self.atom_identity: dict[int, str] = {}
        self._prepare_atom_identity()
        self._prepare_hybrid_topology()


    def _map_perturbed_residue(self, resA, resB, res_entry: dict, offset: int) -> int:
        """Assign hybrid indices for one perturbed residue pair and collect broken bonds.

        Ordering: core (by A index) → unique-A → unique-B.

        Returns the next available hybrid atom index.
        """
        resA_atoms = list(resA.atoms())
        resB_atoms = list(resB.atoms())
        atom_map        = res_entry["atom_map"]
        mapping_dict_AB = {i: j for i, j in atom_map}
        mapped_B_local  = set(mapping_dict_AB.values())
        # None means all mapped atoms are core (key absent → whole residue is core)
        core_A_local = set(res_entry["core_A"]) if "core_A" in res_entry else None

        # Mapped atoms: present in both A and B, ordered by A's local index
        for local_A, local_B in sorted(mapping_dict_AB.items()):
            self.map_A_to_hybrid[resA_atoms[local_A].index] = offset
            self.map_B_to_hybrid[resB_atoms[local_B].index] = offset
            if core_A_local is None or local_A in core_A_local:
                self.core_atoms.add(offset)
            offset += 1

        # Unique-A atoms: only in A, dummy in B
        for local_A in range(len(resA_atoms)):
            if local_A not in mapping_dict_AB:
                self.map_A_to_hybrid[resA_atoms[local_A].index] = offset
                offset += 1

        # Unique-B atoms: only in B, dummy in A
        for local_B in range(len(resB_atoms)):
            if local_B not in mapped_B_local:
                self.map_B_to_hybrid[resB_atoms[local_B].index] = offset
                offset += 1

        # Broken bonds: local indices → hybrid indices
        for local_i, local_j in res_entry.get("broken_bonds_moli", []):
            idx_i = self.map_A_to_hybrid[resA_atoms[local_i].index]
            idx_j = self.map_A_to_hybrid[resA_atoms[local_j].index]
            self.broken_bonds_A.append((
                min(idx_i, idx_j),max(idx_i, idx_j),
            ))
        for local_i, local_j in res_entry.get("broken_bonds_molj", []):
            idx_i = self.map_B_to_hybrid[resB_atoms[local_i].index]
            idx_j = self.map_B_to_hybrid[resB_atoms[local_j].index]
            self.broken_bonds_B.append((
                min(idx_i, idx_j),max(idx_i, idx_j)
            ))

        return offset

    def _map_env_residue(self, resA, resB, offset: int) -> int:
        """Assign hybrid indices for one unchanged environment residue pair.

        Returns the next available hybrid atom index.
        """
        resA_atoms = list(resA.atoms())
        resB_atoms = list(resB.atoms())
        if len(resA_atoms) != len(resB_atoms):
            raise ValueError(
                f"Atom count mismatch in environment residue "
                f"'{resA.name}' (A={len(resA_atoms)}, B={len(resB_atoms)})"
            )
        for atomA, atomB in zip(resA_atoms, resB_atoms):
            self.map_A_to_hybrid[atomA.index] = offset
            self.map_B_to_hybrid[atomB.index] = offset
            offset += 1
        return offset

    def _prepare_atom_identity(self):
        for idx in self.env_atoms:
            self.atom_identity[idx] = "env"
        for idx in self.unique_A_atoms:
            self.atom_identity[idx] = "unique_A"
        for idx in self.unique_B_atoms:
            self.atom_identity[idx] = "unique_B"
        for idx in self.core_atoms:
            self.atom_identity[idx] = "core"

    def _prepare_hybrid_topology(self):
        """Build self.hybrid_top from topologyA and topologyB.

        Two-pass construction:

        1. Iterate topologyA's chain→residue structure to create the matching
           chains and residues in ``hybrid_top``.
        2. Loop over hybrid indices 0…n_atoms-1 in order, looking each atom up
           in topologyA (for A-present atoms) or topologyB (unique-B atoms), so
           ``atom.index`` in ``hybrid_top`` is guaranteed to equal the hybrid index.

        Bonds: all from A, plus any B bond whose hybrid pair was not already added.
        Box vectors are copied from topologyA.
        """
        # Pass 1: create chain/residue structure from topologyA.
        residues_in_hybrid: dict[int, app.topology.Residue] = {}
        for chainA in self.topologyA.chains():
            new_chain = self.hybrid_top.addChain(chainA.id)
            for resA in chainA.residues():
                residues_in_hybrid[resA.index] = self.hybrid_top.addResidue(
                    resA.name, new_chain, resA.id
                )

        # Pass 2: add atoms in hybrid-index order.
        atoms_A = {atom.index: atom for atom in self.topologyA.atoms()}
        atoms_B = {atom.index: atom for atom in self.topologyB.atoms()}
        added: dict[int, app.topology.Atom] = {}

        for h in range(self.n_atoms):
            if h in self.map_hybrid_to_A:
                atom = atoms_A[self.map_hybrid_to_A[h]]
            else:
                atom = atoms_B[self.map_hybrid_to_B[h]]
            added[h] = self.hybrid_top.addAtom(
                atom.name, atom.element, residues_in_hybrid[atom.residue.index]
            )

        for h, at in added.items():
            assert h == at.index, f"Hybrid index {h} != atom.index {at.index}"

        bond_add = set()
        for bond in self.topologyA.bonds():
            h1 = self.map_A_to_hybrid[bond.atom1.index]
            h2 = self.map_A_to_hybrid[bond.atom2.index]
            h1, h2 = min(h1, h2), max(h1, h2)
            if (h1, h2) in bond_add:
                warnings.warn(f"Duplicated bond in topology A.")
            self.hybrid_top.addBond(added[h1], added[h2])
            bond_add.add((h1, h2))

        for bond in self.topologyB.bonds():
            h1 = self.map_B_to_hybrid[bond.atom1.index]
            h2 = self.map_B_to_hybrid[bond.atom2.index]
            h1, h2 = min(h1, h2), max(h1, h2)
            if (h1, h2) not in bond_add:
                self.hybrid_top.addBond(added[h1], added[h2])

        if self.topologyA.getPeriodicBoxVectors() is not None:
            self.hybrid_top.setPeriodicBoxVectors(self.topologyA.getPeriodicBoxVectors())


# Hybrid RBFE REST2
def hybird_constraint_check(mapping_AB_pair: list,
                            system_A: openmm.System,
                            topology_A: app.topology.Topology,
                            system_B: openmm.System,
                            topology_B: app.topology.Topology,
                            ) -> tuple[list, list]:
    """Check constraint lengths for all mapped atom pairs and remove H atoms where length differs.

    Parameters
    ----------
    mapping_AB_pair:
        List of (A_global_idx, B_global_idx) pairs representing the atom mapping.
    system_A, topology_A:
        OpenMM system and topology for state A.
    system_B, topology_B:
        OpenMM system and topology for state B.

    Returns
    -------
    new_mapping_AB_pair : list
        Cleaned mapping with mismatched-H pairs removed.
    removed_pairs : list
        The (A_global_idx, B_global_idx) pairs that were removed.
    """
    ms_A = hybrid_topology.MolecularSystem().gen_from_openmm_system(system_A, topology_A)
    ms_B = hybrid_topology.MolecularSystem().gen_from_openmm_system(system_B, topology_B)
    map_A_to_B = {a: b for a, b in mapping_AB_pair}

    constraint_B = {
        (min(c.atoms[0], c.atoms[1]), max(c.atoms[0], c.atoms[1])): c.potential.length0
        for c in ms_B.constraints_list
    }

    to_remove_A: set[int] = set()
    removed_pairs: list = []
    for c in ms_A.constraints_list:
        at0_A, at1_A = c.atoms[0], c.atoms[1]
        if at0_A not in map_A_to_B or at1_A not in map_A_to_B:
            continue
        at0_B = map_A_to_B[at0_A]
        at1_B = map_A_to_B[at1_A]
        key_B = (min(at0_B, at1_B), max(at0_B, at1_B))
        if key_B not in constraint_B:
            continue
        if not np.isclose(constraint_B[key_B], c.potential.length0):
            for at_A in (at0_A, at1_A):
                if ms_A.atoms[at_A].element == "H" and at_A not in to_remove_A:
                    warnings.warn(
                        f"Constraint length mismatch for A-atom {at_A} (H): "
                        f"A={c.potential.length0:.6f} nm, B={constraint_B[key_B]:.6f} nm."
                    )
                    to_remove_A.add(at_A)
                    removed_pairs.append((at_A, map_A_to_B[at_A]))

    new_mapping_AB_pair = [(a, b) for a, b in mapping_AB_pair if a not in to_remove_A]
    return new_mapping_AB_pair, removed_pairs

def sp3_stereo_solver(angle_A_C_A, angle_B_C_A):
    r"""
    Given A-C-A and B-C-A angle, calculate A-C-A-B improper dihedral.

    Diagram::

         A1
          \
           C - B1
         /   \
        A2   B2

    Assumes the SP3 center is symmetric: all A-C-B angles equal
    *angle_B_C_A*. Coordinate construction: A1 = (cos(aca/2), sin(aca/2), 0)
    [unit, in xy-plane]; A2 = (cos(aca/2), -sin(aca/2), 0) [unit, symmetric];
    B1 in yz-plane s.t. B1·A1 = cos(bca).

    Closed-form result: cos(dihedral) = -tan(aca/2) / tan(bca).

    Parameters
    ----------
    angle_A_C_A : float
        A1-C-A2 angle in radians.
    angle_B_C_A : float
        B1-C-A1 angle in radians.

    Returns
    -------
    dihedral: float
        Dihedral A1-C-A2-B1 in radians, in [0, pi].
    """
    aca = angle_A_C_A
    bca = angle_B_C_A
    cos_dih = np.tan(aca / 2.0) / np.tan(bca)
    return np.arccos(np.clip(cos_dih, -1.0, 1.0))

class HybridRest2TopologyFactoryBase:
    """
    This class generate a topology for REST2 RBFE simulation.
    """
    def __init__(self,
                 system_A, position_A, rotatable_A,
                 system_B, position_B, rotatable_B,
                 index_mapping: HybridIndexMapping, softcore_alpha=0.5, soft_bond_alpha=2
                 ):
        self.index_mapping = index_mapping
        self.position_A = position_A
        self.position_B = position_B
        self._system_A = system_A  # kept for virtual-site copying
        self._system_B = system_B
        self.softcore_alpha = softcore_alpha
        self.soft_bond_alpha = soft_bond_alpha

        self.molecule_system_A = hybrid_topology.MolecularSystem().gen_from_openmm_system(system_A, index_mapping.topologyA)
        self.molecule_system_B = hybrid_topology.MolecularSystem().gen_from_openmm_system(system_B, index_mapping.topologyB)
        self.molecule_system_A.set_rotatable_bonds(rotatable_A)
        self.molecule_system_B.set_rotatable_bonds(rotatable_B)
        self.rotatable_bonds = set()
        self._set_rotatable_bonds()

        for res_id, hyb_list in self.index_mapping.hybridization["A"].items():
            self.molecule_system_A.set_hybridization_for_residues(res_id, hyb_list)
        for res_id, hyb_list in self.index_mapping.hybridization["B"].items():
            self.molecule_system_B.set_hybridization_for_residues(res_id, hyb_list)

        self.system = openmm.System()
        self._prepare_system()                # Add particle, constraint, virtual site, default box vector
        self._prepare_bond()                  # Add Forces for bond
        self._prepare_dummy_anchoring_point() #
        self._prepare_angle()
        self._prepare_dihe()





    def get_hybrid_position(self, lambda_position: float=0):
        """Return hybrid positions at the given lambda.

        Parameters
        ----------
        lambda_position : float
            0.0 = pure state A, 1.0 = pure state B.

        Returns
        -------
        numpy.ndarray, shape (n_atoms, 3), in nm.
            - Core / environment atoms: linear interpolation
              ``(1 - lambda) * posA + lambda * posB``.
            - Unique-A atoms: always ``posA``.
            - Unique-B atoms: always ``posB``.
        """


        def _to_nm(pos):
            if hasattr(pos, "value_in_unit"):
                pos = pos.value_in_unit(unit.nanometer)
            return np.array([[v.x, v.y, v.z] for v in pos])

        pos_A = _to_nm(self.position_A)
        pos_B = _to_nm(self.position_B)

        mapping    = self.index_mapping
        hybrid_to_A = {h: a for a, h in mapping.map_A_to_hybrid.items()}
        hybrid_to_B = {h: b for b, h in mapping.map_B_to_hybrid.items()}

        out = np.empty((mapping.n_atoms, 3))
        for h in range(mapping.n_atoms):
            in_A = h in hybrid_to_A
            in_B = h in hybrid_to_B
            if in_A and in_B:
                out[h] = (1.0 - lambda_position) * pos_A[hybrid_to_A[h]] + lambda_position * pos_B[hybrid_to_B[h]]
            elif in_A:
                out[h] = pos_A[hybrid_to_A[h]]
            else:
                out[h] = pos_B[hybrid_to_B[h]]

        return out * unit.nanometer

    def _set_rotatable_bonds(self):
        self.rotatable_bonds = set()
        for at1, at2 in self.molecule_system_A.rotatable_bonds:
            idx1_hyb = self.index_mapping.map_A_to_hybrid[at1]
            idx2_hyb = self.index_mapping.map_A_to_hybrid[at2]
            idx1_hyb, idx2_hyb = min(idx1_hyb, idx2_hyb), max(idx1_hyb, idx2_hyb)
            self.rotatable_bonds.add((idx1_hyb, idx2_hyb))

        for at1, at2 in self.molecule_system_B.rotatable_bonds:
            idx1_hyb = self.index_mapping.map_B_to_hybrid[at1]
            idx2_hyb = self.index_mapping.map_B_to_hybrid[at2]
            idx1_hyb, idx2_hyb = min(idx1_hyb, idx2_hyb), max(idx1_hyb, idx2_hyb),
            self.rotatable_bonds.add((idx1_hyb, idx2_hyb))

    def _prepare_system(self):
        """Add particles, virtual sites, constraints, and box vectors to self.system.

        Particle masses
        ---------------
        For each hybrid index h:
          - Core / env  →  average of A and B masses (handles e.g. C→N mutations)
          - Unique-A    →  mass from A
          - Unique-B    →  mass from B

        Constraints
        -----------
        All A constraints are added first.  For each B constraint, if the same
        hybrid atom pair already exists (from A), the distances must match to
        within 1e-6 nm — a mismatch raises ``ValueError``.  New pairs (unique-B
        atoms) are added unconditionally.

        Box vectors
        -----------
        Copied from molecule_system_A (identical to B for solvated periodic systems).
        """
        mapping = self.index_mapping
        ms_A    = self.molecule_system_A
        ms_B    = self.molecule_system_B

        # --- Particles ---
        # Core / env: average mass of A and B counterparts (handles C→N type changes).
        # Unique-A: mass from A only.  Unique-B: mass from B only.
        for h in range(mapping.n_atoms):
            in_A = h in mapping.map_hybrid_to_A
            in_B = h in mapping.map_hybrid_to_B
            if in_A and in_B:
                mass = 0.5 * (ms_A.atoms[mapping.map_hybrid_to_A[h]].mass +
                              ms_B.atoms[mapping.map_hybrid_to_B[h]].mass)
            elif in_A:
                mass = ms_A.atoms[mapping.map_hybrid_to_A[h]].mass
            else:
                mass = ms_B.atoms[mapping.map_hybrid_to_B[h]].mass
            self.system.addParticle(mass)

        # --- Virtual sites ---
        # Build hybrid-index → B VirtualSiteInfo lookup for validation.
        b_hybrid_vs: dict[int, hybrid_topology.VirtualSiteInfo] = {
            mapping.map_B_to_hybrid[b]: vs for b, vs in ms_B.virtual_sites.items()
        }

        for a_idx, vs_info_A in ms_A.virtual_sites.items():
            h_vs = mapping.map_A_to_hybrid[a_idx]
            if h_vs not in mapping.unique_A_atoms:
                # env/core: B must have an identical VS at the same hybrid index.
                if h_vs not in b_hybrid_vs:
                    raise ValueError(
                        f"Virtual site at hybrid index {h_vs} exists in A but not in B. "
                        "Core/env virtual sites must be identical in both end states."
                    )
                vs_info_B = b_hybrid_vs[h_vs]
                p_A = [mapping.map_A_to_hybrid[p] for p in vs_info_A.particles]
                p_B = [mapping.map_B_to_hybrid[p] for p in vs_info_B.particles]
                if vs_info_A.type_name != vs_info_B.type_name or p_A != p_B:
                    raise ValueError(
                        f"Virtual site at hybrid index {h_vs} differs between A and B "
                        f"(A: {vs_info_A.type_name} particles={p_A}, "
                        f"B: {vs_info_B.type_name} particles={p_B}). "
                        "Perturbing virtual sites is not supported."
                    )
            self.system.setVirtualSite(h_vs, vs_info_A.to_openmm(mapping.map_A_to_hybrid))

        # unique-B VSes (not present in A — dummy in A)
        for b_idx, vs_info in ms_B.virtual_sites.items():
            h_vs = mapping.map_B_to_hybrid[b_idx]
            if h_vs in mapping.unique_B_atoms:
                self.system.setVirtualSite(h_vs, vs_info.to_openmm(mapping.map_B_to_hybrid))

        # --- Constraints ---
        seen: dict[tuple[int, int], float] = {}  # key → length (nm)
        for c in ms_A.constraints_list:
            h1 = mapping.map_A_to_hybrid[c.atoms[0]]
            h2 = mapping.map_A_to_hybrid[c.atoms[1]]
            key = (min(h1, h2), max(h1, h2))
            self.system.addConstraint(h1, h2, c.potential.length0)
            seen[key] = c.potential.length0

        for c in ms_B.constraints_list:
            h1 = mapping.map_B_to_hybrid[c.atoms[0]]
            h2 = mapping.map_B_to_hybrid[c.atoms[1]]
            key = (min(h1, h2), max(h1, h2))
            if key in seen:
                if abs(seen[key] - c.potential.length0) > 1e-6:
                    at0 = self.molecule_system_B.atoms[c.atoms[0]]
                    at1 = self.molecule_system_B.atoms[c.atoms[1]]
                    raise ValueError(
                        f"Constraint length mismatch for hybrid atoms {key}: "
                        f"A={seen[key]:.6f} nm, B={c.potential.length0:.6f} nm ("
                        f"{at0.id}：{at0.element}-{at1.id}：{at1.element})"
                    )
            else:
                self.system.addConstraint(h1, h2, c.potential.length0)
                seen[key] = c.potential.length0

        # --- Box vectors ---
        if ms_A.box_vectors is not None:
            self.system.setDefaultPeriodicBoxVectors(*ms_A.box_vectors)

    def _prepare_bond(self):
        """Classify all hybrid bonds and populate ``self.hybrid_bond_info``.

        Each bond is assigned to one of three groups:

        * **h**   — ``HarmonicBondForce``: parameters are identical in A and B
          (unique-anything bonds, env-env bonds).
        * **c_h** — ``CustomBondForce`` harmonic: at least one core atom; ``length0``
          and ``k`` are linearly interpolated between states A and B via
          ``lambda_bonds``.
        * **c_s** — ``CustomBondForce`` soft-core: bond is present in only one end
          state (listed in ``broken_bonds_A`` or ``broken_bonds_B``).  These need
          soft-core treatment so the bond potential vanishes smoothly.
          energy = 0.5 * lambda * k (r - r0)^2 / (1 + soft_bond_alpha * (1-lambda) * (r - r0)^2)

        Classification table (symmetric, ``Nan`` = impossible pair):

        +----------+----------+----------+------+------+
        |          | unique_A | unique_B | core | env  |
        +==========+==========+==========+======+======+
        | unique_A | h        | Nan      | h    | h    |
        +----------+----------+----------+------+------+
        | unique_B | Nan      | h        | h    | h    |
        +----------+----------+----------+------+------+
        | core     | h        | h        | c_h  | c_h  |
        +----------+----------+----------+------+------+
        | env      | h        | h        | c_h  | h    |
        +----------+----------+----------+------+------+

        ``BondInfo`` named-tuple fields:

        * ``at0, at1``     — atom identity strings of the two endpoints.
        * ``length0, k0``  — equilibrium length (nm) and force constant
          (kJ mol⁻¹ nm⁻²) in state A; ``"break"`` if the bond is absent in A.
        * ``length1, k1``  — same for state B.
        * ``group``        — ``"h"``, ``"c_h"``, or ``"c_s"``.

        Processing order
        ----------------
        1. Iterate ``ms_A.bonds``: assign group and store A-state parameters.
           For ``c_h`` bonds the B-state parameters are initialised to the A values
           and will be overwritten in step 2.
        2. Iterate ``ms_B.bonds``: update ``length1/k1`` for ``c_h`` bonds;
           add new ``h`` entries for unique-B bonds; add ``c_s`` entries for
           ``broken_bonds_B``; assert env-env parameters match A.
        """
        mapping  = self.index_mapping
        ms_A     = self.molecule_system_A
        ms_B     = self.molecule_system_B

        BondInfo = namedtuple("BondInfo", "at0 at1 length0 k0 length1 k1 group")
        self.hybrid_bond_info = {}
        for bond in ms_A.bonds:
            h1  = mapping.map_A_to_hybrid[bond.atoms[0]]
            h2  = mapping.map_A_to_hybrid[bond.atoms[1]]
            h1, h2 = min(h1, h2), max(h1, h2)
            if (h1, h2) in mapping.broken_bonds_A:
                self.hybrid_bond_info[(h1, h2)] = BondInfo(
                    mapping.atom_identity[h1],
                    mapping.atom_identity[h2],
                    bond.potential.length0,
                    bond.potential.k,
                    "break",
                    "break",
                    "c_s"
                )
            elif mapping.atom_identity[h1] == "unique_A" or mapping.atom_identity[h2] == "unique_A":
                assert (h1, h2) not in self.hybrid_bond_info, "Duplicate bond in state A"
                self.hybrid_bond_info[(h1, h2)] = BondInfo(
                    mapping.atom_identity[h1],
                    mapping.atom_identity[h2],
                    bond.potential.length0,
                    bond.potential.k,
                    bond.potential.length0,
                    bond.potential.k,
                    "h"
                )
            elif mapping.atom_identity[h1] == "core" or mapping.atom_identity[h2] == "core":
                assert (h1, h2) not in self.hybrid_bond_info, "Duplicate bond in state A"
                self.hybrid_bond_info[(h1, h2)] = BondInfo(
                    mapping.atom_identity[h1],
                    mapping.atom_identity[h2],
                    bond.potential.length0,
                    bond.potential.k,
                    bond.potential.length0,
                    bond.potential.k,
                    "c_h"
                )
            elif mapping.atom_identity[h1] == "env" and mapping.atom_identity[h2] == "env":
                assert (h1, h2) not in self.hybrid_bond_info, "Duplicate bond in state A"
                self.hybrid_bond_info[(h1, h2)] = BondInfo(
                    mapping.atom_identity[h1],
                    mapping.atom_identity[h2],
                    bond.potential.length0,
                    bond.potential.k,
                    bond.potential.length0,
                    bond.potential.k,
                    "h"
                )
            else:
                assert False, f"Unclassified bond pair in state A: {mapping.atom_identity[h1]} {mapping.atom_identity[h2]}"

        for bond in ms_B.bonds:
            h1  = mapping.map_B_to_hybrid[bond.atoms[0]]
            h2  = mapping.map_B_to_hybrid[bond.atoms[1]]
            h1, h2 = min(h1, h2), max(h1, h2)
            if (h1, h2) in mapping.broken_bonds_B:
                self.hybrid_bond_info[(h1, h2)] = BondInfo(
                    mapping.atom_identity[h1],
                    mapping.atom_identity[h2],
                    "break",
                    "break",
                    bond.potential.length0,
                    bond.potential.k,
                    "c_s"
                )
            elif mapping.atom_identity[h1] == "unique_B" or mapping.atom_identity[h2] == "unique_B":
                assert (h1, h2) not in self.hybrid_bond_info
                self.hybrid_bond_info[(h1, h2)] = BondInfo(
                    mapping.atom_identity[h1],
                    mapping.atom_identity[h2],
                    bond.potential.length0,
                    bond.potential.k,
                    bond.potential.length0,
                    bond.potential.k,
                    "h"
                )
            elif mapping.atom_identity[h1] == "core" or mapping.atom_identity[h2] == "core":
                assert (h1, h2) in self.hybrid_bond_info
                bond_info_new = BondInfo(
                    self.hybrid_bond_info[(h1, h2)].at0,
                    mapping.atom_identity[h2],
                    self.hybrid_bond_info[(h1, h2)].length0,
                    self.hybrid_bond_info[(h1, h2)].k0,
                    bond.potential.length0,
                    bond.potential.k,
                    "c_h"
                )
                self.hybrid_bond_info[(h1, h2)] = bond_info_new
            elif mapping.atom_identity[h1] == "env" and mapping.atom_identity[h2] == "env":
                # should be identical
                assert np.isclose(self.hybrid_bond_info[(h1, h2)].length0, bond.potential.length0)
                assert np.isclose(self.hybrid_bond_info[(h1, h2)].k0,      bond.potential.k)
            else:
                assert False, f"Unclassified bond pair in state B: {mapping.atom_identity[h1]} {mapping.atom_identity[h2]}"

        h_force = openmm.HarmonicBondForce()

        c_h_force = openmm.CustomBondForce(
            "0.5 * ((1-lambda_bonds)*k0 + lambda_bonds*k1)"
            " * (r - ((1-lambda_bonds)*length0 + lambda_bonds*length1))^2"
        )
        c_h_force.addGlobalParameter("lambda_bonds", 0.0)
        c_h_force.addPerBondParameter("k0")
        c_h_force.addPerBondParameter("length0")
        c_h_force.addPerBondParameter("k1")
        c_h_force.addPerBondParameter("length1")

        # Bonds present in A, absent in B: lambda_bonds_A=1 coupled, 0 decoupled
        c_s_force_A = openmm.CustomBondForce(
            "0.5 * lambda_bonds_A * k * (r - length0)^2"
            " / (1 + soft_bond_alpha * (1 - lambda_bonds_A) * (r - length0)^2)"
        )
        c_s_force_A.addGlobalParameter("lambda_bonds_A", 1.0)
        c_s_force_A.addGlobalParameter("soft_bond_alpha", self.soft_bond_alpha)
        c_s_force_A.addPerBondParameter("k")
        c_s_force_A.addPerBondParameter("length0")

        # Bonds absent in A, present in B: lambda_bonds_B=1 coupled, 0 decoupled
        c_s_force_B = openmm.CustomBondForce(
            "0.5 * lambda_bonds_B * k * (r - length0)^2"
            " / (1 + soft_bond_alpha * (1 - lambda_bonds_B) * (r - length0)^2)"
        )
        c_s_force_B.addGlobalParameter("lambda_bonds_B", 0.0)
        c_s_force_B.addGlobalParameter("soft_bond_alpha", self.soft_bond_alpha)
        c_s_force_B.addPerBondParameter("k")
        c_s_force_B.addPerBondParameter("length0")

        for (h1, h2), info in self.hybrid_bond_info.items():
            if info.group == "h":
                h_force.addBond(h1, h2, info.length0, info.k0)
            elif info.group == "c_h":
                c_h_force.addBond(h1, h2, [info.k0, info.length0, info.k1, info.length1])
            elif info.group == "c_s":
                if info.length0 == "break":
                    c_s_force_B.addBond(h1, h2, [info.k1, info.length1])
                elif info.length1 == "break":
                    c_s_force_A.addBond(h1, h2, [info.k0, info.length0])
                else:
                    assert False, f"Invalid c_s bond with length0={info.length0}, length1={info.length1}"
            else:
                assert False, f"Unknown bond group {info.group}"

        self.system.addForce(h_force)
        self.system.addForce(c_h_force)
        self.system.addForce(c_s_force_A)
        self.system.addForce(c_s_force_B)



    def _prepare_dummy_anchoring_point(self):
        """Find core/env anchor atoms for unique-A and unique-B atoms, separately per state.

        An *anchoring point* is a core or env atom directly bonded or constrained
        to a unique atom.  State A adjacency is built from ``ms_A`` bonds/constraints
        (mapped through ``map_A_to_hybrid``); state B from ``ms_B``.

        Populates four attributes:

        ``anchoring_points_A`` : dict[int, set[int]]
            unique-A hybrid index → set of core/env anchor hybrid indices (from A topology).

        ``anchor_connectivity_A`` : dict[int, dict[str, list[int]]]
            For each anchor in state A, its hybrid-index neighbors grouped by class::

                {"unique_A": [...], "unique_B": [...], "core": [...], "env": [...]}

        ``anchoring_points_B`` : dict[int, set[int]]
            unique-B hybrid index → set of core/env anchor hybrid indices (from B topology).

        ``anchor_connectivity_B`` : dict[int, dict[str, list[int]]]
            Same structure as ``anchor_connectivity_A`` but for state B anchors.

        For different anchoring point, there are differences in potentials to give dummy

        Stereo SP3, 1 angle + 1 improper for 1 Dum atom
            4_SP3 to 3_R + 1_Dum
            4_SP3 to 2_R + 2_Dum

        Stereo flat, 1 angle + 1 improper for 1 Dum atom
            3_SP3 to 2_R + 1_Dum
            3_SP2 to 2_R + 1_Dum

        Keep all angle
            4_SP3 to 1_R + 3_Dum
            3_SP3 to 1_R + 2_Dum
            3_SP2 to 1_R + 2_Dum
            2_SP3 to 1_R + 1_Dum
            2_SP2 to 1_R + 1_Dum
            2_SP  to 1_R + 1_Dum


        """
        mapping = self.index_mapping
        ms_A    = self.molecule_system_A
        ms_B    = self.molecule_system_B
        core_env = mapping.core_atoms | mapping.env_atoms

        def _build_neighbors(bonds, constraints, index_map):
            nb: dict[int, set[int]] = defaultdict(set)
            for bond in bonds:
                h1 = index_map[bond.atoms[0]]
                h2 = index_map[bond.atoms[1]]
                nb[h1].add(h2)
                nb[h2].add(h1)
            for c in constraints:
                h1 = index_map[c.atoms[0]]
                h2 = index_map[c.atoms[1]]
                nb[h1].add(h2)
                nb[h2].add(h1)
            return nb

        def _anchor_connectivity(anchors, neighbors):
            result: dict[int, dict[str, list[int]]] = {}
            for anchor in anchors:
                lists: dict[str, list[int]] = {"unique_A": [], "unique_B": [], "core": [], "env": []}
                for nb in neighbors.get(anchor, set()):
                    if nb in mapping.unique_A_atoms:
                        lists["unique_A"].append(nb)
                    elif nb in mapping.unique_B_atoms:
                        lists["unique_B"].append(nb)
                    elif nb in mapping.core_atoms:
                        lists["core"].append(nb)
                    else:
                        lists["env"].append(nb)
                result[anchor] = lists
            return result

        # State A: unique-A atoms anchored in A's topology
        neighbors_A = _build_neighbors(ms_A.bonds, ms_A.constraints_list, mapping.map_A_to_hybrid)
        self.anchoring_points_A = {
            u: {nb for nb in neighbors_A.get(u, set()) if nb in core_env}
            for u in mapping.unique_A_atoms
        }
        all_anchors_A = {a for s in self.anchoring_points_A.values() for a in s}
        self.anchor_connectivity_A = _anchor_connectivity(all_anchors_A, neighbors_A)

        # State B: unique-B atoms anchored in B's topology
        neighbors_B = _build_neighbors(ms_B.bonds, ms_B.constraints_list, mapping.map_B_to_hybrid)
        self.anchoring_points_B = {
            u: {nb for nb in neighbors_B.get(u, set()) if nb in core_env}
            for u in mapping.unique_B_atoms
        }
        all_anchors_B = {a for s in self.anchoring_points_B.values() for a in s}
        self.anchor_connectivity_B = _anchor_connectivity(all_anchors_B, neighbors_B)

        # Build additional parameters for anchoring points
        ## For all unique_A connected to real system
        AnchorSummary = namedtuple("AnchorSummary", ["hybridization", "real_count", "dummy_count"])
        for center_idx, connection_dict in self.anchor_connectivity_A.items():
            hybridization = self.molecule_system_A.atoms[center_idx].hybridization
            real_count = len(connection_dict["env"]) + len(connection_dict["core"])
            dummy_count = len(connection_dict["unique_A"])
            connection_dict["summary"] = AnchorSummary(hybridization, real_count, dummy_count)
            if real_count == 1:
                # keep all angle
                pass
            elif hybridization=="SP3":
                if real_count + dummy_count == 4:
                    # build stereo SP3 for each dummy
                    pass
                elif real_count + dummy_count == 3:
                    # build stereo flat for each dummy
                    pass
                else:
                    assert False, f"Impossible {hybridization} anchoring point with {real_count=} + {dummy_count=}"
            elif hybridization=="SP2":
                assert real_count == 2
                assert dummy_count == 1
                # build stereo flat
                pass
            elif hybridization not in ["SP3", "SP2", "SP"]:
                # keep all angle
                warnings.warn(f"Unseen hybridization {hybridization=}")




        ## For all unique_B connected to real system
        for center_idx, connection_dict in self.anchor_connectivity_B.items():
            hybridization = self.molecule_system_B.atoms[center_idx].hybridization
            real_count = len(connection_dict["env"]) + len(connection_dict["core"])
            dummy_count = len(connection_dict["unique_B"])
            connection_dict["summary"] = AnchorSummary(hybridization, real_count, dummy_count)

    def _prepare_angle(self):
        pass

    def _prepare_dihe(self):
        pass


class HybridRest2TopologyFactory(HybridRest2TopologyFactoryBase):
    """
    XXX
    """
    def __init__(self):
        HybridRest2TopologyFactoryBase.__init__(self)

    def _prepare_nonbonded(self):
        pass


class HybridRest2TopologyFactoryWaterSwap(HybridRest2TopologyFactoryBase):
    """
    XXX
    """

    def __init__(self):
        HybridRest2TopologyFactoryBase.__init__(self)

    def _prepare_nonbonded(self):
        pass


class HybridRest2TopologyFactoryWaterIonSwap(HybridRest2TopologyFactoryBase):
    """
    XXX
    """

    def __init__(self):
        HybridRest2TopologyFactoryBase.__init__(self)

    def _prepare_nonbonded(self):
        pass






# ABFE REST2

# help function

