hybrid\_topology
=================

.. currentmodule:: grandfep.hybrid_topology

.. contents:: hybrid_topology — Molecular System and Interaction Terms
   :depth: 2
   :local:
   :backlinks: entry

The ``hybrid_topology`` subpackage provides a lightweight, unit-keyed
representation of a molecular system that supports fast bidirectional
lookups between atoms and bonded interactions.  It is the data layer
used when constructing hybrid topologies for alchemical FEP
transformations and REST2 enhanced sampling.

Overview
--------

A :class:`MolecularSystem` is populated from an OpenMM ``System`` and
``Topology`` pair via :meth:`MolecularSystem.gen_from_openmm_system`.
All atom indices match the 0-based OpenMM particle indices, and all
numerical values are stored in **OpenMM native units** (nm, kJ/mol,
radians, elementary charge, Da).

.. code-block:: text

    MolecularSystem
    ├── atoms                : dict[int, Atom]              ← keyed by 0-based index
    ├── residues             : dict[int, Residue]
    ├── bonds                : BondTable                    ← HarmonicBondForce entries
    ├── angles               : AngleTable                   ← HarmonicAngleForce entries
    ├── proper_dihedrals     : DihedralTable                ← proper torsions (bond-chain)
    ├── improper_dihedrals   : DihedralTable                ← improper (out-of-plane) torsions
    ├── nonbonded_exceptions : NonbondedExceptionTable      ← exclusions (1-2/1-3) and 1-4 exceptions
    ├── constraints_list     : list[BondTerm]               ← System constraints (no k)
    ├── virtual_sites        : dict[int, VirtualSiteInfo]   ← virtual site atoms
    └── rotatable_bonds      : set[tuple[int,int]]          ← populated externally (e.g., from RDKit)

Bidirectional lookup
~~~~~~~~~~~~~~~~~~~~

Every :class:`TermTable` (and its subclasses) internally maintains an
``atom_id → [term indices]`` index built at insertion time, so both
directions are O(1)::

    # atom → interactions
    mol_sys.terms_for_atom(atom_id)      # returns dict of lists
    mol_sys.bonds.terms_for_atom(atom_id)

    # interaction → atoms
    mol_sys.atoms_for_term(bond_term)    # returns list[Atom]
    bond_term.atoms                      # raw tuple of atom indices

Multi-term dihedrals
~~~~~~~~~~~~~~~~~~~~

GAFF / Amber force fields assign 2–4 Fourier components to each
rotatable bond.  OpenMM stores each component as a separate row in
``PeriodicTorsionForce``; :class:`DihedralTable` follows the same
convention — one :class:`DihedralTerm` per row, all sharing the same
four atom indices.  ``terms_for_atom`` returns all of them.

Quick-start
-----------

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

    # All bonded terms involving atom 0
    print(mol_sys.terms_for_atom(0))

    # Atom objects that participate in the first stored bond
    first_bond = mol_sys.bonds[0]
    print(mol_sys.atoms_for_term(first_bond))

API Reference
-------------

MolecularSystem
~~~~~~~~~~~~~~~

.. autoclass:: MolecularSystem
   :members:
   :undoc-members:
   :show-inheritance:

Atom and topology objects
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Atom
   :members:

.. autoclass:: Residue
   :members:

Term and potential objects
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BondTerm
   :members:

.. autoclass:: BondPotential
   :members:

.. autoclass:: AngleTerm
   :members:

.. autoclass:: AnglePotential
   :members:

.. autoclass:: DihedralTerm
   :members:

.. autoclass:: DihedralPotential
   :members:

TermTable and subclasses
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TermTable
   :members:

.. autoclass:: BondTable
   :show-inheritance:

.. autoclass:: AngleTable
   :show-inheritance:

.. autoclass:: DihedralTable
   :show-inheritance:

.. autoclass:: NonbondedExceptionTable
   :show-inheritance:

Nonbonded exceptions
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NonbondedExceptionTerm
   :members:

.. autoclass:: NonbondedExceptionPotential
   :members:

Virtual sites
~~~~~~~~~~~~~

.. currentmodule:: grandfep.hybrid_topology

.. autoclass:: VirtualSiteInfo
   :members:
   :undoc-members:
   :noindex:

REST2 factory
-------------

:class:`Rest2TopologyFactory` converts a normal AMBER/GAFF OpenMM system into a
REST2-ready system by scaling the interaction of hot (REST2) atom with global
parameters of ``k_rest2``. The nonbonded interaction and the selected rotatable
bond dihedral will be scaled. ``k_rest2_sqrt`` is also defined


REST2 scaling convention:

.. code-block:: text

    n_hot = 2 → scale by k_rest2      (= k_rest2_sqrt^2)   ← both central torsion atoms hot
    n_hot = 1 → scale by k_rest2_sqrt                      ← one central torsion atom hot
    n_hot = 0 → unscaled

.. currentmodule:: grandfep.hybrid_topology.hybrid_factory

.. autoclass:: Rest2TopologyFactory
   :members:
   :undoc-members:
   :show-inheritance:

Hybrid topology factory
-----------------------

:class:`HybridIndexMapping` maps atom indices from two end-state topologies
(A and B) onto a single hybrid topology.  Each perturbed residue follows the
ordering core → unique-A → unique-B.  It also collects broken bonds (present
in only one end state) and builds the merged ``hybrid_top`` topology.

.. autoclass:: HybridIndexMapping
   :members:
   :undoc-members:
   :show-inheritance:

:class:`HybridRest2TopologyFactoryBase` is the base class for building hybrid
REST2 RBFE systems. Only bonded interaction is handled in this based class,
Nonbonded interaction will differ because of water-swap or ion-swap.

.. autoclass:: HybridRest2TopologyFactoryBase
   :members:
   :undoc-members:
   :show-inheritance:

Helper functions
~~~~~~~~~~~~~~~~

.. autofunction:: hybird_constraint_check

.. autofunction:: sp3_stereo_solver
