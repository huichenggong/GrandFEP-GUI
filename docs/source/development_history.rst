Development History
===================

2.0.0_dev (current)
-------------------

Core data model and topology factories for hybrid alchemical FEP with REST2
enhanced sampling.

Implemented
~~~~~~~~~~~

* ``MolecularSystem`` — flat, ID-keyed data model with bidirectional
  atom↔term lookups built from OpenMM ``System`` + ``Topology``.
* ``VirtualSiteInfo`` — type-independent representation of OpenMM
  virtual sites (TwoParticleAverage, ThreeParticleAverage, OutOfPlane,
  LocalCoordinates) with index remapping for hybrid topologies.
* ``Rest2TopologyFactory`` — plain REST2 topology factory with
  ``addParticleParameterOffset`` / ``addExceptionParameterOffset``
  for correct PME scaling.
* ``HybridIndexMapping`` — maps atom indices from two end-state
  topologies onto a single hybrid topology (core / unique-A /
  unique-B / env classification).
* ``HybridRest2TopologyFactoryBase`` — hybrid RBFE factory handling
  particles, virtual sites, constraints, and the full bonded force
  layer:

  - **Bonds** — classified into ``h`` (harmonic), ``c_h``
    (interpolated via ``lambda_bonds``), and ``c_s`` (soft-core,
    ``lambda_bonds_A`` / ``lambda_bonds_B``) with the Gapsys-style
    soft-core potential.
  - **Dummy-atom anchoring** — ``_prepare_dummy_anchoring_point``
    builds ``anchor_info`` (per-anchor ``AnchorInfo``) and
    ``dummy_restraint``.  For each dummy it either keeps the existing
    angle or adds 1 angle + 1 harmonic improper to preserve SP3 / SP2
    stereochemistry (True-Dummy separability).
  - **Angles** — ``_prepare_angle`` routes env-env-env terms to a
    ``HarmonicAngleForce`` and everything else to a
    ``CustomAngleForce`` interpolating A→B (``lambda_angle``), plus
    separate ``CustomAngleForce_A`` / ``_B`` for broken-bond angles.
  - **Dihedrals** — ``_prepare_dihe`` classifies every proper and
    improper term via ``ProperDihedralInfo`` / ``ImproperDihedralInfo``
    into groups ``env`` / ``normal`` / ``break`` / ``anchor`` / ``uu``
    and routes them into five torsion forces.  Rotatable ``uu`` proper
    dihedrals are scaled by ``dummy_dihe_scaling``; dummy-stereo
    restraints use a harmonic (minimum-image) improper form.

* Hybrid term info classes — ``AnchorInfo``, ``DihedralInfoBase``,
  ``ProperDihedralInfo``, ``ImproperDihedralInfo``.
* Bonded forces tested with bond-breaking and bond-forming cases;
  proper and improper dihedral classification tested.
* Gapsys soft-core potential reference for nonbonded interactions.

Planned
~~~~~~~

* Nonbonded force for hybrid systems (``_prepare_nonbonded``) — the
  three subclasses exist and wire up ``super().__init__``, but their
  ``_prepare_nonbonded`` bodies are still ``pass``.
* REST2 scaling of hybrid bonded terms — ``k_rest2`` / ``k_rest2_sqrt``
  are not yet wired into the hybrid bond / angle / dihedral force
  expressions (only ``Rest2TopologyFactory`` applies them).
* Water-swap and ion-swap hybrid factory subclasses (nonbonded pending).
* Sampler classes (``grandfep/samplers/``).
* GUI layer (FastAPI + htmx + Alpine.js).
