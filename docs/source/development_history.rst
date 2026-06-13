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
* ``HybridRest2TopologyFactoryBase`` — hybrid RBFE factory with bond
  classification (harmonic / interpolated / soft-core) and dummy-atom
  anchor-point detection.
* Bonded forces tested with bond-breaking and bond-forming cases.
* Gapsys soft-core potential reference for nonbonded interactions.

Planned
~~~~~~~

* Nonbonded force for hybrid systems (``_prepare_nonbonded``).
* Angle and dihedral potentials for dummy atoms.
* Water-swap and ion-swap hybrid factory subclasses.
* Sampler classes (``grandfep/samplers/``).
* GUI layer (FastAPI + htmx + Alpine.js).
