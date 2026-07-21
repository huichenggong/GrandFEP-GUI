# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

currently the development uses the mamba env called `gfep_gui_dev`

## Current development focus
`src/grandfep/hybrid_topology`

## Project Overview

GrandFEP-GUI is a reconstruction of the [GrandFEP](https://github.com/deGrootLab/GrandFEP) package — a Python toolkit for Free Energy Perturbation (FEP) simulations with enhanced sampling (water/ion swap). This repo adds a web-based GUI and new features: Ion Swap for GPCRs, Core Hopping (scaffold hopping), and True Dummy atoms (separable in the partition function, no redundant bonded terms).

## Architecture

**`src/grandfep/`** — installable Python library, no GUI dependencies:
- `src/grandfep/hybrid_topology/` — **active development**. Core data model (`molecules.py`: `Atom`, `Residue`, `BondPotential`, `AnglePotential`, `DihedralPotential`, `NonbondedExceptionPotential`, `VirtualSiteInfo`, `MolecularSystem`, and `TermTable` subclasses) and topology factory classes (`hybrid_factory.py`: `Rest2TopologyFactory`, `HybridIndexMapping`, `HybridRest2TopologyFactoryBase`, plus the `AnchorInfo` / `DihedralInfoBase` / `ProperDihedralInfo` / `ImproperDihedralInfo` term-info classes).
- `src/grandfep/utils/` — I/O utilities (`io.py`: `load_amber_sys` for loading AMBER inpcrd/prmtop files).
- `src/grandfep/samplers/` — placeholder for FEP sampler classes (not yet implemented).

**`app/`** — planned GUI layer (not yet implemented):
- Backend: FastAPI with REST endpoints
- Frontend: htmx + Alpine.js (no npm/build pipeline)
- Molecule rendering: RDKit server-side SVG
- 3D viewer: NGL.js (CDN) for protein/ligand/trajectory visualization

## FEP Workflow (6 steps)

The GUI layer (`app/`) and setup modules (`grandfep/setup/`) are not yet implemented. Current development focuses on step 5 (hybrid system construction).

1. **Protein Preparation** (planned) — load PDB → check/add missing residues/side chains → cap termini → assign protonation states → check disulfide bonds → validate `openmm.Topology` + `openmm.System`
2. **Ligand Preparation** (planned) — load SDF(s) with pre-assigned protonation states and explicit H → parameterize via antechamber, OpenFF, or MATCH → create small molecule class
3. **Ligand Mapping** (planned) — build perturbation network linking all ligands; user can manually adjust the network graph
4. **Atom Mapping** (planned) — MCS-based auto-mapping per ligand pair → user reviews/corrects via GUI (click-to-pair on 2D SVGs); exports atom index mapping for perturbation
5. **FEP with Enhanced Sampling** — hybrid system construction via `HybridRest2TopologyFactoryBase` + sampler execution; jobs submitted to cluster (scheduler integration TBD)
6. **Analysis** (planned) — free energy estimate extraction and visualization

## Key Domain Concepts

- **Enhanced sampling here means water-swap + REST2 + ion-swap**: exchanging water/ion positions in the binding site, important for GPCRs (Ion Swap feature)
- **True Dummy atoms**: dummy atoms must be separable in the partition function with no redundant bonded terms
- **Core Hopping**: scaffold hopping transformations require atom mapping that crosses ring systems; the mapping code must handle cases where no single MCS covers the full perturbation

## Key Classes (hybrid_topology)

### `MolecularSystem` (`molecules.py`)
Flat, ID-keyed data model for one simulation system. Provides fast bidirectional lookups between atoms and their bonded terms. Built from an OpenMM `System` + `Topology` via `gen_from_openmm_system()`.

**Atom identities** (five-way classification for hybrid topologies):
- **core** — atoms present in both A and B whose parameters change during the alchemical transformation
- **unique_A** — atoms only in state A (dummy in state B)
- **unique_B** — atoms only in state B (dummy in state A)
- **swap** — water molecules for water-swap Monte Carlo
- **env** — environment atoms, identical in both states (can be scaled by REST2)

**Key containers:**
- `atoms: dict[int, Atom]` — maps particle index → `Atom` dataclass (id, name, element, charge, mass, sigma, epsilon, is_virtual_site, is_rest2, hybridization)
- `residues: dict[int, Residue]` — maps residue index → `Residue`
- `bonds: BondTable`, `angles: AngleTable` — harmonic bonded terms
- `proper_dihedrals: DihedralTable`, `improper_dihedrals: DihedralTable` — torsion terms partitioned by bond-chain connectivity
- `nonbonded_exceptions: NonbondedExceptionTable` — 1-4 exceptions and 1-2/1-3 exclusions
- `constraints_list: list[BondTerm]` — bond-length constraints (kept separate from harmonic bonds)
- `virtual_sites: dict[int, VirtualSiteInfo]` — virtual site atoms
- `rotatable_bonds: set[tuple[int, int]]` — populated externally (e.g., from RDKit); drives REST2 dihedral scaling via `rest2_scalable_dihedrals()`

All numerical values are in OpenMM native units (nm, kJ/mol, radians, elementary charge, Da).

### `VirtualSiteInfo` (`molecules.py`)
Type-independent representation of an OpenMM `VirtualSite`. Supports `TwoParticleAverageSite`, `ThreeParticleAverageSite`, `OutOfPlaneSite`, and `LocalCoordinatesSite`. Key methods:
- `from_openmm(vs)` — extract parameters from an OpenMM `VirtualSite` object
- `to_openmm(index_map=None)` — reconstruct the OpenMM object, optionally remapping particle indices (used when building hybrid topologies)

### `Rest2TopologyFactory` (`hybrid_factory.py`)
Generates a topology for **plain REST2** (non-hybrid, single end-state) enhanced sampling. Controlled by two global parameters: `k_rest2` and `k_rest2_sqrt` (caller must keep `k_rest2 = k_rest2_sqrt^2`).

**Scaling convention** (per bonded term, counting hot atoms involved):
- `n_hot = 2` → scale by `k_rest2_sqrt^2` (= `k_rest2`)
- `n_hot = 1` → scale by `k_rest2_sqrt`
- `n_hot = 0` → unscaled

**Force construction:**
- Bonds and angles: unmodified `HarmonicBondForce` / `HarmonicAngleForce`
- Dihedrals: unscaled terms go to `PeriodicTorsionForce`; REST2-scaled terms (proper dihedrals on rotatable bonds with `n_hot >= 1`) go to a `CustomTorsionForce` with expression `k_rest2_sqrt^n_hot * k * (1 + cos(n * theta - phase))`
- Nonbonded: hot-atom charges/epsilons stored as base=0 with `addParticleParameterOffset` / `addExceptionParameterOffset` keyed to `k_rest2_sqrt` and `k_rest2` — this ensures PME reciprocal space is also correctly scaled

### `HybridIndexMapping` (`hybrid_factory.py`)
Maps atom indices from two end-state topologies (A and B) onto a single hybrid topology. Each residue listed in `index_a2b` is *perturbed*; all others are environment.

**Hybrid atom ordering within a perturbed residue:**
1. Core atoms (mapped A↔B, ordered by A's local index)
2. Unique-A atoms (only in A)
3. Unique-B atoms (only in B)

**Key attributes:**
- `map_A_to_hybrid`, `map_B_to_hybrid` — global index → hybrid index
- `map_hybrid_to_A`, `map_hybrid_to_B` — reverse mappings
- `core_atoms`, `unique_A_atoms`, `unique_B_atoms`, `env_atoms` — disjoint sets covering all hybrid indices. 
`core_atoms`: The parameters (bond, angle, dihedral, vdw, coulomb) around thoes atoms can possibly changed. 
`unique_A_atoms`: They only exist in state A and will be dummy atoms in state B. All the nonbonded (vdw, coulomb) 
interaction around those atoms will be turned off in state B. Bond will be kept untouched on those atoms in state B. 
The uncoupled angle interaction will be kept, for example `core`-`unique`-`unique` and `unique`-`unique`-`unique`. 
The `core`-`core`-`unique` will be specially treated as different types of anchoring point. The uncoupled 
dihedral if rotatable will be scaled down in state B, for example `unique`-`unique`-`unique`-`unique`, 
`core`-`unique`-`unique`-`unique`, `core`-`core`-`unique`-`unique`. The `core`-`core`-`core`-`unique` 
will in general, be turned to 0 in state B. `unique_B_atoms`: They only exist in state B and will be dummy 
atoms in state A.  
- `atom_identity: dict[int, str]` — maps each hybrid index to one of `"core"`, `"unique_A"`, `"unique_B"`, `"env"`
- `broken_bonds_A`, `broken_bonds_B` — bonds present in only one end-state (hybrid-index pairs)
- `hybrid_top: openmm.app.Topology` — the merged hybrid topology
- `topologyA`, `topologyB` — the original end-state `openmm.app.Topology` objects (stored for downstream use, e.g. building a `MolecularSystem` per state)
- `hybridization: dict[str, dict[int, list]]` — per-state (`"A"`/`"B"`) hybridization strings for atoms in perturbed residues (from the `hybridization_moli`/`hybridization_molj` mapping keys); drives SP3/SP2 stereo handling for dummy atoms

### `HybridRest2TopologyFactoryBase` (`hybrid_factory.py`)
Base class for building hybrid REST2 RBFE systems. Takes two end-state systems, positions, rotatable bond sets, and a `HybridIndexMapping`.

**Bond classification** (stored in `hybrid_bond_info` as `BondInfo` named-tuples with fields `at0, at1, length0, k0, length1, k1, group`):

| Group | Force | Description |
|-------|-------|-------------|
| `h` | `HarmonicBondForce` | Parameters identical in A and B (unique-*, env-env) |
| `c_h` | `CustomBondForce` | At least one core atom; length0 and k linearly interpolated via `lambda_bonds` |
| `c_s` | `CustomBondForce` (soft-core) | Bond present in only one end-state (broken); soft-core potential with `soft_bond_alpha` |

**Soft-core bond expression** (for bonds breaking/forming):
```
0.5 * lambda * k * (r - r0)^2 / (1 + soft_bond_alpha * (1 - lambda) * (r - r0)^2)
```

**Anchor points** (`_prepare_dummy_anchoring_point`): For unique (dummy) atoms, finds core/env anchor atoms bonded or constrained to them and summarizes each anchor's connectivity + hybridization. Populates `anchor_info: dict[int, AnchorInfo]` (one entry per anchor atom) and `dummy_restraint` (`{"unique_A": {...}, "unique_B": {...}}`, each mapping dummy hybrid index → `{"angles": [...], "impropers": [...]}`). Per dummy, it either keeps the existing angle (when the anchor has only 1 real reference atom) or adds 1 angle + 1 harmonic improper to preserve SP3/SP2 stereochemistry — this is what makes dummies separable in the partition function (True Dummy).

**Angles** (`_prepare_angle`): env-env-env → `HarmonicAngleForce`; everything else → a `CustomAngleForce` interpolating A→B via `lambda_angle` (per-angle `theta0, k0, theta1, k1`). `unique_A`-(env/core) angles get `k1=0` (off in B); `unique_B`-(env/core) get `k0=0` (off in A); angles with ≥2 unique atoms keep `k0=k1`; core/env angles interpolate A→B. Two extra `CustomAngleForce_A`/`_B` (via `lambda_angle_A`/`lambda_angle_B`) hold broken-bond angles. Dummy-restraint angles from `_prepare_dummy_anchoring_point` are added with the complementary k so they turn on in the dummy state.

**Dihedrals** (`_prepare_dihe`): every proper and improper term is classified by `ProperDihedralInfo`/`ImproperDihedralInfo` (frozen dataclasses with a `.classify()` classmethod) into a group, then routed to one of five torsion forces:

| Group | Force | k0/k1 rule |
|-------|-------|------------|
| `env` | `PeriodicTorsionForce` | constant k; all-env, identical in A and B |
| `normal` | `CustomTorsionForce` (`lambda_dihedral`) | A: k0=k,k1=0; B: k0=0,k1=k (≥1 core, not broken) |
| `break` | `CustomTorsionForce_A`/`_B` (`lambda_dihedral_A`/`_B`) | scales with the break-side lambda |
| `anchor` | `CustomTorsionForce` (`lambda_dihedral`) | A: k0=k,k1=0; B: k0=0,k1=k (1u/2u permitted patterns) |
| `uu` | `CustomTorsionForce` (`lambda_dihedral`) | rotatable proper: scaled by `dummy_dihe_scaling`; else k. Kept in dummy state — integrates to a partition-function constant I₀(βk) |
| `dummy` | `CustomTorsionForce_harmonic` (`lambda_dihedral`) | harmonic minimum-image improper from `_prepare_dummy_anchoring_point`; unique_A turns on at λ→1, unique_B at λ→0 |

Classification collapses atom identity to `r` (core/env) or `u` (unique_A/B) and looks up a LUT; patterns that violate True-Dummy separability (e.g. a dummy group anchored to two distinct real fragments) raise `ValueError`. Classified terms are stored in `hybrid_proper_dihedral_info[(min(at1,at2),max(at1,at2))]["A"/"B"]` and `hybrid_improper_dihedral_info[hub]["A"/"B"]`.

**Note**: REST2 scaling (`k_rest2`/`k_rest2_sqrt`) is not yet wired into the hybrid bond/angle/dihedral force expressions — only `Rest2TopologyFactory` applies it.

**Subclasses** — exist and call `super().__init__()` (so the bonded layer is built), then call `_prepare_nonbonded()`, which is still `pass` in all three (nonbonded not yet implemented):
- `HybridRest2TopologyFactory` — basic RBFE
- `HybridRest2TopologyFactoryWaterSwap` — RBFE with water swap
- `HybridRest2TopologyFactoryWaterIonSwap` — RBFE with water + ion swap

**Helper functions:**
- `hybird_constraint_check()` — validates constraint lengths for mapped atom pairs; removes H atoms where constraint lengths differ between A and B
- `sp3_stereo_solver()` — computes the A-C-A-B improper dihedral angle for an SP3 center given A-C-A and B-C-A angles

## Nonbonded Force

> **Status:** Design spec only — **not yet implemented**. The hybrid subclasses' `_prepare_nonbonded` are `pass`. The tables below describe the planned vdw/Coulomb treatment for the hybrid system.

### vdw
For the vdw interaction, we set a **NonbondedForce** and a **CustomNonbondedForce**. All of the interactions that need soft-core goes to 
**CustomNonbondedForce**.  

Global parameters: 
- **k_rest2**      :  
- **k_rest2_sqrt** :
- **lambda_vdw_core_A**
- **lambda_vdw_core_B**
- **lambda_vdw_unique_A**
- **lambda_vdw_unique_B**


|          | core     | unique_A | unique_B | swap     | env      |
|----------|----------|----------|----------|----------|----------|
| core     |  N       |          |          |          |          |
| unique_A |  Cust    |  Cust    |          |          |          |
| unique_B |  Cust    |          |  Cust    |          |          |
| swap     |  Cust    |  Cust    |  Cust    |  Cust    |          |
| env      |  N       |  Cust    |  Cust    |  Cust    |  N       |

- **core**     : These atoms are changing in state A and B  
- **unique_A** : These atoms only appears in state A and they are dummy in state B  
- **unique_B** : These atoms only appears in state B and they are dummy in state A  
- **swap**     : 2 water molecules for water-swap Monte Carlo  
- **env**      : Other atoms, they have the same vdw in state A and B. they can be scaled by REST2  

0/1 encoding of the atom identity. PerParticleParameter
is_core  
is_unique_A  
is_unique_B  
is_swap  
is_hot

### Coulomb
All the Coulomb is in NonbondedForce. 

Global parameters: 
- **k_rest2**      :  
- **k_rest2_sqrt** : 
- **lambda_coulomb_unique_A** :
- **lambda_coulomb_unique_A_k_rest2_sqrt** :
- **lambda_coulomb_unique_B**   :
- **lambda_coulomb_unique_B_x_k_rest2_sqrt** :
- **lambda_coulomb_core_A**                  :
- **lambda_coulomb_core_A_x_k_rest2_sqrt**   :
- **lambda_coulomb_core_B**                  :
- **lambda_coulomb_core_B_x_k_rest2_sqrt**   :
- **lambda_coulomb_swap1**                   :
- **lambda_coulomb_swap2**                   :

|           | is_hot | ParticleParameterOffset 1            | ParticleParameterOffset 2            |
|-----------|--------|--------------------------------------|--------------------------------------|
| core      | Y      | lambda_coulomb_core_A_x_k_rest2_sqrt | lambda_coulomb_core_B_x_k_rest2_sqrt |
| core      |        | lambda_coulomb_core_A                | lambda_coulomb_core_B                |
| unique_A  | Y      | lambda_coulomb_unique_A_k_rest2_sqrt |                                      | 
| unique_A  |        | lambda_coulomb_unique_A              |                                      |
| unique_B  | Y      | lambda_coulomb_unique_B_k_rest2_sqrt |                                      |
| unique_B  |        | lambda_coulomb_unique_B              |                                      |
| swap1     |        | lambda_coulomb_swap1                 |                                      |
| swap2     |        | lambda_coulomb_swap2                 |                                      |
| else      | Y      | k_rest2_sqrt (Coulomb)               | k_rest2 (vdw)                        |
| else      |        |                                      |                                      |
