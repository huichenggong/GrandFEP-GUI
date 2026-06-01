# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

currently the development uses the mamba env called `gfep_gui_dev`

## Current development focuse
`src/grandfep/hybrid_topology`

## Project Overview

GrandFEP-GUI is a reconstruction of the [GrandFEP](https://github.com/deGrootLab/GrandFEP) package — a Python toolkit for Free Energy Perturbation (FEP) simulations with enhanced sampling (water/ion swap). This repo adds a web-based GUI and new features: Ion Swap for GPCRs, Core Hopping (scaffold hopping), and True Dummy atoms (separable in the partition function, no redundant bonded terms).

## Architecture

**`src/grandfep/`** — installable Python library, no GUI dependencies:
- `src/grandfep/samplers/` — FEP sampler classes (`WaterSwapSamplerMPI`, `BasicSampler`) and sampler utilities
- `src/grandfep/utils/` — hybrid system factory (`relative_REST2_factory`), MD parameters (`md_parameters`)
- `src/grandfep/interaction_table` - topology classes (`Atom`, `Residue`, ...)


### GUI tech stack

- **Backend**: FastAPI with REST endpoints
- **Molecule rendering**: RDKit server-side SVG (2D depictions with atom coordinates sent to browser)
- **Atom mapping canvas**: SVG + vanilla JS overlay for click-to-pair interaction on server-rendered SVGs
- **3D viewer**: NGL.js (CDN) for protein/ligand/trajectory visualization — handles large GPCR systems and supports XTC/DCD trajectory formats needed for FEP analysis
- **Frontend**: htmx + Alpine.js (no npm/build pipeline)

## FEP Workflow (6 steps)

Each step maps to a `grandfep/setup/` module and a `app/routers/` endpoint:

1. **Protein Preparation** — load PDB → check/add missing residues/side chains → cap termini → assign protonation states → check disulfide bonds → validate `openmm.Topology` + `openmm.System`
2. **Ligand Preparation** — load SDF(s) with pre-assigned protonation states and explicit H → parameterize via antechamber, OpenFF, or MATCH → create small molecule class
3. **Ligand Mapping** — build perturbation network linking all ligands; user can manually adjust the network graph
4. **Atom Mapping** — MCS-based auto-mapping per ligand pair → user reviews/corrects via GUI (click-to-pair on 2D SVGs); exports atom index mapping for perturbation
5. **FEP with Enhanced Sampling** — hybrid system construction (`relative_REST2_factory`) + sampler execution; jobs submitted to cluster (scheduler integration TBD)
6. **Analysis** — free energy estimate extraction and visualization

## Key Domain Concepts

- **Enhanced sampling here means water-swap + REST2 + ion-swap**: exchanging water/ion positions in the binding site, important for GPCRs (Ion Swap feature)
- **True Dummy atoms**: dummy atoms must be seperable in the partition function with no redundent bonded terms
- **Core Hopping**: scaffold hopping transformations require atom mapping that crosses ring systems; the mapping code must handle cases where no single MCS covers the full perturbation


