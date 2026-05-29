# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

currently the development uses the mamba env called `gfep_GUI_dev`

## Commands

```bash
# Start dev server (auto-reloads on Python file changes)
mamba run -n gfep_GUI_dev uvicorn app.main:app --reload --port 8000

# Run all API tests (no browser/server needed)
mamba run -n gfep_GUI_dev pytest tests/output/GUI_test/test_api.py -v

# Run a single test
mamba run -n gfep_GUI_dev pytest tests/output/GUI_test/test_api.py::test_ligand_svgs_render_with_rdkit -v
```

See `tests/output/GUI_test/README.md` for the full manual browser test checklist.

## Project Overview

GrandFEP-GUI is a reconstruction of the [GrandFEP](https://github.com/deGrootLab/GrandFEP) package — a Python toolkit for Free Energy Perturbation (FEP) simulations with enhanced sampling (water/ion swap). This repo adds a web-based GUI and new features: Ion Swap for GPCRs, Core Hopping (scaffold hopping), and True Dummy atoms (separable in the partition function, no redundant bonded terms).

## Architecture

### Two-layer design

**`grandfep/`** — installable Python library, no GUI dependencies:
- `grandfep/samplers/` — FEP sampler classes (`WaterSwapSamplerMPI`, `BasicSampler`) and sampler utilities
- `grandfep/utils/` — hybrid system factory (`relative_REST2_factory`), MD parameters (`md_parameters`)
- `grandfep/setup/` — workflow setup modules: `protein_prep`, `ligand_prep`, `ligand_mapping`, `atom_mapping`

**`app/`** — web GUI layer (FastAPI), depends on `grandfep/`:
- `app/main.py` — FastAPI application entry point
- `app/routers/` — one router per workflow step (protein, ligand, mapping, fep, analysis)
- `app/static/` — JavaScript: `atom_map.js` (interactive SVG atom pairing), `viewer.js` (NGL.js wrapper)
- `app/templates/` — Jinja2 HTML templates

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

- **Enhanced sampling here means GCMC/water swap**: exchanging water/ion positions in the binding site, important for GPCRs (Ion Swap feature)
- **True Dummy atoms**: dummy atoms must contribute zero to the partition function with no residual bonded terms leaking between end states — a correctness constraint that shapes the hybrid topology construction in `utils/`
- **Core Hopping**: scaffold hopping transformations require atom mapping that crosses ring systems; the mapping code must handle cases where no single MCS covers the full perturbation

## New Feature Notes

- **Ion Swap**: extends water swap sampling to include ion exchange; relevant for GPCR systems where metal/ion coordination affects binding
- **Core Hopping**: full pipeline support from ligand mapping through perturbation for scaffold-hopping transformations (not just R-group changes)
- **True Dummy**: implementation constraint — dummy atoms must be separable in the partition function; check `grandfep/utils/` hybrid topology construction when modifying perturbation logic

