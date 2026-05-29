# GrandFEP-GUI
Reconstruction of the GrandFEP package with GUI and more features.

# FEP Full Cycle
**1. Protein Preparation**  
**2. Ligand Preparation**  
**3. Ligand Mapping**  
**4. Atom Mapping**  
**5. FEP with enhanced sampling**  
**6. Analysis**  

# New features planned
- **GUI**: Atom selection and mapping  
- **Ion Swap**: GPCRs also need enhanced sampling of Ion  
- **Core Hopping**: Scaffold hopping and core hopping. Full support for from mapping to perturbation.
- **True Dummy**: Dummy atoms should be separable in the partition function. They should have no redundant bonded terms.

# Installation
```bash
# Place Holder For now
```

# Basic Structure
```python
from grandfep import samplers # sampler classes for FEP with enhanced sampling
# samplers.WaterSwapSamplerMPI
# samplers.BasicSampler
# samplers.utils

from grandfep import utils
# utils.relative_REST2_factory # factory function for creating hybrid system
# utils.md_parameters

from grandfep import setup
# setup.protein_prep
# setup.ligand_prep
# setup.ligand_mapping
# setup.atom_mapping
```

```bash
./app/grandfep_gui.py # GUI
./docs/               # Documentation
./tests/              # Tests
```

# Basic workflow
## 1. Protein Preparation
- load pdb file into a protein class
- check for missing residues
- add missing residues/side chains
- Cap or modify termini
- assign protonation states
- check disulfide bonds
- test if we can prepare a `openmm.Topology` and `openmm.System`
## 2. Ligand Preparation
- load sdf file(s) with multiple ligands
- hydrogen must be added before loading (protonation state assigned before loading)
- antechamber or openff or match
- load sdf with parameters to create a small molecule class
## 3. Ligand Mapping
- create a map to link all the ligands
- allow user to manually adjust the map
## 4. Atom Mapping
- create atom mapping for each ligand pair
- allow user to manually check/adjust the atom mapping
## 5. FEP with enhanced sampling
## 6. Analysis
