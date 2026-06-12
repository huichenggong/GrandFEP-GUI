import unittest
from pathlib import Path
from typing import List, Tuple
import copy
import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski

from openmm import app, unit, openmm
from grandfep import utils, hybrid_topology

from utils_test import calc_energy_force, match_force, separate_force



base = Path(__file__).resolve().parent.parent

def get_rotatable_bond_from_sdf(sdf):
    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
    mol = supplier[0]
    query = Lipinski.RotatableBondSmarts
    matches = mol.GetSubstructMatches(query)
    return matches

class MyTestCase(unittest.TestCase):
    def test_hybrid_rest2(self):
        print("# Macrocycle 2B8V")
        ligand_path = base / "public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/macrocycles/2B8V_lig24and25_alpha05/"
        lig1_path = ligand_path / "ligand_preparation/A01"
        lig2_path = ligand_path / "ligand_preparation/A02"
        inpcrdA, prmtopA, systemA = utils.load_amber_sys(
            lig1_path / "02_solv.inpcrd",
            lig1_path / "02_solv.prmtop",
        )
        topA = prmtopA.topology
        rotatable_A = get_rotatable_bond_from_sdf(lig1_path / f"{lig1_path.name}.sdf")
        inpcrdB, prmtopB, systemB = utils.load_amber_sys(
            lig2_path / "02_solv.inpcrd",
            lig2_path / "02_solv.prmtop",
        )
        topB = prmtopB.topology
        rotatable_B = get_rotatable_bond_from_sdf(lig2_path / f"{lig2_path.name}.sdf")
        with open(ligand_path / "edge_0_1/mapping.json") as f:
            mapping = json.load(f)

        index_map = hybrid_topology.HybridIndexMapping(topA, topB, {0: mapping})
        h_factory = hybrid_topology.HybridRest2TopologyFactoryBase(
            systemA, inpcrdA.positions, rotatable_A,
            systemB, inpcrdB.positions, rotatable_B,
            index_map
        )
        print("## Bonded")
        energyA, forceA = calc_energy_force(
            separate_force(systemA, ["HarmonicBondForce"]),
            topA, inpcrdA.positions)
        energyB, forceB = calc_energy_force(
            separate_force(systemB, ["HarmonicBondForce"]),
            topB, inpcrdB.positions)
        energyH, forceH = calc_energy_force(
            separate_force(h_factory.system, ["HarmonicBondForce", "CustomBondForce"]),
            h_factory.index_mapping.hybrid_top, h_factory.get_hybrid_position(0))
        energyH1, forceH1 = calc_energy_force(
            separate_force(h_factory.system, ["HarmonicBondForce", "CustomBondForce"]),
            h_factory.index_mapping.hybrid_top, h_factory.get_hybrid_position(1),
            global_parameters={"lambda_bonds_A":0.0, "lambda_bonds_B":1.0, "lambda_bonds":1.0})

        # Unique atom is constraint, energy should be the same
        self.assertAlmostEqual(energyH, energyA)
        self.assertAlmostEqual(energyH1, energyB)

        reorder_h_2_A = [h_factory.index_mapping.map_A_to_hybrid[i] for i in range(systemA.getNumParticles())]
        all_close, _, error_msg = match_force(forceA, forceH[reorder_h_2_A])
        self.assertTrue(all_close, "Bonded term state A " + error_msg)

        reorder_h_2_B = [h_factory.index_mapping.map_B_to_hybrid[i] for i in range(systemB.getNumParticles())]
        all_close, _, error_msg = match_force(forceB, forceH1[reorder_h_2_B])
        self.assertTrue(all_close, "Bonded term state B " + error_msg)


    def test_hybrid_rest2_hspw(self):
        ligand_path = base / "public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/waterset/hsp90_woodhead/"
        lig1_path = ligand_path / "ligand_preparation/A02"
        lig2_path = ligand_path / "ligand_preparation/A01"
        inpcrdA, prmtopA, systemA = utils.load_amber_sys(
            lig1_path / "02_solv.inpcrd",
            lig1_path / "02_solv.prmtop",
        )
        topA = prmtopA.topology
        rotatable_A = get_rotatable_bond_from_sdf(lig1_path / f"{lig1_path.name}.sdf")
        inpcrdB, prmtopB, systemB = utils.load_amber_sys(
            lig2_path / "02_solv.inpcrd",
            lig2_path / "02_solv.prmtop",
        )
        topB = prmtopB.topology
        rotatable_B = get_rotatable_bond_from_sdf(lig2_path / f"{lig2_path.name}.sdf")
        with open(ligand_path / "edge_1_0/mapping_constraint_checked.json") as f:
            mapping = json.load(f)

        index_map = hybrid_topology.HybridIndexMapping(topA, topB, {0: mapping})
        h_factory = hybrid_topology.HybridRest2TopologyFactoryBase(
            systemA, inpcrdA.positions, rotatable_A,
            systemB, inpcrdB.positions, rotatable_B,
            index_map
        )
        # print(h_factory.anchor_connectivity_A)
        # print(h_factory.anchor_connectivity_B)



if __name__ == '__main__':
    unittest.main()
