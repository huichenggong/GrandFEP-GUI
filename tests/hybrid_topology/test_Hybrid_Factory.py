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


def load_ligand(inpcrd, prmtop, sdf):
    inpcrd, prmtop, system = utils.load_amber_sys(inpcrd, prmtop)
    rotatable = get_rotatable_bond_from_sdf(sdf)
    return inpcrd, prmtop, prmtop.topology, system, rotatable

class MyTestCase(unittest.TestCase):
    def test_hybrid_constraint_check(self):
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
        with open(ligand_path / "edge_1_0/mapping.json") as f:
            mapping = json.load(f)

        new_pairs, removed_pairs = hybrid_topology.hybrid_constraint_check(mapping["atom_map"], systemA, topA, systemB, topB)
        self.assertEqual(len(new_pairs), 39)
        self.assertEqual(removed_pairs, [(40,42)])

    def test_sp3_stereo_solver(self):
        dihe = hybrid_topology.sp3_stereo_solver(np.arccos(-1/3), np.arccos(-1/3))
        self.assertAlmostEqual(dihe, 2/3*np.pi)

        from MDAnalysis.lib.distances import calc_dihedrals
        a1 = np.array([ 1, 1, 0])
        a2 = np.array([-1, 1, 0])
        C  = np.array([ 0, 0, 0])
        b1 = np.array([ 0,-1, 1])
        cos_theta = np.dot(a2, b1) / (np.linalg.norm(a2) * np.linalg.norm(b1))
        dihe = hybrid_topology.sp3_stereo_solver(0.5 * np.pi, np.arccos(cos_theta))
        angle_rad = calc_dihedrals(
            a1.reshape(1, 3),
            a2.reshape(1, 3),
            C .reshape(1, 3),
            b1.reshape(1, 3)
        )
        self.assertAlmostEqual(dihe, angle_rad[0])

    def test_hybrid_rest2_macrocycles_2B8V(self):
        print("\n# Macrocycle 2B8V")
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
        with open(ligand_path / "edge_0_1/mapping_visial_checked.json") as f:
            mapping = json.load(f)

        index_map = hybrid_topology.HybridIndexMapping(topA, topB, {0: mapping})
        h_factory = hybrid_topology.HybridRest2TopologyFactoryBase(
            systemA, inpcrdA.positions, rotatable_A,
            systemB, inpcrdB.positions, rotatable_B,
            index_map
        )
        print("## Bond")
        energyA, forceA = calc_energy_force(
            separate_force(systemA, ["HarmonicBondForce"]),
            topA, inpcrdA.positions)
        energyB, forceB = calc_energy_force(
            separate_force(systemB, ["HarmonicBondForce"]),
            topB, inpcrdB.positions)
        energyH, forceH = calc_energy_force(
            separate_force(h_factory.system, ["HarmonicBondForce", "CustomBondForce",
                                              "CustomBondForce_h", "CustomBondForce_s_A", "CustomBondForce_s_B"]),
            h_factory.index_mapping.hybrid_top, h_factory.get_hybrid_position(0))
        energyH1, forceH1 = calc_energy_force(
            separate_force(h_factory.system, ["HarmonicBondForce", "CustomBondForce",
                                              "CustomBondForce_h", "CustomBondForce_s_A", "CustomBondForce_s_B"]),
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

        print("## Bond, Angle")
        sys_A_bond_angle = separate_force(systemA, ["HarmonicBondForce", "HarmonicAngleForce",])
        sys_B_bond_angle = separate_force(systemB, ["HarmonicBondForce", "HarmonicAngleForce"])
        sys_hyb_bond_angle = separate_force(h_factory.system, ["HarmonicBondForce", "CustomBondForce",
                                                               "CustomBondForce_h", "CustomBondForce_s_A", "CustomBondForce_s_B",
                                                               "HarmonicAngleForce", "CustomAngleForce",
                                                               "CustomAngleForce_A", "CustomAngleForce_B"])
        platform = openmm.Platform.getPlatform('Reference')

        # optimize the coordinate with h_factory.system in state A and compare forces for state A
        print("### State A")
        integrator_A = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
        sim_A = app.Simulation(h_factory.index_mapping.hybrid_top, h_factory.system, integrator_A, platform)
        sim_A.context.setPositions(h_factory.get_hybrid_position(0))
        sim_A.minimizeEnergy()
        state_A = sim_A.context.getState(getPositions=True, getEnergy=True)
        pos_opt_A = state_A.getPositions()
        with open("/tmp/hybrid_opt_stateA.pdb", "w") as f:
            app.PDBFile.writeFile(h_factory.index_mapping.hybrid_top, pos_opt_A, f)
        pos_A_opt = [pos_opt_A[h_factory.index_mapping.map_A_to_hybrid[i]] for i in range(systemA.getNumParticles())]
        energyA_ba, forceA_ba = calc_energy_force(sys_A_bond_angle, topA, pos_A_opt)
        energyH_A, forceH_A = calc_energy_force(sys_hyb_bond_angle, h_factory.index_mapping.hybrid_top, pos_opt_A)
        
        # self.assertAlmostEqual(energyH_A, energyA_ba)
        all_close, _, error_msg = match_force(forceA_ba, forceH_A[reorder_h_2_A])
        self.assertTrue(all_close, "Bond+Angle state A \n" + error_msg)

        # optimize the coordinate with h_factory.system in state B and compare forces for state B
        print("### State B")







    def test_hybrid_rest2_macrocycle_2Q15_0_4_improper(self):
        print("\n macrocycle, bond breaking with improper dihedral")
        ligand_path = base / "public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/macrocycles/2Q15_lig17to21_alpha05/"
        lig1_path = ligand_path / "ligand_preparation/A01"
        lig2_path = ligand_path / "ligand_preparation/A05"

        inpcrdA, prmtopA, topA, systemA, rotatable_A = load_ligand(lig1_path / "02_solv.inpcrd",
                                                                   lig1_path / "02_solv.prmtop",
                                                                   lig1_path / f"{lig1_path.name}.sdf")
        inpcrdB, prmtopB, topB, systemB, rotatable_B = load_ligand(lig2_path / "02_solv.inpcrd",
                                                                   lig2_path / "02_solv.prmtop",
                                                                   lig2_path / f"{lig2_path.name}.sdf")
        with open(ligand_path / "edge_0_4/mapping_visial_checked.json") as f:
            mapping = json.load(f)

    def test_hybrid_rest2_hspw_edge_1_0(self):
        print("\n hsp90 woodhead, break a bond in state B")
        ligand_path = base / "public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/waterset/hsp90_woodhead/"
        lig1_path = ligand_path / "ligand_preparation/A02"
        lig2_path = ligand_path / "ligand_preparation/A01"

        inpcrdA, prmtopA, topA, systemA, rotatable_A = load_ligand(lig1_path / "02_solv.inpcrd",
                                                             lig1_path / "02_solv.prmtop",
                                                             lig1_path / f"{lig1_path.name}.sdf")
        inpcrdB, prmtopB, topB, systemB, rotatable_B = load_ligand(lig2_path / "02_solv.inpcrd",
                                                             lig2_path / "02_solv.prmtop",
                                                             lig2_path / f"{lig2_path.name}.sdf")
        with open(ligand_path / "edge_1_0/mapping_visial_checked.json") as f:
            mapping = json.load(f)
        index_map = hybrid_topology.HybridIndexMapping(topA, topB, {0: mapping})
        self.assertListEqual(["SP3", "SP3", "SP2", "SP2", "SP2"], [index_map.hybridization["A"][0][idx] for idx in [1, 2, 3, 4, 5]])
        self.assertListEqual(["SP3", "SP3", "SP2", "SP2", "SP2"], [index_map.hybridization["B"][0][idx] for idx in [2, 3, 4, 5, 6]])
        h_factory = hybrid_topology.HybridRest2TopologyFactoryBase(
            systemA, inpcrdA.positions, rotatable_A,
            systemB, inpcrdB.positions, rotatable_B,
            index_map
        )
        self.assertListEqual(
            [h_factory.molecule_system_B.atoms[i].hybridization for i in [1, 19, 20, 21]],
            ["SP3", "SP2", "SP2", "SP2"])  # C, C, C, O
        self.assertListEqual(
            [h_factory.molecule_system_B.atoms[i].hybridization for i in [5, 6, 7, 9]],
            ["SP2", "SP2", "SP2", "SP2"]) # C, C, C, N

        self.assertSetEqual({1, 21}, set([hybrid_idx for hybrid_idx in h_factory.anchor_info.keys()]))
        self.assertEqual(    h_factory.anchor_info[1].hybridization_A, "SP3")
        self.assertEqual(    h_factory.anchor_info[1].hybridization_B, "SP3")
        self.assertDictEqual(h_factory.anchor_info[1].unique_A, {39: 'AB'})
        self.assertDictEqual(h_factory.anchor_info[1].unique_B, {41: 'AB'})
        self.assertDictEqual(h_factory.anchor_info[1].core, {0: 'AB', 2: 'AB', 3: 'AB'})
        self.assertSetEqual( h_factory.anchor_info[1].env, set())
        self.assertEqual(len(h_factory.anchor_info[1].angle_A), 6)
        self.assertEqual(len(h_factory.anchor_info[1].angle_B), 6)

        self.assertEqual(    h_factory.anchor_info[21].hybridization_A, "SP2")
        self.assertEqual(    h_factory.anchor_info[21].hybridization_B, "SP2")
        self.assertDictEqual(h_factory.anchor_info[21].unique_A, {40: 'AB'})
        self.assertDictEqual(h_factory.anchor_info[21].unique_B, {41: 'B', 44: 'AB'})
        self.assertDictEqual(h_factory.anchor_info[21].core, {20: 'AB'})
        self.assertSetEqual( h_factory.anchor_info[21].env, set())
        self.assertEqual(len(h_factory.anchor_info[21].angle_A), 1)
        self.assertEqual(len(h_factory.anchor_info[21].angle_B), 3)


        # print(h_factory.anchor_info)

    def test_hybrid_rest2_hspw_edge_2_3(self):
        print("\n hsp90 woodhead, break a bond in state A")
        ligand_path = base / "public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/waterset/hsp90_woodhead/"
        lig1_path = ligand_path / "ligand_preparation/A03"
        lig2_path = ligand_path / "ligand_preparation/A04"

        inpcrdA, prmtopA, topA, systemA, rotatable_A = load_ligand(lig1_path / "02_solv.inpcrd",
                                                             lig1_path / "02_solv.prmtop",
                                                             lig1_path / f"{lig1_path.name}.sdf")
        inpcrdB, prmtopB, topB, systemB, rotatable_B = load_ligand(lig2_path / "02_solv.inpcrd",
                                                             lig2_path / "02_solv.prmtop",
                                                             lig2_path / f"{lig2_path.name}.sdf")
        with open(ligand_path / "edge_2_3/mapping_visial_checked.json") as f:
            mapping = json.load(f)
        index_map = hybrid_topology.HybridIndexMapping(topA, topB, {0: mapping})




if __name__ == '__main__':
    unittest.main()
