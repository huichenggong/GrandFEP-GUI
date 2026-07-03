import unittest
from pathlib import Path
import itertools
import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski

from openmm import app, unit, openmm
from grandfep import utils, hybrid_topology

from utils_test import calc_energy_force, match_force, separate_force, find_term_in_force



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


def check_bond_angle(h_factory, systemA, topA, systemB, topB, test_case):
    """Bond+Angle force comparison for both end-states using original hybrid coordinates.

    For atoms whose forces differ between the hybrid and end-state systems
    (anchors and ref1 atoms of active dummy restraint angles), the comparison
    is skipped and a Newton's-3rd-law check is done instead: the sum of forces
    over each restraint angle (unique atom + anchor + ref1) must be zero.
    """
    print("## Bond, Angle")
    sys_A_ba  = separate_force(systemA, ["HarmonicBondForce", "HarmonicAngleForce"])
    sys_B_ba  = separate_force(systemB, ["HarmonicBondForce", "HarmonicAngleForce"])
    sys_hyb_ba = separate_force(h_factory.system,
        ["HarmonicBondForce", "CustomBondForce",
         "CustomBondForce_h", "CustomBondForce_s_A", "CustomBondForce_s_B",
         "HarmonicAngleForce", "CustomAngleForce", "CustomAngleForce_A", "CustomAngleForce_B"])

    reorder_h_2_A = [h_factory.index_mapping.map_A_to_hybrid[i] for i in range(systemA.getNumParticles())]
    reorder_h_2_B = [h_factory.index_mapping.map_B_to_hybrid[i] for i in range(systemB.getNumParticles())]

    # ---- State A ----
    print("### State A")
    pos_hyb_A = h_factory.get_hybrid_position(0)
    pos_A = [pos_hyb_A[h_factory.index_mapping.map_A_to_hybrid[i]] for i in range(systemA.getNumParticles())]
    _, forceA_ba = calc_energy_force(sys_A_ba, topA, pos_A)
    _, forceH_A  = calc_energy_force(sys_hyb_ba, h_factory.index_mapping.hybrid_top, pos_hyb_A)

    # Collect all atoms in any CustomAngleForce angle involving unique_B atoms.
    # This covers: dummy restraint angles, n_uB>=2 constant-k angles, and n_uB==1
    # dr_B_keep constant-k angles — all of which have non-zero k0 at state A.
    unique_B_set = h_factory.index_mapping.unique_B_atoms
    extra_angle_atoms_uB = set()
    for force in h_factory.system.getForces():
        if force.getName() == "CustomAngleForce":
            for i in range(force.getNumAngles()):
                p0, p1, p2, params = force.getAngleParameters(i)
                if p0 in unique_B_set or p1 in unique_B_set or p2 in unique_B_set:
                    extra_angle_atoms_uB.update([p0, p1, p2])

    # A-local indices of any atom in extra_angle_atoms_uB that exists in the A-system
    unique_B_affected_A = sorted({
        h_factory.index_mapping.map_hybrid_to_A[at]
        for at in extra_angle_atoms_uB
        if at in h_factory.index_mapping.map_hybrid_to_A
    })
    all_close, _, error_msg = match_force(forceA_ba, forceH_A[reorder_h_2_A], excluded_list=unique_B_affected_A)
    test_case.assertTrue(all_close, "Bond+Angle state A\n" + error_msg)

    # Newton's 3rd law: total extra force from ALL unique_B angle terms = 0.
    f_extra_A = np.zeros(3)
    for at_h in extra_angle_atoms_uB:
        if at_h in h_factory.index_mapping.map_hybrid_to_A:
            f_extra_A += forceH_A[at_h] - forceA_ba[h_factory.index_mapping.map_hybrid_to_A[at_h]]
        else:
            f_extra_A += forceH_A[at_h]
    test_case.assertTrue(
        np.allclose(f_extra_A, np.zeros(3)),
        f"Total extra force from unique_B angle terms != 0: {f_extra_A}")

    # ---- State B ----
    print("### State B")
    gp_B = {"lambda_bonds": 1.0, "lambda_bonds_A": 0.0, "lambda_bonds_B": 1.0,
             "lambda_angle": 1.0, "lambda_angle_A": 0.0, "lambda_angle_B": 1.0}
    pos_hyb_B = h_factory.get_hybrid_position(1)
    pos_B = [pos_hyb_B[h_factory.index_mapping.map_B_to_hybrid[i]] for i in range(systemB.getNumParticles())]
    _, forceB_ba = calc_energy_force(sys_B_ba, topB, pos_B)
    _, forceH_B  = calc_energy_force(sys_hyb_ba, h_factory.index_mapping.hybrid_top, pos_hyb_B, global_parameters=gp_B)

    # Collect all atoms in any CustomAngleForce_A angle involving unique_A atoms.
    # This mirrors the state-A logic: covers restraint, n_uA>=2, and dr_A_keep angles.
    unique_A_set = h_factory.index_mapping.unique_A_atoms
    extra_angle_atoms_uA = set()
    for force in h_factory.system.getForces():
        if force.getName() in ("CustomAngleForce", "CustomAngleForce_A"):
            for i in range(force.getNumAngles()):
                p0, p1, p2, params = force.getAngleParameters(i)
                if p0 in unique_A_set or p1 in unique_A_set or p2 in unique_A_set:
                    extra_angle_atoms_uA.update([p0, p1, p2])

    # B-local indices of any atom in extra_angle_atoms_uA that exists in the B-system
    unique_A_affected_B = sorted({
        h_factory.index_mapping.map_hybrid_to_B[at]
        for at in extra_angle_atoms_uA
        if at in h_factory.index_mapping.map_hybrid_to_B
    })
    all_close, _, error_msg = match_force(forceB_ba, forceH_B[reorder_h_2_B], excluded_list=unique_A_affected_B)
    test_case.assertTrue(all_close, "Bond+Angle state B\n" + error_msg)

    # Newton's 3rd law: total extra force from ALL unique_A angle terms = 0.
    f_extra_B = np.zeros(3)
    for at_h in extra_angle_atoms_uA:
        if at_h in h_factory.index_mapping.map_hybrid_to_B:
            f_extra_B += forceH_B[at_h] - forceB_ba[h_factory.index_mapping.map_hybrid_to_B[at_h]]
        else:
            f_extra_B += forceH_B[at_h]
    test_case.assertTrue(
        np.allclose(f_extra_B, np.zeros(3)),
        f"Total extra force from unique_A angle terms != 0: {f_extra_B}")


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

        inpcrdA, prmtopA, topA, systemA, rotatable_A = load_ligand(lig1_path / "02_solv.inpcrd",
                                                                   lig1_path / "02_solv.prmtop",
                                                                   lig1_path / f"{lig1_path.name}.sdf")
        inpcrdB, prmtopB, topB, systemB, rotatable_B = load_ligand(lig2_path / "02_solv.inpcrd",
                                                                   lig2_path / "02_solv.prmtop",
                                                                   lig2_path / f"{lig2_path.name}.sdf")
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

        check_bond_angle(h_factory, systemA, topA, systemB, topB, self)







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

        index_map = hybrid_topology.HybridIndexMapping(topA, topB, {0: mapping})
        h_factory = hybrid_topology.HybridRest2TopologyFactoryBase(
            systemA, inpcrdA.positions, rotatable_A,
            systemB, inpcrdB.positions, rotatable_B,
            index_map
        )
        check_bond_angle(h_factory, systemA, topA, systemB, topB, self)

    def test_hybrid_rest2_hspw_edge_1_0(self):
        print("\n# hsp90 woodhead, break a bond in state B")
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
        print("\n## Hybridization")
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

        print("## Anchor Info")
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

        print("## bond force and angle force")
        check_bond_angle(h_factory, systemA, topA, systemB, topB, self)
        with open(base / "hybrid_topology/output/hspw_edge_1_0_hybrid.pdb", "w") as f:
            app.PDBFile.writeFile(h_factory.index_mapping.hybrid_top, h_factory.get_hybrid_position(0), f)

        # check angle term on dummy atoms
        c_angles_39 = find_term_in_force(h_factory.system, "CustomAngleForce", 39) # unique_A
        self.assertEqual(len(c_angles_39), 4)
        self.assertListEqual([c_angles_39[0][3][3], c_angles_39[1][3][3], c_angles_39[2][3][3], c_angles_39[3][3][1]], [0, 0, 0, 0])

        c_angles_41 = find_term_in_force(h_factory.system, "CustomAngleForce", 41) # unique_B
        h_angles_41 = find_term_in_force(h_factory.system, "HarmonicAngleForce", 41)  # unique_B
        cA_angles_41 = find_term_in_force(h_factory.system, "CustomAngleForce_A", 41)  # unique_B
        cB_angles_41 = find_term_in_force(h_factory.system, "CustomAngleForce_B", 41)  # unique_B

        self.assertEqual(len(h_angles_41), 0)
        self.assertEqual(len(c_angles_41), 7)
        self.assertEqual(len(cA_angles_41), 0)
        self.assertEqual(len(cB_angles_41), 5) # bond 21-41 is broken, 41 is a -CH2-, 21 is a -NH-. H-C21-N41 x2, C-C21-N41, C41-N21-C, C41-N21-N

        print("## Dihedral Info")


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
        h_factory = hybrid_topology.HybridRest2TopologyFactoryBase(
            systemA, inpcrdA.positions, rotatable_A,
            systemB, inpcrdB.positions, rotatable_B,
            index_map
        )
        check_bond_angle(h_factory, systemA, topA, systemB, topB, self)
        with open(base / "hybrid_topology/output/hspw_edge_2_3_hybrid.pdb", "w") as f:
            app.PDBFile.writeFile(h_factory.index_mapping.hybrid_top, h_factory.get_hybrid_position(0), f)

        c_angles_42 = find_term_in_force(h_factory.system, "CustomAngleForce", 42) # unique_B
        h_angles_42 = find_term_in_force(h_factory.system, "HarmonicAngleForce", 42)  # unique_B
        cA_angles_42 = find_term_in_force(h_factory.system, "CustomAngleForce_A", 42)  # unique_B
        cB_angles_42 = find_term_in_force(h_factory.system, "CustomAngleForce_B", 42)  # unique_B

        self.assertEqual(len(h_angles_42), 0)
        self.assertEqual(len(c_angles_42), 7)
        for at0, at1, at2, angle_param in c_angles_42:
            self.assertTupleEqual(angle_param[:2], angle_param[2:])
        self.assertEqual(len(cA_angles_42), 0)
        self.assertEqual(len(cB_angles_42), 0) # bond 21-41 is broken, 41 is a -CH2-, 21 is a -NH-. H-C21-N41 x2, C-C21-N41, C41-N21-C, C41-N21-N

        c_angles_39 = find_term_in_force(h_factory.system, "CustomAngleForce", 39)  # unique_B
        h_angles_39 = find_term_in_force(h_factory.system, "HarmonicAngleForce", 39)  # unique_B
        cA_angles_39 = find_term_in_force(h_factory.system, "CustomAngleForce_A", 39)  # unique_B
        cB_angles_39 = find_term_in_force(h_factory.system, "CustomAngleForce_B", 39)  # unique_B

        self.assertEqual(len(h_angles_39), 0)
        self.assertEqual(len(c_angles_39), 7)
        self.assertEqual(len(cA_angles_39), 4)
        self.assertEqual(len(cB_angles_39), 0)

    def test_brd4_all(self):
        print("\n brd4")
        ligand_path = base / "public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/waterset/brd41_ASH106/"
        for edge, a_name, b_name in [
            ["edge_0_6", "A01", "A07"],
            ["edge_1_0", "A02", "A01"],
            ["edge_1_2", "A02", "A03"],
            ["edge_1_3", "A02", "A04"],
            ["edge_1_4", "A02", "A05"],
            ["edge_1_5", "A02", "A06"],
            ["edge_1_6", "A02", "A07"],
            ["edge_1_7", "A02", "A08"],
            ["edge_2_7", "A03", "A08"],
            ["edge_3_5", "A04", "A06"],
            ["edge_4_6", "A05", "A07"],
        ]:
            print(f"## {edge}")
            lig1_path = ligand_path / f"ligand_preparation/{a_name}"
            lig2_path = ligand_path / f"ligand_preparation/{b_name}"
            with open(ligand_path / f"{edge}/mapping_visial_checked.json") as f:
                mapping = json.load(f)
            inpcrdA, prmtopA, topA, systemA, rotatable_A = load_ligand(lig1_path / "02_solv.inpcrd",
                                                                       lig1_path / "02_solv.prmtop",
                                                                       lig1_path / f"{lig1_path.name}.sdf")
            inpcrdB, prmtopB, topB, systemB, rotatable_B = load_ligand(lig2_path / "02_solv.inpcrd",
                                                                       lig2_path / "02_solv.prmtop",
                                                                       lig2_path / f"{lig2_path.name}.sdf")
            index_map = hybrid_topology.HybridIndexMapping(topA, topB, {0: mapping})
            h_factory = hybrid_topology.HybridRest2TopologyFactoryBase(
                systemA, inpcrdA.positions, rotatable_A,
                systemB, inpcrdB.positions, rotatable_B,
                index_map
            )
            with open(ligand_path / f"{edge}/hybrid_solv.pdb", "w") as f:
                app.PDBFile.writeFile(h_factory.index_mapping.hybrid_top, h_factory.get_hybrid_position(0), f)

            print(f"### Check Number of Dihedral entry")
            ptf_A = next(
                f for f in (systemA.getForce(i) for i in range(systemA.getNumForces()))
                if isinstance(f, openmm.PeriodicTorsionForce)
            )
            ptf_B = next(
                f for f in (systemB.getForce(i) for i in range(systemB.getNumForces()))
                if isinstance(f, openmm.PeriodicTorsionForce)
            )
            n_src_A, n_src_B = ptf_A.getNumTorsions(), ptf_B.getNumTorsions()
            n_classified_A = (
                sum(len(v["A"]) for v in h_factory.hybrid_proper_dihedral_info.values())
                + sum(len(v["A"]) for v in h_factory.hybrid_improper_dihedral_info.values())
            )
            n_classified_B = (
                    sum(len(v["B"]) for v in h_factory.hybrid_proper_dihedral_info.values())
                    + sum(len(v["B"]) for v in h_factory.hybrid_improper_dihedral_info.values())
            )
            self.assertEqual(
                n_src_A, n_classified_A,
                f"{edge}: systemA has {n_src_A} torsion terms but "
                f"hybrid_dihedral_info['A'] has {n_classified_A}"
            )
            self.assertEqual(
                n_src_B, n_classified_B,
                f"{edge}: systemB has {n_src_B} torsion terms but "
                f"hybrid_dihedral_info['A'] has {n_classified_B}"
            )

            
            if edge == "edge_0_6":
                pass
    
    def test_improper_dihedral_star_LUT(self):
        print("\n_IMPROPER_DIHEDRAL_STAR_GROUP_LUT covers all star-topology cases correctly.")
        from grandfep.hybrid_topology import hybrid_factory

        lut = hybrid_factory._IMPROPER_DIHEDRAL_STAR_GROUP_LUT

        # Distinct sorted outer-label tuples (multiplicity matters: 4 patterns, not 2^3=8)
        outer_patterns = {tuple(t) for t in itertools.product(["r", "u"], repeat=3)}
        all_keys = {(hub, outer) for hub in ("r", "u") for outer in outer_patterns}
        self.assertEqual(set(lut.keys()), all_keys,
                         "LUT keys do not match the expected set")

        expected: dict[tuple, str | None] = {
            # ─── 2r, 1u
            ("u", ("r", "r", "u")): None,
            ("u", ("r", "u", "r")): None,
            ("u", ("u", "r", "r")): None,
            # ─── 3r
            ("u", ("r", "r", "r")): None,
        }
        for key, group in expected.items():
            self.assertEqual(lut[key], group,
                             f"LUT[{key}]: expected {group}, got {lut[key]}")


    def test_dihedral_LUT(self):
        print("\n_PROPER_DIHEDRAL_GROUP_LUT covers all 16 (r/u)^4 patterns correctly.")
        from grandfep.hybrid_topology import hybrid_factory

        lut = hybrid_factory._PROPER_DIHEDRAL_GROUP_LUT

        # --- completeness: exactly the 16 (r/u)^4 keys, nothing more ---
        all_keys = set(itertools.product(["r", "u"], repeat=4))
        self.assertEqual(set(lut.keys()), all_keys,
                         "LUT keys do not match the full (r/u)^4 set")

        # None  → forbidden (u at inner position, or c-u-u-c)
        expected: dict[tuple, str | None] = {
            ("r", "r", "u", "r"): None,
            ("r", "u", "r", "r"): None,
            ("r", "u", "u", "r"): None,
            ("u", "r", "u", "r"): None,
            ("r", "u", "r", "u"): None,
        }
        for k,v in expected.items():
            self.assertEqual(lut[k], v,)


if __name__ == '__main__':
    unittest.main()
