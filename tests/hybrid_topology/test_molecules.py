from pathlib import Path
import unittest
from typing import List, Tuple
import copy
import json
from collections import namedtuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski

from openmm import app, unit, openmm
from grandfep import utils, hybrid_topology

from utils_test import calc_energy_force, match_force, separate_force


base = Path(__file__).resolve().parent.parent

def inference_hybridization_rdkit(mol):
    element_list = []
    hyb_list = []
    out = {}
    atom_hyb = namedtuple("atom_hyb", "symbol hybridization")
    for idx, atom in enumerate( mol.GetAtoms()):
        out[idx] = atom_hyb(atom.GetSymbol(), atom.GetHybridization())
    return out


class MyTestCase_MolecularSystem(unittest.TestCase):
    def test_MolecularSystem_gen(self):
        print("# TEST MolecularSystem")
        lig_base = base / "public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/waterset/hsp90_woodhead/ligand_preparation/A01"

        # find all the rotatable bond from using rdkit
        sdf_path = lig_base / "A01.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = supplier[0]

        query = Lipinski.RotatableBondSmarts
        matches = mol.GetSubstructMatches(query)

        inpcrd, prmtop, system = utils.load_amber_sys(
            lig_base / "01_dry.inpcrd",
            lig_base / "01_dry.prmtop",
        )
        mol_system = hybrid_topology.MolecularSystem().gen_from_openmm_system(system, prmtop.topology)

        print("## Find terms according to a atom")
        terms = mol_system.terms_for_atom(6)
        self.assertEqual(len(terms["bonds"]),              3)
        self.assertEqual(len(terms["angles"]),             9)
        self.assertEqual(len(terms["proper_dihedrals"]),  23)
        self.assertEqual(len(terms["improper_dihedrals"]), 4)

        print("## Find dihedral according to the rotatable bond")
        mol_system.set_rotatable_bonds(((0, 2),))
        dihe = mol_system.rest2_scalable_dihedrals()
        self.assertEqual(len(dihe.proper_rest2), 9)
        self.assertEqual(len(dihe.improper), 15)
        self.assertEqual(len(dihe.proper_not_rest2), 120)

        print("## Assign hybridization from sdf")
        mol_system.set_hybridization_for_residues_from_sdf(0, lig_base / "A01.sdf")
        self.assertListEqual([mol_system.atoms[i].hybridization for i in [2, 3, 4]], ["SP3", "SP3", "SP2"])


class MyTestCase_MolecularSystem_REST2(unittest.TestCase):
    def test_Rest2TopologyFactory(self):
        print("# TEST Rest2TopologyFactory")
        from rdkit import Chem
        from rdkit.Chem import Lipinski

        sdf_path = base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/A01.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = supplier[0]
        rot_bonds = mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)
        hot_atoms = list(range(mol.GetNumAtoms())) # set all atoms to hot

        inpcrd, prmtop, system = utils.load_amber_sys(
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.inpcrd",
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.prmtop",
        )

        factory = hybrid_topology.Rest2TopologyFactory(system, prmtop.topology, hot_atoms, rot_bonds)

        # Particle and constraint counts match
        self.assertEqual(factory.system.getNumParticles(), system.getNumParticles())
        self.assertEqual(factory.system.getNumConstraints(), system.getNumConstraints())

        print("## Can we get identical energy and forces?")
        energy1, force1 = calc_energy_force(system, prmtop.topology, inpcrd.positions)
        energy2, force2 = calc_energy_force(factory.system, factory.topology, inpcrd.positions)
        self.assertAlmostEqual(energy1, energy2)
        match_force(force1, force2)

        print("## NonbondedForce should be scaled by k_rest2")
        sys1 = separate_force(system, ["NonbondedForce"])
        sys1.getForces()[0].setUseDispersionCorrection(False)
        energy1, force1 = calc_energy_force(sys1, prmtop.topology, inpcrd.positions)

        sys2 = separate_force(factory.system, ["NonbondedForce"])
        sys2.getForces()[0].setUseDispersionCorrection(False)
        energy2, force2 = calc_energy_force(sys2, factory.topology, inpcrd.positions,
                                            global_parameters={"k_rest2": 1.0, "k_rest2_sqrt":np.sqrt(1.0)})
        self.assertAlmostEqual(energy1, energy2)
        match_force(force1, force2)

        print("## k_rest2=0.5, energy and force should be scaled by k_rest2")
        energy3, force3 = calc_energy_force(sys2, factory.topology, inpcrd.positions,
                                            global_parameters={"k_rest2": 0.5, "k_rest2_sqrt": np.sqrt(0.5)})

        self.assertAlmostEqual(energy1, energy3*2)
        all_close, _, error_msg = match_force(force1, force3 * 2)
        self.assertTrue(all_close, msg=f"NonbondedForce force mismatch:\n{error_msg}")

        print("## HarmonicBondForce: identical energy and forces (no REST2 scaling)")
        sys1_bond = separate_force(system, ["HarmonicBondForce"])
        sys2_bond = separate_force(factory.system, ["HarmonicBondForce"])
        energy1_bond, force1_bond = calc_energy_force(sys1_bond, prmtop.topology, inpcrd.positions)
        energy2_bond, force2_bond = calc_energy_force(sys2_bond, factory.topology, inpcrd.positions)
        self.assertAlmostEqual(energy1_bond, energy2_bond)
        all_close, _, error_msg = match_force(force1_bond, force2_bond)
        self.assertTrue(all_close, msg=f"HarmonicBondForce force mismatch:\n{error_msg}")

        print("## HarmonicAngleForce: identical energy and forces (no REST2 scaling)")
        sys1_angle = separate_force(system, ["HarmonicAngleForce"])
        sys2_angle = separate_force(factory.system, ["HarmonicAngleForce"])
        energy1_angle, force1_angle = calc_energy_force(sys1_angle, prmtop.topology, inpcrd.positions)
        energy2_angle, force2_angle = calc_energy_force(sys2_angle, factory.topology, inpcrd.positions)
        self.assertAlmostEqual(energy1_angle, energy2_angle)
        all_close, _, error_msg = match_force(force1_angle, force2_angle)
        self.assertTrue(all_close, msg=f"HarmonicAngleForce force mismatch:\n{error_msg}")

        print("## Torsion scaling")
        # Combined torsion energy at k_rest2_sqrt=1 should match original PeriodicTorsionForce
        sys1_dihe = separate_force(system, ["PeriodicTorsionForce"])
        sys2_dihe = separate_force(factory.system, ["PeriodicTorsionForce", "CustomTorsionForce"])
        energy1_dihe, force1_dihe = calc_energy_force(sys1_dihe, prmtop.topology, inpcrd.positions)
        energy2_dihe, force2_dihe = calc_energy_force(sys2_dihe, factory.topology, inpcrd.positions)
        self.assertAlmostEqual(energy1_dihe, energy2_dihe)
        all_close, _, error_msg = match_force(force1_dihe, force2_dihe)
        self.assertTrue(all_close, msg=f"Torsion force mismatch at k_rest2_sqrt=1:\n{error_msg}")

        print("## Check CustomTorsionForce")
        sys2_custom = separate_force(factory.system, ["CustomTorsionForce"])
        torsion_num = sys2_custom.getForces()[0].getNumTorsions()
        self.assertEqual(torsion_num, 32, msg=f"There should be 9+9+2+4+8 Torsions in CustomTorsionForce, {torsion_num} was found")
        energy_ct_k1, force_ct_k1 = calc_energy_force(
            sys2_custom, factory.topology, inpcrd.positions)
        energy_ct_k05, force_ct_k05 = calc_energy_force(
            sys2_custom, factory.topology, inpcrd.positions,
            global_parameters={"k_rest2_sqrt": np.sqrt(0.5)})
        self.assertAlmostEqual(
            energy_ct_k1,
            energy_ct_k05 * 2)
        all_close, _, error_msg = match_force(force_ct_k1, force_ct_k05 * 2)
        self.assertTrue(all_close, msg=f"CustomTorsionForce scaling mismatch at k_rest2=0.5:\n{error_msg}")

    def test_Rest2TopologyFactory_OPC_vsite(self):
        print("# TEST Rest2TopologyFactory with OPC and virtual site")
        from rdkit import Chem
        from rdkit.Chem import Lipinski

        ligand_path = base / "public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/macrocycles/2B8V_lig24and25_alpha05/ligand_preparation"
        sdf_path = ligand_path / "A01/A01.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = supplier[0]
        rot_bonds = mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)
        hot_atoms = list(range(mol.GetNumAtoms())) # set all atoms to hot

        inpcrd, prmtop, system = utils.load_amber_sys(
            ligand_path / "A01/02_solv.inpcrd",
            ligand_path / "A01/02_solv.prmtop",
        )

        factory = hybrid_topology.Rest2TopologyFactory(system, prmtop.topology, hot_atoms, rot_bonds)
        print("## Can we get identical energy and forces?")
        energy1, force1 = calc_energy_force(system, prmtop.topology, inpcrd.positions)
        energy2, force2 = calc_energy_force(factory.system, factory.topology, inpcrd.positions)
        self.assertAlmostEqual(energy1, energy2)
        match_force(force1, force2)

class MyTestCase_HybridIndexMapping(unittest.TestCase):
    def test_loading(self):
        # find all the rotatable bond from using rdkit
        ligand_path = base / "public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/macrocycles/2B8V_lig24and25_alpha05/"
        inpcrdA, prmtopA, systemA = utils.load_amber_sys(
            ligand_path / "ligand_preparation/A01/02_solv.inpcrd",
            ligand_path / "ligand_preparation/A01/02_solv.prmtop",
        )
        topA = prmtopA.topology
        inpcrdB, prmtopB, systemB = utils.load_amber_sys(
            ligand_path / "ligand_preparation/A02/02_solv.inpcrd",
            ligand_path / "ligand_preparation/A02/02_solv.prmtop",
        )
        topB = prmtopB.topology
        with open(ligand_path / "edge_0_1/mapping.json") as f:
            mapping = json.load(f)

        index_map = hybrid_topology.HybridIndexMapping(topA, topB, {0:mapping})
        atA_index_list = [at.index for at in topA.atoms()]
        atB_index_list = [at.index for at in topB.atoms()]
        self.assertListEqual(atA_index_list, sorted(index_map.map_A_to_hybrid.keys()))
        self.assertListEqual(atB_index_list, sorted(index_map.map_B_to_hybrid.keys()))
        self.assertEqual(len(list(topA.bonds())) + 2, len(list(index_map.hybrid_top.bonds())))
        self.assertEqual(len(list(topB.bonds())) + 1, len(list(index_map.hybrid_top.bonds())))
        self.assertListEqual(index_map.broken_bonds_A, [(11, 12)])
        self.assertListEqual(index_map.broken_bonds_B, [])

if __name__ == '__main__':
    unittest.main()
