from pathlib import Path
import unittest
from typing import List, Tuple
import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski

from openmm import app, unit, openmm
from grandfep import utils, hybrid_topology


base = Path(__file__).resolve().parent.parent

def calc_energy_force(system: openmm.System, topology: app.topology.Topology, positions, platform=openmm.Platform.getPlatform('Reference'), global_parameters=None):
    """
    Calculate energy and force for the system
    :param system: openmm.System
    :param topology: openmm.Topology
    :param positions: openmm.Vec3
    :return: energy, force
    """
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    if global_parameters:
        for key, value in global_parameters.items():
            if key in simulation.context.getParameters():
                simulation.context.setParameter(key, value)
            # else:
            #     print(f"Global Parameter {key} not found in the system.")
    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    force = state.getForces(asNumpy=True)
    return energy, force

def match_force(force1, force2, excluded_list = None):
    """
    Check if the force is the same
    :param force1: state.getForces(asNumpy=True)
    :param force2: state.getForces(asNumpy=True)
    :param excluded_list: list of atom index to be excluded in the comparison
    :return: bool
    """
    if not excluded_list:
        excluded_list = []
    all_close_flag = True
    mis_match_list = []
    n_matched = 0
    for at_index, (f1, f2) in enumerate(zip(force1, force2)):
        if at_index in excluded_list:
            continue
        at_flag = np.allclose(f1, f2)
        if not at_flag:
            all_close_flag = False
            mis_match_list.append([at_index, f1, f2])
        else:
            n_matched += 1
    print(f"{n_matched} atoms matched.")
    error_msg = "".join([f"{at}\n    {f1}\n    {f2}\n" for at, f1, f2 in mis_match_list])
    return all_close_flag, mis_match_list, error_msg

def separate_force(system, force_name: List, ):
    """
    Create a new system and only keep certain force
    """
    sys_new = openmm.System()
    # box
    sys_new.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())

    for particle_idx in range(system.getNumParticles()):
        particle_mass = system.getParticleMass(particle_idx)
        sys_new.addParticle(particle_mass)

    for f in system.getForces():
        if f.getName() in force_name:
            sys_new.addForce(copy.deepcopy(f))
    return sys_new

class MyTestCase_MolecularSystem_REST2(unittest.TestCase):
    def test_MolecularSystem_gen(self):

        # find all the rotatable bond from using rdkit
        sdf_path = base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/A01.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = supplier[0]

        query = Lipinski.RotatableBondSmarts
        matches = mol.GetSubstructMatches(query)

        print(matches)

        inpcrd, prmtop, system = utils.load_amber_sys(
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.inpcrd",
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.prmtop",
        )
        mol_system = hybrid_topology.MolecularSystem().gen_from_openmm_system(system, prmtop.topology)

        terms = mol_system.terms_for_atom(6)
        self.assertEqual(len(terms["bonds"]),              3)
        self.assertEqual(len(terms["angles"]),             9)
        self.assertEqual(len(terms["proper_dihedrals"]),  23)
        self.assertEqual(len(terms["improper_dihedrals"]), 4)

        mol_system = hybrid_topology.MolecularSystem().gen_from_openmm_system(system, prmtop.topology)
        mol_system.set_rotatable_bonds(((0, 2),))
        dihe = mol_system.rest2_scalable_dihedrals()
        self.assertEqual(len(dihe.proper_rest2), 9)
        self.assertEqual(len(dihe.improper), 15)
        self.assertEqual(len(dihe.proper_not_rest2), 120)

    def test_Rest2TopologyFactory(self):
        print("# TEST Rest2TopologyFactory")
        from rdkit import Chem
        from rdkit.Chem import Lipinski
        from grandfep.hybrid_topology.hybrid_factory import Rest2TopologyFactory

        sdf_path = base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/A01.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = supplier[0]
        rot_bonds = mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)
        hot_atoms = list(range(mol.GetNumAtoms())) # set all atoms to hot

        inpcrd, prmtop, system = utils.load_amber_sys(
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.inpcrd",
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.prmtop",
        )

        factory = Rest2TopologyFactory(system, prmtop.topology, hot_atoms, rot_bonds)

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



if __name__ == '__main__':
    unittest.main()
