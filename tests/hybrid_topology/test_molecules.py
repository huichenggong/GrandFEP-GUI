from pathlib import Path
import unittest
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import Lipinski

from openmm import app, unit, openmm
from grandfep import utils, hybrid_topology


base = Path(__file__).resolve().parent.parent

class MyTestCase(unittest.TestCase):
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

    def test_Rest2TopologyFactory(self):
        print("# Can we construct a proper REST2 system which generate exact same force?")
        from rdkit import Chem
        from rdkit.Chem import Lipinski
        from grandfep.hybrid_topology.hybrid_factory import Rest2TopologyFactory

        sdf_path = base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/A01.sdf"
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = supplier[0]
        rot_bonds = mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)
        hot_atoms = list(range(mol.GetNumAtoms()))

        inpcrd, prmtop, system = utils.load_amber_sys(
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.inpcrd",
            base / "schrodinger_sets/water_set/hsp90_woodhead/test/A01/01_dry.prmtop",
        )

        factory = Rest2TopologyFactory(system, prmtop.topology, hot_atoms, rot_bonds)

        # Particle and constraint counts match
        self.assertEqual(factory.system.getNumParticles(), system.getNumParticles())
        self.assertEqual(factory.system.getNumConstraints(), system.getNumConstraints())

        # Energy identity at k_rest2_sqrt = 1.0
        platform = openmm.Platform.getPlatformByName("Reference")
        ctx_orig  = openmm.Context(system,         openmm.VerletIntegrator(0.001), platform)
        ctx_rest2 = openmm.Context(factory.system, openmm.VerletIntegrator(0.001), platform)
        ctx_orig.setPositions(inpcrd.positions)
        ctx_rest2.setPositions(inpcrd.positions)
        E_orig  = ctx_orig.getState(getEnergy=True).getPotentialEnergy()
        E_rest2 = ctx_rest2.getState(getEnergy=True).getPotentialEnergy()
        self.assertAlmostEqual(
            E_orig.value_in_unit(unit.kilojoule_per_mole),
            E_rest2.value_in_unit(unit.kilojoule_per_mole),
            delta=0.1,
        )

        # Setting k_rest2_sqrt != 1 changes energy (keep k_rest2 = k_rest2_sqrt^2)
        ctx_rest2.setParameter("k_rest2_sqrt", 0.5)
        ctx_rest2.setParameter("k_rest2", 0.25)
        E_scaled = ctx_rest2.getState(getEnergy=True).getPotentialEnergy()
        self.assertGreater(
            abs((E_orig - E_scaled).value_in_unit(unit.kilojoule_per_mole)), 1.0
        )



if __name__ == '__main__':
    unittest.main()
