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



if __name__ == '__main__':
    unittest.main()
