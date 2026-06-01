from openmm import unit, app, openmm
from grandfep import hybrid_topology

# Normal REST2
class Rest2TopologyFactory:
    """
    This class generate a topology for REST2 simulation, set 2 global parameters k_rest2 and k_rest2_sqrt to control
    the scaling
    """
    def __init__(self, system, topology, nb_hot_atoms, rotatable_bonds ):
        self.molecule_system = hybrid_topology.MolecularSystem().gen_from_openmm_system(system, topology)
        pass

    def _prepare_system(self):
        """
        prepare basic property of system, including: atom, mass, constraint
        """
        pass

    def _prepare_topology(self):
        """
        prepare topology
        """
        pass

    def _prepare_bond(self):
        """
        """
        pass

    def _prepare_angle(self):
        """
        No REST2 scaling for angle
        """
        pass

    def _prepare_dihe(self):
        """
        Scale rotatable bond with hot atoms, 1 hot atom in rotatable scale `k_rest2_sqrt`, 2 hot atoms scale `k_rest2`
        """
        pass

    def _prepare_exception(self):
        """
        """
        pass



# Hybrid RBFE REST2

# ABFE REST2
