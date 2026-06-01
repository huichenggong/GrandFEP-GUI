from openmm import unit, app, openmm
from grandfep import hybrid_topology

# Normal REST2
class Rest2TopologyFactory:
    """
    This class generate a topology for REST2 simulation, set 2 global parameters k_rest2 and k_rest2_sqrt to control
    the scaling. The caller is responsible for keeping k_rest2 = k_rest2_sqrt^2.

    Scaling convention (n_hot = number of hot atoms in a term):
      n_hot = 2 → scale by k_rest2      (= k_rest2_sqrt^2)
      n_hot = 1 → scale by k_rest2_sqrt (= sqrt(k_rest2))
      n_hot = 0 → unscaled
    """
    def __init__(self, system, topology, nb_hot_atoms, rotatable_bonds):
        self.molecule_system = hybrid_topology.MolecularSystem().gen_from_openmm_system(system, topology)
        self.hot_set = frozenset(nb_hot_atoms)
        self.topology = topology

        for idx in self.hot_set:
            self.molecule_system.atoms[idx].is_rest2 = True

        self.molecule_system.rotatable_bonds = {
            (min(a, b), max(a, b)) for a, b in rotatable_bonds
        }

        self._orig_nb_force = None
        self._cm_freq = None
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                self._orig_nb_force = force
            elif isinstance(force, openmm.CMMotionRemover):
                self._cm_freq = force.getFrequency()
        # self._orig_system reserved for future virtual-site copying (skipped for now)

        self.system = openmm.System()
        self._prepare_system()
        self._prepare_bond()
        self._prepare_angle()
        self._prepare_dihe()
        self._prepare_exception()

    def _prepare_system(self):
        """
        prepare basic property of system, including: atom, mass, constraint
        """
        ms = self.molecule_system
        for idx in sorted(ms.atoms):
            self.system.addParticle(ms.atoms[idx].mass)

        for c in ms.constraints_list:
            a1, a2 = c.atoms
            self.system.addConstraint(a1, a2, c.potential.length0)

        if ms.box_vectors is not None:
            self.system.setDefaultPeriodicBoxVectors(*ms.box_vectors)

        if self._cm_freq is not None:
            self.system.addForce(openmm.CMMotionRemover(self._cm_freq))

    def _prepare_topology(self):
        """
        prepare topology
        """
        pass

    def _prepare_bond(self):
        """
        HarmonicBondForce — no REST2 scaling for bonds.
        """
        hbf = openmm.HarmonicBondForce()
        for t in self.molecule_system.bonds:
            a1, a2 = t.atoms
            hbf.addBond(a1, a2, t.potential.length0, t.potential.k)
        self.system.addForce(hbf)

    def _prepare_angle(self):
        """
        No REST2 scaling for angle
        """
        haf = openmm.HarmonicAngleForce()
        for t in self.molecule_system.angles:
            a1, a2, a3 = t.atoms
            haf.addAngle(a1, a2, a3, t.potential.theta0, t.potential.k)
        self.system.addForce(haf)

    def _prepare_dihe(self):
        """
        Scale rotatable bond with hot atoms, 1 hot atom in rotatable scale `k_rest2_sqrt`, 2 hot atoms scale `k_rest2`
        """
        ms = self.molecule_system
        part = ms.rest2_scalable_dihedrals()

        # PeriodicTorsionForce: all unscaled torsions (cold + non-rotatable + zero-hot rotatable)
        # CustomTorsionForce:   hot rotatable torsions, fully scaled by k_rest2_sqrt^n_hot
        ptf = openmm.PeriodicTorsionForce()
        ctf = openmm.CustomTorsionForce(
            "k_rest2_sqrt^n_hot * k * (1 + cos(n * theta - phase))"
        )
        ctf.addGlobalParameter("k_rest2_sqrt", 1.0)
        ctf.addPerTorsionParameter("n_hot")
        ctf.addPerTorsionParameter("k")
        ctf.addPerTorsionParameter("n")
        ctf.addPerTorsionParameter("phase")

        for t in part.proper_not_rest2:
            p = t.potential.parameters
            ptf.addTorsion(*t.atoms, p["periodicity"], p["phase"], p["k"])

        for t in part.improper:
            p = t.potential.parameters
            ptf.addTorsion(*t.atoms, p["periodicity"], p["phase"], p["k"])

        for t in part.proper_rest2:
            p = t.potential.parameters
            n_hot = len({t.atoms[1], t.atoms[2]} & self.hot_set)
            if n_hot >= 1:
                ctf.addTorsion(*t.atoms, [float(n_hot), p["k"], float(p["periodicity"]), p["phase"]])
            else:
                ptf.addTorsion(*t.atoms, p["periodicity"], p["phase"], p["k"])

        self.system.addForce(ptf)
        self.system.addForce(ctf)

    def _prepare_exception(self):
        """
        Build a single REST2-ready NonbondedForce using addParticleParameterOffset
        and addExceptionParameterOffset.

        Hot-atom charges and epsilons are stored as base=0 and recovered via
        parameter offsets, so the NonbondedForce natively scales all interactions
        — including PME reciprocal space — when k_rest2_sqrt / k_rest2 change.

        Scaling applied to each offset parameter:
          charge:    multiply hot-atom charge    by k_rest2_sqrt  (linear offset)
          epsilon:   multiply hot-atom epsilon   by k_rest2       (linear offset)
          Mixed pairs follow from the Lorentz-Berthelot combining rules automatically:
            hot-hot charge:    k_rest2_sqrt * q_i * k_rest2_sqrt * q_j = k_rest2 * q_i*q_j
            hot-cold charge:   k_rest2_sqrt * q_i * q_j
            hot-hot epsilon_ij:  sqrt(k_rest2*eps_i * k_rest2*eps_j) = k_rest2 * sqrt(eps_i*eps_j)
            hot-cold epsilon_ij: sqrt(k_rest2*eps_i * eps_j)         = k_rest2_sqrt * sqrt(eps_i*eps_j)

        1-4 exceptions involving hot atoms are handled with addExceptionParameterOffset:
          n_hot=1: param = k_rest2_sqrt
          n_hot=2: param = k_rest2
        """
        if self._orig_nb_force is None:
            return

        ms = self.molecule_system
        hot_set = self.hot_set
        orig = self._orig_nb_force

        nbf = openmm.NonbondedForce()
        # Parameters must be declared on the NonbondedForce itself before addParticleParameterOffset
        nbf.addGlobalParameter("k_rest2_sqrt", 1.0)
        nbf.addGlobalParameter("k_rest2", 1.0)
        nbf.setNonbondedMethod(orig.getNonbondedMethod())
        nbf.setCutoffDistance(orig.getCutoffDistance())
        nbf.setEwaldErrorTolerance(orig.getEwaldErrorTolerance())
        nbf.setUseSwitchingFunction(orig.getUseSwitchingFunction())
        if orig.getUseSwitchingFunction():
            nbf.setSwitchingDistance(orig.getSwitchingDistance())
        nbf.setUseDispersionCorrection(orig.getUseDispersionCorrection())
        nbf.setExceptionsUsePeriodicBoundaryConditions(
            orig.getExceptionsUsePeriodicBoundaryConditions()
        )

        # Particles: cold atoms use original params; hot atoms use base=0 + offset
        for idx in sorted(ms.atoms):
            atom = ms.atoms[idx]
            if idx in hot_set:
                nbf.addParticle(0.0, atom.sigma, 0.0)
            else:
                nbf.addParticle(atom.charge, atom.sigma, atom.epsilon)

        for idx in hot_set:
            atom = ms.atoms[idx]
            nbf.addParticleParameterOffset("k_rest2_sqrt", idx, atom.charge, 0.0, 0.0)
            nbf.addParticleParameterOffset("k_rest2",      idx, 0.0, 0.0, atom.epsilon)

        # Exceptions: exclusions and cold 1-4 are copied as-is; hot 1-4 use offsets
        for term in ms.nonbonded_exceptions:
            a1, a2 = term.atoms
            p = term.potential
            if p.is_exclusion or not (a1 in hot_set or a2 in hot_set):
                nbf.addException(a1, a2, p.chargeProd, p.sigma, p.epsilon)
            else:
                exc_idx = nbf.addException(a1, a2, 0.0, p.sigma, 0.0)
                n_hot = (1 if a1 in hot_set else 0) + (1 if a2 in hot_set else 0)
                param_name = "k_rest2" if n_hot == 2 else "k_rest2_sqrt"
                nbf.addExceptionParameterOffset(param_name, exc_idx, p.chargeProd, 0.0, p.epsilon)

        self.system.addForce(nbf)


# Hybrid RBFE REST2

# ABFE REST2
