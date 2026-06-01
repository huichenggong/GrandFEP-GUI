from openmm import unit, app, openmm
from grandfep import hybrid_topology

# Normal REST2
class Rest2TopologyFactory:
    """
    This class generate a topology for REST2 simulation, set global parameter k_rest2_sqrt to control
    the scaling. k_rest2 = k_rest2_sqrt^2 is implicit via the expression exponent.
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

        # ALL torsions go into PeriodicTorsionForce (unscaled baseline).
        # For hot rotatable-bond torsions, CustomTorsionForce adds the correction
        # (k^n_hot - 1) * energy so that at k=1 the correction is zero.
        ptf = openmm.PeriodicTorsionForce()
        ctf = openmm.CustomTorsionForce(
            "(k_rest2_sqrt^n_hot - 1) * k * (1 + cos(n * theta - phase))"
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
            ptf.addTorsion(*t.atoms, p["periodicity"], p["phase"], p["k"])
            n_hot = len({t.atoms[1], t.atoms[2]} & self.hot_set)
            if n_hot >= 1:
                ctf.addTorsion(*t.atoms, [float(n_hot), p["k"], float(p["periodicity"]), p["phase"]])

        self.system.addForce(ptf)
        self.system.addForce(ctf)

    def _prepare_exception(self):
        """
        Build 4 forces to handle nonbonded interactions with REST2 scaling.

        Uses the additive-correction approach so that at k_rest2_sqrt=1 every
        correction force contributes exactly 0 and total energy matches original:

          A: NonbondedForce         — original, unchanged (handles PME for all atoms)
          B: CustomNonbondedForce HH — (k^2 - 1) × hot-hot direct space
          C: CustomNonbondedForce HC — (k   - 1) × hot-cold direct space
          D: CustomBondForce         — (k^n_hot - 1) × hot-involved 1-4 pairs

        At k=1: B=C=D=0 so total equals original NonbondedForce energy.
        At k≠1: direct-space hot interactions are scaled; PME reciprocal space
                is left unscaled (standard REST2 approximation).
        """
        if self._orig_nb_force is None:
            return

        ms = self.molecule_system
        hot_set = self.hot_set

        # Classify exception terms
        exc_hot_14 = []
        for term in ms.nonbonded_exceptions:
            a1, a2 = term.atoms
            if not term.potential.is_exclusion and (a1 in hot_set or a2 in hot_set):
                exc_hot_14.append(term)

        # Copy settings from original NonbondedForce
        orig = self._orig_nb_force
        orig_method   = orig.getNonbondedMethod()
        cutoff        = orig.getCutoffDistance()
        use_switch    = orig.getUseSwitchingFunction()
        switch_dist   = orig.getSwitchingDistance()

        # Map NonbondedForce method → CustomNonbondedForce method
        NF = openmm.NonbondedForce
        CNF = openmm.CustomNonbondedForce
        _map = {NF.NoCutoff: CNF.NoCutoff, NF.CutoffNonPeriodic: CNF.CutoffNonPeriodic}
        cnb_method = _map.get(orig_method, CNF.CutoffPeriodic)

        # ----------------------------------------------------------------
        # Force A: copy of original NonbondedForce, completely unchanged
        # ----------------------------------------------------------------
        nbf = openmm.NonbondedForce()
        nbf.setNonbondedMethod(orig_method)
        nbf.setCutoffDistance(cutoff)
        nbf.setEwaldErrorTolerance(orig.getEwaldErrorTolerance())
        nbf.setUseSwitchingFunction(use_switch)
        if use_switch:
            nbf.setSwitchingDistance(switch_dist)
        nbf.setUseDispersionCorrection(orig.getUseDispersionCorrection())
        nbf.setExceptionsUsePeriodicBoundaryConditions(
            orig.getExceptionsUsePeriodicBoundaryConditions()
        )
        for idx in sorted(ms.atoms):
            charge, sigma, epsilon = orig.getParticleParameters(idx)
            nbf.addParticle(charge, sigma, epsilon)
        for i in range(orig.getNumExceptions()):
            a1, a2, chargeProd, sigma, epsilon = orig.getExceptionParameters(i)
            nbf.addException(a1, a2, chargeProd, sigma, epsilon)
        self.system.addForce(nbf)

        # ----------------------------------------------------------------
        # Helper: build a CustomNonbondedForce with common setup
        # ----------------------------------------------------------------
        lj_coulomb = (
            "4*sqrt(eps1*eps2)*((0.5*(sig1+sig2)/r)^12 - (0.5*(sig1+sig2)/r)^6)"
            " + 138.935456*q1*q2/r"
        )

        def _make_cnbf(expression):
            f = openmm.CustomNonbondedForce(expression)
            f.addGlobalParameter("k_rest2_sqrt", 1.0)
            f.addPerParticleParameter("q")
            f.addPerParticleParameter("eps")
            f.addPerParticleParameter("sig")
            for idx in sorted(ms.atoms):
                atom = ms.atoms[idx]
                f.addParticle([atom.charge, atom.epsilon, atom.sigma])
            for term in ms.nonbonded_exceptions:
                f.addExclusion(*term.atoms)
            f.setNonbondedMethod(cnb_method)
            if cnb_method != CNF.NoCutoff:
                f.setCutoffDistance(cutoff)
            if use_switch:
                f.setUseSwitchingFunction(True)
                f.setSwitchingDistance(switch_dist)
            return f

        # ----------------------------------------------------------------
        # Force B: CustomNonbondedForce HH — correction for hot-hot pairs
        # ----------------------------------------------------------------
        hh_force = _make_cnbf(f"(k_rest2_sqrt^2 - 1) * ({lj_coulomb})")
        hh_force.addInteractionGroup(set(hot_set), set(hot_set))
        self.system.addForce(hh_force)

        # ----------------------------------------------------------------
        # Force C: CustomNonbondedForce HC — correction for hot-cold pairs
        # ----------------------------------------------------------------
        cold_set = set(ms.atoms) - set(hot_set)
        hc_force = _make_cnbf(f"(k_rest2_sqrt - 1) * ({lj_coulomb})")
        hc_force.addInteractionGroup(set(hot_set), cold_set)
        self.system.addForce(hc_force)

        # ----------------------------------------------------------------
        # Force D: CustomBondForce — correction for hot-involved 1-4 pairs
        # ----------------------------------------------------------------
        cbf = openmm.CustomBondForce(
            "(k_rest2_sqrt^n_hot - 1) * (4*eps*((sig/r)^12-(sig/r)^6) + 138.935456*chargeProd/r)"
        )
        cbf.addGlobalParameter("k_rest2_sqrt", 1.0)
        cbf.addPerBondParameter("n_hot")
        cbf.addPerBondParameter("eps")
        cbf.addPerBondParameter("sig")
        cbf.addPerBondParameter("chargeProd")

        for term in exc_hot_14:
            a1, a2 = term.atoms
            p = term.potential
            n_hot = (1 if a1 in hot_set else 0) + (1 if a2 in hot_set else 0)
            cbf.addBond(a1, a2, [float(n_hot), p.epsilon, p.sigma, p.chargeProd])

        self.system.addForce(cbf)


# Hybrid RBFE REST2

# ABFE REST2
