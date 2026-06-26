import copy

import numpy as np
from openmm import app, unit, openmm


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
    force = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
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

def find_term_in_force(system, force_name: str, idx: int) -> list:
    """
    Find all entries in a named force that involve atom *idx*.

    Parameters
    ----------
    system     : openmm.System
    force_name : name of the force as returned by force.getName()
    idx        : atom index to search for

    Returns
    -------
    List of parameter tuples returned by the corresponding get*Parameters()
    call (atom indices are always the leading elements of each tuple).

    Supported force classes
    -----------------------
    HarmonicBondForce, CustomBondForce
    HarmonicAngleForce, CustomAngleForce
    PeriodicTorsionForce, CustomTorsionForce
    """
    _dispatch = {
        openmm.HarmonicBondForce:    (2, lambda f: [f.getBondParameters(i)    for i in range(f.getNumBonds())]),
        openmm.CustomBondForce:      (2, lambda f: [f.getBondParameters(i)    for i in range(f.getNumBonds())]),
        openmm.HarmonicAngleForce:   (3, lambda f: [f.getAngleParameters(i)   for i in range(f.getNumAngles())]),
        openmm.CustomAngleForce:     (3, lambda f: [f.getAngleParameters(i)   for i in range(f.getNumAngles())]),
        openmm.PeriodicTorsionForce: (4, lambda f: [f.getTorsionParameters(i) for i in range(f.getNumTorsions())]),
        openmm.CustomTorsionForce:   (4, lambda f: [f.getTorsionParameters(i) for i in range(f.getNumTorsions())]),
    }
    results = []
    for force in system.getForces():
        if force.getName() != force_name:
            continue
        handler = _dispatch.get(type(force))
        if handler is None:
            raise ValueError(f"find_term_in_force: unsupported force class {type(force).__name__!r}")
        n_atoms, iter_fn = handler
        for entry in iter_fn(force):
            if idx in entry[:n_atoms]:
                results.append(entry)
    return results


def separate_force(system, force_name: list, ):
    """
    Create a new system and only keep certain force
    """
    sys_new = openmm.System()
    # box
    sys_new.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())

    for particle_idx in range(system.getNumParticles()):
        particle_mass = system.getParticleMass(particle_idx)
        sys_new.addParticle(particle_mass)

    for i in range(system.getNumConstraints()):
        p1, p2, length = system.getConstraintParameters(i)
        sys_new.addConstraint(p1, p2, length)

    for f in system.getForces():
        if f.getName() in force_name:
            sys_new.addForce(copy.deepcopy(f))
    return sys_new
