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

    for f in system.getForces():
        if f.getName() in force_name:
            sys_new.addForce(copy.deepcopy(f))
    return sys_new
