import numpy as np
import simtk.unit as unit
import logging

def conformational_sampling(systems, n_conf, steps_integrator):
    """
    Function that performs conformational sampling on a given ParaMol system and calculated the ab initio properties (energies and forces).

    Notes
    -----
    As currently implemented, previous reference data is erased in the ParaMol system.

    Parameters
    ----------
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List containing instances of ParaMol systems.
    n_conf : int
        How many structure to sample in in total.
    steps_integrator : int
        Number of steps the integrator performs each time it is called.

    Returns
    -------
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List containing instances of ParaMol systems (updated).
    """
    for system in systems:
        system.ref_coordinates = []
        system.ref_energies = []
        system.ref_forces = []
        system.n_structures = 0

        logging.info("Performing conformational sampling of {} conformations of system {}.".format(n_conf, system.name))

        for i in range(n_conf):
            system.engine.context.setVelocitiesToTemperature(system.engine.integrator.getTemperature())

            # Perform classical MD
            system.engine.integrator.step(steps_integrator)

            # Get positions and compute QM energy and forces
            coord = system.engine.context.getState(getPositions=True).getPositions()
            energy, forces = system.qm_engine.qm_engine.run_calculation(coords=coord.in_units_of(unit.angstrom)._value, label=0)

            # Append energies, forces and conformations
            system.ref_energies.append(energy)
            system.ref_forces.append(np.asarray(forces))
            system.ref_coordinates.append(np.asarray(coord._value))
            system.n_structures += 1

        system.ref_forces = np.asarray(system.ref_forces)
        system.ref_energies = np.asarray(system.ref_energies)
        system.ref_coordinates = np.asarray(system.ref_coordinates)

    logging.info("Conformational sampling of all systems performed successfully.")

    return systems

