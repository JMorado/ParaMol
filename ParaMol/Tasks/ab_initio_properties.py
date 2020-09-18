# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Tasks.ab_initio_properties.AbInitioProperties` class, which is a ParaMol task used to calculated ab initio properties.
"""
import copy
import numpy as np

# ParaMol modules
from ..QM_engines.qm_engine import *
from .task import *
from .forces_parallel import qm_parallel
from ..Utils.interface import *

# ------------------------------------------------------------
#                                                            #
#                   ab initio Properties Task                #
#                                                            #
# ------------------------------------------------------------
class AbInitioProperties(Task):
    """
    ParaMol task that calculates ab initio properties.
    """
    def __init__(self):
        pass

    # ---------------------------------------------------------- #
    #                                                            #
    #                       PUBLIC METHODS                       #
    #                                                            #
    # ---------------------------------------------------------- #
    def run_task(self, settings, systems,  interface=None, write_data=False):
        """
        Method that calculates ab initio energies and forces and optionally also classical energies and forces.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        write_data: bool
            Flag that signal whether or not to write calculated properties to an output file (default is `False`)
        interface : :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol interface.
        Notes
        -----
        Classical data is only stored locally (can be written to a file).

        Returns
        -------
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        """
        print("!=================================================================================!")
        print("!                         AB INITIO FORCES AND ENERGIES                           !")
        print("!=================================================================================!")

        # Create QM Engines
        for system in systems:
            if system.interface is None:
                system.interface = ParaMolInterface()

            system.create_qm_engines(settings.qm_engine["qm_engine"], settings.qm_engine[settings.qm_engine["qm_engine"].lower()])

        # Calculate ab initio properties
        for system in systems:
            assert system.ref_coordinates is not None
            print("\nComputing ab initio forces and energies of system {}.".format(system.name))

            # Initiate the arrays
            system.ref_energies = []
            system.ref_forces = []

            if system.n_cpus > 1:
                self._run_parallel(system, write_data)
            else:
                self._run_serial(system, write_data)

            print("Computed QM forces of {} structures of system {}.".format(system.n_structures, system.name))

        print("!=================================================================================!")
        print("!             AB INITIO FORCES AND ENERGIES CALCULATED SUCCESFULLY!               !")
        print("!=================================================================================!")
        return systems

    # ---------------------------------------------------------- #
    #                                                            #
    #                       PRIVATE METHODS                      #
    #                                                            #
    # ---------------------------------------------------------- #
    def _run_serial(self, system, write_data):
        """
        Method that is a a serial version of the ab initio properties calculator.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of a ParaMol System.
        write_data: bool
            Flag that signal whether or not to write calculated properties to an output file (default is `False`)

        Returns
        -------
        system
        """
        print("Serial version. Number of cpus is {}.".format(system.n_cpus))

        mm_energies = []
        mm_forces = []

        # Run serial QM force computation
        count = 0

        # Run serial QM forces computation
        for coord in system.ref_coordinates:
            coord = np.asarray(coord)
            energy, forces = system.qm_engine.qm_engine.run_calculation(coords=coord * 10, label=0)
            system.ref_energies.append(energy)
            system.ref_forces.append(forces)

            # Compute MM forces and energy
            mm_forces.append(system.engine.get_forces(coord))
            mm_energies.append(system.engine.get_potential_energy(coord))

            if count % 10 == 0:
                print("Calculating QM energy and forces of structure number {}.".format(count))

            count += 1

        system.ref_energies = np.asarray(system.ref_energies)
        system.ref_forces = np.asarray(system.ref_forces)

        if write_data:
            system.write_data()

        return system

    def _run_parallel(self, system, write_data):
        """
        Method that is a parallel version of the ab initio properties calculator.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.`
            Instance of a ParaMol System.
        write_data: bool
            Flag that signal whether or not to write calculated properties to an output file (default is `False`)

        Returns
        -------
        system
        """
        import multiprocessing as mp

        print("Parallel version. Number of cpus is {}.".format(system.n_cpus))

        # Define batch sizes and limits
        batch_size = int(len(system.ref_coordinates) / float(system.n_cpus)) + 1
        batch_lims = [[n * batch_size, (n + 1) * batch_size] for n in range(system.n_cpus)]

        # Define callback function arguments
        args = [[copy.deepcopy(system.ref_coordinates[batch_lims[i][0]:batch_lims[i][1]]),
                 copy.deepcopy(system.qm_engine.qm_engine),
                 i] for i in range(system.n_cpus)]

        # Create batch indices
        with mp.get_context("fork").Pool(processes=system.n_cpus) as pool:
            data = pool.starmap(qm_parallel, args)
            pool.close()

        # Convert data no numpy and update the system
        data = np.asarray(data)
        system.ref_forces = np.concatenate(data[:, 0])
        system.ref_energies = np.concatenate(data[:, 1])

        if write_data:
            system.write_data()

        return system

