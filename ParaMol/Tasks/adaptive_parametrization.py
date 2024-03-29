# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Tasks.adaptive_parametrization.AdaptiveParametrization` class, which is a ParaMol task that performs adaptive parametrization.
"""

import numpy as np
import copy
import simtk.unit as unit
import os

# ParaMol library
from .task import *
from .parametrization import *
from ..QM_engines.qm_engine import *
from ..Utils.interface import *


class AdaptiveParametrization(Task):
    """
    ParaMol adaptive parametrization task.

    Attributes
    ----------
    parameters_generations : List of list of float
        Parameters of every generation.
    iteration : int
        Adaptive parametrization iteration number.
    rmsd : float
        Root mean square deviation of the parameters between two successive iterations.
    old_param : list or np.array
        Array containing parameters of the previous iteration.
    new_param : list or np.array
        Array containing parameters of the current iteration.
    sampling_done : List of bool
        List of flags that indicate whether or not, for a given system, sampling was already performed in this iteration.
    """
    def __init__(self):
        self.parameters_generations = None
        self.iteration = None
        self.rmsd = None
        self.old_param = None
        self.new_param = None
        self.sampling_done = None

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_task(self, settings, systems, parameter_space=None, objective_function=None, optimizer=None, interface=None, rmsd_tol=1e-4, max_iter=100, steps_integrator=500, structures_per_iter=1000, wham_reweighing=False, restart=False):
        """
        Method that performs self-consistent parametrization until convergence or maximum number of iterations is achieved.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instances of ParameterSpace.
        objective_function : :obj:`ParaMol.Objective_function.objective_function.ObjectiveFunction`
            Instance of the objective function.
        optimizer : one of the optimizers defined in the subpackage :obj:`ParaMol.Optimizers`
            Instance of the optimizer.
        interface: :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol system instance.
        rmsd_tol : float
            RMSD tolerance used to break the adaptive parametrization loop.
        max_iter : int
            Maximum number of adaptive parametrization iterations (default is 100).
        steps_integrator : int
            Number of steps the integrator performs each time it is called.
        structures_per_iter : int
            How many structure to sample in each adaptive parametrization iteration (default is 1000).
        wham_reweighting : bool
            Flag that signals whether or not to perform WHAM reweighting at the end of every iteration.
        restart : bool
            Flag that controls whether or not to perform a restart.

        Returns
        -------
        systems, parameter_space, objective_function, optimizer
        """
        print("!=================================================================================!")
        print("!                              ADAPTIVE PARAMETRIZATION                           !")
        print("!=================================================================================!")

        # Create QM Engines
        for system in systems:
            if system.interface is None:
                system.interface = ParaMolInterface()

            system.create_qm_engines(settings.qm_engine["qm_engine"], settings.qm_engine[settings.qm_engine["qm_engine"].lower()])

        # Create parametrization
        parametrization = Parametrization()

        # Create IO Interface
        if interface is None:
            interface = ParaMolInterface()
        else:
            assert type(interface) is ParaMolInterface

        # Create ParameterSpace
        if parameter_space is None:
            parameter_space = self.create_parameter_space(settings, systems, interface, restart=restart)
        else:
            assert type(parameter_space) is ParameterSpace

        if restart:
            logging.info("Starting adaptive parametrization from a previous restart.")
            # Read AdaptiveParametrization pickle
            self.__dict__ = self.read_restart_pickle(settings.restart, interface, "restart_adaptive_parametrization_file")

            # Read data into system
            for system in systems:
                system.read_data(os.path.join(settings.restart["restart_dir"], "{}_data_restart.nc".format(system.name)))
                # Perform WHAM re-weighting
                if wham_reweighing:
                    print("Reweighting configurations of system {}.".format(system.name))
                    system.wham_reweighing(self.parameters_generations)
        else:
            # Get copy of current parameters values
            self.old_param = copy.deepcopy(parameter_space.optimizable_parameters_values_scaled)
            # Do not restart calculation
            logging.info("Starting adaptive parametrization without a restart.")
            self.parameters_generations = [parameter_space.optimizable_parameters_values]
            # Global SC loop parameters
            self.iteration = 0
            self.rmsd = 1e18
            self.sampling_done = [False for _ in systems]

        # ================================================================================= #
        #                            START ADAPTIVE PARAMETRIZATION                         #
        # ================================================================================= #

        while self.iteration < max_iter or self.rmsd < rmsd_tol:
            # Run the algorithm while:
            # the number of iterations is less than number of maximum iterations;
            # until the RMSD has not converged to a user-defined threshold.

            print("Iteration no. {} of the adaptive parametrization loop.".format(self.iteration))

            # Iterate over all ParaMol systems and QM Wrappers
            for i in range(len(systems)):
                if not self.sampling_done[i]:
                    system = systems[i]
                    system.convert_system_ref_arrays_to_list()

                    # Perform sampling for this system
                    for j in range(structures_per_iter):
                        if j % 50 == 0:
                            print("Generating new configuration {}.".format(j + 1))

                        # Randomize velocities
                        system.engine.context.setVelocitiesToTemperature(system.engine.integrator.getTemperature())

                        # Perform classical MD
                        system.engine.integrator.step(steps_integrator)

                        # Get positions and compute QM energy and forces
                        coord = system.engine.context.getState(getPositions=True).getPositions()
                        energy, forces = system.qm_engine.qm_engine.run_calculation(coords=coord.in_units_of(unit.angstrom)._value, label=0, calculate_force=settings.properties["include_forces"])

                        # Append energies, forces and conformations
                        system.ref_energies.append(energy)
                        system.ref_forces.append(forces)
                        system.ref_coordinates.append(coord._value)
                        system.n_structures += 1

                    # Perform WHAM re-weighting
                    if wham_reweighing:
                        print("Reweighting configurations of system {}.".format(system.name))
                        system.wham_reweighing(self.parameters_generations)

                    print("Generated new {} MM structures for system {}.".format(structures_per_iter, system.name))

                    # Convert lists back to numpy arrays (format used in parametrization)
                    system.ref_energies = np.asarray(system.ref_energies)
                    system.ref_forces = np.asarray(system.ref_forces)
                    system.ref_coordinates = np.asarray(system.ref_coordinates)

                    # Sampling done for this system, write reference data
                    self.sampling_done[i] = True

                    # Write restarts
                    self.write_restart_pickle(settings.restart, interface, "restart_adaptive_parametrization_file", self.__dict__)
                    system.write_data(os.path.join(settings.restart["restart_dir"], "{}_data_restart.nc".format(system.name)))

            # Perform parametrization
            systems, parameter_space, objective_function, optimizer = parametrization.run_task(settings=settings,
                                                                                               systems=systems,
                                                                                               parameter_space=parameter_space,
                                                                                               objective_function=objective_function,
                                                                                               optimizer=optimizer,
                                                                                               adaptive_parametrization=True,
                                                                                               restart=restart)

            # Save parameters of every system after each iteration
            for system in systems:
                xml_file_name = os.path.join(settings.restart["restart_dir"], "{}_restart.xml".format(system.name))
                # xml_file_name = "{}_sc_iter_{}.xml".format(system.name, iteration + 1)
                system.engine.write_system_xml(xml_file_name)
                xml_file_name = "{}_sc_iter_{}.xml".format(system.name, self.iteration + 1)
                system.engine.write_system_xml(xml_file_name)

                # system.write_data("paramol_data_sc_iter_{}.nc".format(iteration+1))

            # Compute RMSD and check for convergence
            self.new_param = copy.deepcopy(parameter_space.optimizable_parameters_values_scaled)
            self.rmsd = self._get_parameters_rmsd(self.old_param, self.new_param)

            print("Parameter RMSD value is {}".format(self.rmsd))

            # Append parameters of these generation so that they are used to re-weight the conformations
            self.parameters_generations.append(parameter_space.optimizable_parameters_values)
            self.old_param = self.new_param
            self.iteration += 1

            # Write parameters generation to pickle file
            self.sampling_done = [False for _ in systems]
            self.write_restart_pickle(settings.restart, interface, "restart_adaptive_parametrization_file", self.__dict__)

            if self.rmsd < rmsd_tol:
                print("Self-consistent parametrization achieved convergence in {} iterations using a tolerance of {}.".format(self.iteration, rmsd_tol))

                return systems, parameter_space, objective_function, optimizer

            if restart:
                restart = False

        print("!=================================================================================!")
        print("!                  ADAPTIVE PARAMETRIZATION PERFORMED SUCCESSFULLY                !")
        print("!=================================================================================!")
        return systems, parameter_space, objective_function, optimizer

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PRIVATE METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    @staticmethod
    def _get_parameters_rmsd(old_params, new_params):
        """
        Method that computes the RMSD between the old and new set of parameters.
        Parameters
        ----------
        old_params: list
            List of the old parameters.
        new_params: list
            List of the new parameters
        Returns
        -------
        rmsd: float
            Value of the RMSD.
        """
        old_params = np.asarray(old_params)
        new_params = np.asarray(new_params)

        rmsd = np.power((new_params - old_params),2)
        rmsd = np.sum(rmsd) / float(len(old_params))
        rmsd = np.sqrt(rmsd)

        return rmsd



