# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Tasks.adaptive_parametrization.AdaptiveParametrization` class, which is a ParaMol task that performs adaptive parametrization.
"""

import numpy as np
import copy
import simtk.unit as unit
import pickle

# ParaMol library
from .task import *
from .parametrization import *
from ..QM_engines.qm_engine import *
from ..Utils.interface import *

class AdaptiveParametrization(Task):
    """
    ParaMol adaptive parametrization task.
    """
    def __init__(self):
        pass

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_task(self, settings, systems, parameter_space=None, objective_function=None, optimizer=None, interface=None, rmsd_tol=1e-4, max_iter=100, steps_integrator=500, structures_per_iter=1000, wham_reweighing=False,  ):
        """
        Method that performs self-consistent parametrization until convergence or maximum number of iterations is achieved.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instance of the parameter space.
        objective_function : :obj:`ParaMol.Objective_function.objective_function.ObjectiveFunction`
            Instance of the objective function.
        optimizer : one of the optimizers defined in the subpackage :obj:`ParaMol.Optimizers`
            Instance of the optimizer.
        rmsd_tol : float
            RMSD tolerance used to break the adaptive parametrization loop.
        max_iter : int
            Maximum number of adaptive parametrization iterations (default is 100).
        steps_integrator : int
            Number of steps the integrator performs each time it is called.
        structures_per_iter : int
            How many structure to sample in each adaptive parametrization iteration (default is 1000).
        wham_reweighing : bool
            Flag that signals whether or not to perform WHAM reweighing at the end of every iteration.

        Returns
        -------
        systems, parameter_space, objective_function, optimizer
        """
        print("!=================================================================================!")
        print("!                              ADAPTIVE PARAMETRIZATION                           !")
        print("!=================================================================================!")
        # Set settings

        # Create QM Engines
        for system in systems:
            if system.interface is None:
                system.interface = ParaMolInterface()

            system.create_qm_engines(settings.qm_engine["qm_engine"], settings.qm_engine[settings.qm_engine["qm_engine"].lower()])

        # Create Parameter Space
        if parameter_space is None:
            parameter_space = self.create_parameter_space(settings.parameter_space, settings.restart, systems, preconditioning=True)
        else:
            assert type(parameter_space) is ParameterSpace

        # Create parametrization
        parametrization = Parametrization()

        # Get copy of current parameters values
        old_param = copy.deepcopy(parameter_space.optimizable_parameters_values_scaled)

        if settings.restart["restart_file"] is None:
            # DO NOT RESTART CALCULATION
            if settings.restart["parameters_generation_pickle"] is not None:
                print("No restart file was provided but parameters generation pickle was provided. "
                      "ParaMol will not read it.")
            parameters_generations = [parameter_space.optimizable_parameters_values]
        else:
            # RESTART CALCULATION
            if settings.restart["parameters_generation_pickle"] is None:
                print("Restarting calculation but not providing parameters generation pickle. "
                      "This may cause imbalance in the weights of each configuration.")
            else:
                with open(settings.restart["parameters_generation_pickle"], 'rb') as pickle_file:
                    parameters_generations = pickle.load(pickle_file)

        # =================================================================================
        #                            START ADAPTIVE PARAMETRIZATION
        # =================================================================================
        # Global SC loop parameters
        rmsd = 1e18
        iteration = 0
        while iteration < max_iter or rmsd < rmsd_tol:
            # Run the algorithm while:
            # the number of iterations is less than number of maximum iterations;
            # until the RMSD has not converged to a user-defined threshold.

            print("Iteration no. {} of the adaptive parametrization loop.".format(iteration))

            # Iterate over all ParaMol systems and QM Wrappers
            for system in systems:
                self._convert_system_ref_arrays_to_list(system)
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
                    energy, forces = system.qm_engine.qm_engine.run_calculation(coords=coord.in_units_of(unit.angstrom)._value, label=0)

                    # Append energies, forces and conformations
                    system.ref_energies.append(energy)
                    system.ref_forces.append(forces)
                    system.ref_coordinates.append(coord._value)
                    system.n_structures += 1

                # Perform WHAM re-weighing
                if wham_reweighing:
                    print("Reweighting configurations of system {}.".format(system.name))
                    system.wham_reweightning(parameters_generations)

                print("Generated new {} MM structures for system {}.".format(structures_per_iter, system.name))

                # Convert lists back to numpy arrays (format used in parametrization)
                system.ref_energies = np.asarray(system.ref_energies)
                system.ref_forces = np.asarray(system.ref_forces)
                system.ref_coordinates = np.asarray(system.ref_coordinates)

            # Perform parametrization
            systems, parameter_space, objective_function, optimizer = parametrization.run_task(settings=settings,
                                                                                               systems=systems,
                                                                                               parameter_space=parameter_space,
                                                                                               objective_function=objective_function,
                                                                                               optimizer=optimizer,
                                                                                               adaptive_parametrization=True)

            # Save parameters of every system after each iteration
            for system in systems:
                xml_file_name = "{}_sc_iter_{}.xml".format(system.name, iteration + 1)
                system.engine.write_system_xml(xml_file_name)
                system.write_data("paramol_data_sc_iter_{}.nc".format(iteration+1))

            # Write ParaMol restart file
            Task.write_restart_file(parameter_space, "{}_restart_sc_iter_{}.nc".format(system.name, iteration+1))

            # Compute RMSD and check for convergence
            new_param = copy.deepcopy(parameter_space.optimizable_parameters_values_scaled)
            rmsd = self._get_parameters_rmsd(old_param, new_param)

            print("Parameter RMSD value is {}".format(rmsd))

            if rmsd < rmsd_tol:
                print("Self-consistent parametrization achieved convergence in {} iterations using a tolerance of {}.".format(iteration+1,rmsd_tol))

                return systems, parameter_space, objective_function, optimizer

            # Append parameters of these generation so that they are used to re-weight the conformations
            parameters_generations.append(parameter_space.optimizable_parameters_values)
            old_param = new_param

            # Write parameters generation to pickle file
            for system in systems:
                with open("{}_param_gen_sc_iter_{}.pickle".format(system.name, iteration+1), 'wb') as pickle_file:
                    pickle.dump(parameters_generations, pickle_file)

            iteration += 1

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

    @staticmethod
    def _convert_system_ref_arrays_to_list(system):
        """
        Method that checks the data type of the forces, energies and coordinates data structures and converts them to lists.

        Parameters
        ----------
        system: :obj:`ParaMol.System.system.ParaMolSystem`
            ParaMol System instance.

        Returns
        -------
        ref_forces, ref_energies, ref_coordinates: list, list, list
            Forces, energies and coordinates as lists.
        """
        if system.ref_forces is None:
            system.ref_forces = []
        elif type(system.ref_forces) == np.ndarray:
            system.ref_forces = system.ref_forces.tolist()
        elif type(system.ref_forces) is not list:
            print("QM Forces array type is unknown!")
            exit(-1)

        if system.ref_energies is None:
            system.ref_energies = []
        elif type(system.ref_energies) is np.ndarray:
            system.ref_energies = system.ref_energies.tolist()
        elif type(system.ref_energies) is not list:
            print("QM energies array type is unknown!")
            exit(-1)

        if system.ref_coordinates is None:
            system.ref_coordinates = []
        else:
            system.ref_coordinates = system.ref_coordinates.tolist()

        return system.ref_forces, system.ref_energies, system.ref_coordinates

