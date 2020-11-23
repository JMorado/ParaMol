# -*- coding: utf-8 -*-

"""
Description
-----------

This module defines the :obj:`ParaMol.Objective_function.objective_function.ObjectiveFunction` class, which is a ParaMol representation of the objective function.
"""

import time
import numpy as np
import multiprocessing as mp
import logging

# ParaMol modules
from .pickable_swig import *
from .cpu_objective_function import *
from .gpu_objective_function import *


# Simtk modules
from simtk.openmm import *


class ObjectiveFunction:
    """
    ParaMol representation of the objective function.

    Parameters
    ----------
    restart_settings : dict
        Dictionary containing restart ParaMol settings.
    parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
        ParaMol representation of ParameterSpace.
    properties : list of :obj:`ParaMol.ObjectiveFunction.Properties`
        List of property object instances.
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List containing instances of ParaMol systems.
    platform_name :
        Name of the OpenMM platform. Only options are 'Reference', 'CPU' and 'OpenCL'.
    parallel : bool, default=`False`
        Flag that signals if the objective function calculation is to be performed in parallel.
    weighting_method : str, default="uniform"
        Method used to weigh the conformations. Available methods are "uniform, "boltzmann" and "non-boltzmann".
    weighting_temperature : unit.simtk.Quantity, default=300.0*unit.kelvin
        Temperature used in the weighing. Only relevant if `weighting_method` is "boltzmann" or "non_boltzmann".
    checkpoint_freq : int
        Frequency at which checkpoint files are saved.

    Attributes
    ----------
    parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
        ParaMol representation of ParameterSpace.
    properties : list of :obj:`ParaMol.ObjectiveFunction.Properties`
        List of property object instances.
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List containing instances of ParaMol systems.
    """
    def __init__(self, restart_settings, parameter_space, properties, systems, platform_name, parallel=False, weighting_method="uniform", weighting_temperature=300*unit.kelvin, checkpoint_freq=1000):
        self.restart_settings = restart_settings
        # OpenMM platform used to compute the objective function
        self.parameter_space = parameter_space
        self.properties = properties
        self.systems = systems
        # Private variables
        self._platform = platform_name
        self._parallel = parallel
        self._checkpoint_freq = checkpoint_freq
        self._weighting_method = weighting_method
        self._weighting_temperature = weighting_temperature
        self._f_count = 0

        if self._parallel:
            # Parallel objective function variables
            self._total_n_batches = None
            self._batch_lims = None
            self._parallel_function = None
            self.init_parallel()

        os.environ['OPENMM_DEFAULT_PLATFORM'] = 'Reference'
        os.environ['OPENMM_NUM_THREADS'] = '1'

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def reset_f_count(self):
        """
        Method that resets the objective function evaluation counter.

        Returns
        -------
        int
            Number of times that the objective function has been evaluated (should be equal to 0).
        """
        self._f_count = 0
        return self._f_count

    def get_f_count(self):
        """
        Method that retrieves the objective function evaluation counter.

        Returns
        -------
        int
            Number of times that the objective function has been evaluated.
        """
        return self._f_count

    def init_parallel(self):
        """
        Method that initializes the variables and calls the functions necessary to perform parallel objective function evaluation.

        Returns
        -------
        int
            Number of parallel calculation batches.
        """

        logging.info("Initializing parallel objective function.")
        logging.info("System / Number of cpus")
        for system in self.systems:
            logging.info("{} {}".format(system.name, system.n_cpus))

        # Create Parallel Objective Function Object and set values of shared variables
        if self._platform  in ["CPU", "Reference"]:
            self._parallel_function = CPUObjectiveFunction()
        elif self._platform in ["CUDA", "OpenCL"]:
            self._parallel_function = GPUObjectiveFunction()

        self._total_n_batches = 0
        self._parallel_function.X = []
        self._parallel_function.n_atoms = []
        self._parallel_function.batch_lims = []
        self._parallel_function.calculate_energies = []
        self._parallel_function.calculate_forces = []
        for system in self.systems:
            # Append shared data for this system
            self._parallel_function.X.append(system.ref_coordinates)
            self._parallel_function.n_atoms.append(system.n_atoms)

            self._parallel_function.calculate_energies.append("ENERGY" in [property.name for property in self.properties])
            self._parallel_function.calculate_forces.append("FORCE" in [property.name for property in self.properties])

            # Determine number of batches of this system and sum it up to the number of total batches
            n_structures = len(system.ref_coordinates)
            self._total_n_batches = self._total_n_batches + system.n_cpus

            # Define batch sizes and limits for this system
            batch_size = int(n_structures / float(system.n_cpus)) + 1

            batch_lims = [[n * batch_size, (n + 1) * batch_size] for n in range(system.n_cpus)]
            self._parallel_function.batch_lims.append(batch_lims)

            # Initiate the workers
            self._parallel_function.init_worker()

        return self._total_n_batches

    def calculate_objective_function(self, parameters_values, fmm, emm, esp):
        """
        Method that performs the necessary steps to calculate the objective function.

        Notes
        -----
        This is a shared function of both parallel and serial version since performance-speaking it does not compensate to parallelize this function.
        The conformations weights for the current force field parameters are calculated inside this method.

        Parameters
        ----------
        parameters_values : list of floats
            Lists containing the optimizable values of the parameters.
        fmm : np.array
            Forces array.
        emm : np.array
            Energies array.
        esp : np.array
            Electrostatic potential array.

        Returns
        -------
        objective_function : float
            Value of the objective function.
        """

        # Compute conformations weigths:
        objective_function = 0.0

        for system in self.systems:
            system.compute_conformations_weights(temperature=self._weighting_temperature, emm=emm, weighting_method=self._weighting_method)

        for property in self.properties:
            if property.name == "ENERGY":
                energy_decomposed = property.calculate_property(emm)
                objective_function += property.weight * np.sum(energy_decomposed)
            elif property.name == "FORCE":
                force_decomposed = property.calculate_property(fmm)
                objective_function += property.weight * np.sum(force_decomposed)
            elif property.name == "ESP":
                esp_decomposed = property.calculate_property(esp)
                objective_function += property.weight * np.sum(esp_decomposed)
            elif property.name == "REGULARIZATION":
                objective_function += property.weight * property.calculate_property(parameters_values)

        """
        # TODO: neat this print
        print("!=================================================================================!")
        print("!                               Properties Analysis                               !")
        print("! {:<13s}{:<13s}{:<13s} {:<13s}{:<13s}{:<13s} !".format("System", "Property", "Units", "(Ref-Obs)", "RMS", "(Ref-Obs)/RMS"))
        print("!---------------------------------------------------------------------------------!")
        for property in self.properties:
            if property.systems is not None:
                for system in property.systems:
                    if property.name == "ENERGY":
                        num, denom, ratio = system.energy_statistics(emm)
                    elif property.name == "FORCE":
                        num, denom, ratio = system.force_statistics(fmm)
                    print("! {:<13s}{:<13s}{:<13s} {:<13.8f}{:<13.8f}{:<13.8f} !".format(system.name,
                                                                                         property.name,
                                                                                         property.units.get_name()[:13],
                                                                                         num, denom, ratio))
        print("!=================================================================================!\n")
        """
        return objective_function

    # ------------------------------------------------------------------------------------------------------- #
    #                                          Objective Function                                             #
    # ------------------------------------------------------------------------------------------------------- #
    def f(self, parameters_values, opt_mode=True):
        """
        Method that calculates the objective function for a given set of parameters.

        Parameters
        ----------
        parameters_values : list of floats
            Lists containing the optimizable values of the parameters.
        opt_mode : bool
            Flag that signal whether this objective function . If it is not, there is more verbose.

        Returns
        -------
        objective_function : float
            Value of the objective function.
        """

        start_time = time.time()

        def run_serial():
            # Calculate MM energies and forces
            fmm = []
            emm = []
            esp = []

            for property in self.properties:
                if property.systems is not None:
                    for system in property.systems:
                        if property.name == "ENERGY":
                            emm.append(system.get_energies_ensemble())
                        elif property.name == "FORCE":
                            fmm.append(system.get_forces_ensemble())
                        elif property.name == "ESP":
                            esp.append(system.get_esp_ensemble())

            return fmm, emm, esp

        def run_parallel():
            """
            Run the parallel callback function. :return (tuple): fmm_data, emm_data
            """

            if self._platform in ["CPU", "Reference"]:
                args = []
                for system_id in range(len(self.systems)):
                    # Create a pickalable context for this system
                    pickalable_context = PickalableContext(self.systems[system_id].engine.system,
                                                           copy.deepcopy(self.systems[system_id].engine.integrator))

                    args = args + [[pickalable_context, system_id, worker_id] for worker_id in range(self.systems[system_id].n_cpus)]

            elif self._platform in ["CUDA", "OpenCL"]:
                args = []
                for system_id in range(len(self.systems)):
                    # Create a pickalable context for this system
                    args = args + [[self.systems[system_id].engine.system,
                                    self.systems[system_id].engine.integrator,
                                    system_id,
                                    worker_id] for worker_id in range(self.systems[system_id].n_cpus)]
            else:
                raise NotImplementedError("Platform {} not implemented for the calculation of the objective function.".format(self._platform))

            # Parallel Region
            with mp.get_context("fork").Pool(processes=self._total_n_batches,
                                             initializer=self._parallel_function.init_worker) as pool:
                data = pool.starmap(self._parallel_function.f_callback_function, args)
                pool.terminate()

            # Collect data from parallel calculation of classical energies and forces
            # Pool.starmap will return the values in order of the arguments passed into them
            emm_data = []
            fmm_data = []
            data = np.asarray(data)
            start_batch = 0
            for system in self.systems:
                end_batch = start_batch + system.n_cpus
                emm_data.append(np.concatenate(data[start_batch:end_batch, 0]))
                fmm_data.append(np.concatenate(data[start_batch:end_batch, 1]))
                start_batch = system.n_cpus

            return fmm_data, emm_data

        # Update the parameters in the force field
        self.parameter_space.update_systems(self.systems, parameters_values)

        if opt_mode:
            self._f_count += 1

        if self._parallel:
            fmm, emm = run_parallel()
            esp = None
        else:
            fmm, emm, esp = run_serial()

        objective_function = self.calculate_objective_function(parameters_values, fmm, emm, esp)

        if np.isnan(objective_function):
            # Set objective function to a very high value so that the parameters are discarded
            objective_function = 1e16

        if self._f_count % self._checkpoint_freq == 1:
            from ParaMol.Tasks.task import Task
            from ParaMol.Utils.interface import ParaMolInterface

            Task("").write_restart_pickle(self.restart_settings, ParaMolInterface(), "restart_parameter_space_file", self.parameter_space.__dict__)

            for system in self.systems:
                system.engine.write_system_xml("{}_checkpoint.xml".format(system.name))

        if not opt_mode:
            print("!=================================================================================!")
            print("!                           Objective Function Analysis                           !")
            print("! {:<25s}{:^17s}{:<1s}{:^17s}{:<2s}{:^17s} !".format("PROPERTY", "Value",  "x",  "Weight", "=", "Contribution"))
            print("!---------------------------------------------------------------------------------!")
            obj_fun_total = 0.0
            for property in self.properties:
                print("! {:<25s}{:^17.8f} {:^17.8f} {:^17.8f}  !".format(property.name,
                                                                         property.value,
                                                                         property.weight,
                                                                         property.value * property.weight))
                obj_fun_total = obj_fun_total + property.value * property.weight

            print("! {:<61s}{:^17.8f}  !".format("TOTAL", obj_fun_total))
            print("!---------------------------------------------------------------------------------!")
            print("! {:<25s}{:<10}{:<25s}{:<8.3f}{:11s} !".format(
                "Function Evaluations:", self._f_count, "Time (s):", time.time()-start_time, " "))
            print("!=================================================================================!\n")

        return objective_function


