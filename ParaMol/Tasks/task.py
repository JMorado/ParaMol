# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Objective_function.Tasks.task.Task` class, which is the Task base class.
"""

import logging
import pickle

# ParaMol libraries
from ..System.system import *
from ..Optimizers.optimizer import *
from ..Parameter_space.parameter_space import *
from ..Objective_function.objective_function import *
from ..Objective_function.Properties.regularization import *
from ..Objective_function.Properties.esp_property import *
from ..Objective_function.Properties.force_property import *
from ..Objective_function.Properties.energy_property import *


class Task:
    """
    ParaMol Task base class

    Parameters
    ----------
    task_name : str
        Name of the task.
    """

    def __init__(self, task_name):
        self._task_name = task_name

    def __str__(self):
        return "Instance of task {}.".format(self._task_name)

    def run_task(self, *args, **kwargs):
        """
        Method that is used when a task is not implemented.

        Parameters
        ----------
        args : any
        kwargs : any

        Returns
        -------
        Raises NotImplementedError
        """

        raise NotImplementedError("Task {} is not implemented yet.". format(self._task_name))

    @staticmethod
    def create_parameter_space(settings, systems, interface=None, preconditioning=True, restart=False):
        """
        Method that created the ParameterSpace object instance.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        interface: :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol interface instance.
        preconditioning : bool
            Flag that signal whether or not the preconditioning of the parameters is done when this method is run.
        restart : bool
            Flag that controls whether or not to perform a restart.

        Returns
        -------
        parameter_space: :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
        """
        for system in systems:
            assert type(system) is ParaMolSystem, "ParaMol System provided is invalid."

        parameter_space = ParameterSpace(**settings.parameter_space)

        if restart:
            assert interface is not None

            for system in systems:
                system.force_field.get_optimizable_parameters(symmetry_constrained=True)

            parameter_space.__dict__ = Task.read_restart_pickle(settings.restart, interface, "restart_parameter_space_file")

            if parameter_space.preconditioned:
                parameter_space.update_systems(systems, parameter_space.optimizable_parameters_values_scaled)
            else:
                parameter_space.update_systems(systems, parameter_space.optimizable_parameters_values)
        else:
            parameter_space.get_optimizable_parameters(systems)

            if preconditioning:
                parameter_space.calculate_scaling_constants()
                parameter_space.jacobi_preconditioning()
            
            parameter_space.initial_optimizable_parameters_values = copy.deepcopy(parameter_space.optimizable_parameters_values)
            parameter_space.initial_optimizable_parameters_values_scaled = copy.deepcopy(parameter_space.optimizable_parameters_values_scaled)

        return parameter_space

    @staticmethod
    def create_properties(properties_settings, parameter_space_settings, systems, parameter_space):
        """
        Method that creates the Property instances
        Parameters
        ----------
        properties_settings : dict
            Properties settings dictionary.
        parameter_space_settings : dict
            Properties settings dictionary.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instance of the ParameterSpace.
        Returns
        -------
        properties : list of ParaMol properties
            List containing the instances of the properties that will enter the objective function.
        """
        assert type(parameter_space) is ParameterSpace, "Parameter space provided is invalid."

        properties = []

        # Create physical properties
        if properties_settings["include_energies"]:
            properties.append(EnergyProperty(systems, **properties_settings["energies"]))
        if properties_settings["include_forces"]:
            properties.append(ForceProperty(systems, **properties_settings["forces"]))
        if properties_settings["include_esp"]:
            properties.append(ESPProperty(systems, **properties_settings["esp"]))

        # Calculate variance of physical quantities
        for prop in properties:
            prop.calculate_variance()

        # Create regularization if required
        if properties_settings["include_regularization"]:
            parameter_space.calculate_prior_widths(parameter_space_settings["prior_widths_method"])

            regularization = Regularization(initial_parameters_values=parameter_space.initial_optimizable_parameters_values_scaled,
                                            prior_widths=parameter_space.prior_widths,
                                            **properties_settings["regularization"])

            regularization.set_initial_parameters_values(parameter_space.initial_optimizable_parameters_values_scaled)
            properties.append(regularization)

        return properties

    @staticmethod
    def create_objective_function(objective_function_settings, restart_settings, parameter_space, properties, systems):
        """
        Method that creates the ObjectiveFunction instances

        Parameters
        ----------
        objective_function_settings : dict
            ObjectiveFunction settings dictionary.
        restart_settings : dict
            Restart settings dictionary.
        parameter_space: :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instance of the ParameterSpace.
        properties : list of ParaMol properties
            List containing the instances of the properties that will enter the objective function.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.

        Returns
        -------
        properties : list of ParaMol properties
            List containing the instances of the properties that will enter the objective function.
        """

        assert type(parameter_space) is ParameterSpace, "ParameterSpace provided is invalid."

        objective_function = ObjectiveFunction(restart_settings=restart_settings,
                                               parameter_space=parameter_space,
                                               properties=properties,
                                               systems=systems,
                                               **objective_function_settings)

        return objective_function

    @staticmethod
    def create_optimizer(method, optimizer_settings):
        """
        Method that creates the Optimizer instance.

        Parameters
        ----------
        method: `str`
            Optimizer method.
        optimizer_settings: dict
            Dictionary containing the optimizer settings.

        Returns
        -------
        optimizer
        """
        # Create optimizer
        optimizer = Optimizer(method=method,
                              settings=optimizer_settings)

        return optimizer

    # ------------------------------------------------------------ #
    #                                                              #
    #                       PRIVATE METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    @staticmethod
    def read_restart_pickle(restart_settings, interface, restart_dict_key):
        """
        Method that reads restart pickle.

        Parameters
        ----------
        restart_settings: dict
            Dictionary containing global ParaMol settings.
        interface: :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol system instance.
        restart_dict_key: `str`
            Key that defines the name of the restart file.

        Returns
        -------
        class_dict : dict
            self.__dict__ of the class to be stored into a pickle.
        """
        # Check restart directory exists
        restart_dir = os.path.join(interface.base_dir, restart_settings["restart_dir"])
        interface.check_dir_exists(restart_dir)

        # Check restart file exists
        if restart_dict_key in restart_settings:
            scan_restart_file = os.path.join(restart_dir, restart_settings[restart_dict_key])
            interface.check_file_exists(scan_restart_file)
        else:
            raise KeyError("{} does not exist.".format(restart_dict_key))

        logging.info("Reading restart file from file {}".format(scan_restart_file))

        with open(scan_restart_file, 'rb') as restart_file:
            class_dict = pickle.load(restart_file)

        return class_dict

    @staticmethod
    def write_restart_pickle(restart_settings, interface, restart_dict_key, class_dict):
        """
        Method that writes restart pickle.

        Parameters
        ----------
        restart_settings: dict
            Dictionary containing global ParaMol settings.
        interface: :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol system instance.
        restart_dict_key: `str`
            Key that defines the name of the restart file.
        class_dict : dict
            self.__dict__ of the class to be stored into a pickle.

        Returns
        -------
        class_dict : dict
            self.__dict__ of the class to be stored into a pickle.
        """
        # Create restart if it does not exist
        restart_dir = os.path.join(interface.base_dir, restart_settings["restart_dir"])
        if not os.path.exists(restart_dir):
            os.makedirs(restart_dir)

        # Check restart directory exists
        interface.check_dir_exists(restart_dir)

        # Check restart file exists
        if restart_dict_key in restart_settings:
            scan_restart_file = os.path.join(restart_dir, restart_settings[restart_dict_key])
        else:
            raise KeyError("{} does not exist.".format(restart_dict_key))

        logging.info("Writing restart file to file {}".format(scan_restart_file))

        with open(scan_restart_file, 'wb') as restart_file:
            pickle.dump(class_dict, restart_file, protocol=pickle.HIGHEST_PROTOCOL)

        return class_dict
