# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Objective_function.Tasks.task.Task` class, which is the Task base class.
"""

import logging
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
    def create_parameter_space(parameter_space_settings, restart_settings, systems, preconditioning=True, ):
        """
        Method that created the ParameterSpace object instance.

        Parameters
        ----------
        parameter_space_settings : dict
            Parameter Space ParaMol settings dictionary.
        restart_settings : dict
            Restart ParaMol settings dictionary.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        preconditioning : bool
            Flag that signal whether or not the preconditioning of the parameters is done when this method is run.

        Returns
        -------
        parameter_space: :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
        """
        for system in systems:
            assert type(system) is ParaMolSystem, "ParaMol System provided is invalid."

        parameter_space = ParameterSpace(systems, **parameter_space_settings)
        parameter_space.get_optimizable_parameters()

        if restart_settings["restart_file"] is None:
            if preconditioning:
                parameter_space.calculate_scaling_constants()
                parameter_space.jacobi_preconditioning()
            
            parameter_space.initial_optimizable_parameters_values = copy.deepcopy(parameter_space.optimizable_parameters_values)
            parameter_space.initial_optimizable_parameters_values_scaled = copy.deepcopy(parameter_space.optimizable_parameters_values_scaled)
        else:
            # Read restart file
            Task.read_restart_file(parameter_space, restart_settings["restart_file"])

        return parameter_space

    @staticmethod
    def create_properties(properties_settings, systems, parameter_space):
        """
        Method that creates the Property instances

        Parameters
        ----------
        properties_settings : dict
            Properties settings dictionary.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instance of the parameter space.

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
            parameter_space.calculate_prior_widths()
            regularization = Regularization(initial_parameters_values=parameter_space.initial_optimizable_parameters_values_scaled,
                                            prior_widths=parameter_space.prior_widths,
                                            **properties_settings["regularization"])

            regularization.set_initial_parameters_values(parameter_space.initial_optimizable_parameters_values_scaled)
            properties.append(regularization)

        return properties

    @staticmethod
    def create_objective_function(objective_function_settings, parameter_space, properties):
        assert type(parameter_space) is ParameterSpace, "Parameter space provided is invalid."

        objective_function = ObjectiveFunction(parameter_space=parameter_space,
                                               properties=properties,
                                               **objective_function_settings)

        return objective_function

    @staticmethod
    def create_optimizer(method, optimizer_settings):
        # Create optimizer
        optimizer = Optimizer(method=method,
                              settings=optimizer_settings)

        return optimizer
    
    @staticmethod
    def write_restart_file(parameter_space, restart_file_name=None):
        import netCDF4 as nc

        if restart_file_name is None:
            restart_file_name = 'restart_paramol.nc'

        logging.info("Writing restart file {}.".format(restart_file_name))

        # Open a new netCDF file for writing.
        ncfile = nc.Dataset(restart_file_name, 'w')

        #
        n_parameters = len(parameter_space.optimizable_parameters)

        # Create parameters dimension
        ncfile.createDimension('n_parameters', n_parameters)

        data_optimizable_parameters_values = ncfile.createVariable('optimizable_parameters_values', np.dtype('float64').char,
                                                            ('n_parameters'))
        data_optimizable_parameters_values.units = "standard_paramol_units"
        if parameter_space.optimizable_parameters_values is not None:
            data_optimizable_parameters_values[:] = parameter_space.optimizable_parameters_values
        else:
            logging.info("There are no optimizable parameters to be written.")
            data_optimizable_parameters_values[:] = np.empty(n_parameters)

        data_optimizable_parameters_values_scaled = ncfile.createVariable('optimizable_parameters_values_scaled',
                                                                   np.dtype('float64').char, ('n_parameters'))
        data_optimizable_parameters_values_scaled.units = "adimensional"
        if parameter_space.optimizable_parameters_values_scaled is not None:
            data_optimizable_parameters_values_scaled[:] = parameter_space.optimizable_parameters_values_scaled
        else:
            logging.info("There are no scaled optimizable parameters to be written.")
            data_optimizable_parameters_values_scaled[:] = np.empty(n_parameters)

        data_initial_optimizable_parameters_values = ncfile.createVariable('initial_optimizable_parameters_values',
                                                                   np.dtype('float64').char,
                                                                   ('n_parameters'))
        data_initial_optimizable_parameters_values.units = "standard_paramol_units"
        if parameter_space.initial_optimizable_parameters_values is not None:
            data_initial_optimizable_parameters_values[:] = parameter_space.initial_optimizable_parameters_values
        else:
            logging.info("There are no initial optimizable parameters to be written.")
            data_initial_optimizable_parameters_values[:] = np.empty(n_parameters)

        data_initial_optimizable_parameters_values_scaled = ncfile.createVariable('initial_optimizable_parameters_values_scaled',
                                                                          np.dtype('float64').char, ('n_parameters'))
        data_initial_optimizable_parameters_values_scaled.units = "adimensional"
        if parameter_space.initial_optimizable_parameters_values is not None:
            data_initial_optimizable_parameters_values_scaled[:] = parameter_space.initial_optimizable_parameters_values_scaled
        else:
            logging.info("There are no scaled initial optimizable parameters to be written.")
            data_initial_optimizable_parameters_values_scaled[:] = np.empty(n_parameters)

        data_scaling_constants = ncfile.createVariable('scaling_constants', np.dtype('float64').char, ('n_parameters'))
        data_scaling_constants.units = "standard_paramol_units"
        if parameter_space.scaling_constants is not None:
            data_scaling_constants[:] = parameter_space.scaling_constants
        else:
            logging.info("There are no scaling constants to be written.")
            data_scaling_constants[:] = np.empty(n_parameters)

        data_prior_widths = ncfile.createVariable('prior_widths', np.dtype('float64').char, ('n_parameters'))
        data_prior_widths.units = "standard_paramol_units"
        if parameter_space.prior_widths is not None:
            data_prior_widths[:] = parameter_space.prior_widths
        else:
            logging.info("There are no prior widths to be written.")
            data_prior_widths[:] = np.empty(n_parameters)

        logging.info("SUCCESS! Restart file was written to file {}".format(restart_file_name))

        return ncfile.close()

    @staticmethod
    def read_restart_file(parameter_space, restart_file_name=None):
        import netCDF4 as nc

        if restart_file_name is None:
            restart_file_name = 'restart_paramol.nc'

        logging.info("\nReading restart file {}.".format(restart_file_name))

        # Open a new netCDF file for writing.
        ncfile = nc.Dataset(restart_file_name, 'r')

        if 'optimizable_parameters_values' in ncfile.variables:
            if all(ncfile.variables["optimizable_parameters_values"] == np.empty(ncfile.variables["optimizable_parameters_values"].shape)):
                parameter_space.optimizable_parameters_values = None
            else:
                parameter_space.optimizable_parameters_values = np.asarray(parameter_space.optimizable_parameters_values)
        else:
            logging.info("{} does not contain optimizable parameters data.".format(restart_file_name))

        if 'optimizable_parameters_values_scaled' in ncfile.variables:
            if all(ncfile.variables["optimizable_parameters_values_scaled"] == np.empty(ncfile.variables["optimizable_parameters_values_scaled"].shape)):
                parameter_space.optimizable_parameters_values_scaled = None
            else:
                parameter_space.optimizable_parameters_values_scaled = np.asarray(parameter_space.optimizable_parameters_values_scaled)
        else:
            logging.info("{} does not contain scaled optimizable parameters data.".format(restart_file_name))

        if 'scaling_constants' in ncfile.variables:
            if all(ncfile.variables["scaling_constants"] == np.empty(ncfile.variables["scaling_constants"].shape)):
                parameter_space.scaling_constants = None
            else:
                parameter_space.scaling_constants = np.asarray(parameter_space.scaling_constants)
        else:
            logging.info("{} does not contain scaling constants data.".format(restart_file_name))

        if 'prior_widths' in ncfile.variables:
            if all(ncfile.variables["prior_widths"] == np.empty(ncfile.variables["prior_widths"].shape)):
                parameter_space.prior_widths = None
            else:
                parameter_space.prior_widths = np.asarray(parameter_space.prior_widths)
        else:
            logging.info("{} does not contain scaling constants data.".format(restart_file_name))

        if 'initial_optimizable_parameters_values' in ncfile.variables:
            if all(ncfile.variables["initial_optimizable_parameters_values"] == np.empty(ncfile.variables["initial_optimizable_parameters_values"].shape)):
                parameter_space.initial_optimizable_parameters_values = None
            else:
                parameter_space.initial_optimizable_parameters_values = np.asarray(parameter_space.initial_optimizable_parameters_values)
        else:
            logging.info("{} does not contain initial optimizable parameters data.".format(restart_file_name))

        if 'initial_optimizable_parameters_values_scaled' in ncfile.variables:
            if all(ncfile.variables["initial_optimizable_parameters_values_scaled"] == np.empty(ncfile.variables["initial_optimizable_parameters_values_scaled"].shape)):
                parameter_space.initial_optimizable_parameters_values_scaled = None
            else:
                parameter_space.initial_optimizable_parameters_values_scaled = np.asarray(parameter_space.initial_optimizable_parameters_values_scaled)
        else:
            logging.info("{} does not contain scaled initial optimizable parameters data.".format(restart_file_name))

        logging.info("SUCCESS! Restart file was read from file {}".format(restart_file_name))

        return ncfile.close()
