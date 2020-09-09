# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace` class, which is a ParaMol representation of the parameter space used in the optimization.
"""

import numpy as np
import simtk.unit as unit
import logging

from ..MM_engines.resp import *
from ..MM_engines.openmm import *


# ------------------------------------------------------------
#                                                            #
#                       PARAMETER SPACE                      #
#                                                            #
# ------------------------------------------------------------
class ParameterSpace:
    """
    ParaMol representation of the set of mathematical parameters that are used in the optimization.

    Parameters
    ----------
    parameters_magnitudes : dict
        Dictionary containing the mapping between parameter keys and their natural magnitude values.
        This is used to calculate the scaling constants and prior widths if the 'default' option is chosen.
    prior_widths_method : str
        Method used to calculate the scaling constants. Available methods are 'default', 'arithmetic' and 'geometric'.
    scaling_constants_method : str
        Method used to calculate the scaling constants. Available methods are 'default', 'arithmetic' and 'geometric'.

    Attributes
    ----------
    optimizable_parameters_values_scaled : list or np.array
        Array containing the scaled optimizable parameters values.
    optimizable_parameters_values : list or np.array
        Array containing the unscaled optimizable parameters values.
    optimizable_parameters : list or np.array
        Array that contains instances of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` that are optimizable,
    initial_optimizable_parameters_values_scaled : list or np.array
        Array containing the initial scaled optimizable parameters values.
    initial_optimizable_parameters_values : list or np.array
        Array containing the initial unscaled optimizable parameters values.
    prior_widths : list or np.array
        Array containing the prior wdiths for all parameters.
    prior_widths_dict : dict
        Dictionary that maps a param_key to the correspondent prior width.
    scaling_constants : list or np.array
        Array containing the scaling constants used to perform the Jacobi preconditioning for all parameters.
    scaling_constants_dict : dict
        Dictionary that maps a param_key to the correspondent scaling constants used to perform the Jacobi preconditioning.
    """

    def __init__(self, parameters_magnitudes={'charge': 0.5, 'lj_sigma': 0.30, 'lj_eps': 0.20,
                                              'torsion_phase': np.pi, 'torsion_k': 4 * 4.184,
                                              'bond_eq': 0.05, 'bond_k': 100000, 'angle_eq': np.pi / 16.0,
                                              'angle_k': 100.0, 'scee': 1.0, 'scnb': 1.0},
                 prior_widths_method="default", scaling_constants_method="default"):

        self.optimizable_parameters_values_scaled = None
        self.optimizable_parameters_values = None
        self.optimizable_parameters = None

        self.parameters_magnitudes = parameters_magnitudes
        # Prior widths
        self.prior_widths = None
        self.prior_widths_dict = None

        # Scaling constants
        self.scaling_constants = None
        self.scaling_constants_dict = None

        # Initial parameters values
        self.initial_optimizable_parameters_values = None
        self.initial_optimizable_parameters_values_scaled = None

        # Methods
        self.scaling_constants_method = scaling_constants_method
        self.prior_widths_method = prior_widths_method

        # Indicates whether preconditioning was already performed or not
        self.preconditioned = False

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #

    def calculate_scaling_constants(self, method=None):
        """
        Method that calculates the scaling constants and sets the class attribute self.scaling_constants.

        Parameters
        ----------
        method : str
            Available methods are 'default', 'arithmetic' and 'geometric'.

        Returns
        -------
        scaling_constants : np.array, shape=(n_params)
            Numpy array containing the scaling constants for each parameter.
        """
        if method is None:
            method = self.scaling_constants_method

        self.scaling_constants_dict, self.scaling_constants = self.calculate_parameters_magnitudes(method=method)

        # 83 columns
        print("!=================================================================================!")
        print("!                                Scaling Constants                                !")
        print("! {:<30s}{:<30s}{:<19s} !".format("Term type", "Value", " "))
        print("!---------------------------------------------------------------------------------!")
        for term_type in self.scaling_constants_dict:
            print('! {:<30s}{:<30.8f}{:<19s} !'.format(term_type, self.scaling_constants_dict[term_type], " "))
        print("!=================================================================================!")

        return self.scaling_constants

    def calculate_prior_widths(self, method=None):
        """
        Method that calculates the prior widths and sets the class attribute self.prior_widths.

        Parameters
        ----------
        method : str
            Available methods are 'default', 'arithmetic' and 'geometric'.

        Returns
        -------
        scaling_constants : np.array, shape=(n_params)
            Numpy array containing the prior widths for each parameter.
        """
        if method is None:
            method = self.scaling_constants_method

        logging.info("Calculating prior widths using the method {}".format(method))

        self.prior_widths_dict, self.prior_widths = self.calculate_parameters_magnitudes(method=method)

        # 83 columns
        print("!=================================================================================!")
        print("!                                   Prior Widths                                  !")
        print("! {:<30s}{:<30s}{:<19s} !".format("Term type", "Value", " "))
        print("!---------------------------------------------------------------------------------!")
        for term_type in self.prior_widths_dict:
            print('! {:<30s}{:<30.8f}{:<19s} !'.format(term_type, self.prior_widths_dict[term_type], " "))
        print("!=================================================================================!")

        return self.prior_widths_dict, self.prior_widths

    def calculate_parameters_magnitudes(self, method=None):
        """
        Method that calculates the widths of the prior distributions or the scalings constants of the parameters according to different recipes.

        Parameters
        ----------
        method : str
            Recipe used to calculate the prior widths. Options are "default", "geometric" or "default".

        Returns
        -------
        prior_widths : np.array(n_parameters)
            Array containing the prior widths for each parameter.
        """
        assert method is not None, "No method was chosen to calculate the parameters's magnitudes."

        # Get the parameters for every key
        param_keys = {}
        parameters_magnitudes_dict = {}
        parameters_magnitudes = []

        for parameter in self.optimizable_parameters:
            if parameter.param_key in param_keys:
                param_keys[parameter.param_key].append(parameter.value)
            else:
                param_keys[parameter.param_key] = []
                param_keys[parameter.param_key].append(parameter.value)

        if method.lower() == "geometric":
            # Compute the geometric mean
            for param_key in param_keys:
                geometric_mean = 1.0
                n = 0.0
                for value in param_keys[param_key]:
                    if abs(value) > 1e-8:
                        # If value is not zero
                        geometric_mean = geometric_mean * np.abs(value)
                        n = n + 1
                if abs(geometric_mean) > 1e-8 and n > 0:
                    geometric_mean = geometric_mean ** (1.0 / n)
                    parameters_magnitudes_dict[param_key] = geometric_mean
                else:
                    parameters_magnitudes_dict[param_key] = self.parameters_magnitudes[param_key]

        elif method.lower() == "arithmetic":
            # Arithmetic mean
            for param_key in param_keys:
                arithmetic_mean = 0.0
                n = 0.0
                for value in param_keys[param_key]:
                    arithmetic_mean = arithmetic_mean + np.abs(value)
                    n = n + 1

                if abs(arithmetic_mean) > 1e-8 and n > 0:
                    arithmetic_mean = arithmetic_mean / n
                    parameters_magnitudes_dict[param_key] = arithmetic_mean
                else:
                    parameters_magnitudes_dict[param_key] = self.parameters_magnitudes[param_key]

        elif method.lower() == "default":
            for param_key in param_keys:
                parameters_magnitudes_dict[param_key] = self.parameters_magnitudes[param_key]
        else:
            raise NotImplementedError(
                "\t * Mean type {} not available to guess the prior widths.".format(method))

        for parameter in self.optimizable_parameters:
            parameters_magnitudes.append(parameters_magnitudes_dict[parameter.param_key])

        # Convert to numpy array
        prior_widths = np.asarray(parameters_magnitudes)

        return parameters_magnitudes_dict, prior_widths

    def jacobi_preconditioning(self):
        """
        Method that applies Jacobi (diagonal) preconditioning to the parameters that will enter in the optimization.

        Notes
        -----
        This method should be called before any optimization and after the prior widths were calculated.

        Returns
        -------
        optimizable_parameters_values_scaled : np.array(n_parameters)
            Array containing the scaled optimizable parameters values.
        """

        assert self.scaling_constants is not None
        assert self.optimizable_parameters_values is not None

        self.optimizable_parameters_values_scaled = self.optimizable_parameters_values / self.scaling_constants

        self.preconditioned = True

        return self.optimizable_parameters_values_scaled

    def get_optimizable_parameters(self, system, symmetry_constrained=True):
        """
        Method that returns a list with the optimizable parameters to be passed to the optimization.

        Parameters
        ----------
        symmetry_constrained : bool
            Flag that signal if there are any symmetry constraints.
        system: `ParaMol.System.system.ParaMolSystem`
            ParaMol System instance.

        Returns
        -------
        optimizable_parameters, optimizable_parameters_values : np.array(n_parameters), np.array(n_parameters)
            array that contains instances of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` that are optimizable, array containing the  optimizable parameters values.
        """
        # Get parameters for optimization:
        self.optimizable_parameters = []
        self.optimizable_parameters_values = []

        system.force_field.get_optimizable_parameters(symmetry_constrained=symmetry_constrained)
        # Concatenate lists
        self.optimizable_parameters += system.force_field.optimizable_parameters
        self.optimizable_parameters_values += system.force_field.optimizable_parameters_values

        return self.optimizable_parameters, self.optimizable_parameters_values

    def update_system(self, system, parameters_values):
        """
        Method that defines defines the point of contact with the external world.

        Parameters
        ----------
        parameters_values : list of floats
            1D list with the adimensional mathematical parameters used in the optimization.
        system: `ParaMol.System.system.ParaMolSystem`
            ParaMol System instance.

        Notes
        -----
        Firstly, it converts the adimensional mathematical parameters used in the optimization to parameters with physical units.
        Then, it updates the ParaMol ForceField instance associated with every system in system and updates the parameters in the OpenMM system.

        Returns
        -------
        optimizable_parameters_values : np.array(n_parameters)
            Array containing the scaled optimizable parameters values.
        """
        if self.preconditioned:
            self.optimizable_parameters_values_scaled = parameters_values
            self.optimizable_parameters_values = parameters_values * self.scaling_constants
        else:
            self.optimizable_parameters_values = parameters_values

        # Update the parameters in the system's ParaMolForce Field
        system.force_field.update_force_field(self.optimizable_parameters_values)

        # Update the parameters in the OpenMM context
        system.engine.set_bonded_parameters(system.force_field.force_field_optimizable)
        system.engine.set_nonbonded_parameters(system.force_field.force_field_optimizable)

        if type(system.resp_engine) is RESP:
            system.resp_engine.set_charges(system.force_field.force_field)
        elif system.resp_engine is not None:
            raise NotImplementedError("MM RESP Engine {} is not implemented.".format(type(system.resp_engine)))

        return self.optimizable_parameters_values
