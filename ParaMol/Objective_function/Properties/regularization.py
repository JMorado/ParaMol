# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Objective_function.Properties.regularization.Regularization` class, which is a ParaMol representation of the regularization property.
"""

import numpy as np

from .property import *


# -----------------------------------------------------------#
#                                                            #
#                       REGULARIZATION                       #
#                                                            #
# -----------------------------------------------------------#
class Regularization(Property):
    """
    ParaMol representation of the regularization property.

    Parameters
    ----------
    initial_parameters_values : list or np.array of floats
        List or np.array containing the initial parameters's values.
    prior_widths : list or np.array of floats
        List or np.array containing the prior width of each parameter.
    method : str
        Type of regularization. Options are 'L1', 'L2' or 'hyperbolic' ('hyperbolic' only for RESP calculations)
    weight : float
        Weight of this property in the objective function.
    scaling_factor : float
        Scaling factor of the regularization value.
    hyperbolic_beta : float
        Hyperbolic beta value. Only used if `method` is `hyperbolic`.

    Attributes
    ----------
    name : str
        'REGULARIZATION'
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List of ParaMol Systems. Currently not used and it is set to None.
    units : str
        'ADIMENSIONAL'
    value : float
        Current value of this property
    weight : float
        Weight of this property in the objective function.
    """
    def __init__(self, initial_parameters_values, prior_widths, method, weight=1.0, scaling_factor=1.0, hyperbolic_beta=0.01):
        self.name = "REGULARIZATION"
        self.systems = None
        self._regularization_type = method
        self._scaling_factor = scaling_factor
        self._hyperbolic_beta = hyperbolic_beta
        self._initial_parameters_values = initial_parameters_values
        self._prior_widths = prior_widths
        self.units = 'ADIMENSIONAL'
        self.value = None
        self.weight = weight

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def set_initial_parameters_values(self, initial_parameters_values):
        """
        Method that sets the initial parameters's values as a private attribute of this instance.

        Parameters
        ----------
        initial_parameters_values : list or np.array of floats
            List or np.array containing the initial parameters's values.

        Returns
        -------
        initial_parameters_values : list of floats
            List containing the prior width of each parameter (private attribute).
        """

        self._initial_parameters_values = initial_parameters_values
        return self._initial_parameters_values

    def set_prior_widths(self, prior_widths):
        """
        Method that sets the prior widths of the variables as a private attribute of this instance.

        Parameters
        ----------
        prior_widths : list or np.array of floats
            List or np.array containing the prior width of each parameter.

        Returns
        -------
        prior_widths: list of floats
            List containing the prior width of each parameter (private attribute).
        """

        self._prior_widths = prior_widths
        return self._prior_widths

    def calculate_property(self, current_parameters, a=None, b=None):
        """
        Method that wraps private regularization methods in order to calculate the regularization term of the objective function.

        Parameters
        ----------
        current_parameters : list of floats
            Lists containing the optimizable values of the parameters.
        a : float, default=`None`
            a parameter (scaling factor). If not `None`, instance attribute `self._scaling_factor` is ignored.
        b : float, default=`None`
            Hyperbolic beta parameter. If not `None`, instance attribute `self._hyperbolic_beta` is ignored.

        Returns
        -------
        float
            Regularization value.
        """

        if self._regularization_type == "L2":
            return self._l2_regularization(current_parameters, a)
        elif self._regularization_type == "L1":
            return self._l1_regularization(current_parameters, a)
        elif self._regularization_type == "HYPERBOLIC":
            return self._hyperbolic_regularization(current_parameters, a, b)
        else:
            raise NotImplementedError("Regularization {} scheme not implement.".format(self._regularization_type))

    # ------------------------------------------------------------ #
    #                                                              #
    #                       PRIVATE METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    def _l2_regularization(self, current_parameters, a=None):
        """
        Method that computes the value of the L2 regularization.

        Parameters
        ----------
        current_parameters : list of floats
            Lists containing the optimizable values of the parameters.
        a : float, default=`None`
            a parameter (scaling factor). If not `None`, instance attribute `self._scaling_factor` is ignored.

        Notes
        -----
        :math:`L2 = a(param-param_0)^2` where a is a scaling factor.

        Returns
        -------
        value : float
            Value of the regularization.
        """
        if a is None:
            a = self._scaling_factor

        diff = (np.asarray(current_parameters) - self._initial_parameters_values)

        reg = np.power(diff, 2)
        self.value = a * np.sum(reg)

        return self.value

    def _l1_regularization(self, current_parameters, a=None):
        """
        Method that computes the value of the L1 regularization.

        Parameters
        ----------
        current_parameters : list of floats
            Lists containing the optimizable values of the parameters.
        a : float, default=`None`
            a parameter (scaling factor). If not `None`, instance attribute `self._scaling_factor` is ignored.

        Notes
        -----
        :math:`L1 = a|param-param_0|` where a is a scaling factor.

        Returns
        -------
        value : float
            Value of the regularization.
        """
        if a is None:
            a = self._scaling_factor

        diff = (np.asarray(current_parameters) - self._initial_parameters_values)
        reg = np.abs(diff)
        self.value = a * np.sum(reg)

        return self.value

    def _hyperbolic_regularization(self, current_parameters, a=None, b=None):
        """
        Method that computes the value of the hyperbolic regularization.

        Parameters
        ----------
        current_parameters : list of floats
            Lists containing the optimizable values of the parameters.
        a : float, default=`None`
            a parameter (scaling factor). If not `None`, instance attribute `self._scaling_factor` is ignored.
        b : float, default=`None`
            Hyperbolic beta parameter. If not `None`, instance attribute `self._hyperbolic_beta` is ignored.

        Notes
        -----
        :math:`hyperbolic = a\sum_{m}^{N_{charges}} ((q_m^2 + b^2 )^{1/2} - b)`

        Returns
        -------
        value : float
            Value of the regularization.
        """
        if a is None:
            a = self._scaling_factor
        if b is None:
            b = self._hyperbolic_beta

        reg = np.sum( ((np.asarray(current_parameters) )**2 + b**2)**(1/2.) - b)

        self.value = a * reg

        return self.value
