# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Objective_function.Properties.esp_property.ESPProperty` class, which is a ParaMol representation of the electrostatic potential property.
"""


import numpy as np
import simtk.unit as unit

from .property import *


class ESPProperty(Property):
    """
    ParaMol representation of the electrostatic potential (ESP) property.

    Parameters
    ----------
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List of ParaMol Systems.
    weight : float
        Weight of this property in the objective function.

    Attributes
    ----------
    name : str
        'ESP'
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List of ParaMol Systems. Currently not used and it is set to None.
    units : simtk.unit.Quantity
        kj/mol
    value : float
        Current value of this property
    weight : float
        Weight of this property in the objective function.
    variance : list of floats
        Variance of the ESP for each system.
    """
    def __init__(self, systems=[], weight=None):
        self.name = 'ESP'
        self.value = None
        self.weight = weight
        self.systems = systems
        self.variance = []
        self.units = unit.kilojoule_per_mole

    def add_system(self, system):
        """
        Method that adds a system to the property.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            ParaMol System.

        Returns
        -------
        system: list of
            Attribute system that contains a list of ParaMol Systems.
        """
        self.systems.append(system)

        return self.systems

    def calculate_property(self, esp_data):
        """
        Method that calculates the value of the energy property.

        Parameters
        ----------
        esp_data : np.array
            Array containing the MM electrostatic potential for each conformation.

        Notes
        -----
        This method should be modified by the user if another objective function form is to be used.

        Returns
        -------
        value : float
            Value of the ESP contribution to the objective function.
        """
        self.value = 0

        for system, esp, var in zip(self.systems, esp_data, self.variance):
            # Iterate over all conformations
            # Iterate over all conformations
            tmp_value = np.zeros((system.n_structures))
            for m in range(system.n_structures):
                # Determine number of ESP points
                esp_points = len(system.ref_esp[m])

                # Iterate over all grid points
                sq_diff_sum = 0
                for i in range(esp_points):
                    diff = system.ref_esp[m][i] - esp[m][i]
                    sq_diff_sum += diff * diff

                tmp_value[m] = sq_diff_sum # / (var * esp_points)

            self.value = np.sum(self.weight * tmp_value, axis=0)

        return self.value

    def calculate_variance(self):
        """
        Method that calculates the variance of the electrostatic potential.

        Notes
        ------
        The variance will be stored in the attribute self.variance. It is used as a normalization factor in the objective function because it introduces the correct rescaling that make the residuals.

        Returns
        -------
        variance : np.array of floats
            Array containing the variance of the QM (reference) ESP for each system.
        """
        self.variance = []
        for system in self.systems:
            assert system.ref_esp is not None, "ERROR: It is not possible to calculate the variance, data was not set."
            for m in range(system.n_structures):
                self.variance.append(np.var(system.ref_esp[m]))

        self.variance = np.asarray(self.variance)

        return self.variance