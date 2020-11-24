# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Objective_function.Properties.energy_property.EnergyProperty` class, which is a ParaMol representation of the energy property.
"""

import numpy as np
import simtk.unit as unit

from .property import *


# -----------------------------------------------------------#
#                                                            #
#                      ENERGY PROPERTY                       #
#                                                            #
# -----------------------------------------------------------#
class EnergyProperty(Property):
    """
    ParaMol representation of the energy property.

    Parameters
    ----------
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List of ParaMol Systems.
    weight : float
        Weight of this property in the objective function.

    Attributes
    ----------
    name : str
        'ENERGY'
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List of ParaMol Systems. Currently not used and it is set to None.
    units : simtk.unit.Quantity
        kj/mol
    value : float
        Current value of this property
    weight : float
        Weight of this property in the objective function.
    variance : list of floats
        Variance of the energy for each system.
    """
    def __init__(self, systems=[], weight=1.0):
        self.name = 'ENERGY'
        self.value = None
        self.weight = weight
        self.systems = systems
        self.variance = None
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
        system: list of :obj:`ParaMol.System.system.ParaMolSystem`
            Attribute system that contains a list of ParaMol Systems.
        """
        self.systems.append(system)

        return self.systems

    def calculate_property(self, emm_data):
        """
        Method that calculates the value of the energy property.

        Notes
        -----
        This method should be modified by the user if another objective function form is to be used.
        Calculates the square of the difference of the MM and QM energies, i.e.:

        :math:`\sum_k^{N_{conf}} (E_k^{MM} - E_k^{QM}-<E^{MM}-E^{QM}>)^2`

        Parameters
        ----------
        emm_data : np.array
            Array containing the MM energies for each conformation

        Returns
        -------
        obj_fun_energies : list of float
            Value of the energy property contribution to the objective function for every system.
        """

        obj_fun_energies = []
        emm_data = np.asarray(emm_data)

        for system, emm, var in zip(self.systems, emm_data, self.variance):
            # (E^{MM} - E^{QM})
            diff = system.ref_energies - emm
            avg_diff = np.mean(diff)
            # (E^{MM} - E^{QM}) - <E^{MM} - E^{QM}>
            obj_fun = diff - avg_diff
            # [ (E^{MM} - E^{QM}) - <E^{MM} - E^{QM}> ]^2
            obj_fun = obj_fun * obj_fun
            # \sum_i \omega_i [ (E_i^{MM} - E_i^{QM}) - <E^{MM} - E^{QM}> ]^2
            obj_fun = np.sum(system.weights * system.wham_weights *  obj_fun)
            # [ (E^{MM} - E^{QM}) - <E^{MM} - E^{QM}> ]^2 / [ var(E^{QM}) * Ns ]
            obj_fun_energies.append(obj_fun / var)

        self.value = np.sum(obj_fun_energies)

        return obj_fun_energies

    def calculate_variance(self):
        """
        Method that calculates the variance of the QM energies, i.e., :math:`var(E_{QM})=<E^2_{QM}>-<E_{QM}>^2`.

        Notes
        ------
        The variance will be stored in the attribute self.variance. It is used as a normalization factor in the objective function because it introduces the correct rescaling that make the residuals, i.e., :math:`\Delta E=E^{QM}-E^{MM}`, adimensional and with unit magnitude.

        Returns
        -------
        variance : np.array of floats
            Array containing the variance of the QM (reference) energies for each system.
        """
        self.variance = []

        for system in self.systems:
            assert system.ref_energies is not None, "ERROR: It is not possible to calculate the variance, data was not set."
            self.variance.append(np.var(system.ref_energies))

        self.variance = np.asarray(self.variance)
        return self.variance
