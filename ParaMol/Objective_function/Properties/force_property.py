# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Objective_function.Properties.force_property.ForceProperty` class, which is a ParaMol representation of the force property.
"""


import numpy as np
import simtk.unit as unit
import logging

from .property import *


class ForceProperty(Property):
    """
    ParaMol representation of the force property.

    Parameters
    ----------
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List of ParaMol Systems.
    weight : float
        Weight of this property in the objective function.
    term_type : str
        Forces term type. Available options are "norm" or "components".

    Attributes
    ----------
    name : str
        'FORCE'
    systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
        List of ParaMol Systems. Currently not used and it is set to None.
    units : simtk.unit.Quantity
        kj/mol/nm
    value : float
        Current value of this property
    weight : float
        Weight of this property in the objective function.
    variance : list of np.array(n_atoms)
        Variance.
    inv_covariance : list of np.array
        Inverse covariance.
    """
    def __init__(self, systems=[], term_type="components", weight=1.0):
        self.name = "FORCE"
        self.value = None
        self.weight = weight
        self.systems = systems
        self.units = unit.kilojoule_per_mole / unit.nanometer
        self.term_type=term_type

        self.variance = None
        self.inv_covariance = None

    def calculate_property(self, fmm_data, term_type=None):
        """
        Method that computes the forces term of the objective function.

        Notes
        -----
        This method should be modified by the user if another objective function form is to be used.
        The two term types available are:

        - "components": :math:`\sum_i^{N_s} \omega_i \sum_j^{N_a} \Delta F_{i,j}<F^{QM} * F^{QM}>^{-1}\Delta F_{i,j}` where :math:`\Delta F_{i,j}=F_{i,j}^{MM}-F_{i,j}^{QM}` and :math:`*` means tensorial multiplication. This type becomes very slow as the number of atoms of the system increases (faster implementation soon to be implemented).
        - "norm": :math:`\sum_i^{N_s} \omega_i \sum_j^{N_a} | \Delta F_{i,j}|^2/var(|F_{i,j}^{QM}|)` where :math:`\Delta F_{i,j}=F_{i,j}^{MM}-F_{i,j}^{QM}`.

        Parameters
        ----------
        fmm_data : np.array
            array containing the MM energies for each conformation
        term_type : str
            Available options are "components", "norm".

        Returns
        -------
        value : float
            Value of the forces property contribution to the objective function for every system.
        """
        if term_type is None:
            term_type = self.term_type

        if term_type.lower() == "components":
            # TODO: make this more efficient.
            obj_fun_forces = []

            for system, fmm, inv_covariance_forces in zip(self.systems, fmm_data, self.inv_covariance):
                diff = fmm - system.ref_forces

                force_error = []
                for i in range(system.n_structures):
                    tmp = 0.0
                    for j in range(system.n_atoms):
                        tmp += np.matmul(np.matmul(diff[i, j, :].T, inv_covariance_forces[j, :, :]), diff[i, j, :])
                    force_error.append(tmp)

                force_error = np.asarray(force_error)
                force_error = np.sum(system.weights * force_error) / (3 * system.n_atoms)
                obj_fun_forces.append(force_error)
        elif term_type.lower() == "norm":
            obj_fun_forces = []
            for system, fmm, var in zip(self.systems, fmm_data, self.variance):
                # (F^{MM} - F^{QM})
                diff = fmm - system.ref_forces
                # [ (F^{MM} - F^{QM}) ]^2
                obj_fun = np.sum(np.power(diff, 2), axis=2)
                # [ (F^{MM} - F^{QM}) ]^2 / var(F^{QM}) - divide by the variance of the norm of each atomic force
                obj_fun = obj_fun / var
                # \sum_j [ (F_i^{MM} - F_i^{QM}) - <F^{MM} - F^{QM}> ]^2 - sum along the conformations axis
                obj_fun = np.sum(obj_fun, 1)
                # \sum_i \omega_i [ (F_i^{MM} - F_i^{QM}) - <F^{MM} - F^{QM}> ]^2
                obj_fun = np.sum(system.weights * obj_fun )
                # Normalize
                obj_fun = obj_fun / (3 * system.n_atoms)
                obj_fun_forces.append(obj_fun)

        else:
            raise NotImplementedError("Force property term of type {} is not implemented.".format(type))

        self.value = np.sum(obj_fun_forces)

        return obj_fun_forces

    def calculate_inverse_covariance_qm_forces(self):
        """
        Method that calculates the inverse covariance of the QM forces, i.e., :math:`<F^{QM} * F^{QM}>^{-1}`, where :math:`*` means tensorial multiplication.

        Notes
        -----
        This is useful when the forces term type used in the objective function is "COMPONENTS".
        The inverted covariance will be stored in the attribute variable self.inv_covariance.
        It is used as a normalization factor in the objective function because it introduces the correct re-scalings.

        Returns
        -------
        inv_covariance: list of np.array
        """
        from numpy.linalg import inv

        for system in self.systems:
            assert system.ref_forces is not None, \
                "\t * Impossible to calculate the covariance of the QM forces since these were not set yet."

            self.inv_covariance = []
            inv_covariance_forces = np.zeros((system.n_atoms, 3, 3))

            for i in range(system.n_atoms):
                avg_outer_product = np.zeros((3, 3))
                for j in range(system.n_structures):
                    outer_product = np.outer(system.ref_forces[j, i, :], system.ref_forces[j, i, :])
                    avg_outer_product = avg_outer_product + outer_product

                # Divide by number of conformations to calculate the average
                avg_outer_product = avg_outer_product / system.n_structures

                # Invert covariance of QM forces
                if abs(np.linalg.det(avg_outer_product)) < 1e-8:
                    # If determinant is very close to zero; assume not invertible
                    logging.info("Determinant is zero and therefore de covariance matrix is not invertible.")
                    avg_outer_product = np.identity(3)
                else:
                    # Det(A)=!0 and, therefore, it should be invertible
                    avg_outer_product = inv(avg_outer_product)

                # Store calculated inverted covariance matrix
                inv_covariance_forces[i, :, :] = avg_outer_product

            self.inv_covariance.append(inv_covariance_forces)

        return self.inv_covariance

    def calculate_variance(self):
        """
        Method that calculates the variance covariance of the QM forces, :math:`var(|F_{i,j}^{QM}|)`.

        Notes
        -----
        This is useful when the forces term type used in the objective function is "NORM.".
        The variance will be stored in the attribute variable self.variance.
        It is used as a normalization factor in the objective function because it introduces the correct re-scalings.

        Returns
        -------
        variance: list of np.array
        """
        self.variance = []
        for system in self.systems:
            assert system.ref_forces is not None, "ERROR: It is not possible to calculate the variance, data was not set."

            # Calculate the norms of the QM atomic forces (Nstruct,Na,3)--> (Nstruct,Na)
            qm_forces_norm = np.linalg.norm(system.ref_forces, axis=2)
            # Calculate the variance along the conformation axis (Nstruct,Na) -> (Na)
            var = np.var(qm_forces_norm, axis=0)
            self.variance.append(var)

        # Also calculate covariance
        self.calculate_inverse_covariance_qm_forces()

        return self.variance
