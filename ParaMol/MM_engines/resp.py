# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.MM_engines.resp.RESP` class which is the ParaMol RESP engine.
"""

# ParaMol modules
from ..System.system import *
import numpy as np


# ------------------------------------------------------------ #
#                                                              #
#                             RESP                             #
#                                                              #
# ------------------------------------------------------------ #
class RESP:
    """
    ParaMol's RESP engine.

    Parameters
    ----------
    total_charge : int
        Total charge of the system
    constraint_tolerance : float
        Constraint tolerance.

    Attributes
    ----------
    charges : list of floats
        List with system's charges.
    initial_charges : list of floats
        List with initial system's charges.
    inv_rij : list or np.array
        (nconformations, natoms, n_esp_values) Inverse distances (1/rij) between the atomic centre j and the electrostatic point i for every every conformation of every system.
    """
    def __init__(self, total_charge, constraint_tolerance=1e-8):
        # Variables related to total charge and symmetry constraints
        self._n_constraints = 1 # There is always at least one constraint (total charge one). This value is increased if there are more symmetry constraints.
        self._total_charge = total_charge
        self._contraint_tol = constraint_tolerance
        self._symmetry_constraints = None

        self.charges = None
        self.initial_charges = None

        # Variables that defines matrices used in the charge fitting procedure
        self.inv_rij = None

        # Matrices used in the explicit solution of the RESP equations
        self._A = None
        self._B = None
        # Auxiliary matrices that store values that do not change along the calculation
        self._A_aux = None
        self._B_aux = None

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def calculate_inverse_distances(self, system):
        """
        Method that calculates the inverse distances (1/rij) between the atomic centre j and the electrostatic point i for every every conformation of every system.

        Notes
        -----
        This will be used in the fitting procedure and should be calculated only once before any optimization.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            ParaMol System.

        Returns
        -------
        inv_rij : np.array
            (nconformations, natoms, n_esp_values) distance matrix
        """

        # Create empty array where result will be stored
        self.inv_rij = np.zeros((system.n_structures,
                                 system.n_atoms,
                                 system.ref_esp.shape[1]))

        # Compute 1/rij for all conformations
        # Iterate over all conformations
        for m in range(system.n_structures):
            # Iterate over all atomic centers
            for j in range(system.n_atoms):
                # Iterate over all grid points
                for i in range(system.ref_esp.shape[1]):
                    v = system.ref_coordinates[m, j, :] - system.ref_esp_grid[m, i, :]
                    rij = np.linalg.norm(v)
                    self.inv_rij[m, j, i] = 1.0 / rij

        return self.inv_rij

    def set_charges(self, force_field):
        """
        Method that sets the `charges` attribute by constructing a list with the current ParaMol Force Field charges.

        Parameters
        ----------
        force_field : :obj:`ParaMol.Force_field.force_field.ForceField`
            ParaMol Force Field.

        Returns
        -------
        charges : list of float
            List with system's charges.
        """

        self.charges = []
        if "NonbondedForce" in force_field:
            for nonbonded_term in force_field["NonbondedForce"]:
                self.charges.append(nonbonded_term.parameters["charge"].value)

        return self.charges

    def set_initial_charges(self, force_field):
        """
        Method that sets the `charges_initial` attribute by constructing a list with the current ParaMol Force Field charges.

        Parameters
        ----------
        force_field : :obj:`ParaMol.Force_field.force_field.ForceField`
            ParaMol Force Field.

        Returns
        -------
        charges : list of float
            List with system's charges.
        """
        self.initial_charges = []
        if "NonbondedForce" in force_field:
            for nonbonded_term in force_field["NonbondedForce"]:
                self.initial_charges.append(nonbonded_term.parameters["charge"].value)

        return self.initial_charges

    # ------------------------------------------------------------ #
    #           Explicit solution of the RESP problem              #
    # ------------------------------------------------------------ #
    def fit_resp_charges_explicitly(self, system):
        """
        Method that explicitly solves RESP equations

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            ParaMol System.

        Returns
        -------
        charges : list of float
            List with system's charges.
        """
        from numpy.linalg import inv

        #system.compute_conformations_weights(temperature=self._weighting_temperature, emm=emm, weighting_method=self._weighting_method)
        # Compute A matrix
        self._calculate_a(system, initialize=True)
        # Compute B matrix
        self._calculate_b(system, initialize=True)

        print("Initial net charge: {}.".format(self._total_charge))

        # Solve system of equations in matrix form (it is necessary to invert a matrix; may be costly for large mols)
        # q=A^{-1}B
        self.charges = np.matmul(inv(self._A), self._B)

        print("Initial charges {}".format(self.charges[:system.n_atoms]))
        max_iter = 100000
        # Self-consistent solution
        n_iter = 0

        rmsd_tol = 1e-5
        rmsd = 1.0
        while n_iter < max_iter and rmsd > rmsd_tol:
            # Advance one iteration
            n_iter = n_iter + 1

            # Compute A matrix
            self._calculate_a(system, initialize=False)

            # Compute B matrix
            self._calculate_b(system, initialize=False)

            # q=A^{-1}B
            old_charges = self.charges
            self.charges = np.matmul(inv(self._A), self._B)
            new_charges = self.charges

            rmsd = np.sqrt(np.sum( (old_charges-new_charges)**2 ) / system.n_atoms )

        for charge in self.charges[:system.n_atoms]:
            print("charge: {}".format(charge))

        print("Final net charge: {}.".format(np.sum(self.charges[:system.n_atoms])))

        return self.charges

    def set_symmetry_constraints(self, system, symmetry_constrained):
        """
        Method that sets symmetry constraints.

        Notes
        -----
        This is only necessary for the explicit solution case.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            ParaMol System.
        symmetry_constrained : bool
            Whether or not the optimization is constrained by symmetries.

        Returns
        -------
        charges : list of list of int
            List of lists, wherein the inner lists contain symmetric-equivalent pairs of atoms.
        """
        self._symmetry_constraints = []

        if symmetry_constrained:
            symmetry_groups = []  # List used to keep track of symmetry groups

            for i in range(len(system.force_field.force_field['NonbondedForce'])):
                parameter_i = system.force_field.force_field['NonbondedForce'][i].parameters['charge']
                if parameter_i.symmetry_group != system.force_field.symmetry_group_default \
                        and parameter_i.symmetry_group not in symmetry_groups:
                    # If parameter belong to the non-default symmetry group and this symmetry group was not already set
                    symmetry_groups.append(parameter_i.symmetry_group)
                    for j in range(i+1,len(system.force_field.force_field['NonbondedForce'])):
                        parameter_j = system.force_field.force_field['NonbondedForce'][j].parameters['charge']
                        if parameter_j.symmetry_group == symmetry_groups[-1]:
                            # Add this symmetry constraint
                            self._symmetry_constraints.append([i, j])
                            self._n_constraints = self._n_constraints + 1

        return self._symmetry_constraints

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PRIVATE METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def _calculate_a(self, system, initialize=False):
        """
        Method that calculates the auxiliary A matrix.

        Notes
        -----
        This is only necessary for the explicit solution case.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            ParaMol System.
        initialize : bool
            Whether or not A has been already initialized previously. It is only necessary to compute non-diagonal elements when A is initialized.

        Returns
        -------
        A : np.array
            Array of the A matrix.
        """
        if initialize:
            # It is only necessary to compute non-diagonal elements when A is initialized.
            # This is due to the fact that these terms do not change during the SC procedure (they do not depend on q_j)
            self._A_aux = np.zeros((system.n_structures,
                                    system.n_atoms + self._n_constraints,
                                    system.n_atoms + self._n_constraints))

            # (Nconf,Natoms+Nconstraints,Natoms+Nconstraints)

            # Compute the immutable part of non-diagonal and diagonal elements
            # Only compute upper diagonal part + diagonal elements
            for m in range(system.n_structures):
                for j in range(system.n_atoms):
                    for k in range(j,system.n_atoms):
                        for i in range(system.ref_esp.shape[1]):
                            # A_{jk} = \sum_{i} 1/(r_{ij}*r_{ik})
                            self._A_aux[m, j, k] = self._A_aux[m, j, k] + self.inv_rij[m, j, i] * self.inv_rij[m, k, i]

                            if k != j:
                                # Lower diagonal part
                                self._A_aux[m, k, j] = self._A_aux[m, j, k]

            # Add row/column information relative to the Lagrange multiplier
            self._A_aux[:, system.n_atoms, 0:system.n_atoms] = 1.0
            self._A_aux[:, 0:system.n_atoms, system.n_atoms] = 1.0

            # Add row/column information relative to the symmetry-constraints
            for i in range(len(self._symmetry_constraints)):
                self._A_aux[:, system.n_atoms + i + 1, self._symmetry_constraints[i][0]] = 1.0
                self._A_aux[:, system.n_atoms + i + 1, self._symmetry_constraints[i][1]] = -1.0
                self._A_aux[:, self._symmetry_constraints[i][0], system.n_atoms + i + 1] = 1.0
                self._A_aux[:, self._symmetry_constraints[i][1], system.n_atoms + i + 1] = -1.0

        # Set matrix A equal to its immutable part
        self._A = self._A_aux

        # Compute diagonal elements restraints
        # They are necessary to re-calculate in every iteration because they depend on the charge value
        # A_{jj} = \sum_{i} 1/(r_{ij}^2) + \frac{\partial X^2_{rstr}}{\partial q_j}
        # Iterate over all conformations
        for m in range(system.n_structures):
            # Iterate over all atomic centers
            for j in range(system.n_atoms):
                # Add derivative of restrain w.r.t. atomic charge
                self._A[m, j, j] = self._A[m, j, j] + self._regularization_derivative(j)

        # Sum over all conformations
        self._A = np.sum(self._A, axis=0)

        return self._A

    def _calculate_b(self, system, initialize=False):
        """
        Method that calculates the auxiliary B matrix.

        Notes
        -----
        This is only necessary for the explicit solution case. :math:`B_j = \sum_{i} V_{i} / r_{ij} + q_{0j} + dX^2_{rstr}/dq_{j}`

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            ParaMol System.
        initialize : bool
            Whether or not B has been already initialized previously.

        Returns
        -------
        A : np.array
            Array of the A matrix.
        """

        if initialize:
            self._B_aux = np.zeros((system.n_structures, system.n_atoms + self._n_constraints))

            # Iterate over all conformations
            for m in range(system.n_structures):
                # Iterate over all atomic centers
                for j in range(system.n_atoms):
                    # Iterate over all grid points
                    for i in range(system.ref_esp.shape[1]):
                        # B_{j} = \sum_{i} \frac{V_i}{r_{ij}}
                        self._B_aux[m, j] = self._B_aux[m, j] + system.ref_esp[m, i] * self.inv_rij[m, j, i]

            # Set last element of the row equal to the total charge
            self._B_aux[:, system.n_atoms] = self._total_charge

        # Set matrix B equal to its immutable part
        self._B = self._B_aux

        # Iterate over all conformations
        for m in range(system.n_structures):
            # Iterate over all atomic centers
            for j in range(system.n_atoms):
                # Add derivative of restrain w.r.t. atomic charge
                self._B[m, j] = self._B[m, j] + self._regularization_derivative(j) * self.initial_charges[j]

        # Sum results over all conformations
        self._B = np.sum(self._B, axis=0)

        return self._B

    def _regularization_derivative(self, at):
        """
        Method that calculates the derivative of the regularization.

        Parameters
        ----------
        at : int
            Atom index

        Notes
        -----
        This is only necessary for the explicit solution case.

        Returns
        -------
        reg_deriv : float
            Value of the regularization derivative.
        """
        type = "NONE"

        if type.upper() == "L2":
            a = 1.0
            reg_deriv = - 2.0 * a * (self.initial_charges[at]-self.charges[at])
        elif type.upper() == "HYPERBOLIC":
            a = 0.01
            b = 0.1
            reg_deriv = a * self.charges[at] * (self.charges[at]**2+b**2)**(-1/2.)
        elif type.upper() == "NONE":
            reg_deriv = 0

        return reg_deriv

