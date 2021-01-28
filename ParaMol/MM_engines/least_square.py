import numpy as np
import copy

from ..Utils.geometry import *


class LinearLeastSquare:
    """
    Linear Least Square Fitting solution.

    Parameters
    ----------
    parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
        Instance of the parameter space.
    include_regulatization : bool
        Flag that signal whether or not to include regularization.
    method : str
        Type of regularization. Options are 'L2' or 'hyperbolic'.
    scaling_factor : float
        Scaling factor of the regularization value.
    hyperbolic_beta : float
        Hyperbolic beta value. Only used if `regularization_type` is `hyperbolic`.
    weighting_method : str
        Method used to weight the conformations. Available methods are "uniform, "boltzmann" and "manual".
    weighting_temperature : unit.simtk.Quantity
        Temperature used in the weighting. Only relevant if `weighting_method` is "boltzmann".

    Attributes
    ----------
    include_regulatization : bool
        Flag that signal whether or not to include regularization.
    regularization_type : str
        Type of regularization. Options are 'L2' or 'hyperbolic'.
    scaling_factor : float
        Scaling factor of the regularization value.
    hyperbolic_beta : float
        Hyperbolic beta value. Only used if `regularization_type` is `hyperbolic`.
    weighting_method : str
        Method used to weight the conformations. Available methods are "uniform, "boltzmann" and "manual".
    weighting_temperature : unit.simtk.Quantity
        Temperature used in the weighting. Only relevant if `weighting_method` is "boltzmann".
    """
    def __init__(self, parameter_space, include_regularization, method, scaling_factor, hyperbolic_beta, weighting_method, weighting_temperature, **kwargs):
        # Matrices used in the explicit solution of the LLS equations
        self._parameter_space = parameter_space

        self._parameters = None
        self._n_parameters = None
        # Private variables
        self._A = None
        self._B = None
        self._param_keys_list = None
        self._p0 = None
        self._initial_param_regularization = None

        # Regularization variables
        self._include_regularization = include_regularization
        self._regularization_type = method
        self._scaling_factor = scaling_factor
        self._hyperbolic_beta = hyperbolic_beta

        # Weighting variables
        self._weighting_method = weighting_method
        self._weighting_temperature = weighting_temperature

    def fit_parameters_lls(self, systems, alpha_bond=0.05, alpha_angle=0.001):
        """
        Method that fits bonded parameters using LLS.

        Notes
        -----
        Only one ParaMol system is supported at once.

        Parameters
        ----------
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        alpha_bond : float
            P
        alpha_angle : float

        Returns
        -------
        systems, parameter_space, objective_function, optimizer
        """
        # TODO: change this
        system = systems[0]

        # Get optimizable parameters; symmetry constrained is False to allow to get all parameters.
        self._parameter_space.get_optimizable_parameters([system], symmetry_constrained=False)

        # Compute A matrix
        self._calculate_a(system, alpha_bond, alpha_angle)
        self._n_parameters = self._A.shape[1]

        # Compute B matrix
        self._calculate_b(system)

        # ---------------------------------------------------------------- #
        #                 Calculate conformations weights                  #
        # ---------------------------------------------------------------- #
        system.compute_conformations_weights(temperature=self._weighting_temperature, weighting_method=self._weighting_method, emm=self.mm_energies_zero)

        # Calculate variance of QM energies
        var_QM = np.var(system.ref_energies)

        # Weight conformations
        for row in range(system.n_structures):
            self._A[row, :] = self._A[row, :] * np.sqrt(system.weights[row] / (system.n_structures*var_QM))

        self._B = self._B * np.sqrt(system.weights[row] / (system.n_structures*var_QM))
        # ---------------------------------------------------------------- #

        # ---------------------------------------------------------------- #
        #                           Preconditioning                        #
        # ---------------------------------------------------------------- #
        # Preconditioning
        self._calculate_scaling_constants()

        for row in range(system.n_structures):
            self._A[row, :] = self._A[row, :] / self._scaling_constants
        # ---------------------------------------------------------------- #

        # ---------------------------------------------------------------- #
        #                           Regularization                         #
        # ---------------------------------------------------------------- #
        if self._include_regularization:
            # Add regularization
            self._A, self._B = self._add_regularization()
        # ---------------------------------------------------------------- #

        # ---------------------------------------------------------------- #
        #                           Symmetries                             #
        # ---------------------------------------------------------------- #
        self._add_symmetries(system)
        # ---------------------------------------------------------------- #

        # Perform LLS
        self._parameters = np.linalg.lstsq(self._A, self._B, rcond=None)[0]

        # Revert scaling
        self._parameters = self._parameters / self._scaling_constants

        # Reconstruct parameters
        self._reconstruct_parameters(self._parameters)

        # Get optimizable parameters
        self._parameter_space.get_optimizable_parameters([system], symmetry_constrained=False)

        return self._parameter_space.optimizable_parameters_values

    def _add_regularization(self):
        """
        Method that adds the regularization part of the A and B matrices.

        Returns
        -------
        self._A, self._B
        """
        # Create alpha=scaling_factor / scaling_constants
        alpha = self._scaling_factor / self._scaling_constants

        # Calculate prior widths
        self._calculate_prior_widths()

        # Calculate A_reg
        A_reg = np.identity(self._n_parameters)

        for row in range(A_reg.shape[0]):
            A_reg[row, :] = (A_reg[row, :]) / self._prior_widths

        A_reg = A_reg * alpha

        # Update A matrix
        self._A = np.vstack((self._A, A_reg))

        # Calculate B_reg
        #B_reg = np.zeros((n_parameters))
        B_reg = alpha * self._initial_param_regularization

        # Update B matrix
        self._B = np.concatenate((self._B, B_reg))

        print("Added regularization.")

        return self._A, self._B

    def _add_symmetries(self, system):
        """
        Method that adds the symmetrie part of the A and B matrices.

        Returns
        -------
        self._A, self._B
        """
        n_symmetries = 0

        symm_covered = []
        A_symm = []
        for i in range(len(self._param_symmetries_list)):
            symm_i = self._param_symmetries_list[i]

            if symm_i in symm_covered or symm_i in ["X_x", "X_y", "X"]:
                continue

            for j in range(i+1, len(self._param_symmetries_list)):
                symm_j = self._param_symmetries_list[j]

                if symm_i == symm_j:
                    A_symm_row = np.zeros((self._n_parameters))
                    A_symm_row[i] = 1.0
                    A_symm_row[j] = 1.0
                    A_symm.append(A_symm_row)
                    n_symmetries += 1

            symm_covered.append(symm_i)

        A_symm = np.asarray(A_symm)

        # Update matrices
        if n_symmetries > 0:
            self._A = np.vstack((self._A, A_symm))

            # Calculate B_reg
            B_symm = np.zeros((n_symmetries))

            # Update B matrix
            self._B = np.concatenate((self._B, B_symm))

        print("{} symmetries were found".format(n_symmetries))

        return self._A, self._B

    def _calculate_prior_widths(self, method=None):
        """"
        Method that generates the prior_widths vector.

        Parameters
        ----------
        method : str, optional
            Method used to generate the prior widths.

        Returns
        -------
        self._prior_widths : np.array
            Array containing the prior widths.
        """
        self._prior_widths = []
        prior_widths_dict, prior_widths = self._parameter_space.calculate_prior_widths(method=method)

        for i in range(self._n_parameters):
            self._prior_widths.append(prior_widths_dict[self._param_keys_list[i]])

        self._prior_widths = np.asarray(self._prior_widths)
        return self._prior_widths

    def _calculate_scaling_constants(self, method=None):
        """
        Method that generates the scaling constant's vector.

        Parameters
        ----------
        method : str, optional
            Method used to generate the prior widths.

        Returns
        -------
        self._prior_widths : np.array
            Array containing the scaling constants.
        """
        self._scaling_constants = []

        scaling_constants_dict, scaling_constants = self._parameter_space.calculate_scaling_constants(method=method)

        for i in range(self._n_parameters):
            self._scaling_constants.append(scaling_constants_dict[self._param_keys_list[i]])

        self._scaling_constants = np.asarray(self._scaling_constants)
        return self._scaling_constants

    def _reconstruct_parameters(self, final_parameters):
        """
        Method that reconstructs the parameters after the LLS.

        Parameters
        ----------
        final_parameters : np.array or list
            List containing the final parameters.

        Returns
        -------
        """
        m = 0
        for parameter in self._parameter_space.optimizable_parameters:
            ff_term = parameter.ff_term

            # ---------------------------------------------------------------- #
            #                               Bonds                              #
            # ---------------------------------------------------------------- #
            if parameter.param_key == "bond_k":
                if ff_term.parameters["bond_eq"].optimize:
                    k_xy = np.asarray(final_parameters[m:m + 2])
                    x0_xy = np.asarray(self._p0[m:m+2])
                    # Update value of "bond_k"
                    parameter.value = np.sum(k_xy)
                    # Update value of "bond_eq"
                    ff_term.parameters["bond_eq"].value = np.sum(k_xy*x0_xy) / np.sum(k_xy)
                    m += 2
                else:
                    k_xy = final_parameters[m]
                    # Update value of "bond_k"
                    parameter.value = k_xy
                    m += 1

            # ---------------------------------------------------------------- #
            #                              Angles                              #
            # ---------------------------------------------------------------- #
            elif parameter.param_key == "angle_k":
                if ff_term.parameters["angle_eq"].optimize:
                    k_xy = np.asarray(final_parameters[m:m + 2])
                    theta0_xy = np.asarray(self._p0[m:m+2])
                    # Update value of "bond_k"
                    parameter.value = np.sum(k_xy)
                    # Update value of "bond_eq"
                    ff_term.parameters["angle_eq"].value = np.sum(k_xy*theta0_xy) / np.sum(k_xy)
                    m += 2
                else:
                    k_xy = final_parameters[m]
                    # Update value of "bond_k"
                    parameter.value = k_xy
                    m += 1

            # ---------------------------------------------------------------- #
            #                             Torsions                             #
            # ---------------------------------------------------------------- #
            elif parameter.param_key == "torsion_k":
                if ff_term.parameters["torsion_phase"].optimize:
                    k_xy = np.asarray(final_parameters[m:m + 2])
                    delta_xy = np.asarray(self._p0[m:m + 2])
                    # Update value of "bond_k"
                    parameter.value = np.sqrt(np.sum(k_xy*k_xy))
                    # Update value of "bond_eq"
                    ff_term.parameters["torsion_phase"].value = np.arctan2(k_xy[1], k_xy[0])
                    m += 2
                else:
                    k_xy = final_parameters[m]
                    # Update value of "bond_k"
                    parameter.value = k_xy
                    m += 1

            elif parameter.param_key not in ["torsion_phase", "bond_eq", "angle_eq"]:
                raise NotImplementedError("Fitting of {} not implemented in LLS.".format(parameter.param_key))

        return

    def _calculate_a(self, system, alpha_bond=None, alpha_angle=None):
        """
        Method that calculates the A matrix.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of a ParaMol System.
        alpha_bond : float
        alpha_angle : float

        Returns
        -------
        self._A : np.array
            Array containing the A matrix.
        """

        self._initial_param_regularization = []
        self._param_keys_list = []
        self._p0 = []
        self._param_symmetries_list = []

        r_matrix = []
        for parameter in self._parameter_space.optimizable_parameters:
            ff_term = parameter.ff_term

            # ---------------------------------------------------------------- #
            #                               Bonds                              #
            # ---------------------------------------------------------------- #
            if parameter.param_key == "bond_k":

                # Calculate distances
                distances = []
                at1, at2 = ff_term.atoms

                for conformation in system.ref_coordinates:
                    distances.append(calculate_distance(conformation[at1], conformation[at2]))

                distances = np.asarray(distances)

                if ff_term.parameters["bond_eq"].optimize:
                    #x0_x = np.min(distances)
                    #x0_y = np.max(distances)
                    # Alternative way of calculating x0_x and x0_y, leave it here
                    x0_x = ff_term.parameters["bond_eq"].value * (1 - alpha_bond)
                    x0_y = ff_term.parameters["bond_eq"].value * (1 + alpha_bond)
                    r_vec = np.empty((system.n_structures, 2))
                    for m in range(system.n_structures):
                        r_vec[m, 0] = 0.5 * (distances[m] - x0_x) * (distances[m] - x0_x)
                        r_vec[m, 1] = 0.5 * (distances[m] - x0_y) * (distances[m] - x0_y)

                    r_matrix.append(r_vec)
                    self._p0.append(x0_x)
                    self._p0.append(x0_y)
                    self._param_keys_list.append(parameter.param_key)
                    self._param_keys_list.append(parameter.param_key)
                    self._initial_param_regularization.append(parameter.value)
                    self._initial_param_regularization.append(parameter.value)
                    self._param_symmetries_list.append(parameter.symmetry_group+"_x")
                    self._param_symmetries_list.append(parameter.symmetry_group+"_y")
                else:
                    x0 = ff_term.parameters["bond_eq"].value
                    r_vec = np.empty((system.n_structures, 1))
                    for m in range(system.n_structures):
                        r_vec[m, 0] = 0.5 * (distances[m] - x0) * (distances[m] - x0)

                    r_matrix.append(r_vec)
                    self._p0.append(x0)
                    self._param_keys_list.append(parameter.param_key)
                    self._initial_param_regularization.append(parameter.value)

            # ---------------------------------------------------------------- #
            #                              Angles                              #
            # ---------------------------------------------------------------- #
            elif parameter.param_key == "angle_k":
                # Calculate angles
                angles = []
                at1, at2, at3 = ff_term.atoms

                for conformation in system.ref_coordinates:
                    v1 = conformation[at1]-conformation[at2]
                    v2 = conformation[at3]-conformation[at2]

                    angles.append(calculate_angle(v1, v2))

                angles = np.asarray(angles)

                if ff_term.parameters["angle_eq"].optimize:
                    theta0_x = ff_term.parameters["angle_eq"].value * (1 - alpha_angle)
                    theta0_y = ff_term.parameters["angle_eq"].value * (1 + alpha_angle)
                    #theta0_x = np.min(angles)
                    #theta0_y = np.max(angles)

                    r_vec = np.empty((system.n_structures, 2))
                    for m in range(system.n_structures):
                        r_vec[m, 0] = 0.5 * (angles[m] - theta0_x) * (angles[m] - theta0_x)
                        r_vec[m, 1] = 0.5 * (angles[m] - theta0_y) * (angles[m] - theta0_y)

                    r_matrix.append(r_vec)
                    self._p0.append(theta0_x)
                    self._p0.append(theta0_y)
                    self._param_keys_list.append(parameter.param_key)
                    self._param_keys_list.append(parameter.param_key)
                    self._initial_param_regularization.append(parameter.value)
                    self._initial_param_regularization.append(parameter.value)
                    self._param_symmetries_list.append(parameter.symmetry_group+"_x")
                    self._param_symmetries_list.append(parameter.symmetry_group+"_y")
                else:
                    theta0 = ff_term.parameters["angle_eq"].value
                    r_vec = np.empty((system.n_structures, 1))
                    for m in range(system.n_structures):
                        r_vec[m, 0] = 0.5 * (angles[m] - theta0) * (angles[m] - theta0)

                    r_matrix.append(r_vec)
                    self._p0.append(theta0)
                    self._param_keys_list.append(parameter.param_key)
                    self._initial_param_regularization.append(parameter.value)

            # ---------------------------------------------------------------- #
            #                              Torsions                            #
            # ---------------------------------------------------------------- #
            elif parameter.param_key == "torsion_k":
                # Calculate dihedral angles
                dihedrals = []
                at1, at2, at3, at4 = ff_term.atoms

                for conformation in system.ref_coordinates:
                    p1 = conformation[at1]
                    p2 = conformation[at2]
                    p3 = conformation[at3]
                    p4 = conformation[at4]

                    dihedrals.append(calculate_dihedral(p1, p2, p3, p4))

                dihedrals = np.asarray(dihedrals)

                if ff_term.parameters["torsion_phase"].optimize:
                    phase_x = 0 #-np.pi/4
                    phase_y = np.pi/2 #np.pi/4

                    r_vec = np.empty((system.n_structures, 2))
                    for m in range(system.n_structures):
                        r_vec[m, 0] = (1+np.cos(ff_term.parameters["torsion_periodicity"].value*dihedrals[m]-phase_x))
                        r_vec[m, 1] = (1+np.cos(ff_term.parameters["torsion_periodicity"].value*dihedrals[m]-phase_y))

                    r_matrix.append(r_vec)
                    self._p0.append(phase_x)
                    self._p0.append(phase_y)
                    self._param_keys_list.append(parameter.param_key)
                    self._param_keys_list.append(parameter.param_key)
                    self._initial_param_regularization.append(parameter.value)
                    self._initial_param_regularization.append(parameter.value)
                    self._param_symmetries_list.append(parameter.symmetry_group+"_x")
                    self._param_symmetries_list.append(parameter.symmetry_group+"_y")
                else:
                    phase = ff_term.parameters["angle_eq"].value
                    r_vec = np.empty((system.n_structures, 1))
                    for m in range(system.n_structures):
                        r_vec[m, 0] = (1+np.cos(ff_term.parameters["torsion_periodicity"].value*dihedrals[m]-phase))

                    r_matrix.append(r_vec)
                    self._p0.append(phase)
                    self._param_keys_list.append(parameter.param_key)
                    self._initial_param_regularization.append(parameter.value)
            elif parameter.param_key not in ["torsion_phase", "bond_eq", "angle_eq"]:
                raise NotImplementedError("Fitting of {} not implemented in LLS.".format(parameter.param_key))

        # Form final A matrix
        self._A = np.hstack(r_matrix)
        for col in range(self._A.shape[1]):
            self._A[:, col] = self._A[:, col] - np.mean(self._A[:, col])

        self._initial_param_regularization = np.asarray(self._initial_param_regularization)

        return self._A

    def _calculate_b(self, system):
        """
        Method that calculates the B matrix.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of a ParaMol System.

        Returns
        -------
        self._B : np.array
            Array containing the B matrix.
        """
        mm_energies_initial = system.get_energies_ensemble()

        # Calculate the MM energies of all conformations, for the case in hiwch all FF terms are present except the ones being fitted
        # This is equivalent o zero all parameters and update the FF
        old_parameter_values = copy.deepcopy(self._parameter_space.optimizable_parameters_values)

        # Generate vector of zeros with same length as the parameters' vector
        zero_values = np.zeros(len(old_parameter_values)).tolist()

        # Update FF and OpenMM Engine
        self._parameter_space.update_systems([system], zero_values)

        mm_energies_zero = system.get_energies_ensemble()

        self.mm_energies_zero = mm_energies_zero
        # Energy contribution associated with the initial guess parameters
        corr = mm_energies_initial-mm_energies_zero

        # Update FF and OpenMM Engine
        self._parameter_space.update_systems([system], old_parameter_values)

        # Calculate self._B
        self._B = np.asarray(system.ref_energies) - mm_energies_zero
        if self._include_regularization:
            pass

        self._B = self._B - np.mean(self._B)

        print("Calculated B matrix.")

        return self._B

