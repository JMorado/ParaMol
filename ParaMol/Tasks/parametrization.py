# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Tasks.parametrization.Parametrization` class, which is a ParaMol task that performs force field parametrization.
"""
import numpy as np
import logging

# ParaMol libraries
from .task import *
from ..Optimizers.optimizer import *
from ..Parameter_space.parameter_space import *
from ..Objective_function.objective_function import *


# ------------------------------------------------------------
#                                                            #
#                       PARAMETRIZATION TASK                 #
#                                                            #
# ------------------------------------------------------------
class Parametrization(Task):
    """
    ParaMol parametrization task.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------- #
    #                                                            #
    #                       PUBLIC METHODS                       #
    #                                                            #
    # ---------------------------------------------------------- #
    def run_task(self, settings, systems, parameter_space=None, objective_function=None, optimizer=None, adaptive_parametrization=False):
        """
        Method that performs the standard ParaMol parametrization.
        
        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instance of the parameter space.
        objective_function : :obj:`ParaMol.Objective_function.objective_function.ObjectiveFunction`
            Instance of the objective function.
        optimizer : one of the optimizers defined in the subpackage :obj:`ParaMol.Optimizers`
            Instance of the optimizer.
        adaptive_parametrization: bool
            Flag that signals if this parametrization is being done inside a an adaptive parametrization loop. If `False` the sytem xml file is not written in this method (default is `False`).

        Returns
        -------
        systems, parameter_space, objective_function, optimizer
        """

        print("!=================================================================================!")
        print("!                                 PARAMETRIZATION                                 !")
        print("!=================================================================================!")
        for system in systems:
            # Perform basic assertions
            self._perform_assertions(settings, system)
            # Create force field optimizable for every system
            system.force_field.create_force_field_optimizable()

        # Create Parameter Space
        if parameter_space is None:
            parameter_space = self.create_parameter_space(settings.parameter_space, settings.restart, systems)
        else:
            assert type(parameter_space) is ParameterSpace

        # Create properties and objective function
        if objective_function is None or parameter_space is None:
            properties = self.create_properties(settings.properties, systems, parameter_space)
            objective_function = self.create_objective_function(settings.objective_function, parameter_space, properties)
        else:
            assert type(objective_function) is ObjectiveFunction
            if settings.objective_function["parallel"]:
                # Number of structures might have been changed and therefore it is necessary to re-initialize
                # the parallel objective function
                objective_function.init_parallel()
            for prop in objective_function.properties:
                if prop.name == "REGULARIZATION":
                    # TODO: if commented, reg in adaptive parametrization is done w.r.t. to the initial parameters at iter 0
                    #prop.set_initial_parameters_values(parameter_space.initial_optimizable_parameters_values_scaled)
                    pass

        # Write restart file
        Task.write_restart_file(parameter_space)

        # Print Initial Info of Objective Function
        objective_function.f(parameter_space.optimizable_parameters_values_scaled, opt_mode=False)

        # Create optimizer
        if optimizer is None:
            optimizer = self.create_optimizer(settings.optimizer["method"],
                                              settings.optimizer[settings.optimizer["method"].lower()])
        else:
            assert type(optimizer) is Optimizer

        # Charge correction
        for system in systems:
            # Apply charge correction
            self._apply_charge_correction(system)
            # Create optimizable force field
            system.force_field.create_force_field_optimizable()
            # Get optimizable parameters
            parameter_space.get_optimizable_parameters()
            # Calculate prior widths, scaling constants and apply jacobi preconditioning (they may have changes if charges changed).
            # Otherwise, we may assume that the change is so small that this has no effect... quite good approximation, hence these lines may be commented
            #parameter_space.calculate_scaling_constants()
            #parameter_space.calculate_prior_widths()
            parameter_space.jacobi_preconditioning()
            # Update the OpenMM context
            parameter_space.update_systems(parameter_space.optimizable_parameters_values_scaled)

        # Perform Optimization
        print("Using {} structures in the optimization.".format(system.n_structures))
        parameters_values = self._perform_optimization(settings, optimizer, objective_function, parameter_space)

        # Print Final Info of Objective Function
        objective_function.f(parameters_values, opt_mode=False)

        # Update the parameters in the force field
        parameter_space.update_systems(parameters_values)

        # Write final system to xml file
        if not adaptive_parametrization:
            for system in systems:
                system.engine.write_system_xml("{}_reparametrized.xml".format(system.name))

        print("!=================================================================================!")
        print("!                   PARAMETRIZATION TERMINATED SUCCESSFULLY :)                    !")
        print("!=================================================================================!")
        return systems, parameter_space, objective_function, optimizer

    # -----------------------------------------------------------#
    #                                                            #
    #                       PRIVATE METHODS                      #
    #                                                            #
    # -----------------------------------------------------------#
    def _perform_assertions(self, settings, system):
        """
        Method that asserts if the parametrization asked by the user contains the necessary data (coordinates, forces, energies, esp).

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of ParaMol System.

        Returns
        -------
        True
        """
        assert system.ref_coordinates is not None, "Conformations data was not set."

        if settings.properties["include_energies"]:
            assert system.ref_energies is not None, "Energies were not set."
        if settings.properties["include_forces"]:
            assert system.ref_forces is not None, "Forces were not set."
        if settings.properties["include_esp"]:
            assert system.ref_esp is not None, "ESP was not set."
            assert system.ref_esp_grid is not None, "ESP was not set."

        return True

    def _get_constraints(self, scipy_method, parameter_space, total_charge=0.0, threshold=1e-8):
        """
        Method that gets the constraints to be passed into the SciPy optimizer.

        Parameters
        ----------
        scipy_method : str
            SciPy method. Should be "COBYLA", SLSQP" or "trust-consr".
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instance of parameter space.
        total_charge : float
            System's total charge
        threshold : float
            Constraint's threshold.

        Returns
        -------
        list
            List with constraints.
        """

        if scipy_method == "COBYLA":
            # Constraint functions must all be >=0 (a single function if only 1 constraint).
            # Each function takes the parameters x as its first argument, and it can return either a single number or an array or list of numbers.

            constraint_vector_charges = [param.multiplicity if param.param_key == "charge" else 0 for param in parameter_space.optimizable_parameters]

            constraints = [{'type': 'ineq', 'fun': lambda x, b: x.dot(np.asarray(b)*parameter_space.scaling_constants_dict["charge"]) - total_charge + threshold, 'args': (constraint_vector_charges,)},
                           {'type': 'ineq', 'fun': lambda x, b: -x.dot(np.asarray(b)*parameter_space.scaling_constants_dict["charge"]) + total_charge + threshold, 'args': (constraint_vector_charges,)}]

            return constraints

        elif scipy_method == "SLSQP":
            # Total charge constraint defined as an equality
            constraint_vector_charges = [param.multiplicity if param.param_key == "charge" else 0 for param in parameter_space.optimizable_parameters]

            constraints = [{'type': 'ineq', 'fun': lambda x, b: x.dot(np.asarray(b)*parameter_space.scaling_constants_dict["charge"]) - total_charge + threshold, 'args': (constraint_vector_charges,)},
                           {'type': 'ineq', 'fun': lambda x, b: -x.dot(np.asarray(b)*parameter_space.scaling_constants_dict["charge"]) + total_charge + threshold, 'args': (constraint_vector_charges,)}]

            return constraints

        elif scipy_method == "trust-constr":
            from scipy.optimize import LinearConstraint
            constraint_vector = [param.multiplicity if param.param_key == "charge" else 0 for param in parameter_space.optimizable_parameters]

            return LinearConstraint(constraint_vector, [total_charge-threshold], [total_charge+threshold])
        else:
            raise NotImplementedError("SciPy method {} does not support constraints.".format(scipy_method))

    def _perform_optimization(self, settings, optimizer, objective_function, parameter_space):
        """
        Method that wraps the functions used to perform the optimization of the parameters.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instance of parameter space.
        objective_function : :obj:`ParaMol.Objective_function.objective_function.ObjectiveFunction`
            Instance of objective function.
        optimizer : :obj:`ParaMol.Optimizers.optimizer.Optimizer`
            Instance of optimizer.

        Returns
        -------
        parameters_values: list
            List of optimized parameters
        """

        # Determine whether to perform constrained or unconstrained optimization
        constrained = False
        for parameter in parameter_space.optimizable_parameters:
            if parameter.param_key == "charge":
                # If charges are present in the optimizable parameters, perform constrained optimization
                constrained = True
                break

        print("Number of parameters to be optimized: {}.".format(len(parameter_space.optimizable_parameters_values_scaled)))
        if constrained:
            print("ParaMol will perform constrained optimization.")
            constraints = self._get_constraints(scipy_method=settings.optimizer["scipy"]["method"],
                                                parameter_space=parameter_space)

            parameters_values = optimizer.run_optimization(f=objective_function.f,
                                                           parameters_values=parameter_space.optimizable_parameters_values_scaled,
                                                           constraints=constraints)
        else:
            print("ParaMol will perform unconstrained optimization.")
            parameters_values = optimizer.run_optimization(f=objective_function.f,
                                                           parameters_values=parameter_space.optimizable_parameters_values_scaled)

        return parameters_values

    def _apply_charge_correction(self, system):
        """
        Method that applies charge correction to the system.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of ParaMol System.
        
        Notes
        ----
        Due to numerical errors, the total charge of the system may not be equal to the real total charge of the system.
        Hence, in order to overcome this problem, which causes unexpected behaviour specially when constraints are being applied, the excess or deficiency of charge is shared equally amongst all atoms. This usually changes the charge in each atom by a very small amount.
        Note that this method only changes the charges in the ParaMol ForceField of the ParaMolSystem. Therefore, it is required to update the OpenMM systems after this method is called.
        
        Returns
        -------
        total_charge : float
            Final total charge of the system.
        """
        if "NonbondedForce" in system.force_field.force_field:
            # Get total charge and calculate charge correction
            total_charge = self._get_total_charge(system)
            logging.info("Applying charge correction.")
            logging.info("Total charge before correction: {}e .".format(total_charge))
            charge_correction = total_charge / system.n_atoms
            logging.info("Charge correction {}e per atom.".format(charge_correction))

            # Add charge correction to all atoms
            for nonbonded_term in system.force_field.force_field["NonbondedForce"]:
                nonbonded_term.parameters["charge"].value -= charge_correction

            total_charge = self._get_total_charge(system)
            logging.info("Total charge after correction: {}e .\n".format(total_charge))

            return total_charge

        else:
            logging.info("Not applying charge correction.")
            return 1
    # -----------------------------------------------------------#
    #                                                            #
    #                       STATIC METHODS                       #
    #                                                            #
    # -----------------------------------------------------------#
    @staticmethod
    def _get_total_charge(system):
        """
        Method that gets the system's total charge as in the ParaMol ForceField of the ParaMolSystem.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of ParaMol System.

        Returns
        -------
        total_charge : float
            Final total charge of the system.
        """
        total_charge = 0.0
        if "NonbondedForce" in system.force_field.force_field:
            for nonbonded_term in system.force_field.force_field["NonbondedForce"]:
                total_charge += nonbonded_term.parameters["charge"].value

        return total_charge

