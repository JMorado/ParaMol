# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Tasks.resp.RESPFitting` class, which is a ParaMol task that performs fitting of the electrostatic potential.
"""
import logging

from .task import *
from .parametrization import *
from ..MM_engines.resp import *
from ..Utils.interface import *


# ------------------------------------------------------------ #
#                                                              #
#                          RESP Task                           #
#                                                              #
# ------------------------------------------------------------ #
class RESPFitting(Task):
    """
    ParaMol RESP fitting task.
    """
    resp_solvers = ["EXPLICIT", "SCIPY"]

    def __init__(self):
        pass

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_task(self, settings, systems, parameter_space=None, objective_function=None, optimizer=None, interface=None, solver="SCIPY", total_charge=None, constraint_tolerance=1e-6):
        """
        Method that performs a RESP calculation.

        Notes
        -----
        Only one ParaMol system is supported at once.

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
        interface: :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol system instance.
        solver : str
            RESP solver. Options are "EXPLICTI" or "SCIPY" (default is "SCIPY").
        total_charge : int
            System's total charge (default is `None`).
        constraint_tolerance : float
            Tolerance used to impose total charge or symmetry constraints. Only used if `solver` is "SCIPY" (default is 1e-6).

        Returns
        -------
        systems, parameter_space, objective_function, optimizer
        """
        print("!=================================================================================!")
        print("!                               RESP CHARGE FITTING                               !")
        print("!=================================================================================!")

        assert total_charge is not None, "System's total charge was not specified."
        assert len(systems) == 1, "RESP task currently only supports parametrization of one system at once."

        if solver.lower() == "scipy":
            logging.info("ParaMol will solve fit to ESP using a SciPy optimimzer.")

            for system in systems:
                system.resp_engine = RESP(total_charge=total_charge,
                                          constraint_tolerance=constraint_tolerance)

                system.resp_engine.calculate_inverse_distances(system)

            # Perform parametrization
            parametrization = Parametrization()
            systems, parameter_space, objective_function, optimizer = parametrization.run_task(settings=settings,
                                                                                               systems=systems,
                                                                                               parameter_space=parameter_space,
                                                                                               objective_function=objective_function,
                                                                                               optimizer=optimizer,
                                                                                               interface=interface,
                                                                                               adaptive_parametrization=False)

        elif solver.lower() == "explicit":
            logging.info("ParaMol will solve RESP equations explicitly.")

            # Create interface
            if interface is None:
                interface = ParaMolInterface()
            else:
                assert type(interface) is ParaMolInterface

            # Create Parameter Space
            if parameter_space is None:
                parameter_space = self.create_parameter_space(settings, systems, preconditioning=False)
            else:
                assert type(parameter_space) is ParameterSpace

            # Create RESP object instance
            for system in systems:
                assert system.ref_esp is not None
                assert system.ref_coordinates is not None
                assert system.ref_esp_grid is not None

                system.resp_engine = RESP(total_charge=total_charge)

                # Set initial/current charges
                system.resp_engine.set_initial_charges(system.force_field.force_field)
                system.resp_engine.set_charges(system.force_field.force_field)

                # Calculate 1/r_{ij} matrix before the RESP procedure
                system.resp_engine.calculate_inverse_distances(system)
                # Set symmetry constraints
                system.resp_engine.set_symmetry_constraints(system, True)
                # Perform RESP charge fitting
                charges = system.resp_engine.fit_resp_charges_explicitly(system)

                # Update system
                parameter_space.update_systems(systems, charges)
        else:
            raise NotImplementedError("RESP solver {} is not implemented.".format(solver))

        print("!=================================================================================!")
        print("!                     RESP FITTING TERMINATED SUCCESSFULLY :)                     !")
        print("!=================================================================================!")
        return systems, parameter_space, objective_function, optimizer

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PRIVATE METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #

