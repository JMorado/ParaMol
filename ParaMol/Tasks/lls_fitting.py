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
from ..Parameter_space.parameter_space import *
from ..Utils.interface import *
from ..MM_engines.least_square import *


# ------------------------------------------------------------
#                                                            #
#                       PARAMETRIZATION TASK                 #
#                                                            #
# ------------------------------------------------------------
class LLSFitting(Task):
    """
    ParaMol Linear Least Square Fitting task.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------- #
    #                                                            #
    #                       PUBLIC METHODS                       #
    #                                                            #
    # ---------------------------------------------------------- #
    def run_task(self, settings, systems, parameter_space=None, interface=None, adaptive_parametrization=False, restart=False):
        """
        Method that performs the standard ParaMol parametrization.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instances of ParameterSpace.
        interface: :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol system instance.
        adaptive_parametrization: bool
            Flag that signals if this parametrization is being done inside a an adaptive parametrization loop. If `False` the system's xml file is not written in this method (default is `False`).
        restart : bool
            Flag that controls whether or not to perform a restart.

        Returns
        -------
        systems, parameter_space, objective_function, optimizer
        """

        print("!=================================================================================!")
        print("!                               LLS ENERGY FITTING                                !")
        print("!=================================================================================!")
        for system in systems:
            # Perform basic assertions
            self._perform_assertions(settings, system)
            # Create force field optimizable for every system
            system.force_field.create_force_field_optimizable()

        # Create IO Interface
        if interface is None:
            interface = ParaMolInterface()
        else:
            assert type(interface) is ParaMolInterface

        # Create ParameterSpace
        if parameter_space is None:
            parameter_space = self.create_parameter_space(settings, systems, interface, restart=restart, preconditioning=False, symmetry_constrained=False)
        else:
            assert type(parameter_space) is ParameterSpace

        lst_sqr = LinearLeastSquare(parameter_space, settings.properties["include_regularization"], **settings.properties["regularization"], **settings.objective_function)
        parameters_values = lst_sqr.fit_parameters_lls(systems)

        # Update the parameters in the ParaMol Force Field and in the Engine
        # Symmetry constraint has to be False
        parameter_space.update_systems(systems, parameters_values, symmetry_constrained=False)

        # Write ParameterSpace restart file
        self.write_restart_pickle(settings.restart, interface, "restart_parameter_space_file", parameter_space.__dict__)

        # Write final system to xml file
        if not adaptive_parametrization:
            for system in systems:
                system.engine.write_system_xml("{}_reparametrized.xml".format(system.name))

        print("!=================================================================================!")
        print("!                   LLS ENERGY FITTING TERMINATED SUCCESSFULLY :)                 !")
        print("!=================================================================================!")
        return systems, parameter_space

    @staticmethod
    def _perform_assertions(settings, system):
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
            raise NotImplementedError("Currently forces cannot be fitted using LLS.")
            #assert system.ref_forces is not None, "Forces were not set."
        if settings.properties["include_esp"]:
            assert system.ref_esp is not None, "ESP was not set."
            assert system.ref_esp_grid is not None, "ESP was not set."

        return True