# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Tasks.objective_function_plot.ObjectiveFunctionPlot` class, which is a ParaMol task that plots the objective function.
"""

# ParaMol modules
from .task import *
from .ab_initio_properties import *
from ..Parameter_space.parameter_space import *


# ---------------------------------------------------------- #
#                                                            #
#                  OBJECTIVE FUNCTION PLOT                   #
#                                                            #
# ---------------------------------------------------------- #
class ObjectiveFunctionPlot(Task):
    """
    ParaMol task that plots the objective function.
    """
    def __init__(self):
        pass

    # ---------------------------------------------------------- #
    #                                                            #
    #                       PUBLIC METHODS                       #
    #                                                            #
    # ---------------------------------------------------------- #
    def run_task(self, settings, systems, grid_1, grid_2=None, parameter_space=None, objective_function=None, write_data=False, plot=True):
        """
        This method can be used to generate plots of the objective function.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        grid_1 : list or np.array
            Array with values of parameter 1.
        grid_2 : list or np.array
            Array with values of parameter 2.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instance of parameter space. (default is `None`).
        objective_function : :obj:`ParaMol.Objective_function.objective_function.ObjectiveFunction`
            Instance of ` (default is `None`).
        write_data : bool
            Flag that signal whether the data is going to be written to a file.
        plot : bool
            Flag that signal wheter to plot the data using matplotlib.

        Returns
        -------
        data : np.array
            Objective function plot data
        """
        import numpy as np

        print("!=================================================================================!")
        print("!                           Objective Function Plot                               !")
        print("!=================================================================================!")

        original_ff = []
        # Basic assertions
        for system in systems:
            assert system.ref_coordinates is not None, "Conformations data was not set for system {}.".format(system.name)
            original_ff.append(copy.deepcopy(system.force_field.force_field))

            if system.ref_forces is None or system.ref_energies is None:
                ab_initio = AbInitioProperties()
                ab_initio.run_task([system])

        # Create Parameter Space
        if parameter_space is None:
            parameter_space = self.create_parameter_space(settings.parameter_space, settings.restart, systems, preconditioning=True)
        else:
            assert type(parameter_space) is ParameterSpace

        # Create properties and objective function
        if objective_function is None or parameter_space is None:
            properties = self.create_properties(objective_function_settings=settings.properties,
                                                systems=systems,
                                                parameter_space=parameter_space)

            objective_function = self.create_objective_function(objective_function_settings=settings.objective_function,
                                                                parameter_space=parameter_space,
                                                                properties=properties)
        else:
            assert type(objective_function) is ObjectiveFunction

        print("Parameter 1: {}.".format(parameter_space.optimizable_parameters[0]))
        print("Grid parameter 1: \n {}".format(grid_1))
        if grid_2 is not None:
            print("Parameter 2: {}.".format(parameter_space.optimizable_parameters[1]))
            print("Grid parameter 2: \n {}".format(grid_2))

        data = []
        # Scale grid
        grid_1 = np.asarray(grid_1)
        grid_1 = grid_1

        # --------------------------------------- #
        #              1D SCAN                    #
        # --------------------------------------- #
        if grid_2 is None:
            for x in grid_1:
                parameter_space.optimizable_parameters[0].value = x

                # Get optimizable parameters after change
                parameter_space.get_optimizable_parameters()
                parameter_space.jacobi_preconditioning()
                parameter_space.update_systems(parameter_space.optimizable_parameters_values_scaled)

                # Print Initial Info of Objective Function
                f_val = objective_function.f(parameter_space.optimizable_parameters_values_scaled, opt_mode=True)
                data.append([x, f_val])
                print("{:^17.8f}  {:^17.8f}".format(x,  f_val))

            #self._write_data()
            data = np.asarray(data)
            if plot:
                self._plot(data[:, 0], data[:, 1])

        # --------------------------------------- #
        #              2D SCAN                    #
        # --------------------------------------- #
        else:
            # Scale grid
            grid_2 = grid_2
            grid_2 = np.asarray(grid_2)

            #X, Y = np.meshgrid(grid_1, grid_2)

            for x in grid_1:
                parameter_space.optimizable_parameters[0].value = x
                for y in grid_2:
                    parameter_space.optimizable_parameters[1].value = y

                    # Get optimizable parameters after change
                    parameter_space.get_optimizable_parameters()
                    parameter_space.jacobi_preconditioning()
                    parameter_space.update_systems(parameter_space.optimizable_parameters_values_scaled)

                    # Print Initial Info of Objective Function
                    f_val = objective_function.f(parameter_space.optimizable_parameters_values_scaled, opt_mode=True)
                    data.append([x *  parameter_space.scaling_constants[0], y * parameter_space.scaling_constants[1], f_val])
                    print("{:^17.8f} {:^17.8f} {:^17.8f}".format(x, y, f_val))

            #self._write_data(data)
            data = np.asarray(data)
            if plot:
                self._plot(data[:, 0], data[:, 1],data[:, 2])

        # Re-set original parameters in ParaMol's ForceField representation and in OpenMM engine
        for system_idx in range(len(systems)):
            # ParaMol Force Field
            systems[system_idx].force_field.force_field = original_ff[system_idx]
            systems[system_idx].force_field.create_force_field_optimizable()
            # OpenMM Engine
            systems[system_idx].engine.set_nonbonded_parameters(original_ff[system_idx])
            systems[system_idx].engine.set_bonded_parameters(original_ff[system_idx])

        print("!=================================================================================!")
        print("!                   Objective Function Plot Completed Successfully                !")
        print("!=================================================================================!")
        return data

    # ---------------------------------------------------------- #
    #                                                            #
    #                      PRIVATE METHODS                       #
    #                                                            #
    # ---------------------------------------------------------- #
    @staticmethod
    def _write_data(data=None):
        pass

    @staticmethod
    def _plot(x, y, z=None):
        """
        Method that plots the objective function.
        Parameters
        ----------
        x : list or np.array
            x points.
        y : list or np.array
            y points.
        z : list or np.array
            z points.

        Returns
        -------
        `None`

        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d

        if z is None:
            plt.plot(x, y)
            plt.ylabel("Objective function")
            plt.xlabel("Parameter 1")
            plt.show()
        else:
            ax = plt.axes(projection='3d')
            ax.scatter(x, y, z)
            plt.show()

        return
