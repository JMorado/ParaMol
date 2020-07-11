# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Optimizers.optimizer.Optimizer` class, which is the main Optimizer class.
"""


class Optimizer:
    """
    ParaMol wrapper of the optimization methods.

    Notes
    -----
    This class is a wrapper for the currently implemented optimization methods, viz. "scipy", "monte_carlo", "simulated_annealing" and "gradient_descent".
    It creates the optimizer when called if `create_optimizer` is set to `True`.


    Parameters
    ----------
    settings : dict
        Dictionary containing the optimizer settings.
    method : str
        Name of the optimizer to be created. Available optimizers are "monte_carlo", "scipy", "simulated_annealing", "gradient_descent" and "bayesian" (still being developed, not recommended).
    create_optimizer : bool
        Flag that determines whether or not an instance of the available optimizers is created.

    Attributes
    ----------
    method_name : str
        Name of the created optimizer.
    settings : dict
        Dictionary containing the optimizer settings.
    """
    optimizers = ["scipy",
                  "monte_carlo",
                  "simulated_annealing",
                  "gradient_descent",
                  "bayesian"]

    def __init__(self, method, settings, create_optimizer=True):
        # Optimizer settings
        self.settings = settings
        self.method_name = None

        # Private optimizer
        self._optimizer = None
        if create_optimizer:
            self._create_optimizer(method, settings)

    def __str__(self):
        return "Instance of optimizer {}.".format(self.method_name)

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_optimization(self, f, parameters_values, constraints=None):
        """
        Method to run the parameter's optimization per se.

        Parameters
        ----------
        f : callable
            Reference to the objective function method.
        parameters_values : list
            1D list containing the values of the parameters that will be optimized.
        constraints : list of constraints
            Constraints to be applied during the optimization.

        Returns
        -------
        pameters_values : list
            List containing the optimized parameter values
        """
        parameters_values = self._optimizer.run_optimization(f, parameters_values, constraints)

        return parameters_values

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PRIVATE METHODS                     #
    #                                                              #
    # ------------------------------------------------------------ #
    def _create_optimizer(self, method, settings):
        """
        Method that creates an instance of a chosen optimizer.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        method : str
            Name of the optimizer to be created. Available optimizers are "monte_carlo", "scipy", "simulated_annealing", "gradient_descent" and "bayesian" (still being developed, not recommended).

        Returns
        -------
        optimizer : any optimizer defined in the subpackage :obj:`ParaMol.Optimizers`
            Instance of the created optimizer.
        """
        assert method.lower() in self.optimizers

        if method.lower() == "monte_carlo":
            import ParaMol.Optimizers.monte_carlo as mc
            self._optimizer = mc.MonteCarlo(**settings)

        elif method.lower() == "simulated_annealing":
            import ParaMol.Optimizers.simulated_annealing as sa
            self._optimizer = sa.SimmulatedAnnealing(**settings)

        elif method.lower() == "scipy":
            import ParaMol.Optimizers.scipy_optimizers as scipy_opt
            self._optimizer = scipy_opt.ScipyOptimizer(**settings)

        elif method.lower() == "gradient_descent":
            import ParaMol.Optimizers.gradient_descent as gd
            self._optimizer = gd.GradientDescent(**settings)

        elif method.lower() == "bayesian":
            import ParaMol.Optimizers.devel.bayesian as bayesian
            print("WARNING! Bayesian optimizer is still under development.")
            self._optimizer = bayesian.Bayesian(**settings)

        self.method_name = settings["method"].lower()

        return self._optimizer

