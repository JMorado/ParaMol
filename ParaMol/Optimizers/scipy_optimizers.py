# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Optimizers.scipy_optimizers.ScipyOptimizer` class, which is the ParaMol wrapper of the SciPy optimizers.
"""


class ScipyOptimizer:
    """
    ParaMol wrapper of the SciPy optimizers.

    Notes
    ------
    This class is mainly a wrapper around the minimize SciPy function.
    For more information see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
    Tested optimizers:
    scipy_constrained_methods = ['trust-constr', 'SLSQP', 'COBYLA']
    scipy_unconstrained_methods = ['Powell', 'BFGS', 'Nelder-Mead']

    Parameters
    ----------
    **minimizer_params : dict
        Keyword arguments passed to scipy.optimize.minimize function (except the objective function callable, method name and x0, i.e., the parameter's array).

    Attributes
    ----------
    **minimizer_params : dict
    Keyword arguments passed to scipy.optimize.minimize function (except the objective function callable, method name and x0, i.e., the parameter's array).
    """

    def __init__(self, **minimizer_params):
        self.__dict__.update(**minimizer_params)

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_optimization(self, f, parameters, constraints=None):
        """
        Method that runs a SciPy optimization.

        Parameters
        ----------
        f: callable
            Reference to the objective function.
        parameters: list
            1D list with the adimensional mathematical parameters that will be used in the optimization.
        constraints: list of constraints.
            Constraints to apply.

        Returns
        -------
        parameters: list
            1D list with the updated adimensional mathematical parameters.
        """
        from scipy.optimize import minimize

        print("!=================================================================================!")
        print("!                           STARTING SCIPY OPTIMIZER                              !")
        print("!=================================================================================!")

        if constraints is None:
            # Perform unconstrained optimization
            optimization = minimize(fun=f, x0=parameters, **self.__dict__)
        else:
            # Perform constrained optimization
            optimization = minimize(fun=f, x0=parameters, constraints=constraints, **self.__dict__)

        print("!=================================================================================!")
        print("!                      SCIPY OPTIMIZER TERMINATED SUCCESSFULLY! :)                !")
        print("!=================================================================================!")
        return optimization.x


