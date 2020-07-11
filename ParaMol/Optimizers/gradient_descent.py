# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Optimizers.gradient_descent.GradientDescent` class, which is the ParaMol implementation of the gradient descent method.
"""
import numpy as np
import copy


class GradientDescent:
    """
    ParaMol implementation of the gradient descent optimizer.

    Parameters
    ----------
    max_iter : float
        Maximum number of iterations.
    derivative_calculation : str
        When to calculate the derivatives, which are the most computational expensive part of the algorithm.
        A fair approximation is to re-compute the derivative only if the value of the objective function has increased in successive iterations. Available options are "f-increase" and "always".
    derivative_type : str
        Type of numerical differentiation to perform. Available options are "1-point" or "2-point".
    g_tol : float
        Threshold that defines when the gradient is deemed to be converged, i.e., if the change in the gradient is lower than this threshold than we assume convergence has been reached.
    f_tol : float
        Threshold that defines when the objective function is deemed to be converged, i.e., if the change in the objective function is lower than this threshold than we assume convergence has been reached.
    dx : float
        Change in x for the numerical differentiation (denominator).
    derivative_h : float
        Scaling  factor  that multiplies the  step  size  of  the  descent.
    """
    def __init__(self, max_iter, derivative_calculation, derivative_type, g_tol, f_tol, dx, derivative_h):
        self._max_iter = max_iter
        self._derivative_calculation = derivative_calculation
        self._derivative_type = derivative_type
        self._g_tol = g_tol
        self._f_tol = f_tol
        self._dx = dx
        self._derivative_h = derivative_h

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_optimization(self, f, parameters, constraints=None):
        """
        Method that performs optimization using the gradient descent method.

        Notes
        -----
        When the number of conformations being used in the optimization is large (>= 1000) it becomes prohibitive
        to numerically calculate the derivatives every iteration due to the requirement of calculating the
        objective function. A fair approximation for this situations is to re-compute the derivative only if the
        value of the objective function has increased in successive iterations.

        Source:
        'Developing Consistent Molecular Dynamics Force Fields for Biological Chromophores via Force Matching'
        K. Claridge and Troisi A.
        J. Phys. Chem. B 2019, 123, 2, 428–438

        Parameters
        ----------
        f: callable
            Reference of the objective function.
        parameters: list
            1D list with the adimensional mathematical parameters that will be used in the optimization.
        constraints: None
            Should be None. Monte Carlo optimizer cannot handle restraints.

        Returns
        -------
        parameters:
            1D list with the updated adimensional mathematical parameters.
        """

        assert constraints is None, "Gradient Descent cannot handle restraints."

        print("!=================================================================================!")
        print("!                        STARTING GRADIENT DESCENT OPTIMIZER                      !")
        print("!=================================================================================!")

        parameters = np.asarray(parameters)
        grad = np.ones(len(parameters))
        dp = np.ones(len(parameters)) * self._dx
        alpha = np.ones(len(parameters)) * self._derivative_h

        param_min = copy.deepcopy(parameters)
        f_best = f(parameters)
        f_new = f_best
        f_old = 0.0

        iteration = 0
        while np.linalg.norm(grad) > self._g_tol or iteration < self._max_iter:
            # Increment iteration counter
            iteration = iteration + 1

            # Only compute derivatives of the objective function
            # with respect to each parameters if f has increased in the previous iteration
            parameters_dummy = copy.deepcopy(parameters)

            if self._derivative_calculation == "always" or (f_new > f_old and self._derivative_calculation == "f_increase"):
                # Compute the gradient:
                if self._derivative_type == "1-point":
                    for i in range(len(parameters)):
                        parameters[i] += dp[i]
                        f_plus = f(parameters)
                        parameters[i] -= dp[i]
                        grad[i] = (f_plus - f_new) / (dp[i])

                elif self._derivative_type == "2-point":
                    # 2-point derivative
                    for i in range(len(parameters)):
                        parameters[i] += dp[i]
                        f_plus = f(parameters)
                        parameters[i] -= 2 * dp[i]
                        f_minus = f(parameters)
                        parameters[i] += dp[i]
                        grad[i] = (f_plus - f_minus) / (dp[i])
                else:
                    raise NotImplementedError("derivate_type {} is not implemented".format(self._derivative_type))

            parameters = parameters - alpha * grad

            # Compute new objective function value
            f_old = f_new
            f_new = f(parameters)

            # Check if a best parameter set has been found
            if f_new < f_best:
                if (f_best - f_new) < self._f_tol:
                    break
                # A new best solution was found
                print("Iter: {} F: {:<15.8f} Gradient norm: {:<15.8f} deltaF: {:<15.8f}".format(iteration, f_new, np.linalg.norm(grad), -(f_new-f_best)))
                param_min = copy.deepcopy(parameters)
                f_old = f_new
                f_best = f_new
            else:
                parameters = parameters_dummy

        print("!=================================================================================!")
        print("!                GRADIENT DESCENT OPTIMIZER TERMINATED SUCCESSFULLY! :)           !")
        print("!=================================================================================!")

        return param_min




