# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Optimizers.monte_carlo.MonteCarlo` class, which is the ParaMol implementation of the Monte Carlo method.
"""


class MonteCarlo:
    """
    ParaMol implementation of the Monte Carlo optimizer.

    Parameters
    ----------
    n_blocks : int
        Number of iteration (attempted moves) per block. Within each block the maximum displacements are not updated.
    max_iter : int
        Maximum number of iterations.
    f_tol : float
        Threshold that defines when the objective function is deemed to be converged, i.e., if the change in the objective function is lower than this threshold than we assume convergence has been reached.
    avg_acceptance_rate : float
        Acceptance rate threshold.
        If at the start of a new MC block the acceptance rate of a given parameter is larger (lower) than `avg_acceptance_rate`, the maximum displacement for that parameter is increased (decreased).
    """
    def __init__(self,  n_blocks, max_iter, f_tol, avg_acceptance_rate):
        self._n_blocks = n_blocks
        self._max_iter = max_iter
        self._f_tol = f_tol
        self._avg_acceptance_rate = avg_acceptance_rate

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_optimization(self, f, parameters, constraints=None):
        """
        Method that performs optimization using the Monte Carlo method.

        Notes
        -----
        Source:
        'Developing accurate molecular mechanics force fields for conjugated molecular systems'
        Do H. and Troisi A.
        PCCP 2015, 17, 25123-25132

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
        parameters: list
            1D list with the updated adimensional mathematical parameters.
        """
        import numpy as np
        import copy
        assert constraints is None, "Monte Carlo optimizer cannot handle restraints."

        print("!=================================================================================!")
        print("!                           STARTING MONTE CARLO OPTIMIZER                        !")
        print("!=================================================================================!")

        np.random.seed(np.random.randint(2 ** 32 - 1))
        n_param = len(parameters)

        # Acceptance rates
        acc = [0.0 for _ in range(n_param)]

        # Range
        p_max = [copy.deepcopy(parameters[p]) * 0.5 if abs(parameters[p]) > 1e-3 else 1 for p in range(n_param)]

        old_f = f(parameters)
        block_counter = 0

        for sweep in range(1, self._max_iter):
            if block_counter == self._n_blocks:
                # Update max displacement
                p_max = [p_max[p] * ((acc[p] / float(sweep) - self._avg_acceptance_rate) + 1) for p in range(n_param)]
                block_counter = 0

            block_counter += 1

            for n in range(n_param):
                # Select randomly a parameter
                p = np.random.randint(0, n_param)

                # Keep the old parameters
                p_dummy = copy.deepcopy(parameters[p])

                # Add displacement
                parameters[p] += np.random.uniform(-p_max[p], p_max[p])

                # Compute objective function
                new_f = f(parameters)

                if new_f < old_f:
                    acc[p] += 1.0

                    print("MC move accepted. Objective function value: {}".format(new_f))

                    if abs(new_f - old_f) < self._f_tol:
                        print("\nFinal Acceptance rate: " + str(sum(acc) / ((sweep) * n_param)))
                        print("Convergence was achieved after {} MC sweeps.".format(sweep))
                        print("Last objective function value is {} .".format(new_f))
                        print("!=================================================================================!")
                        print("!                MONTE CARLO OPTIMIZER TERMINATED SUCCESSFULLY! :)                !")
                        print("!=================================================================================!")
                        return parameters

                    old_f = new_f
                else:
                    parameters[p] = p_dummy

        print("Final Acceptance rate: " + str(sum(acc) / (sweep * n_param)))
        print("Maximum number of iterations was reached.")
        print("!=================================================================================!")
        print("!                MONTE CARLO OPTIMIZER TERMINATED SUCCESSFULLY! :)                !")
        print("!=================================================================================!")

        return parameters


