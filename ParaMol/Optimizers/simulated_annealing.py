# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Optimizers.simulated_annealing.SimulatedAnnealing` class, which is the ParaMol implementation of the Monte Carlo method.
"""


import copy
import numpy as np


class SimulatedAnnealing:
    """
    ParaMol implementation of the Simulated Annealing optimizer.

    Parameters
    ----------
    Parameters
    ----------
    n_iter : float
        Number of iterations to perform in total.
    p_init : float
        Probability of accepting worse solution at the beginning. The initial temperature is given by :math:`-1/log(p_init)`.
    p_final
        Probability of accepting worse solution at the end. The final temperature is given by :math:`-1/log(p_{final})`.
    avg_acceptance_rate : float
        Average acceptance rate to aim to.
        If at the start of a new MC block the acceptance rate of a given parameter is larger (lower) than `prob`, the maximum displacement for that parameter is increased (decreased).
    """
    def __init__(self, n_iter, p_init, p_final, avg_acceptance_rate):
        self._n_iter = n_iter
        self._p_init = p_init
        self._p_final = p_final
        self._avg_acceptance_rate = avg_acceptance_rate

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_optimization(self, f, parameters, constraints=None):
        """
        Method that performs optimization using the simulated annealing method.

        Notes
        -----
        Source:
        #TODO. include source

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
        assert constraints is None, "Simulated Annealing optimizer cannot handle restraints."

        print("!=================================================================================!")
        print("!                      STARTING SIMULATED ANNEALING OPTIMIZER                     !")
        print("!=================================================================================!")

        # Initial temperature
        t_init = - 1.0 / np.log(self._p_init)
        # Final temperature
        t_final = - 1.0 / np.log(self._p_final)
        # Fractional reduction every cycle
        frac = (t_final / t_init) ** (1.0 / (self._n_iter - 1.0))

        temp = t_init

        # Choose random seed for the process
        np.random.seed(np.random.randint(2 ** 32 - 1))
        n_param = len(parameters)

        # First objective function minimization
        error, old_f = f(parameters)
        best_f = old_f
        best_parameters = copy.deepcopy(parameters)

        for ntemp in range(self._n_iter):
            # Initiate another MC optimization at a given temperature
            acc = [0.0 for p in range(n_param)]
            p_max = [copy.deepcopy(parameters[p]) * 0.5 for p in range(n_param)]
            print("Starting new temperature...")
            for sweep in range(1, 100):
                p_max = [p_max[p] * ((acc[p] / float(sweep) - self._avg_acceptance_rate) + 1) for p in range(n_param)]

                # parameters_temp = copy.deepcopy(parameters)
                for n in range(n_param):
                    # Create neighbour solution
                    p = np.random.randint(0, n_param)  # Select randomly a parameter
                    p_dummy = copy.deepcopy(parameters[p])
                    parameters[p] += np.random.uniform(-p_max[p], p_max[p])
                    error, new_f = f(parameters)
                    delta_f = new_f - old_f
                    if delta_f < 0:
                        if new_f < best_f:
                            best_f = new_f
                            best_parameters = copy.deepcopy(parameters)

                        old_f = new_f
                        acc[p] += 1.0

                        print("\nMC move accepted (delta_f < 0).")
                        print("Error: ", error)
                        print("Objective function value: {}".format(new_f))
                    else:
                        prob = np.exp(- (new_f - old_f) / temp)
                        if prob > np.random.random():
                            old_f = new_f
                            acc[p] += 1.0

                            print("\n MC move accepted (Metropolis).")
                            print("Error: ", error)
                            print("Objective function value: {}".format(new_f))
                        else:
                            parameters[p] = p_dummy

                # print(np.sqrt(np.sum(( (np.asarray(parameters_temp)-np.asarray(parameters)) /  np.asarray(parameters_temp) )**2) / n_param))
            # Lower the temperature for next cycle
            temp = temp * frac

        print("Acceptance rate: " + str(sum(acc) / ((sweep) * n_param)))
        print("Convergence was achieved after {} MC sweeps.".format(sweep))
        print("Last objective function value is {} .".format(new_f))
        print("!=================================================================================!")
        print("!               SIMULATED ANNEALING OPTIMIZER TERMINATED SUCCESSFULLY! :)         !")
        print("!=================================================================================!")

        return best_parameters


