# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Objective_function.cpu_objective_function.CPUObjectiveFunction` class used by ParaMol to perform parallel evaluation of the objective function using CPUs.
"""

import multiprocessing as mp
import numpy as np

# Shared memory variables
shared_dict = {}


class CPUObjectiveFunction:
    """
    ParaMol's wrapper for the CPU parallel callback function.

    Attributes
    ----------
    X : list
        List containing conformations data.
    calculate_energies : bool
        Flag that controls whether or not the energies will be calculated.
    calculate_energies : bool
        Flag that controls whether or not the forces will be calculated.
    n_atoms : int
        Number of atoms of the system-

    Notes
    ------
    Unix implementation of multiprocessing uses forks. This will not work under Windows.
    When running under Unix, all workers may share the same object, due to how fork works (i.e., the child processes have separate memory but it's copy-on-write, so it may be shared as long as nobody modifies it).
    In copy-on-write the fundamental idea is that if multiple callers ask for resources which are initially indistinguishable, you can give them pointers to the same resource. This function can be maintained until a caller tries to modify its "copy" of the resource, at which point a true private copy is created to prevent the changes becoming visible to everyone else. All of this happens transparently to the callers. The primary advantage is that if a caller never makes any modifications, no private copy need ever be created.
    """
    def __init__(self):
        """

        """
        self.X = None
        self.batch_lims = None
        self.n_atoms = None
        self.calculate_energies = None
        self.calculate_forces = None
        self._lock = None

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    @staticmethod
    def init_worker():
        """
        Method that is the constructor for the Pool of workers.
        It Contains a global dictionary with variables shared between threads.

        Returns
        -------
        None
        """

        global shared_dict
        shared_dict['lock'] = mp.Lock()

    def f_callback_function(self, context, system_id, worker_id):
        """
        Method that may be used as a callback function for the parallel computation of the objective function using CPUs.

        Parameters
        ----------
        context: :obj:`ParaMol.Objective_function.pickable_swig.PickalableContext`
            Pickalable OpenMM context.
        system_id : int
            Index of the OpenMM system copy.
        worker_id : int
            Index of the parallel worker.

        Returns
        -------
        emm_data, fmm_data: np.array(batch_size), np.array(batch_size, n_atoms, 3)
            Arrays containing MM energies and forces.
        """
        conformations = self.X[system_id][self.batch_lims[system_id][worker_id][0]:self.batch_lims[system_id][worker_id][1], :, :]
        batch_size = conformations.shape[0]
        fmm_data = np.zeros((batch_size, self.n_atoms[system_id], 3))
        emm_data = np.zeros((batch_size))

        # Perform necessary computation (energies and forces, only forces or only energies)
        if self.calculate_energies[system_id] and self.calculate_forces[system_id]:
            for i in range(batch_size):
                context.setPositions(conformations[i])
                state = context.getState(getEnergy=True, getForces=True)
                emm_data[i] = state.getPotentialEnergy()._value
                fmm_data[i, :, :] = state.getForces(asNumpy=True)._value
        elif self.calculate_forces[system_id]:
            for i in range(batch_size):
                context.setPositions(conformations[i])
                state = context.getState(getForces=True)
                fmm_data[i, :, :] = state.getForces(asNumpy=True)._value
        elif self.calculate_energies[system_id]:
            for i in range(batch_size):
                context.setPositions(conformations[i])
                state = context.getState(getEnergy=True)
                emm_data[i] = state.getPotentialEnergy()._value

        return emm_data, fmm_data




"""
Depracted code
        #This method calculates the classical potential energy and forces for a given context.

        try:
            lock = shared_dict['lock']
            lock.acquire()
            conformations = self._X[self._batch_lims[worker_id][0]:self._batch_lims[worker_id][1], :, :]
            batch_size = conformations.shape[0]
            fmm_data = np.zeros((batch_size, self._n_atoms, 3))
        except Exception as error:
            print("Could not access shared variables {}".format(error))
        finally:
            lock.release()        

        emm_data = np.zeros((batch_size))
        for i in range(batch_size):
            context.setPositions(conformations[i])
            state = context.getState(getEnergy=True, getForces=True)
            emm_data[i] = state.getPotentialEnergy()._value
            fmm_data[i, :, :] = state.getForces(asNumpy=True)._value

        return fmm_data, emm_data
"""
