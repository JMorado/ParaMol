# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Objective_function.cpu_objective_function.GPUObjectiveFunction` class used by ParaMol to perform parallel evaluation of the objective function using CPUs.
"""
import multiprocessing as mp
import numpy as np
from simtk.openmm import *

# Shared memory variables
shared_dict = {}


class GPUObjectiveFunction:
    """
    ParaMol's wrapper for the GPU parallel callback function.

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
    Only usable with OpenMM's 'OpenCL' platform.
    """
    def __init__(self):
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
        pass

    def f_callback_function(self, system, integrator, system_id, worker_id):
        """
        Method that may be used as a callback function for the parallel computation of the objective function using GPUs.

        Parameters
        ----------
        system: :obj:`ParaMol.Objective_function.pickable_swig.PickalableContext`
            OpenMM system.
        integrator : any OpenMM integrator object
            OpenMM integrator.
        system_id : int
            Index of the OpenMM system copy.
        worker_id : int
            Index of the parallel worker.

        Notes
        ------
        Unfortunately, for GPUs, context has to be created here, which slows down the computation as it is a costly operation.
        Only 'OpenCL' was tested. 'CUDA' platform hasn't been tested yet.

        Returns
        -------
        emm_data, fmm_data: np.array(batch_size), np.array(batch_size, n_atoms, 3)
            Arrays containing MM energies and forces.
        """
        # Create context
        platform = Platform.getPlatformByName('OpenCL')
        platform.setPropertyDefaultValue('DeviceIndex', '%d' % worker_id) # select OpenCL device index
        context = Context(system, copy.deepcopy(integrator), platform)

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

        del platform, context

        return emm_data, fmm_data

