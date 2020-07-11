# -*- coding: utf-8 -*-
"""
ParaMol module with auxiliary functions to perform parallel computation of ab initio properties.
"""
import numpy as np


# -----------------------------------------------------------#
#                                                            #
#                 PARALLEL CALLBACK FUNCTION                 #
#                                                            #
# -----------------------------------------------------------#
def qm_parallel(conformations, qm_wrapper, label):
    """
    Method that can be used as a callback function for the parallel computation of QM forces and energies.

    Parameters
    ----------
    conformations : np.array
        Array with conformations
    qm_wrapper : any ParaMol QM_engine
        Instance of QM wrapper
    label : str
        Label of the

    Returns
    -------

    """
    """

    Args:
        conformations
        qm_wrapper (QMEngine):
        label (str): Label of the calculation.

    Returns:
        fqm, eqm
    """

    # proc = mp.current_process()
    n_conformations = len(conformations)
    fqm = np.zeros((n_conformations, qm_wrapper._n_atoms, 3))
    eqm = np.zeros(n_conformations)

    # Iterate over all conformations and calculate energies and forces,
    for i in range(n_conformations):
        energy, forces = qm_wrapper.run_calculation(coords=conformations[i] * 10.0, label=label)
        eqm[i] = energy
        fqm[i] = forces

    return fqm, eqm
