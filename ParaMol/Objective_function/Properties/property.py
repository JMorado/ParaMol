# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.Objective_function.Properties.property.Property` class, which is the Property base class.
"""


class Property:
    def __init__(self):
        self.name = None
        self.value = None
        self.weight = None

    # ------------------------------------------------------------------------------------------------------- #
    #                                        Data Weighting Methods                                           #
    # ------------------------------------------------------------------------------------------------------- #
    def calculate_variance(self):
        pass

    """
    TODO: deprecated method
    def compute_property_weights(self):
        '''
        Compute the weights to be attributed to the data points of a given property.

        :return: array containing the weight of each data point.
        :rtype: np.ndarray
        '''
        assert self._force_matching.fqm_residual is not None, "No quantum corrected forces were set yet."

        self._w_property = np.zeros(self._force_matching.fqm_residual.shape)
        for i in range(self._force_matching.n_atoms):
            for j in range(3):
                hist, bin_edges = np.histogram(self._force_matching.fqm_residual[:, i, j], bins=100, density=False)

                f = []
                for force in self._force_matching.fqm_residual[:, i, j]:
                    for k in range(1,len(bin_edges)):
                        if bin_edges[k] >= force >= bin_edges[k-1]:
                            f.append(hist[k-1])

                self._w_property[:, i, j] = f

        self._w_property = 1.0 / self._w_property

        return self._w_property
    """