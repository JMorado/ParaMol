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
