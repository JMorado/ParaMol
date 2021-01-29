# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Utils.Symmetrizers.charmm_symmetrizer.CharmmSymmetrizer` used to handle CHARMM files..
"""
import parmed as pmd
import numpy as np

# ParaMol modules
from .symmetrizer import *


class CharmmSymmetrizer(Symmetrizer):
    """
    ParaMol class that implements methods to symmetrize the ParaMol Force Field so that it respects CHARMM atom-types.

    Parameters
    ----------
    top_file : str
        AMBER prmtop file
    xyz : str or array, optional
        If provided, the coordinates and unit cell dimensions from the provided CHARMM .CRD file will be loaded into the molecule, or the coordinates will be loaded from the coordinate array
    """
    def __init__(self, top_file, xyz=None):
        self._charmm_top = pmd.charmm.CharmmParameterSet(top_file)
        super(CharmmSymmetrizer, self).__init__(self._charmm_top)

    def __str__(self):
        return "CharmmSymmetrizer module. CHARMM file in use is {}".format(self._charmm_top)

    def get_symmetries(self, force_field_instance=None):
        pass

    def symmetrize_force_field(self, force_field_instance):
        pass

    def save(self, output_file, format=None):
        pass