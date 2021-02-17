# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Utils.Symmetrizers.amber_symmetrizer.AmberSymmetrizer` used to handle AMBER atom types.
"""
import parmed as pmd
import numpy as np

# ParaMol modules
from .symmetrizer import *


class AmberSymmetrizer(Symmetrizer):
    """
    ParaMol class that implements methods to symmetrize the ParaMol Force Field so that it respects AMBER atom-types.

    Parameters
    ----------
    top_file : str
        AMBER prmtop file
    xyz : str or array, optional
        If provided, the coordinates and unit cell dimensions from the provided Amber inpcrd/restart file will be loaded into the molecule, or the coordinates will be loaded from the coordinate array
    """
    def __init__(self, top_file, xyz=None):
        self._amber_top = pmd.amber.AmberParm(top_file, xyz=xyz)
        super(AmberSymmetrizer, self).__init__(self._amber_top)

    def __str__(self):
        return "AmberSymmetrizer module. AMBER .prmtop file in use is {}".format(self._amber_top)

    def save_frcmod(self, output_file):
        """
        Method that saves the .frcmod AMBER file with the current force field parameters of the self._amber_prmtop instance.

        Notes
        -----
        In order to update the self._amber_prmtop instance with the optimal parameters, the method update_term_types_parameters should be run before this one.

        Parameters
        ----------
        output_file : str
            Name of the output file

        Returns
        -------
        None
        """
        frcmod = pmd.tools.writeFrcmod(self._amber_top, output_file)

        return frcmod.execute()
