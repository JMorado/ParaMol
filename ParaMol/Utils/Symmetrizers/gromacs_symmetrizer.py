# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Utils.Symmetrizers.gromacs_symmetrizer.GromacsSymmetrizer` used to handle GROMACS files.
"""
import parmed as pmd
import numpy as np

# ParaMol modules
from .symmetrizer import *


class GromacsSymmetrizer(Symmetrizer):
    """
    ParaMol class that implements methods to symmetrize the ParaMol Force Field so that it respects GROMACS atom-types.

    Parameters
    ----------
    top_file : str
        GROMACS prmtop file
    xyz : str or array, optional
        If provided, the coordinates and unit cell dimensions from the provided GROMACS crd file will be loaded into the molecule, or the coordinates will be loaded from the coordinate array.
    """
    def __init__(self, top_file, xyz=None):
        self._gromacs_top = pmd.gromacs.GromacsTopologyFile(top_file, xyz=xyz)
        super(GromacsSymmetrizer, self).__init__(self._gromacs_top)

    def __str__(self):
        return "GromacsSymmetrizer module. GROMACS file in use is {}".format(self._gromacs_top)

