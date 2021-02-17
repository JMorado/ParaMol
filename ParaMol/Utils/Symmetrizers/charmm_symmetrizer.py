# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Utils.Symmetrizers.charmm_symmetrizer.CharmmSymmetrizer` used to handle CHARMM files..
"""
import parmed as pmd

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
    def __init__(self, prm_file, psf_file, xyz=None):
        self._charmm_prm = pmd.charmm.CharmmParameterSet(prm_file)
        self._charmm_prm.condense(do_dihedrals=True)
        self._charmm_psf = pmd.charmm.psf.CharmmPsfFile(psf_file)
        self._charmm_psf.load_parameters(self._charmm_prm, copy_parameters=False)
        super(CharmmSymmetrizer, self).__init__(self._charmm_psf)

    def __str__(self):
        return "CharmmSymmetrizer module. CHARMM file in use is {}".format(self._charmm_top)

    def save_parameter_set(self, output_file):
        """
        Method that saves the CHARMM parameter file with the current force field parameters of the self._charmm_prm instance

        Notes
        -----
        In order to update the self._charmm_prm instance with the optimal parameters, the method update_term_types_parameters should be run before this one.

        Parameters
        ----------
        output_file : str
            Name of the output file

        Returns
        -------
        None
        """
        return self._charmm_prm.write(str=output_file)