# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.QM_engines.amber_wrapper.AmberWrapper` class, which is a ParaMol implementation of a AMBER wrapper.
"""

import numpy as np
import simtk.unit as unit
from ..Utils.interface import *


class AmberWrapper:
    """
    ParaMol implementation of a AMBER SQM wrapper.

    Parameters
    ----------
    interface : :obj:`ParaMol.Utils.interface.ParaMolInterface`
        ParaMol interface object instance.
    prmtop_file : str
        Path to AMBER topology file.
    inpcrd_file : str
        Path to AMBER coordinates file.
    cell : list or np.array, shape=(3,3)
        Array with cell vectors
    n_atoms : int
        Number of atoms.
    atom_list : list of str
        List with atom symbols.
    atomic_number_list : list of int
        List with atomic number of system's atoms.
    sqm_params : dict, default={"maxcyc": "0", "qm_theory": "AM1", "dftb_disper": "0", "qm_charge": "0", "scfconv": "1.0d-8", "pseudo_diag": "0", "verbosity": "5"}
        Keyword arguments used in the sqm input file.
    work_dir : str, default="AMBERWorkDir"
        Path to the working directory.
    calc_file_prefix : str, default="sqm"
        Prefix given to the sqm calculation file.
    """

    def __init__(self, interface, prmtop_file, inpcrd_file, cell, n_atoms, atom_list, atomic_number_list,
                 sqm_params={"maxcyc": "0", "qm_theory": "AM1", "dftb_disper": "0", "qm_charge": "0", "scfconv": "1.0d-8", "pseudo_diag": "0", "verbosity": "5"},
                 work_dir="AMBERWorkDir", calc_file_prefix="sqm_"):
        self._prmtop_file = prmtop_file
        self._inpcrd_file = inpcrd_file
        self._cell = cell
        self._n_atoms = n_atoms
        self._atom_list = atom_list
        self._atomic_number_list = atomic_number_list
        self._interface = interface
        self._sqm_params = sqm_params

        # Names
        self._work_dir = work_dir
        self._calc_file_prefix = calc_file_prefix

        # Run final preparations
        #self._check_cell_units()
        self._prepare_workdir()

    # ------------------------------------------------------------ #
    #                                                              #
    #                        PRIVATE METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def _prepare_workdir(self):
        # Go to base directory
        self._interface.chdir_base()
        # Create working directory
        self._interface.create_dir(self._work_dir)
        return 1

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_calculation(self, coords, label):
        """
        Method that runs a calculation using AMBER's quantum chemisty package sqm.

        Parameters
        ----------
        coords : list or np.array
            (n_atoms,3) coordinates array
        label : str
            Seed for the calculation

        Returns
        -------
        energy : float
            Energy is given in kJ/mol.
        forces : float, np.array(n_atoms,3)
            Forces array in kJ/mol/nm.
        """
        # Change to working directory
        self._interface.chdir(self._work_dir, relative_to_base=True)

        sqm_input_file = self._calc_file_prefix + '{}.in'.format(label)
        sqm_output_file = self._calc_file_prefix + '{}.out'.format(label)

        # Write SQM input file
        with open(sqm_input_file, 'w') as f:
            f.write("SQM calculation from ParaMol \n")
            f.write("&qmmm \n")
            for key, value in self._sqm_params.items():
                f.write(" {}={} \n".format(key, value))
            f.write(" / \n")

            for i in range(self._n_atoms):
                f.write(" {} {} {} {} {}\n".format(self._atomic_number_list[i], self._atom_list[i], *coords[i]))

        # Run AMBER
        self._interface.run_subprocess("sqm -O -i " + sqm_input_file + " -o " + sqm_output_file, shell=True)

        # Extract Potential Energy
        energy = float(self._interface.run_subprocess("grep 'QMMM: SCF Energy =' {}".format(sqm_output_file),
                                                      "awk {'print $7'}", pipe=True))

        # Extract forces
        forces = self._interface.run_subprocess("grep -A{} 'QMMM: Forces on QM atoms from SCF calculation' {}".format(self._n_atoms, sqm_output_file),
                                                "head -n +{}".format(self._n_atoms+1),
                                                "tail -n {}".format(self._n_atoms),
                                                "awk {'print $4 \" \" $5 \" \" $6'}",
                                                pipe=True)

        forces = np.asarray(forces.split(), dtype=np.float64).reshape(self._n_atoms, 3)

        self._interface.chdir_base()

        # Do the necessary conversions
        forces = - forces * 10.0 * 4.184 # SQM gives dE/dR and not -dE/dR

        return energy, forces

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PRIVATE METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def _check_cell_units(self):
        """
        Method that removes unit from the cell array.

        Returns
        -------
        cell : np.array(3,3)
            Array with cell vectors
        """
        if self._cell[0][0].unit == unit.nanometer:
            cell_temp = np.zeros((3, 3))
            for i in range(3):
                cell_temp[i, i] = self._cell[i][i].in_units_of(unit.angstrom)._value
            self._cell = cell_temp

        return self._cell
