# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.QM_engines.amber_wrapper.AmberWrapper` class, which is a ParaMol implementation of a AMBER wrapper.
"""

import numpy as np
import simtk.unit as unit
import os
from ..Utils.interface import *


class AmberWrapper:
    """
    ParaMol implementation of a AMBER SQM wrapper.

    Parameters
    ----------
    settings : dict
        Dictionary containing the settings of the QM wrapper.
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
    sqm_params : dict
        Keyword arguments used in the sqm input file.
    """

    def __init__(self, settings, interface, prmtop_file, inpcrd_file, cell, n_atoms, atom_list, atomic_number_list, sqm_params,
                 qm_wrapper_name, work_dir, calc_file_prefix, energies_file_prefix, forces_file_prefix):
        self._prmtop_file = prmtop_file
        self._inpcrd_file = inpcrd_file
        self._cell = cell
        self._n_atoms = n_atoms
        self._atom_list = atom_list
        self._atomic_number_list = atomic_number_list
        self._interface = interface
        self._settings = settings
        self._sqm_params = sqm_params

        # Run final preparations
        self._check_cell_units()
        self._prepare_workdir()

        # Names
        self._qm_wrapper_name = qm_wrapper_name
        self._work_dir = work_dir
        self._calc_file_prefix = calc_file_prefix
        self._energies_file_prefix = energies_file_prefix
        self._forces_file_prefix = forces_file_prefix

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

        # Write SQM input file
        with open(self._calc_file_prefix + '{}.in'.format(label), 'w') as f:
            f.write("SQM calculation from ParaMol \n")
            f.write("&qmmm \n")
            for key, value in self._sqm_params.items():
                f.write(" {}={} \n".format(key, value))
            f.write(" / \n")

            for i in range(self._n_atoms):
                f.write(" {} {} {} {} {}\n".format(self._atomic_number_list[i], self._atom_list[i], *coords[i]))

        # Run calculation
        os.system("./{} {} {}".format(self.bash_exe, label, self._n_atoms))

        # Read energy
        with open(self._energies_file_prefix+'{}'.format(label), 'r') as f:
            sp_energy = float(f.read())

        # Read forces
        forces = np.loadtxt(self._forces_file_prefix + "{}".format(label), dtype=np.float64) * 10.0 * 4.184  # convert to kJ/mol/nm
        forces = - forces # SQM gives dE/dR and not -dE/dR

        self._interface.chdir_base()

        return sp_energy, forces

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
    

"""
    TODO: This part is only necessary when using other modules rather than SQM.
    TODO: Check this later
    
    def run_calculation(self, n_atoms, coord, label, isolated=True):
        #
        #Output units: kJ/mol
        #kJ/mol/A
        #
        coordinate_file = "zero_step_{}.nc".format(label)
        self.write_netcdf(n_atoms, coord, "AmberWorkDir/{}".format(coordinate_file))

        # Run calculation and extract potential energy and forces
        os.chdir("AmberWorkDir/")
        os.system("./run_qmmm_zero_step.sh {} {} {}".format(label, self.prmtop_file, coordinate_file))
        with open('epot_{}'.format(label),'r') as f:
            sp_energy = float(f.read()) * 4.184
        forces = np.asarray(Dataset('forces_' + str(label) + ".nc").variables['forces'])[0] * 4.184 * 10.0
        os.chdir("../")

        return sp_energy, forces

    def write_netcdf(self, n_atoms, coord, filename):
    

        if filename == '':
            filename = self.inpcrd_file

        from scipy.io import netcdf

        # Open output file
        fout = netcdf.netcdf_file(filename, 'w')
        fout.Conventions = 'AMBERRESTART'
        fout.ConventionVersion = "1.0"
        fout.title = 'ParamMol'
        fout.application = "AMBER"
        fout.program = "HybridMC"
        fout.programVersion = "1.0"
        fout.createDimension('cell_spatial', 3)
        fout.createDimension('label', 5)
        fout.createDimension('cell_angular', 3)
        fout.createDimension('time', 1)
        time = fout.createVariable('time', 'd', ('time',))
        time.units = 'picosecond'
        fout.createDimension('spatial', 3)
        spatial = fout.createVariable('spatial', 'c', ('spatial',))
        spatial[:] = np.asarray(list('xyz'))

        # Coordinates
        fout.createDimension('atom', n_atoms)
        coordinates = fout.createVariable('coordinates', 'd', ('atom', 'spatial'))
        coordinates.units = 'angstrom'
        coordinates[:] = coord

       
        # Velocities, only written if available
        #if structure.getVelocities() is not None:
        #    velocities = fout.createVariable('velocities', 'd', ('atom', 'spatial'))
        #    velocities.units = 'angstrom/ picosecond'
        #    velocities[:] = structure.getVelocities()[:]*(1./20.455) #Correct conversion factor if velocities are given in angstrom/picseocnd

        # Cell information
        # title
        cell_angular = fout.createVariable('cell_angular', 'c', ('cell_angular', 'label'))
        cell_angular[0] = np.asarray(list('alpha'))
        cell_angular[1] = np.asarray(list('beta '))
        cell_angular[2] = np.asarray(list('gamma'))
        # title
        cell_spatial = fout.createVariable('cell_spatial', 'c', ('cell_spatial',))
        cell_spatial[0], cell_spatial[1], cell_spatial[2] = 'a', 'b', 'c'
        # data
        cell_lengths = fout.createVariable('cell_lengths', 'd', ('cell_spatial',))
        cell_lengths.units = 'angstrom'
        cell_lengths[0] = self._cell[0, 0]
        cell_lengths[1] = self._cell[1, 1]
        cell_lengths[2] = self._cell[2, 2]

        cell_angles = fout.createVariable('cell_angles', 'd', ('cell_angular',))

        box_alpha, box_beta, box_gamma = 90.0, 90.0, 90.0
        cell_angles[0] = box_alpha
        cell_angles[1] = box_beta
        cell_angles[2] = box_gamma

        cell_angles.units = 'degree'

        return fout.close()
"""
