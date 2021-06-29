# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.QM_engines.dftb_wrapper.DFTBWrapper` class, which is a ParaMol implementation of a DFTB+ wrapper.
"""
import numpy as np
import simtk.unit as unit
from ..Utils.interface import *


class DFTBWrapper:
    """
    ParaMol implementation of a DFTB+ wrapper.

    Parameters
    ----------
    system_name : str
        Name of the system to which this wrapper is associated.
    interface : :obj:`ParaMol.Utils.interface.ParaMolInterface`
        ParaMol interface object instance.
    n_atoms : int
        number of atoms
    atom_list : list of str
        List with the atom symbols
    n_calculations : int
        Number of parallel calculations.
    slater_koster_files_prefix : str
        Path to the Slater-Koster files.
    work_dir : str, default="DFTBWorkDir"
        Path to the working directory.
    calc_file : str, default="dftb_in.hsd"
        Name given to the DFTB+ calculation file.
    calc_file : str, default="dftb_output.out"
        Name given to the DFTB+ stdout file.
    detailed_file_output : str, default="detailed.out"
        Name of the detailed output file written by DFTB+.
        By default DFTB+ writes this to 'detailed.out'. Don't change this unless you know what you're doing.
    geometry_file : str, default="geometry.gen"
        Name given to the DFTB+ .gen geometry file.
    calc_dir_prefix : str, default="dftb\_"
        Prefix given to the directories where the calculations will be performed.
    max_ang_mom : dict, default={"H": "s", "C": "p", "N": "p", "O": "p", "F": "p", "S": "p"}
        Dictionary that defines the maximum angular momentum for each chemical element.
    """
    def __init__(self, system_name, interface, n_atoms, atom_list, n_calculations,
                 slater_koster_files_prefix, work_dir_prefix="DFTBWorkDir_", calc_file="dftb_in.hsd", geometry_file="geometry.gen",
                 calc_dir_prefix="dftb_", detailed_file_output="detailed.out", calc_file_output="dftb_output.out",
                 max_ang_mom={"H": "s", "C": "p", "N": "p", "O": "p", "F": "p", "S": "p"}):

        # Name of the system to which this wrapper is associated to
        self._system_name = system_name

        # Natoms, atom list, n calculations
        self._n_atoms = n_atoms
        self._atom_list = atom_list
        self._n_calculations = n_calculations

        # Get unique atomic element and atomic indexes list
        self._unique_elements = list(set(self._atom_list))
        self._atom_list_idx = [(self._unique_elements.index(element) + 1) for element in self._atom_list]

        # ParaMol interface
        self._interface = interface
        self._calculation_dirs = []

        # Constants
        self._slater_koster_files_prefix = slater_koster_files_prefix
        self._work_dir = "{}{}".format(work_dir_prefix, self._system_name)
        self._calc_file = calc_file
        self._calc_file_output = calc_file_output
        self._geometry_file = geometry_file
        self._calc_dir_prefix = calc_dir_prefix
        self._detailed_file_output = detailed_file_output

        # Maximum angular momentum dictionary
        self._max_ang_mom = max_ang_mom

        # Prepare working directory
        self._prepare_work_dir()

    # ------------------------------------------------------------ #
    #                                                              #
    #                       PUBLIC  METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_calculation(self, coords, label, *args, **kwargs):
        """
        Method that perform a single-point calculation with forces.

        Parameters
        ----------
        coords : np.ndarray, shape=(n_atoms, 3), dtype=float
            Coordinates array.
        label : str
            Seed for the calculation.

        Returns
        -------
        energy : float
            Energy in kJ/mol.
        forces: np.ndarray, shape=np.array(n_atoms,3)
            Forces in kJ/mol/nm
        """

        # Run calculation and extract potential energy

        # Change to calculation directory
        self._interface.chdir(self._calculation_dirs[label], absolute=True)

        # Write geometry in GEN format
        self._write_geometry_gen_format(coords)

        # Run DFTB+
        self._interface.run_subprocess("dftb+", self._calc_file, output_file=self._calc_file_output)

        # Extract Potential Energy
        energy = float(self._interface.run_subprocess("grep 'Total energy' {}".format(self._detailed_file_output),
                                                      "awk {'print $3'}", pipe=True))

        # Extract forces
        forces = (self._interface.run_subprocess("grep -A{} 'Total Forces' {}".format(self._n_atoms, self._detailed_file_output),
                                                 "awk {'print $2 \" \" $3 \" \" $4'}",
                                                 "tail -n +2",
                                                 pipe=True))

        forces = np.asarray(forces.split(), dtype=np.float64).reshape(self._n_atoms, 3)

        self._interface.chdir_base()

        # Do the necessary conversions
        energy = energy * 2625.5002 # Hartree -> kJ/mol
        forces = forces * 2625.5002 / 0.529177 * 10 # Hartree/a.u. -> kJ/mol/nm

        return energy, forces

    # ------------------------------------------------------------ #
    #                                                              #
    #                      PRIVATE  METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    def _prepare_work_dir(self):
        """
        Method that prepares the working directory.

        Returns
        -------
        None
        """
        # Go to base directory
        self._interface.chdir_base()
        # Create working directory
        self._interface.create_dir(self._work_dir)
        # Change to working directory
        self._interface.chdir(self._work_dir, relative_to_base=True)

        for i in range(self._n_calculations):
            self._interface.create_dir(self._calc_dir_prefix+"{}".format(i))
            self._interface.chdir(self._calc_dir_prefix+"{}".format(i))
            self._write_input_file()
            self._calculation_dirs.append(os.getcwd())
            self._interface.chdir_previous()

        self._interface.chdir_base()

        return 1

    def _write_input_file(self, dispersion_correction=False):
        """
        Method that writes the input file necessary to run single-point with forces calculation.

        Notes
        -----
        It only needs to be run upon creation of the QM wrapper.

        Parameters
        ----------
        dispersion_correction : bool
            Whether or not to include D3 dispersion correction section.

        Returns
        -------
        `True` if file was closed successfully. `False` otherwise.
        """

        input_file = open(self._calc_file, 'w')
        input_file.write("Geometry = GenFormat { \n")
        input_file.write("<<< '{}' \n".format(self._geometry_file))
        input_file.write("} \n \n")

        input_file.write("Driver = {} \n \n")

        input_file.write("Hamiltonian = DFTB { \n")
        input_file.write("Scc = yes \n")

        input_file.write("  SlaterKosterFiles = Type2FileNames {\n")
        input_file.write("Prefix = '{}' \n".format(self._slater_koster_files_prefix))
        input_file.write("Separator = '-' \n")
        input_file.write("Suffix = '.skf' \n")
        input_file.write("}\n")

        input_file.write("  MaxAngularMomentum {\n")
        for element in self._unique_elements:
            input_file.write("{} = '{}' \n".format(element, self._max_ang_mom[element]))
        input_file.write("}\n")

        if dispersion_correction:
            # Write dispersion correction part
            input_file.write("Dispersion = DftD3 { Damping = BeckeJohnson { \n")
            input_file.write("a1 = 0.5719 \n")
            input_file.write("a2 = 3.6017 } \n")
            input_file.write("s6 = 1.0 \n")
            input_file.write("s8 = 0.5883 }\n")

        # End of Hamiltonian
        input_file.write("}\n \n")

        input_file.write("Options { \n")
        input_file.write("    WriteDetailedOut = Yes \n")
        input_file.write("} \n")

        input_file.write("Analysis { \n ")
        input_file.write("    CalculateForces = Yes \n")
        input_file.write("} \n")

        input_file.write("ParserOptions { \n")
        input_file.write("    ParserVersion = 7 \n")
        input_file.write("} \n")

        return input_file.close()

    def _write_geometry_gen_format(self, coords):
        """
        Method that writes the geometry in gen format.

        Parameters
        ----------
        coords: list or np.array
            (n_atoms,3) coordinates array

        Returns
        -------
        `True` if file was closed successfully. `False` otherwise.
        """

        gen_file = open(self._geometry_file, "w")
        gen_file.write("{} C \n".format(self._n_atoms))

        # Write unique elements
        for unique_element in self._unique_elements:
            gen_file.write("{} ".format(unique_element))
        gen_file.write("\n")

        # Write coordinates
        for i in range(self._n_atoms):
            gen_file.write("{} {} {} {} {} \n".format(i+1, self._atom_list_idx[i], *coords[i]))

        return gen_file.close()
