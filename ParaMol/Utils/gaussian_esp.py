# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Utils.gaussian_esp.GaussianESP` that  contains function to extract an ESP potential from a gaussian output file so that it can be used as
input for RESP calculations in ParaMol.
"""

import numpy as np
import logging


class GaussianESP:
    def __init__(self, conformations=None, grids=None, esps=None):
        self.conformations = conformations
        self.grids = grids
        self.esps = esps

    # ---------------------------------------------------------- #
    #                                                            #
    #                       PUBLIC METHODS                       #
    #                                                            #
    # ---------------------------------------------------------- #
    @staticmethod
    def gaussian_read_log(log_file):
        """
        Method that reads the output of a gaussian ESP calculation.

        Notes
        -----
        Energy stored will be the last occurrence of "SCF Done:" in the gaussian log file.

        Parameters
        ----------
        log_file : str
            Path to the Gaussian output file.

        Returns
        -------
        conformation, grid, esp, energy
        """
        angstrom_to_au = 1.8897259886
        hartree_to_kj_mol = 2625.5002
        conformation = []
        grid = []
        esp = []

        logging.info("Reading Gaussian file {}.".format(log_file))

        # Read gaussian file
        gaussian_file = open(log_file, 'r')
        for line in gaussian_file:
            if "Fit Center" in line:
                line_split = line.split()
                x = float(line_split[6]) * angstrom_to_au
                y = float(line_split[7]) * angstrom_to_au
                z = float(line_split[8]) * angstrom_to_au
                grid.append([x, y, z])

            elif "Atomic Center" in line:
                line_split = line.split()
                x = float(line_split[5]) * angstrom_to_au
                y = float(line_split[6]) * angstrom_to_au
                z = float(line_split[7]) * angstrom_to_au
                conformation.append([x, y, z])

            elif "Fit    " in line:
                line_split = line.split()
                esp_value = float(line_split[2])
                esp.append(esp_value)

            elif "SCF Done:" in line:
                line_split = line.split()
                energy = float(line_split[4]) * hartree_to_kj_mol

        gaussian_file.close()

        return conformation, grid, esp, energy

    def read_log_files(self, files_names):
        """
        Method that reads multiple files that correspond to the output of a gaussian ESP calculation.

        Parameters
        ----------
        files_names: str or list of str
            Path(s) to Gaussian output files.

        Returns
        -------
        conformations : np.array, shape=(n_conformations,n_atoms,3)
            Array with conformations.
        grids : np.array, shape=(n_conformations,n_esp_point,3)
            Array with grid points.
        esps : np.array, shape=(n_esp_points,3)
            Array with electrostatic potential values.
        energies : np.array, shape=(n_conformations)
            Array with energies
        """
        # Read ALL gaussian log files
        self.conformations = []
        self.grids = []
        self.esps = []
        self.energies = []

        if type(files_names) is str:
            files_names = [files_names]

        for gaussian_file in files_names:
            conformation, grid, esp, energy = self.gaussian_read_log(gaussian_file)
            assert len(grid) == len(esp), "Number of grid points and ESP values does not coincide."
            self.conformations.append(np.asarray(conformation))
            self.grids.append(np.asarray(grid))
            self.esps.append(np.asarray(esp))
            self.energies.append(energy)

        return self.conformations, self.grids, self.esps, self.energies

    def write_esp_paramol_format(self):
        """
        Method that writes ESP files in ParaMol format.

        Returns
        -------
        True if everything run smoothly.

        """
        assert self.conformations is not None
        assert self.grids is not None
        assert self.esps is not None

        # Write all to file
        output_file_esp = open('esp', 'w')
        output_file_conformations = open('conf', 'w')
        for i in range(len(self.conformations)):
            #output_file_esp.write("Conformation {} \n".format(i))
            #output_file_conformations.write("Conformation {} \n".format(i))
            for j in range(len(self.grids[i])):
                output_file_esp.write("{} {} {} {} \n".format(*self.grids[i][j],self.esps[i][j]))
            for j in range(len(self.conformations[i])):
                output_file_conformations.write("{} {} {} \n".format(*self.conformations[i][j]))

        output_file_esp.close()
        output_file_conformations.close()

        return True

