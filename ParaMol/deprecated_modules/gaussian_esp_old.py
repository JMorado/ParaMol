"""
This module contains function to extract an ESP potential from a gaussian output file so that it can be used as
input for RESP calculations in ParaMol.

Joao Morado
"""
import numpy as np


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
        import numpy

        grid_tmp = []
        conformation = []
        esp_tmp = []
        grid = []
        esp = []
        angstrom_to_nm = 0.1
        angstrom_to_au = 1.8897259886
        hartree_to_kjmol = 2625.5002
        hartree_to_kjmol = 1.0

        # Read gaussian file
        gaussian_file = open(log_file, 'r')
        for line in gaussian_file:
            if "Fit Center" in line:
                line_split = line.split()
                x = float(line_split[6]) * angstrom_to_au
                y = float(line_split[7]) * angstrom_to_au
                z = float(line_split[8]) * angstrom_to_au
                grid_tmp.append([x,y,z])

            elif "Atomic Center" in line:
                line_split = line.split()
                x = float(line_split[5]) * angstrom_to_au
                y = float(line_split[6]) * angstrom_to_au
                z = float(line_split[7]) * angstrom_to_au
                conformation.append([x,y,z])

            elif "Fit    " in line:
                line_split = line.split()
                esp_value = float(line_split[2])
                esp_tmp.append(esp_value)

        gaussian_file.close()

        vdw =  [1.7,1.7,1.7,1.7,1.7,1.7,1.2,1.2,1.2,1.2,1.2,1.2,1.55,1.2,1.2]

        prune=False
        n = 0
        if prune:
            # The points must lie outside the van der Waals radius of the molecule for reasons
            for esp_val, gp in zip(esp_tmp, grid_tmp):
                at_id = 0
                for atom in conformation:
                    include = True
                    v = numpy.linalg.norm(np.asarray(atom) - np.asarray(gp))

                    if v < (vdw[at_id] * angstrom_to_au):
                        include = False
                        break
                    at_id +=1

                if include:
                    n += 1
                    grid.append(gp)
                    esp.append(esp_val)
        else:
            grid = grid_tmp
            esp = esp_tmp

        return conformation, grid, esp

    def read_log_files(self, files_names):
        # Read ALL gaussian log files
        self.conformations = []
        self.grids = []
        self.esps = []
        for gaussian_file in files_names:
            conformation, grid, esp = self.gaussian_read_log(gaussian_file)
            assert len(grid) == len(esp), "Number of grid points and ESP values does not coincide."
            self.conformations.append(conformation)
            self.grids.append(grid)
            self.esps.append(esp)

    def write_esp_paramol_format(self):
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

        return

