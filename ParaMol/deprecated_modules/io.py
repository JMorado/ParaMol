from simtk.openmm.app import *
from Force_field.force_field import *


class IO:
    def __init__(self):
        self._verbose = True
        
    @staticmethod
    def write_system_xml(file_name, system):
        """
        This methods writes an OpenMM system to a XML file.

        :param (str) file_name: name of the file to be written.
        :param (simtk.openmm.openmm.System) system: system instance..
        :return: None
        """
        from simtk.openmm import XmlSerializer

        print("Writing serialized system to XML file {}.".format(file_name))
        
        serialized_system = XmlSerializer.serializeSystem(system)
        outfile = open(file_name, 'w')
        outfile.write(serialized_system)
        
        return outfile.close()

    @staticmethod
    def read_system_xml(file_name, force_field):
        """
        This methods reads an OpenMM system from a XML file.

        :param (str) file_name: name of the file to be read.
        :param (ForceField) force_field: force field instance.
        :return: the system instance.
        :rtype: simtk.openmm.openmm.System
        """
        from simtk.openmm import XmlSerializer
        
        assert type(force_field) is ForceField, "Force Field object passed is not of ForceField type."

        print("\t * Reading serialized system from XML file {}.".format(file_name))
        force_field.system = XmlSerializer.deserializeSystem(open(file_name, 'r').read())

        return force_field.system
    
    def read_coordinates_forces_energies(self, coords_file, forces_file, energies_file, force_matching, data_stride, n_structures):
        """
        Read forces and coordinates from NetCDF files.
        :return:
        """
        import constants.constants_without_units as const
        import numpy as np
        from netCDF4 import Dataset

        # Read coordinates
        if coords_file is not None:
            assert os.path.isfile(coords_file), "Coordinates file does not exist."
            if self._verbose:
                print("\t * Reading coordinates from {} ...".format(coords_file), end='', flush=True)
            coords = np.asarray(Dataset(coords_file).variables['coordinates'])[1:]
            if self._verbose:
                print(" done!")

            force_matching.conformations_data = coords[::data_stride][:n_structures] * const.angstrom_to_nm
        else:
            print(" \t * Not reading coordinates file.")

        # Read forces
        if forces_file is not None:
            assert os.path.isfile(forces_file), "Forces file does not exist."
            if self._verbose:
                print("\t * Reading forces from {} ...".format(forces_file), end='', flush=True)
            fqm = np.asarray(Dataset(forces_file).variables['forces'])[1:]
            if self._verbose:
                print(" done!")
            force_matching.qm_forces_data = fqm[::data_stride][:n_structures] * const.kcal_mol_to_kj_mol / const.angstrom_to_nm
        else:
            print(" \t * Not reading forces file.")

        # Read energies
        if energies_file is not None:
            assert os.path.isfile(energies_file), "Energies file does not exist."
            if self._verbose:
                print("\t * Reading energies from {} ...".format(energies_file), end='', flush=True)
            eqm = np.loadtxt(energies_file)

            if self._verbose:
                print(" done!")
            # Set the data in the force matching instance
            force_matching.qm_energy_data = eqm[::data_stride][:n_structures] 
        else:
            print(" \t * Not reading energies file.")

        if self._verbose:
            print("\t * NOTE: Assuming units of kcal/mol/A (forces), Angstroms (coordinates) and kcal/mol (energies).")
            print("\t * NOTE: Converting to kJ/mol/A (forces), nanometers (coordinates) and kJ/mol (energies).")

        return force_matching.qm_forces_data, force_matching.qm_energy_data, force_matching.conformations_data
"""
# TODO: wrap this in io module
# Write energies and forces
np.savetxt("energies_qm.dat", force_matching.eqm, fmt='%s')

with open('forces_qm.dat', 'w') as f:
    struct = 0
    for forces in force_matching.fqm:
        f.write(str(struct) + "\n")
        np.savetxt(f, forces, fmt='%s')
        struct += 1

with open('conformations.dat', 'w') as f:
    struct = 0
    for conform in force_matching.conformations:
        f.write(str(struct) + "\n")
        
        np.savetxt(f, conform, fmt='%s')
        struct += 1
"""
