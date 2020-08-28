# -*- coding: utf-8 -*-

"""
Description
-----------

This module defines the :obj:`ParaMol.System.system.ParaMolSystem` class, which is a ParaMol representation of a chemical system.
This object is used to store reference data such as:
    - coordinates,
    - reference energies,
    - reference forces,
    - number of atoms,
    - MM and QM engines,
    - the :obj:`ParaMol.Force_field.force_field.ForceField` associated with this system,
    - etc ...

It also contains methods that aid the calculation of ensemble properties.
"""
# General imports
import simtk.unit as unit
import ase.calculators
import logging

# ParaMol library imports
from ParaMol.Force_field.force_field import *
from ..MM_engines.openmm import *
from ..QM_engines.qm_engine import *


class ParaMolSystem:
    """
    ParaMol representation of a chemical system.

    Parameters
    ----------
    name : str
        Name of the system.
    n_atoms : int
        Number of atoms of the system.
    engine : :obj:`ParaMol.MM_engines.openmm.OpenMMEngine`
        Instance of the MM engine object to be used. Currently only OpenMM is supported.
    ref_coordinates : list or np.array
        (n_structures,n_atoms,3) 3D list or numpy array containing the reference coordinates.
    ref_energies : list or np.array
        (n_structures) 1D list or numpy array containing the reference energies.
    ref_forces : list or np.array
        (n_structures,n_atoms,3) 3D list or numpy array containing the reference forces.

    Attributes
    ----------
    name : str
        Name of the system.
    n_atoms : int
        Number of atoms of the system.
    weights : int or np.array
        Weight of each configuration.
    wham_weights : int or np.array
        WHAM weight of each configuration. It is equal to 1 unless adaptive parametrization is being performed with WHAM reweighing.
    n_structures : int
        Number of configurations.
    ref_coordinates : list or np.array
        (n_structures,n_atoms,3) 3D list or numpy array containing the reference coordinates.
    ref_energies : list or np.array
        (n_structures) 1D list or numpy array containing the reference energies.
    ref_forces : list or np.array
        (n_structures,n_atoms,3)  3D list or numpy array containing the reference forces.
    ref_esp : list or np.array
        (n_esp) 1D list or numpy array containing values of the electrostatic potential.
    ref_esp_grid : list or np.array
        (n_esp,3)  2D list or numpy array containing the coordinates of the electrostatic potential grid.
    engine : :obj:`ParaMol.MM_engines.openmm.OpenMMEngine`
        Instance of the MM engine object to be used. Currently only OpenMM is supported.
    resp_engine : :obj:`ParaMol.MM_engines.resp.RESP`
        Instance of the RESP engine object to be used.
    qm_engine : any instance of the classes defined in the modules of the subpackage :obj:`ParaMol.QM_engines`
        Instance of the QM engine object to be used.
    interface : :obj:`ParaMol.Utils.interface.ParaMolInterface`
        ParaMol interface object instance.
    n_cpus : int
        Number of cpus that this system uses.
    """

    def __init__(self, name, engine, n_atoms, resp_engine=None, ref_coordinates=None, ref_energies=None, ref_forces=None,
                 ref_esp=None, ref_esp_grid=None, n_cpus=1):

        self.name = name
        self.n_atoms = n_atoms
        self.weights = 1.0
        self.wham_weights = 1.0

        # Ensemble of conformations, energies and forces
        self.n_structures = 0
        self.ref_coordinates = ref_coordinates
        self.ref_energies = ref_energies
        self.ref_forces = ref_forces

        # Electrostatic potential
        self.ref_esp = ref_esp
        self.ref_esp_grid = ref_esp_grid

        # Interface
        self.interface = None

        # Engine and Force Field
        self.engine = engine
        self.resp_engine = resp_engine
        self.n_cpus = n_cpus

        # QM Engine
        self.qm_engine = None

        if type(self.engine) is OpenMMEngine:
            self.force_field = ForceField(engine)
        else:
            self.force_field = None
            raise NotImplementedError("Currently only OpenMMEngine is supported.")

        logging.info("Created system named {}.".format(self.name))

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def create_qm_engines(self, qm_engine_name, qm_engine_settings, interface=None):
        """
        Method (top-level) that creates a QM engine for the system.

        Parameters
        ----------
        qm_engine_name : str
            Name of the QM engine. Available QM engines are "amber", "dftb+" and "ase".
        qm_engine_settings : dict
            Keyword arguments passed to the QM engine wrapper.
        interface : :obj:`ParaMol.Utils.interface.ParaMolInterface`, default=None
            ParaMol interface.

        Returns
        -------
        qm_engine : any instance of the classes defined in the modules of the subpackage :obj:`ParaMol.QM_engines`
            Instance of the QM engine object to be used.
        """

        if interface is None:
            assert self.interface is not None, "System's ParaMol interface was not set."
            interface = self.interface

        if self.qm_engine is None:
            # If there are no QM engine associated with this stystem
            self.qm_engine = QMEngine(self, qm_engine_name, qm_engine_settings, interface)

        return self.qm_engine

    def compute_conformations_weights(self, temperature=None, emm=None, weighing_method="UNIFORM"):
        """
        Method that calculates the weights of every configuration of the ensemble.

        Notes
        -----
        For more info about the non-Boltzmann weighing see:
        "Communication: Hybrid ensembles for improved force matching"
        Lee-Ping Wang and Troy Van Voorhis
        J. Chem. Phys. 133, 231101 (2010)
        https://doi.org/10.1063/1.3519043

        Parameters
        ----------
        temperature : simtk.unit.Quantity
            Temperature of the ensemble in Kelvin.
        weighing_method : str
            Available weighing methods are "UNIFORM", "BOLTZMANN" and "NON-BOLTZMANN".
        emm: list or np.array
            (n_structures) 1D list or numpy array containing the MM energies.

        Returns
        -------
        weights : np.array
            (n_structures) 1D array containing the weight of each configuration.
        """
        np.seterr(all='raise')

        if weighing_method.upper() == "UNIFORM":
            # Equal weight to each conformation.
            # P(r_i) =  P(r_j) = 1/N_structures for any two configurations i and j.
            self.weights = np.ones(self.n_structures)

            # Normalize
            self.weights = self.weights / np.sum(self.weights)

        elif weighing_method.upper() == "NON_BOLTZMANN":
            # Weight given by the Boltzmann distribution of the difference of QM-MM.
            # P(r_i) = exp(-beta*(E^ref(r_i)-E^mm(r_i)-<E^ref-E^mm>))
            assert temperature is not None, "Temperature was not chosen."
            assert emm is not None, "Non-Boltzmann weighing was chosen but MM energies were not provided."

            diff = np.asarray(emm)-np.asarray(self.ref_energies)
            diff = diff - np.mean(diff)
            beta = (unit.BOLTZMANN_CONSTANT_kB / unit.kilojoule_per_mole * temperature * unit.kelvin) * unit.AVOGADRO_CONSTANT_NA
            beta = 1.0 / beta

            # Handle exponential overflow which occurs whenever
            try:
                self.weights = np.exp(- beta * diff)
                self.weights = self.weights / np.sum(self.weights)

            except FloatingPointError:
                # Apply uniform weighting
                self.weights = np.ones((self.n_structures)) / self.n_structures

        elif weighing_method.upper() == "BOLTZMANN":
            # Weight given by the Boltzmann distribution of the reference energies.
            # P(r_i) = exp(-beta*(E^ref(r_i)-<E^ref>))
            assert temperature is not None, "Temperature was not chosen."
            beta = (unit.BOLTZMANN_CONSTANT_kB / unit.kilojoule_per_mole * temperature * unit.kelvin) * unit.AVOGADRO_CONSTANT_NA
            beta = 1.0 / beta

            # Handle exponential overflow which occurs whenever
            try:
                self.weights = np.exp(- np.asarray(self.ref_energies - np.mean(self.ref_energies)) * beta)
                self.weights = self.weights / np.sum(self.weights)

            except FloatingPointError:
                # Apply uniform weighting
                self.weights = np.ones((self.n_structures)) / self.n_structures

        return self.weights

    def wham_reweighing(self, parameters_generation):
        """
        Method that performs WHAM reweighing.

        Notes
        -----
        For more info see the source:
        "Systematic Parametrization of Polarizable Force Fields from Quantum Chemistry Data"
        Lee-Ping Wang, Jiahao Chen, and Troy Van Voorhis
        J. Chem. Theory Comput. 2013, 9, 1, 452–460

        Parameters
        ----------
        parameters_generation : list of lists
            Each inner list contains the parameters for a given generation.

        Returns
        -------
        wham_weights : np.array of floats
            Array with new weights.
        """

        n_gen = len(parameters_generation)
        current_gen = n_gen-1
        rmsd = 9999.0
        threshold = 1e-6

        if n_gen < 2:
            return np.ones(self.n_structures)

        # Initial ansatz of equal weights for each generation
        wham_weights = np.ones((n_gen, self.n_structures)) / self.n_structures
        A = np.ones(n_gen) / n_gen

        # Determine weights for all generations
        weights = np.zeros((n_gen, self.n_structures))

        for j in range(n_gen):
            # Update parameters of the selfs and Update the parameters in the OpenMM context
            self.force_field.update_force_field(parameters_generation[j])
            self.engine.set_bonded_parameters(self.force_field.force_field_optimizable)
            self.engine.set_nonbonded_parameters(self.force_field.force_field_optimizable)

            # Get energies and weights of each configuration for the parameters of generation j
            exp_energies_generation = np.exp(-0.5*self.get_energies_ensemble())

            weights[j, :] = exp_energies_generation / np.sum(exp_energies_generation)

        prev_weight = weights[current_gen, :]

        while rmsd > threshold:
            # Calculate \sum_{j}^{N_{gen}} A^{(j)} *  w_i(k_j) / w_i (k_G)
            final_weights = np.zeros((n_gen, self.n_structures))

            for G in range(n_gen):
                for j in range(G+1):
                    final_weights[G, :] = final_weights[G, :] + A[j] * ( weights[G, :] / weights[j, :] )

                wham_weights[G, :] = final_weights[G, :]
                A[G] = np.sum(wham_weights[G,:])
                A = A / np.sum(A)

            rmsd = np.sqrt( np.sum((final_weights[current_gen, :] - prev_weight)**2) / self.n_structures )
            prev_weight = final_weights[current_gen, :]

        self.wham_weights = final_weights[G, :]

        # Update parameters of the selfs and update the parameters in the OpenMM context back to the current values
        self.force_field.update_force_field(parameters_generation[current_gen])
        self.engine.set_bonded_parameters(self.force_field.force_field_optimizable)
        self.engine.set_nonbonded_parameters(self.force_field.force_field_optimizable)

        return self.wham_weights

    def get_forces_ensemble(self):
        """
        Method that calculates the MM forces of the ensemble of configurations contained in the ref_coordinates attribute.

        Returns
        -------
        fmm : list
            (n_structures,n_atoms,3) 3D list containing the MM forces.
        """
        fmm = np.zeros((self.n_structures, self.n_atoms, 3))

        for n in range(self.n_structures):
            fmm[n, :, :] = self.engine.get_forces(self.ref_coordinates[n])

        return fmm

    def get_energies_ensemble(self):
        """
        Method that calculates the MM energies of the ensemble of configurations contained in the ref_coordinates attribute.

        Returns
        -------
        emm : list
            (n_structures) 1D list containing the MM energies.
        """
        emm = np.zeros(self.n_structures)

        for n in range(self.n_structures):
            emm[n] = self.engine.get_potential_energy(self.ref_coordinates[n])

        return emm

    def get_esp_ensemble(self):
        """
        Method that calculates MM electrostatic potential of the ensemble of configurations contained in the ref_coordinates attribute.

        Notes
        -----
        This method requires that the attribute resp_engine is not `None`.

        Returns
        -------
        mm_esp : list
            (n_structures, n_esp) 2D list containing the calculated ESP values for every configuration.
        """
        assert self.resp_engine is not None
        mm_esp = np.zeros((self.n_structures, self.ref_esp.shape[1]))

        # Iterate over all points of the ESP
        for m in range(self.n_structures):
            for i in range(self.ref_esp.shape[1]):
                mm_esp_dummy = 0.0
                for j in range(self.n_atoms):
                    mm_esp_dummy += self.resp_engine.charges[j] * self.resp_engine.inv_rij[m, j, i]

                mm_esp[m, i] = mm_esp_dummy

        return mm_esp

    def energy_statistics(self, emm):
        """
        Method that calculates statistics of energies.

        Parameters
        ----------
        emm : list
            (n_structures) 1D list containing the MM energies.

        Returns
        -------
        diff, rms, ratio
        """

        assert self.ref_energies is not None
        emm = np.concatenate(np.asarray(emm))
        diff = np.sum(np.abs(self.ref_energies - emm - np.mean(self.ref_energies - emm))) / self.n_structures
        rms = np.sqrt(np.mean((self.ref_energies - np.mean(self.ref_energies)) ** 2))
        ratio = diff / rms

        return diff, rms, ratio

    def force_statistics(self, fmm):
        """
        Method that calculates statistics of forces.

        Parameters
        ----------
        fmm : list
            (n_structures,n_atoms,3) 3D list containing the MM forces.

        Returns
        -------
        num, denom, ratio
        """
        fmm = np.asarray(fmm)
        num = np.sum(np.sqrt(np.sum((self.ref_forces - fmm) ** 2, 2))) / (3 * self.n_atoms * self.n_structures)
        denom = np.sqrt(np.mean(np.sum(np.sum(self.ref_forces ** 2, 2), 1) / (3 * self.n_atoms)))
        ratio = num / denom

        return num, denom, ratio

    def filter_conformations(self, energy_threshold):
        """
        Method that removes configurations for which the energy reference (QM energy) is greater than a given threshold
        with respect to the lowest energy of the ensemble. That is, all configurations for which the :math:`E-E_{min} > \delta`
        are removed, where :math:`E_{min}` is the lowest energy of the ensemble and :math:`\delta` is the energy threshold.

        Parameters
        ----------
        energy_threshold : float
            Energy threshold in kJ/mol.

        Returns
        -------
        ref_coordinates, ref_energies, ref_forces
            Updated reference arrays.
        """
        forces_tmp = []
        energies_tmp = []
        coordinates_tmp = []

        # Get minimum energy
        min_energy = np.min(self.ref_energies)
        for i in range(self.n_structures):
            if (self.ref_energies[i]-min_energy) <= energy_threshold:
                coordinates_tmp.append(self.ref_coordinates[i])
                energies_tmp.append(self.ref_energies[i])
                forces_tmp.append(self.ref_forces[i])

        self.ref_coordinates = coordinates_tmp
        self.ref_energies = energies_tmp
        self.ref_forces = forces_tmp
        self.n_structures = len(self.ref_coordinates)

        return self.ref_coordinates, self.ref_energies, self.ref_forces

    def append_data_to_system(self, conformations_list, qm_energies_list, qm_forces_list):
        """
        Method that appends data to the ParaMol System instance (self).

        Notes
        -----
        Appends conformations and/or QM energies and/or QM forces to the instance variables ref_coordinates, ref_energies
        and ref_forces, respectively.

        Parameters
        ----------
        conformations_list : list or np.array
            (N,n_atoms,3) 3D list or numpy array containing the reference coordinates.
        qm_energies_list : list or np.array
            (N) 1D list or numpy array containing the reference energies.
        qm_forces_list : list or np.array
            (N,n_atoms,3)  3D list or numpy array containing the reference forces.

        Returns
        -------
        ref_coordinates, ref_energies, ref_forces : np.array, np.array, np.array
        """

        if self.n_structures == 0 or self.ref_coordinates is None:
            self.ref_coordinates = []
            self.ref_forces = []
            self.ref_energies = []
        elif isinstance(self.ref_coordinates, np.ndarray):
            # Numpy array; convert to list
            self.ref_coordinates = self.ref_coordinates.tolist()
            self.ref_energies = self.ref_energies.tolist()
            self.ref_forces = self.ref_forces.tolist()
        else:
            pass

        # Set conformations, forces and energies data in the self instance
        if conformations_list is not None:
            self.ref_coordinates += conformations_list
            self.n_structures = len(self.ref_coordinates)
        if qm_energies_list is not None:
            self.ref_energies += qm_energies_list
        if qm_forces_list is not None:
            self.ref_forces += qm_forces_list

        # Convert to numpy array
        self.ref_coordinates = np.asarray(self.ref_coordinates)
        self.ref_energies = np.asarray(self.ref_energies)
        self.ref_forces = np.asarray(self.ref_forces)
        self.n_structures = len(self.ref_energies)

        return self.ref_coordinates, self.ref_energies, self.ref_forces

    # ------------------------------------------------------------ #
    #                                                              #
    #                           I/O METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    def read_data(self, input_file_name=None, append=False):
        """
        Method that reads coordinates, reference energies and reference forces from a NetCDF 4 (.nc) file.

        Parameters
        ----------
        input_file_name : str
            Name of the .nc file to be read.
        append:
            Whether to append or overwrite new data to current data.

        Returns
        -------
        True if file was closed successfully. False otherwise.
        """
        import netCDF4 as nc

        if input_file_name is None:
            input_file_name = '{}_parmol.nc'.format(self.name)

        print("\nReading {} file for system {}.".format(input_file_name, self.name))

        # Open a new netCDF file for writing.
        ncfile = nc.Dataset(input_file_name, 'r')

        if 'reference_coordinates' in ncfile.variables:
            ref_coordinates = ncfile.variables["reference_coordinates"]

            if append:
                self.ref_coordinates = np.concatenate((self.ref_coordinates,ref_coordinates))
            else:
                self.ref_coordinates = np.asarray(ref_coordinates)

            self.n_structures = len(self.ref_coordinates)
            if len(ref_coordinates) == 0:
                self.ref_coordinates = None
        else:
            print("{} does not contain coordinates data.".format(input_file_name))

        if 'reference_forces' in ncfile.variables:
            ref_forces = ncfile.variables["reference_forces"]

            if append:
                self.ref_forces = np.concatenate((self.ref_forces, ref_forces))
            else:
                self.ref_forces = np.asarray(ref_forces)

            if len(ref_forces) == 0:
                self.ref_forces = None
        else:
            print("{} does not contain forces data.".format(input_file_name))

        if 'reference_energies' in ncfile.variables:
            ref_energies = ncfile.variables["reference_energies"]

            if append:
                self.ref_energies = np.concatenate((self.ref_energies, ref_energies))
            else:
                self.ref_energies = np.asarray(ref_energies)

            if len(ref_energies) == 0:
                self.ref_energies = None
        else:
            print("{} does not contain reference energies data.".format(input_file_name))

        print("SUCCESS! Data of system {} was read from file {}".format(self.name, input_file_name))
        return ncfile.close()

    def write_data(self, output_file_name=None):
        """
        Method that writes coordinates, reference energies and reference forces to a NetCDF 4 (.nc) file.

        Parameters
        ----------
        output_file_name : str
            Name of the .nc file to be written.

        Returns
        -------
        True if file was closed successfully. False otherwise.

        """
        import netCDF4 as nc

        if output_file_name is None:
            output_file_name = '{}_parmol.nc'.format(self.name)

        print("\nWriting {} file for system {}.".format(output_file_name, self.name))

        # Open a new netCDF file for writing.
        ncfile = nc.Dataset(output_file_name, 'w')

        # create the x and y dimensions.
        ncfile.createDimension('n_structures', self.n_structures)
        ncfile.createDimension('n_atoms', self.n_atoms)
        ncfile.createDimension('spatial_dim', 3)

        if self.ref_coordinates is not None:
            # Create the variable
            data_coordinates = ncfile.createVariable('reference_coordinates', np.dtype('float64').char, ('n_structures', 'n_atoms', 'spatial_dim'))
            data_coordinates.units = "nanometers"
            # first argument is name of variable, second is datatype, third is a tuple with the name of the dimensions
            data_coordinates[:] = self.ref_coordinates

        if self.ref_forces is not None:
            data_forces = ncfile.createVariable('reference_forces', np.dtype('float64').char, ('n_structures', 'n_atoms', 'spatial_dim'))
            data_forces.units = "kilojoules/mol/nanometers"
            # first argument is name of variable, second is datatype, third is a tuple with the name of the dimensions
            data_forces[:] = self.ref_forces

        if self.ref_energies is not None:
            data_energies = ncfile.createVariable('reference_energies', np.dtype('float64').char, ('n_structures'))
            data_energies.units = "kilojoules/mol"
            data_energies[:] = self.ref_energies

        print("SUCCESS! Data of system {} was written to file {}".format(self.name, output_file_name))
        return ncfile.close()

    def write_coordinates_xyz(self, output_file_name=None, xyz_comment="comment"):
        """
        Method that writes all stored conformations to a .xyz file.

        Parameters
        ----------
        output_file_name : str
            Name of the .xyz file to be written.
        xyz_comment : str
            Comment to be written on the header of each xyz block.

        Returns
        -------
        True if file was closed successfully. False otherwise.
        """
        if output_file_name is None:
            output_file_name = '{}_traj.xyz'.format(self.name)

        atom_list = self.engine.get_atom_list()

        # Write conformations
        with open(output_file_name, 'w') as xyz_file:
            config_id = 0
            for conformation in self.ref_coordinates:
                xyz_file.write("{} \n {} {} \n".format(self.n_atoms, config_id, xyz_comment))
                for atom, symbol in zip(conformation, atom_list):
                    xyz_file.write("{} {} {} {} \n".format(symbol, *atom * 10.0))

                # Keep track of the configuration number
                config_id += 1

        print("SUCCESS! xyz file of system {} was written to file {}".format(self.name, output_file_name))
        return xyz_file.close()

    ########
    # Deprecated
    #########
    def _load_data(self, n_tot, n_conform, stride, coordinates=None, energies=None, forces=None):
        # TODO: eliminate - deprecated; this is not used anymore and is only kept here in order
        # TODO: to convert old files to newer version
        if coordinates is not None:
            self.ref_coordinates = np.loadtxt(coordinates).reshape((n_tot, self.n_atoms, 3))
            self.ref_coordinates = self.ref_coordinates[::stride]
            self.ref_coordinates = self.ref_coordinates[:n_conform]
            self.n_structures = len(self.ref_coordinates)

        if forces is not None:
            self.ref_forces = np.loadtxt(forces).reshape((n_tot, self.n_atoms, 3))
            self.ref_forces = self.ref_forces[::stride]
            self.ref_forces = self.ref_forces[:n_conform]

        if energies is not None:
            self.ref_energies = np.loadtxt(energies).reshape((n_tot))
            self.ref_energies = self.ref_energies[::stride]
            self.ref_energies = self.ref_energies[:n_conform]

        return self.ref_coordinates, self.ref_energies, self.ref_forces



