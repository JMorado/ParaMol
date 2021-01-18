# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.MM_engines.openmm.OpenMMEngine` class which is the ParaMol wrapper for OpenMM.
"""

import logging
import numpy as np
import simtk.unit as unit
import simtk.openmm as openmm
import simtk.openmm.app as app


# ------------------------------------------------------------ #
#                                                              #
#                          OpenMMEngine                        #
#                                                              #
# ------------------------------------------------------------ #
class OpenMMEngine:
    """
    ParaMol's OpenMM wrapper.

    Parameters
    ----------
    init_openmm : bool, optional, default=False
        Whether or not to create the OpenMM system, topology, integrator and context and platform upon creation of an OpenMMEngine instance.
        Note that the only objects created are the ones not passed as parameters.
    topology_format : str, optional, default=None
        Available options are "AMBER" or "XML".
    top_file : str, optional, default=None
        Path to the AMBER topology file.
    xml_file : str, optional, default=None
        Path to the .xml OpenMM system file.
    crd_file : str, optional, default=None
        Path to the AMBER coordinates file.
    platform_name : str, optional, default='Reference'
        Name of the OpenMM platform.
    integrator_params : dict, optional, default={'temperature' : 300.0 * unit.kelvin, 'stepSize' : 0.001 * unit.picoseconds, 'frictionCoeff' : 2.0 / unit.picoseconds}
        Keyword arguments passed to the simtk.openmm.openmm.LangevinIntegrator. Ignored if an OpenMM Integrator is provided through the integrator parameter.
    create_system_params : dict, optional, default={'temperature' : 300.0 * unit.kelvin, 'stepSize' : 0.001 * unit.picoseconds, 'frictionCoeff' : 2.0 / unit.picoseconds}
        Keyword arguments passed to simtk.openmm.app.amberprmtopfile.createSystem. Ignored if an OpenMM System is provided through the system parameter.
    system : simtk.openmm.openmm.System, optional, default=None
        OpenMM system.
    integrator : any OpenMM integrator, optional, default=None
        OpenMM integrator.
    platform : simtk.openmm.openmm.Platform, optional, default=None
        OpenMM platform
    context : simtk.openmm.openmm.Context, optional, default=None
        OpenMM context.
    topology : simtk.openmm.app.topology.Topology, optional, default=None
        OpenMM topology.

    Attributes
    ----------
    topology_format : str
        Available options are "AMBER" or "XML".
    top_file : str
        Path to the AMBER topology file.
    xml_file : str
        Path to the .xml OpenMM system file.
    crd_file : str
        Path to the AMBER coordinates file.
    platform_name : str, optional, default='Reference'
        Name of the OpenMM platform.
    system : simtk.openmm.openmm.System
        OpenMM system.
    integrator : any OpenMM integrator
        OpenMM integrator.
    platform : simtk.openmm.openmm.Platform
        OpenMM platform
    context : simtk.openmm.openmm.Context
        OpenMM context.
    topology : simtk.openmm.app.topology.Topology
        OpenMM topology.
    force_groups : list of int
        List containing all the force groups present in the system.
    atom_list : list of str
        List containing the atom symbols of the system. Method get_atom_list has to be run to set this attribute variable.
    atomic_number_list : list of int
        List containing the atomic numbers of the system. Method get_atomic_numbers has to be run to set this attribute variable.
    masses_list : list of float
        List containing the masses of the atoms of the system. Method get_masses has to be run to set this attribute variable.
    n_atoms : int
        Number of atoms of the system.
    cell : np.ndarray, shape=(3, 3), dtype=float
        Array containing the box size cell vectors (in angstroms). Method get_cell has to be run to set this attribute variable.
    """
    force_groups_dict = {'HarmonicBondForce': 0,
                         'HarmonicAngleForce': 1,
                         'PeriodicTorsionForce': 2,
                         'NonbondedForce': 11,
                         'CMMotionRemover': 3,
                         'CustomBondForce': 5,
                         'CustomAngleForce': 6,}

    def __init__(self, init_openmm=False, topology_format=None, top_file=None, crd_file=None, xml_file=None,
                 platform_name='Reference', system=None, integrator=None, platform=None, context=None, topology=None,
                 integrator_params={'temperature': 300.0 * unit.kelvin, 'stepSize': 0.001 * unit.picoseconds, 'frictionCoeff': 2.0 / unit.picoseconds},
                 create_system_params= {'nonbondedMethod': app.NoCutoff, 'nonbondedCutoff': 1.2 * unit.nanometer, 'constraints': None, 'rigidWater': True}):

        self.topology_format = topology_format
        self.xml_file = xml_file
        self.top_file = top_file
        self.crd_file = crd_file

        # OpenMM essential object instances
        self.system = system
        self.integrator = integrator
        self.platform = platform
        self.context = context
        self.topology = topology

        # Platform-specific variables
        self.platform_name = platform_name if platform_name is not None else 'Reference'

        # Molecule-specific variables
        self.force_groups = []
        self.atom_list = None
        self.atomic_number_list = None
        self.masses_list = None
        self.n_atoms = None
        self.cell = None

        # Params to be passed to OpenMM
        self._create_system_params = create_system_params
        self._integrator_params = integrator_params

        if init_openmm:
            self.init_openmm(self._integrator_params, self._create_system_params)

        if self.system is not None:
            self._set_force_groups()

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def init_openmm(self, integrator_params=None, create_system_params=None):
        """
        Method that initiates OpenMM by creating

        Parameters
        ----------
        integrator_params : dict
            Keyword arguments passed to the simtk.openmm.openmm.LangevinIntegrator
        create_system_params : dict
            Keyword arguments passed to simtk.openmm.app.amberprmtopfile.createSystem

        Returns
        -------
        system : simtk.openmm.openmm.System
            OpenMM system created.
        """
        from simtk.openmm import XmlSerializer

        assert self.topology_format is not None, "No topology_format was provided."
        if self.topology_format == "XML":
            assert self.xml_file is not None, "Topology format is XML but no XML file was provided."
        elif self.topology_format == "AMBER":
            assert self.top_file is not None, "Topology format is XML but no XML file was provided."
        else:
            raise NotImplementedError("Topology format {} is not known.".format(self.topology_format))

        assert self.crd_file is not None, "create_system flag is True but no crd_file was provided."
        if self.platform_name is None:
            logging.info("No platform set. Will use reference.")
            self.platform_name = "Reference"
        else:
            assert self.platform_name in ["Reference", "CPU", "OpenCL", "CUDA"], """create_system flag is True but no
               correct platform was provided."""

        if self.topology is None:
            top = app.AmberPrmtopFile(self.top_file)
            self.topology = top.topology

        crd = app.AmberInpcrdFile(self.crd_file)

        if self.system is None:
            if self.topology_format.upper() == "AMBER":
                assert create_system_params is not None, "No settings to create the system were provided."

                logging.info("Creating OpenMM System from AMBER file.")
                self.system = top.createSystem(**create_system_params)
            elif self.topology_format.upper() == "XML":
                logging.info("Creating OpenMM System from XML file.")
                xml_file = open(self.xml_file)
                self.system = XmlSerializer.deserializeSystem(xml_file.read())
                xml_file.close()
            else:
                raise NotImplementedError("Topology format {} is not recognized.".format(self.topology_format))

        if self.integrator is None:
            assert integrator_params is not None, "No settings to create the integrator were provided."

            self.integrator = openmm.LangevinIntegrator(integrator_params['temperature'], integrator_params["frictionCoeff"], integrator_params["stepSize"])
            logging.info("Creating OpenMM integrator.")
        if self.platform is None:
            self.platform = openmm.Platform.getPlatformByName(self.platform_name)
            logging.info("Creating OpenMM platform.")
        if self.context is None:
            self.context = openmm.Context(self.system, self.integrator, self.platform)
            logging.info("Creating OpenMM Context.")

        # Set positions in context
        self.context.setPositions(crd.positions)

        # Set force groups of the system
        self.force_groups = self._set_force_groups()

        return self.system

    def get_atom_list(self):
        """
        Method that gets a list of the atom symbols.

        Returns
        -------
        atom_list : list of str
            List of the atom symbols of the system.
        """
        assert self.topology is not None, "OpenMM topology is not set."

        self.atom_list = []
        for atom in self.topology.atoms():
            self.atom_list.append(atom.element.symbol)

        return self.atom_list

    def get_atomic_numbers(self):
        """
        Method that gets a list of the atomic numbers of the system.

        Returns
        -------
        atom_list : list of str
            List of the atomic numbers of the system.
        """
        assert self.topology is not None, "OpenMM topology is not set."

        self.atomic_number_list = []
        for atom in self.topology.atoms():
            self.atomic_number_list.append(atom.element.atomic_number)

        return self.atomic_number_list

    def get_number_of_atoms(self):
        """
        Method that gets the number of atoms of the system.

        Returns
        -------
        n_atoms : n_int
            Number of atoms of the system.
        """
        assert self.system is not None, "OpenMM system is not set."

        self.n_atoms = self.system.getNumParticles()

        return self.n_atoms

    def get_masses(self):
        """
        Method that gets the masses of atoms of the system (in amu).

        Returns
        -------
        masses : list of floats
            Masses of the atoms of the system.
        """
        self.masses_list = []
        for atom_index in range(self.get_number_of_atoms()):
            self.masses_list.append(self.system.getParticleMass(atom_index))

        return self.masses_list

    def get_cell(self):
        """
        Method that gets the cell vectors.

        Returns
        -------
        cell : np.array
            (3,3) array containing the cell vectors in angstrom but no simtk.units.
        """
        assert self.system is not None, "OpenMM system is not set."

        self.cell = np.zeros((3,3))
        openmm_cell = self.system.getDefaultPeriodicBoxVectors()

        for i in range(3):
            self.cell[i, i] = openmm_cell[i][i]._value

        self.cell = self.cell * 10.0 # nanometers to angstrom

        return self.cell

    def write_system_xml(self, file_name):
        """
        Method that writes the OpenMM system stored in the `system` attribute to an XML file.

        Parameters
        ----------
        file_name : str
            Name of the XML file to be written.

        Returns
        -------
            `True` if file was closed successfully. `False` otherwise.
        """

        from simtk.openmm import XmlSerializer

        logging.info("Writing serialized system to XML file {}.".format(file_name))

        serialized_system = XmlSerializer.serializeSystem(self.system)
        outfile = open(file_name, 'w')
        outfile.write(serialized_system)
        outfile.close()

        return outfile.close()

    def minimize_system(self, tolerance=1, max_iter=0):
        """
        Method that minimizes the system's energy starting from the state stored at the context attribute.

        Notes
        -----
        More information can be found at: https://simtk.org/api_docs/openmm/api3_1/classOpenMM_1_1LocalEnergyMinimizer.html

        Parameters
        ----------
        tolerance : float
            Specifies how precisely the energy minimum must be located. Minimization will be halted once the root-mean-square value of all force components reaches this tolerance.
        max_iter : int
            Maximum number of iterations to perform. If this is 0, minimation is continued until the results converge without regard to how many iterations it takes. The default value is 0.

        Returns
        -------
        context : simtk.openmm.openmm.Context
            Updated OpenMM Context.
        """

        # Minimizing System
        openmm.LocalEnergyMinimizer.minimize(self.context, tolerance=tolerance, maxIterations=max_iter)

        return self.context

    # -----------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- Custom Forces -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def add_torsion_terms(self, periodicities=[1,2,3,4], phase_default=0.0, v_default=0.0):
        """
        Method that adds the torsional terms with `periodicities` to the OpenMM system 'PeriodicTorsionForce' force group.

        Parameters
        ----------
        periodicities : list of int
            Torsional terms periodicities to be added. If these already exist nothing happens.
        phase_default :
            Value of the phase angle upon creation of the torsional term in radians.
        v_default :
            Value of the torsion barrier height upon creation of the torsional term in kJ/mol.

        Notes
        -----
        This should be used before creating the ParaMol representation of the Force Field.

        Returns
        -------
        context : simtk.openmm.openmm.Context
            Updated OpenMM Context.
        """
        n_added = 0

        if self.force_groups_dict['PeriodicTorsionForce'] in self.force_groups:
            dihedral_force = self.system.getForce(self.force_groups_dict['PeriodicTorsionForce'])
            prev_dihedral = None
            for i in range(dihedral_force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = dihedral_force.getTorsionParameters(i)

                curr_dihedral = [p1, p2, p3, p4]
                if curr_dihedral == prev_dihedral:
                    prev_dihedral = curr_dihedral
                    continue

                prev_dihedral = curr_dihedral
                for n in periodicities:
                    if n != per:
                        dihedral_force.addTorsion(p1, p2, p3, p4, n, phase_default, v_default)
                        n_added += 1

        logging.info("Added {} extra-torsions.".format(n_added))
        # Since the number of torsions has changes it is necessary to reinitialize the context
        # It is also convenient to set the positions so that MD simulations can be started without problems.
        positions_tmp = self.context.getState(getPositions=True).getPositions()
        self.context.reinitialize()
        self.context.setPositions(positions_tmp)

        return self.context

    # -----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Energies and Forces ------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def get_potential_energy(self, positions):
        """
        Method that, given an array of positions (in nanometers), sets the context atomic positions and computes the potential energy.

        Parameters
        ----------
        positions: list or np.array
            Positions array

        Returns
        -------
        epot : float
            Potential energy value in kJ/mol.
        """
        self.context.setPositions(positions)

        epot = self.context.getState(getEnergy=True).getPotentialEnergy()._value

        return epot
 
    def get_kinetic_energy(self, velocities=None):
        """
        Method that computes the kinetic energy.

        Returns
        -------
        ekin : float
            Kinetic energy value in kJ/mol.
        """
        if velocities is not None:
            self.set_velocities(velocities)

        ekin = self.context.getState(getEnergy=True).getKineticEnergy()._value

        return ekin

    def get_forces(self, positions):
        """
        Method that, given an array of positions (in nanometers), sets the context atomic positions and computes the forces.

        Parameters
        ----------
        positions: list or np.array
            Positions array

        Returns
        -------
        forces : np.array
            (Natoms,3) arrary containing forces in kJ/mol/nm.
        """
        self.context.setPositions(positions)

        forces = self.context.getState(getForces=True).getForces(asNumpy=True)._value

        return forces

    # -----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- Bonded Terms methods ------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def set_harmonic_bond_force_parameters(self, ff_bond_terms):
        """
        Method that updates in the OpenMM system the parameters of the terms belonging to the force group 'HarmonicBondForce'.

        Parameters
        ----------
        ff_bond_terms : list of :obj:`ParaMol.Force_field.force_field_term.FFTerm`
            List containing instances of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` which belong to the force group 'HarmonicBondForce'.

        Returns
        -------
        context : simtk.openmm.openmm.Context
            Updated OpenMM Context.
        """
        bond_force = self.system.getForce(self.force_groups_dict["HarmonicBondForce"])

        for bond_term in ff_bond_terms:
            bond_force.setBondParameters(bond_term.idx, *bond_term.atoms, bond_term.parameters["bond_eq"].value,
                                         bond_term.parameters["bond_k"].value)

        bond_force.updateParametersInContext(self.context)

        return self.context

    def set_harmonic_angle_force_parameters(self, ff_angle_terms):
        """
        Method that updates in the OpenMM system the parameters of the terms belonging to the force group 'HarmonicAngleForce'.

        Parameters
        ----------
        ff_angle_terms : list of :obj:`ParaMol.Force_field.force_field_term.FFTerm`
            List containing instances of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` which belong to the force group 'HarmonicAngleForce'.

        Returns
        -------
        context : simtk.openmm.openmm.Context
            Updated OpenMM Context.
        """
        angle_force = self.system.getForce(self.force_groups_dict["HarmonicAngleForce"])

        for angle_term in ff_angle_terms:
            angle_force.setAngleParameters(angle_term.idx, *angle_term.atoms, angle_term.parameters["angle_eq"].value,
                                           angle_term.parameters["angle_k"].value)

        angle_force.updateParametersInContext(self.context)

        return self.context

    def set_periodic_torsion_force_parameters(self, ff_torsion_terms):
        """
        Method that updates in the OpenMM system the parameters of the terms belonging to the force group 'PeriodicTorsionForce'.

        Parameters
        ----------
        ff_torsion_terms : list of :obj:`ParaMol.Force_field.force_field_term.FFTerm`
            List containing instances of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` which belong to the force group 'PeriodicTorsionForce'.

        Returns
        -------
        context : simtk.openmm.openmm.Context
            Updated OpenMM Context.
        """
        torsion_force = self.system.getForce(self.force_groups_dict["PeriodicTorsionForce"])

        for torsion_term in ff_torsion_terms:
            div_value = np.sign(torsion_term.parameters["torsion_phase"].value) * 2.0 * np.pi

            if div_value == 0.0:
                div_value = 2.0 * np.pi

            torsion_force.setTorsionParameters(torsion_term.idx, *torsion_term.atoms,
                                               torsion_term.parameters["torsion_periodicity"].value,
                                               torsion_term.parameters["torsion_phase"].value % div_value,
                                               torsion_term.parameters["torsion_k"].value)

        torsion_force.updateParametersInContext(self.context)

        return self.context

    def set_bonded_parameters(self, force_field_optimizable):
        """
        Method that wraps the methods set_harmonic_bond_force_parameters, set_harmonic_angle_force_parameters, and set_periodic_torsion_force_parameters in order to ease the procedure of updating the system's bonded parameters.

        Parameters
        ----------
        force_field_optimizable : dict
            Dictionary that contains as keys force groups names as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.

        Returns
        -------
        context : simtk.openmm.openmm.Context
            Updated OpenMM Context.
        """

        # Compute all bond term contributions
        if "HarmonicBondForce" in force_field_optimizable:
            self.set_harmonic_bond_force_parameters(force_field_optimizable["HarmonicBondForce"])
        if "HarmonicAngleForce" in force_field_optimizable:
            self.set_harmonic_angle_force_parameters(force_field_optimizable["HarmonicAngleForce"])
        if "PeriodicTorsionForce" in force_field_optimizable:
            self.set_periodic_torsion_force_parameters(force_field_optimizable["PeriodicTorsionForce"])

        return self.context

    # -----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- Non-Bonded Terms methods ----------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def set_non_bonded_parameters_to_zero(self):
        """
        Method that sets all non bonded parameters to zero, namely sigma, epsilon and charge to zero. This is done for all the particles and exceptions.

        Notes
        -----
        When a Context is created, it decides which interactions need to be calculated as exceptions and which ones are "just exclusions". Hence, any exception to be included has to be given a nonzero chargeprod initially. Once the Context is created, the number of exceptions can't be changed.
        More information: https://github.com/pandegroup/openmm/issues/252

        Returns
        -------
        context : simtk.openmm.openmm.Context
            Updated OpenMM Context.
        """

        if self.force_groups_dict['NonbondedForce'] in self.force_groups:
            nonbonded_force = self.system.getForce(self.force_groups.index(self.force_groups_dict['NonbondedForce']))
            for i in range(nonbonded_force.getNumParticles()):
                q, sigma, eps = nonbonded_force.getParticleParameters(i)

                nonbonded_force.setParticleParameters(i, 0.0, 0.0, 0.0)

            if self.context.getPlatform().getName() in ["CPU", "Reference"]:
                # CPU platform raises the following exception if sigma, epsilon and charge are set to 0
                # Exception: updateParametersInContext: The number of non-excluded exceptions has changed
                # Solution is to set them to very small numbers.
                for i in range(nonbonded_force.getNumExceptions()):
                    at1, at2, q, sigma, eps = nonbonded_force.getExceptionParameters(i)

                    if abs(q._value) > 1e-16 and eps._value > 1e-16:
                        nonbonded_force.setExceptionParameters(i, at1, at2, 1e-16, 1e-16, 1e-16)
            else:
                for i in range(nonbonded_force.getNumExceptions()):
                    at1, at2, q, sigma, eps = nonbonded_force.getExceptionParameters(i)
                    nonbonded_force.setExceptionParameters(i, at1, at2, 0.0, 0.0, 0.0)

        # Update parameters in context
        nonbonded_force.updateParametersInContext(self.context)

        return self.context

    def set_nonbonded_parameters(self, force_field_optimizable):
        """
        Method that updates the non-bonded parameters of the OpenMM system.

        Parameters
        ----------
        force_field_optimizable : dict
            Dictionary that contains as keys force groups names as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.

        Returns
        -------
        context : simtk.openmm.openmm.Context
            Updated OpenMM Context.
        """

        if "NonbondedForce" in force_field_optimizable and "Scaling14" not in force_field_optimizable:
            nonbonded_force_terms = force_field_optimizable["NonbondedForce"]
            nonbonded_force = self.system.getForce(nonbonded_force_terms[0].force_group)
            for nonbonded_term in nonbonded_force_terms:
                nonbonded_force.setParticleParameters(nonbonded_term.idx, nonbonded_term.parameters["charge"].value,
                                                      nonbonded_term.parameters["lj_sigma"].value, nonbonded_term.parameters["lj_eps"].value)

            nonbonded_force.updateParametersInContext(self.context)

            # Scaling 1-4 parameters are not being optimized
            scnb = 0.833333  # scee
            scee = 0.5       # scnb
            for i in range(nonbonded_force.getNumExceptions()):
                at1, at2, charge_prod, sigma, eps, = nonbonded_force.getExceptionParameters(i)

                if abs(charge_prod._value) < 1e-8:
                    continue

                # Lorentz-Berthelot rules:
                # \epsilon_{ij} = \sqrt{\epsilon_{ii} * \epsilon_{jj}}
                epsilon = scee * np.sqrt(nonbonded_force_terms[at1].parameters["lj_eps"].value*nonbonded_force_terms[at2].parameters["lj_eps"].value)
                #epsilon = scee * np.sqrt(np.abs(nonbonded_force_terms[at1].parameters["lj_eps"].value) *
                #                         np.abs(nonbonded_force_terms[at2].parameters["lj_eps"].value)) * \
                #          np.sign(nonbonded_force_terms[at1].parameters["lj_eps"].value) * \
                #          np.sign(nonbonded_force_terms[at2].parameters["lj_eps"].value)

                # \sigma_{ij} = (\sigma_{ii} + \sigma_{jj}) / 2
                # Not necessary to scale this value because epsilon controls the LJ 12-6 interaction scaling.
                sigma = 0.5 * (nonbonded_force_terms[at1].parameters["lj_sigma"].value +
                               nonbonded_force_terms[at2].parameters["lj_sigma"].value)

                charge_prod = scnb * \
                              nonbonded_force_terms[at1].parameters["charge"].value * \
                              nonbonded_force_terms[at2].parameters["charge"].value

                nonbonded_force.setExceptionParameters(i, at1, at2, charge_prod, sigma, epsilon)
            nonbonded_force.updateParametersInContext(self.context)

        elif "NonbondedForce" not in force_field_optimizable and "Scaling14" in force_field_optimizable:
            # Scaling 1-4 parameters are being optimized
            scaling_constants = force_field_optimizable["Scaling14"]

            nonbonded_force = self.system.getForce(scaling_constants[0].force_group)

            for i in range(len(scaling_constants)):
                at1, at2, _, sigma, _ = nonbonded_force.getExceptionParameters(i)
                chg1, sigma1, eps1 = nonbonded_force.getParticleParameters(at1)
                chg2, sigma2, eps2 = nonbonded_force.getParticleParameters(at2)
                scee = scaling_constants[i].parameters['scee'].value
                scnb = scaling_constants[i].parameters['scnb'].value

                scee = abs(scee)
                scnb = abs(scnb)

                # Lorentz-Berthelot rules:
                # \epsilon_{ij} = \sqrt{\epsilon_{ii} * \epsilon_{jj}}
                epsilon = scnb * np.sqrt(eps1*eps2)

                # scee*q1*q2
                charge_prod = scee * chg1 * chg2

                nonbonded_force.setExceptionParameters(i, at1, at2, charge_prod, sigma, epsilon)
            nonbonded_force.updateParametersInContext(self.context)

        else:
            pass

        return self.context

    # ------------------------------------------------------------ #
    #                                                              #
    #                  MOLECULAR DYNAMICS METHODS                  #
    #                                                              #
    # ------------------------------------------------------------ #
    def set_positions(self, positions):
        """
        Method that sets the Context positions.

        Parameters
        ----------
        positions : np.array
            Array containing the positions.
        """
        assert self.context is not None, "OpenMM context was not set."

        return self.context.setPositions(positions)

    def set_velocities(self, velocities):
        """
        Method that sets the Context positions.

        Parameters
        ----------
        velocities : np.array
            Array containing the velocities.
        """
        assert self.context is not None, "OpenMM context was not set."

        return self.context.setVelocities(velocities)

    def get_positions(self):
        """
        Method that gets the Context positions.

        Returns
        ----------
        positions : np.array
            Array containing the positions.
        """
        assert self.context is not None, "OpenMM context was not set."

        return self.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)

    def get_velocities(self):
        """
        Method that gets the Context velocities.

        Returns
        ----------
        velocities : np.array
            Array containing the velocities.
        """
        assert self.context is not None, "OpenMM context was not set."

        return self.context.getState(getVelocities=True).getVelocities(asNumpy=True)

    def generate_maxwell_boltzmann_velocities(self, temperature):
        """
        Generate random velocities for the solute.

        """
        assert self.masses_list is not None
        assert self.n_atoms is not None

        # Initiate array
        vel = unit.Quantity(np.zeros([self.n_atoms, 3], np.float64), unit.nanometer / unit.picosecond) # velocities[i,k] is the kth component of the velocity of atom i
        kT = temperature * unit.BOLTZMANN_CONSTANT_kB
        kT = kT.in_units_of(unit.kilogram * unit.meter*unit.meter / (unit.second*unit.second))

        # Assign velocities from the Maxwell-Boltzmann distribution.
        for atom_index in range(self.n_atoms):
            mass = self.masses_list[atom_index]
            if mass._value > 1e-8:
                mass = unit.Quantity(mass._value * 1.66054e-27, unit.kilogram)
                # Standard deviation of velocity distribution for each coordinate for this atom
                sigma = unit.sqrt(kT / mass)
            else:
                sigma = 0.0 * unit.nanometer / unit.picosecond

            for k in range(3):
                # 0.001 is to take into account the ns / ps
                vel[atom_index, k] = (sigma * np.random.standard_normal())

        return vel.in_units_of(unit.nanometer / unit.picosecond)

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PRIVATE METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def _set_force_groups(self):
        """
        Method that sets the force groups of all forces present in the system accordingly to the mapping defined in the forces_groups_dict dictionary.

        Returns
        -------
        force_groups : list of int
            List containing all the force groups present in the system.
        """

        self.force_groups = []
        for force in self.system.getForces():
            # Get force group name
            force_key = force.__class__.__name__
            # Set force group number
            force.setForceGroup(self.force_groups_dict[force_key])
            # Get force group number
            self.force_groups.append(force.getForceGroup())

        return self.force_groups
