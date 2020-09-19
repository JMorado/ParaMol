"""
Description
-----------
This module defines the :obj:`ParaMol.Force_field.force_field.ForceField` class which is the ParaMol representation of a force field that contains all the information about the force field terms and correspondent parameters (even relatively to those that will not enter the optimization).
"""

import os
import numpy as np
import logging

# ParaMol imports
from .force_field_term import *


class ForceField:
    """
    ParaMol representation of a force field.

    Parameters
    ----------
    openmm_engine : :obj:`ParaMol.MM_engines.openmm.OpenMMEngine`
        ParaMol representation of the OpenMMEngine

    Attributes
    ----------
    force_field : dict
        Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`. This mapping is constructed as given by OpenMM.
    force_field_optimizable : dict
        Same as before but only containing optimizable force field terms. Force groups that do not have optimizable force field terms will not be part of this dictionary.
    force_groups : dict
        Dictionary that defines the mapping between force group names and force group numbers, which is defined accordingly to the information obtained form the OpenMM System.
    optimizable_parameters : list
        List that contains instances of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` that are optimizable.
    optimizable_parameters_values : list of float/int
        List that contains the values of the optimizable force field parameters. This is usually fed into the optimization itself.
    """
    symmetry_group_default = "X"

    def __init__(self, openmm_engine):
        self._openmm = openmm_engine
        self.force_field = None
        self.force_field_optimizable = None
        self.force_groups = None
        self.optimizable_parameters = None
        self.optimizable_parameters_values = None

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def update_force_field(self, optimizable_parameters_values, symmetry_constrained=True):
        """
        Method that updates the value of each Parameter object instance.

        Parameters
        ----------
        optimizable_parameters_values : list of float/int
            List that contains the values of the optimizable force field parameters.
        symmetry_constrained : bool
            Whether or not the optimization is constrained by symmetries.

        Returns
        -------
        optimizable_parameters : list of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter`
            List that contains instances of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` that are optimizable.
        """

        if symmetry_constrained:
            # Update the parameter list taking into account the symmetry constraints
            symm_groups = {}

            # Iterate over all optimizable parameters; update all the parameters that belong do the default
            # symmetry group and save the new paramter values of the others
            for i in range(len(self.optimizable_parameters)):
                parameter = self.optimizable_parameters[i]
                if parameter.symmetry_group == self.symmetry_group_default:
                    # If symmetry group of optimizable parameter is default just update it
                    parameter.value = optimizable_parameters_values[i]
                else:
                    # Optimizable parameter does not belong to default symmetry group
                    if parameter.symmetry_group in symm_groups.keys():
                        # If symmetry group is already in symm_groups
                        if parameter.param_key not in symm_groups[parameter.symmetry_group].keys():
                            # If the param_key is not in symm_groups
                            symm_groups[parameter.symmetry_group][parameter.param_key] = optimizable_parameters_values[i]
                    else:
                        symm_groups[parameter.symmetry_group] = {}
                        symm_groups[parameter.symmetry_group][parameter.param_key] = optimizable_parameters_values[i]

            # The only parameters that were not updated yet were the ones that do not belong to the default
            # symmetry group. We have to iterate over force_field_optimizable and update them.
            for force in self.force_field_optimizable:
                # For a given force, iterate over all force field terms
                for force_field_term in self.force_field_optimizable[force]:
                    # For each term, iterate over all its Parameter instances
                    for parameter in force_field_term.parameters.values():
                        if parameter.optimize and parameter.symmetry_group != self.symmetry_group_default:
                            parameter.value = symm_groups[parameter.symmetry_group][parameter.param_key]
        else:
            for i in range(len(self.optimizable_parameters)):
                self.optimizable_parameters[i].value = optimizable_parameters_values[i]

        # TODO: check if there's a better way do this
        # Make all scee, scnb positive and eps and sigma positive
        if "Scaling14" in self.force_field_optimizable:
            for ff_term in self.force_field_optimizable["Scaling14"]:
                ff_term.parameters["scee"].value = abs(ff_term.parameters["scee"].value)
                ff_term.parameters["scnb"].value = abs(ff_term.parameters["scnb"].value)

        if "NonbondedForce" in self.force_field_optimizable:
            for ff_term in self.force_field_optimizable["NonbondedForce"]:
                ff_term.parameters["lj_eps"].value = abs(ff_term.parameters["lj_eps"].value)
                ff_term.parameters["lj_sigma"].value = abs(ff_term.parameters["lj_sigma"].value)

        return self.optimizable_parameters

    def create_force_field(self, opt_bonds=False, opt_angles=False, opt_torsions=False, opt_charges=False, opt_lj=False, opt_sc=False, ff_file=None):
        """
        Method that wraps the methods create_force_field_from_openmm/read_ff_file and create_force_field_optimizable in order to ease the procedure of creating a ParaMol representation of a force field.

        Notes
        -----
        If `ff_file` is not `None` the force field will be created from the provided file. The system stored in :obj:`ParaMol.MM_engines.openmm.OpenMMEngine` should contain exactly the same forces and force field terms as the ones in this file.

        Parameters
        ----------
        opt_bonds : bool
            Flag that signals whether or not the bond parameters will be optimized.
        opt_angles : bool
             Flag that signals whether or not the angle parameters will be optimized.
        opt_torsions : bool
            Flag that signals whether or not the dihedral parameters will be optimized.
        opt_charges : bool
            Flag that signal whether or not the charges will be optimized.
        opt_lj : bool
            Flag that signal whether or not the charges will be optimized.
        opt_sc : bool
            Flag that signal whether or not the 1-4 Lennard-Jones and electrostatic scaling factor will be optimized.
        ff_file : str
            Name of the ParaMol force field file to be read.

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """

        # Set empty Force Field and Force Group dictionaries
        self.force_field = {}
        self.force_groups = {}

        if ff_file is None:
            # No .ff file was provided - create parameter list from force field
            logging.info("Creating force field directly from OpenMM.")
            assert self._openmm is not None, "OpenMM was not set."
            self.create_force_field_from_openmm(opt_bonds, opt_angles, opt_torsions, opt_charges, opt_lj, opt_sc)
        else:
            logging.info("Creating force from .ff file named '{}'.".format(ff_file))
            # A .param file was provided - create parameter list from the information contained in this file.
            assert os.path.exists(ff_file), "\t * ERROR: .param file provided - {} - does not exist."
            self.read_ff_file(ff_file)

        self.create_force_field_optimizable()

        return self.force_field

    def create_force_field_from_openmm(self, opt_bonds, opt_angles, opt_torsions, opt_charges, opt_lj, opt_sc):
        """
        Method that creates the force field dictionary that contains all the FFTerms of the force field as given by OpenMM.
        The FFTerms are grouped in lists that can be accessed by the key of the correspondent force group.

        Notes
        -----
        This method constructs the force_groups dictionary, and calls the methods create_harmonic_bond_force_field,create_harmonic_angle_force_field, create_periodic_torsion_force_field, create_nonbonded_force_field in order to construct the force_filed dictionary.

        Parameters
        ----------
        opt_bonds : bool
            Flag that signals whether or not the bond parameters will be optimized.
        opt_angles : bool
             Flag that signals whether or not the angle parameters will be optimized.
        opt_torsions : bool
            Flag that signals whether or not the dihedral parameters will be optimized.
        opt_charges : bool
            Flag that signal whether or not the charges will be optimized.
        opt_lj : bool
            Flag that signal whether or not the charges will be optimized.
        opt_sc : bool
            Flag that signal whether or not the 1-4 Lennard-Jones and electrostatic scaling factor will be optimized.

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """
        # Iterate over all forces present in the system and determine the force groups
        forces = self._openmm.system.getForces()
        for i in range(len(forces)):
            force = forces[i]
            # Get force group name
            # Alternatively,force_key = force.__str__().split(".")[3].split(";")[0]
            force_key = force.__class__.__name__
            # Set force group number
            force.setForceGroup(i)
            assert force_key not in self.force_groups, "\t * ERROR: Force {} already in the dictionary.".format(force_key)
            self.force_groups[force_key] = i

        # Add extra force group for 1-4 scaling factors
        force_key = "Scaling14"
        assert force_key not in self.force_groups, "\t * ERROR: Force {} already in the dictionary.".format(force_key)
        self.force_groups["Scaling14"] = self.force_groups["NonbondedForce"]

        # Create the force field from OpenMM
        self.create_harmonic_bond_force_field(opt_bonds)
        self.create_harmonic_angle_force_field(opt_angles)
        self.create_periodic_torsion_force_field(opt_torsions)
        self.create_nonbonded_force_field(opt_charges, opt_lj, opt_sc)

        return self.force_field

    def create_force_field_optimizable(self):
        """
        Method that creates the optimizable force field dictionary that contains all the optimizable FFTerms.
        The FFTerms are grouped in lists that can be accessed by the key of the correspondent force group.

        Returns
        -------
        force_field_optimizable : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """

        assert self.force_field is not None, "\t * force_field dictionary was not created yet. Run create_force_field " \
                                             "method before"
        self.force_field_optimizable = {}

        # Iterate over all existent forces
        for force in self.force_field:
            # For a given force, iterate over all force field terms
            for force_field_term in self.force_field[force]:
                # For each term, iterate over all its Parameter instances
                for parameter in force_field_term.parameters.values():
                    if parameter.optimize:
                        if force not in self.force_field_optimizable:
                            self.force_field_optimizable[force] = []
                        self.force_field_optimizable[force].append(force_field_term)

                        break

        return self.force_field_optimizable

    def get_optimizable_parameters(self, symmetry_constrained=True):
        """
        Method that gets the lists containing all optimizable Parameter instances and parameter values.

        Parameters
        ----------
        symmetry_constrained : bool
            Whether or not the optimization is constrained by symmetries.

        Returns
        -------
        optimizable_parameters, optimizable_parameters_values : list of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter`, list of int/float
            Attributes of self.
        """

        assert self.force_field_optimizable is not None, "\t * force_field_optimizable dictionary was not created yet." \
                                                         " First run create_force_field_optimizable method."

        self.optimizable_parameters = []
        self.optimizable_parameters_values = []

        # Multiplicity of the parameters
        ref_parameters = {}

        if symmetry_constrained:
            # Keep track of symmetry groups already included
            symm_groups = {}
            # Iterate over all existent forces
            for force in self.force_field_optimizable:
                # For a given force, iterate over all force field terms
                for force_field_term in self.force_field_optimizable[force]:
                    # For each term, iterate over all its Parameter instances
                    for parameter in force_field_term.parameters.values():
                        if parameter.optimize:
                            if parameter.symmetry_group == self.symmetry_group_default:
                                # If symmetry group is the default ("X")
                                self.optimizable_parameters.append(parameter)
                                self.optimizable_parameters_values.append(parameter.value)
                            elif parameter.symmetry_group in symm_groups.keys():
                                # If group is not the default one ("X")
                                # but that symmetry_group is already in symm_groups
                                if parameter.param_key not in symm_groups[parameter.symmetry_group]:
                                    # Add missing param_key
                                    symm_groups[parameter.symmetry_group].append(parameter.param_key)
                                    self.optimizable_parameters.append(parameter)
                                    self.optimizable_parameters_values.append(parameter.value)
                                    # Parameter multiplicity
                                    ref_parameters[parameter.symmetry_group].update({parameter.param_key : parameter})
                                    parameter.multiplicity = 1
                                else:
                                    # Increase multiplicity of the reference parameter
                                    ref_parameters[parameter.symmetry_group][parameter.param_key].multiplicity += 1
                            else:
                                # If group is not the default one ("X") and not in symm_groups
                                symm_groups[parameter.symmetry_group] = []
                                symm_groups[parameter.symmetry_group].append(parameter.param_key)
                                self.optimizable_parameters.append(parameter)
                                self.optimizable_parameters_values.append(parameter.value)

                                # Parameter multiplicity
                                ref_parameters[parameter.symmetry_group] = {parameter.param_key : parameter}
                                parameter.multiplicity = 1

        else:
            # Iterate over all existent forces
            for force in self.force_field_optimizable:
                # For a given force, iterate over all force field terms
                for force_field_term in self.force_field_optimizable[force]:
                    # For each term, iterate over all its Parameter instances
                    for parameter in force_field_term.parameters.values():
                        if parameter.optimize:
                            self.optimizable_parameters.append(parameter)
                            self.optimizable_parameters_values.append(parameter.value)

        return self.optimizable_parameters, self.optimizable_parameters_values

    def create_harmonic_bond_force_field(self, opt_bonds):
        """
        Method that creates the part of the force field regarding OpenMM's force 'HarmonicBondForce'.

        Parameters
        ----------
        opt_bonds : bool
            Flag that signals whether or not the bond parameters will be optimized.

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """
        force_key = "HarmonicBondForce"
        
        assert force_key not in self.force_field, "\t * ERROR: " \
                                                  "Force group {} already exists.".format(force_key)
        # Create empty list for
        self.force_field[force_key] = []

        bond_force = self._openmm.system.getForce(self.force_groups[force_key])
        for i in range(bond_force.getNumBonds()):
            # Create the FFTerm for this bond term
            at1, at2, length, k = bond_force.getBondParameters(i)
            force_field_term = FFTerm(self.force_groups[force_key], i, [at1, at2])

            # Add parameters to this FFTerm
            force_field_term.add_parameter(self.symmetry_group_default, int(opt_bonds), "bond_eq", length._value)
            force_field_term.add_parameter(self.symmetry_group_default, int(opt_bonds), "bond_k", k._value)

            # Append FFTerm to ForceField
            self.force_field[force_key].append(force_field_term)

        return self.force_field

    def create_harmonic_angle_force_field(self, opt_angles):
        """
        Method that creates the part of the force field regarding OpenMM's force 'HarmonicAngleForce'.

        Parameters
        ----------
        opt_angles : bool
            Flag that signals whether or not the angle parameters will be optimized.

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """
        force_key = "HarmonicAngleForce"
        assert force_key not in self.force_field, "\t * ERROR: " \
                                                  "Force group {} already exists.".format(force_key)
        # Create empty list for
        self.force_field[force_key] = []

        angle_force = self._openmm.system.getForce(self.force_groups[force_key])
        for i in range(angle_force.getNumAngles()):
            # Create the FFTerm for this bond term
            at1, at2, at3, angle, k = angle_force.getAngleParameters(i)
            force_field_term = FFTerm(self.force_groups[force_key], i, [at1, at2, at3])

            # Add parameters to this FFTerm
            force_field_term.add_parameter(self.symmetry_group_default, int(opt_angles), "angle_eq", angle._value)
            force_field_term.add_parameter(self.symmetry_group_default, int(opt_angles), "angle_k", k._value)

            # Append FFTerm to ForceField
            self.force_field[force_key].append(force_field_term)

        return self.force_field

    def create_periodic_torsion_force_field(self, opt_torsions):
        """
        Method that creates the part of the force field regarding OpenMM's force 'PeriodicTorsionForce'.

        Parameters
        ----------
        opt_torsions : bool
            Flag that signals whether or not the torsion parameters will be optimized.

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """
        force_key = "PeriodicTorsionForce"
        assert force_key not in self.force_field, "\t * ERROR: " \
                                                  "Force group {} already exists.".format(force_key)
        # Create empty list for
        self.force_field[force_key] = []

        dihedral_force = self._openmm.system.getForce(self.force_groups[force_key])
        for i in range(dihedral_force.getNumTorsions()):
            # Create the FFTerm for this bond term
            at1, at2, at3, at4, per, phase, k = dihedral_force.getTorsionParameters(i)
            force_field_term = FFTerm(self.force_groups[force_key], i, [at1, at2, at3, at4])

            # Add parameters to this FFTerm
            # OBS: currently not possible to optimize the periodicity
            force_field_term.add_parameter(self.symmetry_group_default, 0, "torsion_periodicity", int(per))
            force_field_term.add_parameter(self.symmetry_group_default, int(opt_torsions), "torsion_phase", phase._value)
            force_field_term.add_parameter(self.symmetry_group_default, int(opt_torsions), "torsion_k", k._value)

            # Append FFTerm to ForceField
            self.force_field[force_key].append(force_field_term)

        return self.force_field

    def create_nonbonded_force_field(self, opt_charges, opt_lj, opt_sc):
        """
        Method that creates the part of the force field regarding OpenMM's force 'NonbondedForce'.

        Parameters
        ----------
        opt_charges : bool
            Flag that signals whether or not the charge parameters will be optimized.
        opt_lj : bool
            Flag that signals whether or not the Lennard-Jones 12-6 parameters will be optimized.
        opt_sc : bool
            Flag that signals whether or not the 1-4 Lennard-Jones and electrostatic scaling factors's parameters will be optimized.

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """
        force_key = "NonbondedForce"
        assert force_key not in self.force_field, "\t * ERROR: " \
                                                  "Force group {} already exists.".format(force_key)
        # Create empty list for
        self.force_field[force_key] = []

        nonbonded_force = self._openmm.system.getForce(self.force_groups[force_key])
        for i in range(nonbonded_force.getNumParticles()):
            # Create the FFTerm for this bond term
            charge, sigma, eps = nonbonded_force.getParticleParameters(i)
            force_field_term = FFTerm(self.force_groups[force_key], i, [i])

            # Add parameters to this FFTerm
            force_field_term.add_parameter(self.symmetry_group_default, int(opt_charges), "charge", charge._value)
            force_field_term.add_parameter(self.symmetry_group_default, int(opt_lj), "lj_sigma", sigma._value)
            force_field_term.add_parameter(self.symmetry_group_default, int(opt_lj), "lj_eps", eps._value)

            # Append FFTerm to ForceField
            self.force_field[force_key].append(force_field_term)

        # Create empty list for 1-4 scaling
        force_key = "Scaling14"
        assert force_key not in self.force_field, "\t * ERROR: " \
                                                  "Force group {} already exists.".format(force_key)
        # Create empty list for
        self.force_field[force_key] = []

        for i in range(nonbonded_force.getNumExceptions()):
            at1, at2, charge_prod, sigma, eps, = nonbonded_force.getExceptionParameters(i)
            force_field_term = FFTerm(self.force_groups[force_key], i, [at1, at2])

            if abs(charge_prod._value) < 1e-8 and abs(eps._value) < 1e-8:
                # No scaling
                scee = 0.0
                scnb = 0.0
                force_field_term.add_parameter(self.symmetry_group_default, 0, "scee", float(scee))
                force_field_term.add_parameter(self.symmetry_group_default, 0, "scnb", float(scnb))
                continue
            else:
                # Determine default scaling
                charge_at1, sigma_at1, eps_at1 = nonbonded_force.getParticleParameters(at1)
                charge_at2, sigma_at2, eps_at2 = nonbonded_force.getParticleParameters(at2)

                try:
                    scee = charge_prod / (charge_at1 * charge_at2)
                except:
                    scee = 1 / 1.2

                try:
                    scnb = eps / np.sqrt(eps_at1 * eps_at2)
                except:
                    scnb = 1 / 2.0

                # Add parameters to this FFTerm
                force_field_term.add_parameter(self.symmetry_group_default, int(opt_sc), "scee", float(scee))
                force_field_term.add_parameter(self.symmetry_group_default, int(opt_sc), "scnb", float(scnb))

            # Append FFTerm to ForceField
            self.force_field[force_key].append(force_field_term)

        return self.force_field

    def write_ff_file(self, file_name):
        """
        Method that writes the force field parameters in the standard format used by ParaMol (usually .ff extension).

        Parameters
        ----------
        file_name : str
            Name of the file to be written.

        Returns
        -------
        `True` if file was closed successfully. `False` otherwise.
        """

        logging.info("Writing force field to .ff file named '{}'.".format(file_name))
        # Open file for writing
        ff_file = open(file_name, 'w')

        # Iterate over all existent forces
        for force in self.force_field:
            ff_file.write("{} {:3d} \n".format(force, self.force_groups[force]))
            # For a given force, iterate over all force field terms
            for force_field_term in self.force_field[force]:
                ff_term_line = ("{:3d} " + "{:3d} " * len(force_field_term.atoms)).format(force_field_term.idx, *force_field_term.atoms)
                # For each term, iterate over all its Parameter instances
                optimization_flags = ""
                for parameter in force_field_term.parameters.values():
                    ff_term_line += "{:16.8f} ".format(parameter.value)
                    optimization_flags += "{:3d} ".format(int(parameter.optimize))

                ff_term_line += optimization_flags
                ff_term_line += "  " + str(parameter.symmetry_group) + " \n"

                ff_file.write(ff_term_line)

        ff_file.write("END \n")
        return ff_file.close()

    def read_ff_file(self, file_name):
        """
        Method that reads the force field parameters in the standard format used by ParaMol (usually .ff extension) and creates its ParaMol representation.

        Parameters
        ----------
        file_name : str
            Name of the file to be read.

        Returns
        -------
        `True` if file was closed successfully. `False` otherwise.
        """
        # Open file for writing
        ff_file = open(file_name, 'r')

        # Iterate over all existent forces
        for line in ff_file:
            line_split = line.split()

            if 'END' in line_split:
                break
            elif len(line_split) == 2:
                # A new force was found; set the force key and force group
                force_key = line_split[0]
                force_group = int(line_split[1])
                self.force_groups[force_key] = force_group
                # Create empty list for the force_key
                self.force_field[force_key] = []
                continue
            else:
                if force_key == 'HarmonicBondForce':
                    idx, at1, at2, bond_eq, bond_k, bond_eq_opt, bond_k_opt, symm_group = line_split
                    force_field_term = FFTerm(self.force_groups[force_key], int(idx), [int(at1), int(at2)])
                    # Add parameters to this FFTerm
                    force_field_term.add_parameter(symm_group, int(bond_eq_opt), "bond_eq", float(bond_eq))
                    force_field_term.add_parameter(symm_group, int(bond_k_opt), "bond_k", float(bond_k))
                    # Append FFTerm to ForceField
                    self.force_field[force_key].append(force_field_term)

                elif force_key == 'HarmonicAngleForce':
                    idx, at1, at2, at3, angle_eq, angle_k, angle_eq_opt, angle_k_opt, symm_group = line_split
                    force_field_term = FFTerm(self.force_groups[force_key], int(idx), [int(at1), int(at2), int(at3)])
                    # Add parameters to this FFTerm
                    force_field_term.add_parameter(symm_group, int(angle_eq_opt), "angle_eq", float(angle_eq))
                    force_field_term.add_parameter(symm_group, int(angle_k_opt), "angle_k", float(angle_k))
                    # Append FFTerm to ForceField
                    self.force_field[force_key].append(force_field_term)

                elif force_key == 'PeriodicTorsionForce':
                    idx, at1, at2, at3, at4, torsion_periodicity, torsion_phase,\
                    torsion_k, torsion_periodicity_opt, torsion_phase_opt, torsion_k_opt, symm_group = line_split
                    force_field_term = FFTerm(self.force_groups[force_key], int(idx), [int(at1), int(at2), int(at3), int(at4)])
                    # Add parameters to this FFTerm
                    # OBS: currently not possible to optimize the periodicity
                    assert int(torsion_periodicity_opt) == 0, \
                        "Flag to parameterize torsions was set to {} but this is not possible.".format(torsion_periodicity_opt)

                    force_field_term.add_parameter(symm_group, int(0), "torsion_periodicity", int(float(torsion_periodicity)))
                    force_field_term.add_parameter(symm_group, int(torsion_phase_opt), "torsion_phase", float(torsion_phase))
                    force_field_term.add_parameter(symm_group, int(torsion_k_opt), "torsion_k", float(torsion_k))
                    # Append FFTerm to ForceField
                    self.force_field[force_key].append(force_field_term)

                elif force_key == 'NonbondedForce':
                    idx, at, charge, sigma, eps, charge_opt, sigma_opt, eps_opt, symm_group = line_split
                    force_field_term = FFTerm(self.force_groups[force_key], int(idx), [int(at)])

                    # Add parameters to this FFTerm
                    force_field_term.add_parameter(symm_group, int(charge_opt), "charge", float(charge))
                    force_field_term.add_parameter(symm_group, int(sigma_opt), "lj_sigma", float(sigma))
                    force_field_term.add_parameter(symm_group, int(eps_opt), "lj_eps", float(eps))
                    # Append FFTerm to ForceField
                    self.force_field[force_key].append(force_field_term)

                elif force_key == 'Scaling14':
                    idx, at1, at2, scee, scnb, scee_opt, scnb_opt, symm_group = line_split
                    force_field_term = FFTerm(self.force_groups[force_key], int(idx), [int(at1), int(at2)])

                    # Add parameters to this FFTerm
                    force_field_term.add_parameter(symm_group, int(scee_opt), "scee", float(scee))
                    force_field_term.add_parameter(symm_group, int(scnb_opt), "scnb", float(scnb))
                    # Append FFTerm to ForceField
                    self.force_field[force_key].append(force_field_term)

        return ff_file.close()

    def optimize_selection(self, lower_idx, upper_idx, change_other=False):
        """
        Methods that sets a parameter as optimizable if it belongs to a force field term for which at least one of the atoms's indices is greather than lower_idx and lower than upper_idx.

        Notes
        -----
        If [10,20] is given a the lower_idx list and [15,25] is given as the upper_idx list, the selection will comprise the atoms between 10-15 and 20-25.

        Parameters
        ----------
        lower_idx : list of int
            Lower index limits.
        upper_idx : list of int
            Upper index limits.
        change_other : bool
            Whether or not the remaining parameter's optimization state is to be set to False. (default is False, i.e., their optimization state is not change)

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """

        assert len(lower_idx) == len(upper_idx)

        # Iterate over all forces
        for force in self.force_field:
            # Iterate over all force field term
            for force_field_term in self.force_field[force]:
                # Iterate over all atoms of a given force field term
                for at in force_field_term.atoms:
                    for i in range(len(lower_idx)):
                        low_limit = lower_idx[i]
                        upper_limit = upper_idx[i]

                        if (at >= low_limit) and (at <= upper_limit):
                            for parameter in force_field_term.parameters.values():
                                parameter.optimize = 1
                        elif (at < low_limit) or (at > upper_limit) and change_other:
                            for parameter in force_field_term.parameters.values():
                                parameter.optimize = 0
                        else:
                            # If outside range but change other is False
                            pass

        return self.force_field

    def optimize_torsions(self, torsions, change_other_torsions=False, change_other_parameters=False):
        """
        Methods that sets as optimizable all parameters of the torsions contained in the listed passed as an argument.

        Parameters
        ----------
        torsions : list of lists
            List of list, wherein the inner lists contain indices of the quartets of atoms that define the torsion to be optimized.
        change_other_torsions : bool
            Whether or not the remaining torsions's optimization state is to be set to False. (default is False, i.e., their optimization state is not changed)
        change_other_parameters : bool
            Whether or not the remaining parameters' optimization state is to be set to False. (default is False, i.e., their optimization state is not changed)

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """
        # ----------------------------------------------------------------------------------------------
        # Set optimization flag in ParaMol Force Field representation for given dihedrals
        # ----------------------------------------------------------------------------------------------

        for force in self.force_field:
            if force == 'PeriodicTorsionForce':
                for force_field_term in self.force_field[force]:
                    for parameter in force_field_term.parameters.values():
                        # If the param key is not torsion periodicity since this are not handled by ParaMol
                        if parameter.param_key != "torsion_periodicity":
                            if force_field_term.atoms in torsions:
                                parameter.optimize = 1
                            elif change_other_torsions:
                                parameter.optimize = 0
            elif change_other_parameters:
                for force_field_term in self.force_field[force]:
                    for parameter in force_field_term.parameters.values():
                        parameter.optimize = 0
            else:
                pass

        return self.force_field

    def optimize_scaling_constants(self, atom_pairs, change_other_sc=False, change_other_parameters=False):
        """
        Methods that sets as optimizable all parameters of the scaling factors contained in the listed passed as an argument.

        Parameters
        ----------
        atom_pairs : list of lists
            List of list, wherein the inner lists contain indices of the pair of atoms for which the scaling factors are to be optimized.
        change_other_sc : bool
            Whether or not the remaining scaling constants's optimization state is to be set to False. (default is False, i.e., their optimization state is not changed)
        change_other_parameters : bool
            Whether or not the remaining parameters' optimization state is to be set to False. (default is False, i.e., their optimization state is not changed)

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """
        # ----------------------------------------------------------------------------------------------
        # Set optimization flag in ParaMol Force Field representation for given dihedrals
        # ----------------------------------------------------------------------------------------------

        for force in self.force_field:
            if force == 'Scaling14':
                for force_field_term in self.force_field[force]:
                    for parameter in force_field_term.parameters.values():
                        if force_field_term.atoms in atom_pairs:
                            parameter.optimize = 1
                        elif change_other_sc:
                            parameter.optimize = 0
            elif change_other_parameters:
                for force_field_term in self.force_field[force]:
                    for parameter in force_field_term.parameters.values():
                        parameter.optimize = 0
            else:
                pass

        return self.force_field

    def optimize_torsions_by_symmetry(self, torsions, change_other_torsions=False, change_other_parameters=False, set_zero=False):
        """
        Methods that sets as optimizable all parameters of the torsions with the same symmetry groups as the ones contained in the listed passed as an argument.

        Parameters
        ----------
        torsions : list of lists
            List of list, wherein the inner lists contain indices of the quartets of atoms that define the torsion to be optimized.
        change_other_torsions : bool
            Whether or not the remaining torsions's optimization state is to be set to False. (default is False, i.e., their optimization state is not changed)
        change_other_parameters : bool
            Whether or not the remaining parameters' optimization state is to be set to False. (default is False, i.e., their optimization state is not changed)
        set_zero : bool
            Whether or not to set the force constant of the optimizable torsions to 0.

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """

        # ----------------------------------------------------------------------------------------------
        # Set optimization flag in ParaMol Force Field representation for given dihedrals
        # ----------------------------------------------------------------------------------------------

        # Get symmetry groups of given dihedrals
        dihedral_types = []
        for force in self.force_field:
            if force == 'PeriodicTorsionForce':
                for force_field_term in self.force_field[force]:
                    for parameter in force_field_term.parameters.values():
                        if parameter.param_key is not "torsion_periodicity":
                            if force_field_term.atoms in torsions:
                                dihedral_types.append(parameter.symmetry_group)

        # Change the necessary optimization states
        for force in self.force_field:
            if force == 'PeriodicTorsionForce':
                for force_field_term in self.force_field[force]:
                    for parameter in force_field_term.parameters.values():
                        # If the param key is not torsion periodicity since this are not handled by ParaMol
                        if parameter.param_key != "torsion_periodicity":
                            if parameter.symmetry_group in dihedral_types:
                                parameter.optimize = 1
                                if parameter.param_key == "torsion_k" and set_zero:
                                    parameter.value = 0.0
                            elif change_other_torsions:
                                parameter.optimize = 0
            elif change_other_parameters:
                for force_field_term in self.force_field[force]:
                    for parameter in force_field_term.parameters.values():
                        parameter.optimize = 0
            else:
                pass

        return self.force_field

    def set_parameter_optimization(self, force_key, idx, param_key, optimize):
        """
        Method that for the force field term with index `idx` of the force `force_key` set the parameter with name `param_key` to the optimization state in `optimize`.

        Parameters
        ----------
        force_key : str
            Name of the force.
        idx : int
            Index of the force field term.
        param_key : str
            Name of the parameter.
        optimize : bool
            Optimization state (0 or 1).

        Returns
        -------
        force_field : dict
            Dictionary that contains as keys force groups names and as values and the correspondent :obj:`ParaMol.Force_field.force_field_term.FFTerm`.
        """
        self.force_field[force_key][idx].parameters[param_key].optimize = optimize

        return self.force_field
