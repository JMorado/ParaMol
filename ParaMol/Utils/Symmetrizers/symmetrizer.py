# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Utils.symmetrizer.symmetrizer.Symmetrizer` .
"""
import parmed as pmd
import numpy as np


class Symmetrizer:
    """
    ParaMol bse class that implements methods to symmetrize the ParaMol Force Field.

    Parameters
    ----------
    top_file : str
        Topology file instance.
    """

    def __init__(self, top_file):
        self._top_file = top_file

        # Bonded terms
        self._bond_types = None
        self._angle_types = None
        self._torsion_types = None

        # Non-bonded terms
        self._lj_types = None
        self._charge_types = None
        self._sc_types = None

    def get_symmetries(self, force_field_instance=None):
        """
        Method that gets symmetries with the aid of Parmed. User-defined charge symmetries require the passing of the force_field_instance argument.

        Notes
        -----
        Currently the implemented term types are: bonds, angles, torsions.

        Parameters
        ----------
        force_field_instance : :obj:`ParaMol.Force_field.force_field.ForceField`
            Instance of the ParaMol ForceField (unsymmetrized).

        Returns
        -------
        dict, dict, dict, dict, dict
            Dictionaries containing the symmetry group as keys and information about the parameters of that symmetry group as values.
        """
        # Types to implement:
        # - improper_types;
        # - cmap_types;
        # - pi_torsion_types;
        # - out_of_plane_bend_types;
        # - torsion_torsion_types;
        # - trigonal_angle_types;
        # - urey_bradley_types;
        # - rb_torsion_types;

        self._bond_types = {}
        for i in range(len(self._top_file.bond_types)):
            bond_type = self._top_file.bond_types[i]
            self._bond_types["B{}".format(i)] = {"idx": bond_type.idx,
                                                 "bond_eq": bond_type.req,
                                                 "bond_k": bond_type.k}
        self._angle_types = {}
        for i in range(len(self._top_file.angle_types)):
            angle_type = self._top_file.angle_types[i]
            self._angle_types["A{}".format(i)] = {"idx": angle_type.idx,
                                                  "angle_eq": angle_type.theteq,
                                                  "angle_k": angle_type.k}

        self._torsion_types = {}
        for i in range(len(self._top_file.dihedral_types)):
            dihedral_type = self._top_file.dihedral_types[i]
            self._torsion_types["T{}".format(i)] = {"idx": dihedral_type.idx,
                                                    "torsion_periodicity": dihedral_type.per,
                                                    "torsion_phase": dihedral_type.phase,
                                                    "torsion_k": dihedral_type.phi_k}
        self._sc_types = {}
        for i in range(len(self._top_file.dihedral_types)):
            dihedral_type = self._top_file.dihedral_types[i]
            self._sc_types["SC{}".format(i)] = {"idx": dihedral_type.idx,
                                                "scee": dihedral_type.scee,
                                                "scnb": dihedral_type.scnb}
        """
        # Leave it here to implement LJ types in the future. This code works fort AMBER.
        
        self._lj_types = {}
        for atom_type in self._top_file.LJ_types:
            lj_type = self._top_file.LJ_types[atom_type] - 1  # LJ type starts at 1
            self._lj_types["{}".format(atom_type)] = {"idx": lj_type,
                                                      "eps": self._amber_prmtop.LJ_depth[lj_type],
                                                      "sigma": self._amber_prmtop.LJ_radius[lj_type],
                                                      'lj_type_id': lj_type}
       
        for i in range(len(self._amber_prmtop.LJ_types)):
            print(self._amber_prmtop.LJ_types[i])
            lj_type = self._amber_prmtop.LJ_types[i]
            self._lj_types["{}".format(lj_type)] = {"eps": lj_type.LJ_depth[i],
                                                    "sigma": lj_type.LJ_radius[i],
                                                    'id': lj_type[i]}
        """
        if force_field_instance is not None:
            self.get_charge_symmetries(force_field_instance)

        return self._bond_types, self._angle_types, self._torsion_types, self._sc_types, self._lj_types, self._charge_types

    def get_charge_symmetries(self, force_field_instance):
        """
        Method that gets the user-defined charge symmetries .

        Parameters
        ----------
        force_field_instance : :obj:`ParaMol.Force_field.force_field.ForceField`
            Instance of the ParaMol ForceField (unsymmetrized) to respect AMBER symmetries.

        Returns
        -------
        force_field_instance : :obj:`ParaMol.Force_field.force_field.ForceField`
            Instance of the ParaMol ForceField symmetrized to respect AMBER symmetries.
        """
        self._charge_types = {}
        for sub_force in force_field_instance.force_field["NonbondedForce"]:
            idx = 0
            for force_field_term in sub_force:
                for parameter in force_field_term.parameters.values():
                    # If the param key is not torsion periodicity since this are not handled by ParaMol
                    if parameter.param_key == "charge" and parameter.param_key != "X":
                        if parameter.symmetry_group not in self._charge_types:
                            self._charge_types[parameter.symmetry_group] = {"atoms_idx": [idx]}
                            # , "charge": [parameter.value]}
                        else:
                            self._charge_types[parameter.symmetry_group]["atoms_idx"].append(idx)
                            # self._charge_types[parameter.symmetry_group]["charge"].append(parameter.value)

                        idx += 1

        return self._charge_types

    def symmetrize_force_field(self, force_field_instance):
        """
        Method that symmetrizes the ParaMol force field so that it respect atom-type symmetries.

        Parameters
        ----------
        force_field_instance : :obj:`ParaMol.Force_field.force_field.ForceField`
            Instance of the ParaMol ForceField (unsymmetrized).

        Returns
        -------
        force_field_instance : :obj:`ParaMol.Force_field.force_field.ForceField`
            Instance of the ParaMol ForceField symmetrized to respect.
        """
        # Set bonds to AMBER format
        # Iterate over terms in harmonic bond force
        for sub_force in force_field_instance.force_field["HarmonicBondForce"]:
            for force_field_term in sub_force:
                amber_bonds = [bond for bond in self._top_file.bonds]
                # Iterate over all AMBER bonds
                for bond_idx in range(len(amber_bonds)):
                    bond = amber_bonds[bond_idx]
                    if bond.atom1.idx == force_field_term.atoms[0] and bond.atom2.idx == force_field_term.atoms[1]:
                        force_field_term.parameters['bond_eq'].symmetry_group = "B{}".format(bond.type.idx)
                        force_field_term.parameters['bond_k'].symmetry_group = "B{}".format(bond.type.idx)
                        force_field_term.symmetry_group = "B{}".format(bond.type.idx)

                        # Pop this bond out and break the loop
                        amber_bonds.pop(bond_idx)
                        break

        # Iterate over terms in harmonic angle force
        for sub_force in force_field_instance.force_field["HarmonicAngleForce"]:
            for force_field_term in sub_force:
                amber_angles = [angle for angle in self._top_file.angles]
                # Iterate over all amber angles
                for angle_idx in range(len(amber_angles)):
                    angle = amber_angles[angle_idx]
                    if angle.atom1.idx == force_field_term.atoms[0] and angle.atom2.idx == force_field_term.atoms[
                        1] and angle.atom3.idx == force_field_term.atoms[2]:
                        force_field_term.parameters['angle_eq'].symmetry_group = "A{}".format(angle.type.idx)
                        force_field_term.parameters['angle_k'].symmetry_group = "A{}".format(angle.type.idx)
                        force_field_term.symmetry_group = "A{}".format(angle.type.idx)
                        # Pop this angle out and break the loop
                        amber_angles.pop(angle_idx)
                        break

        # Iterate over terms in torsions
        for sub_force in force_field_instance.force_field["PeriodicTorsionForce"]:
            for force_field_term in sub_force:
                amber_torsions = [torsion for torsion in self._top_file.dihedrals]
                # Iterate over all amber torsions
                for torsion_idx in range(len(amber_torsions)):
                    torsion = amber_torsions[torsion_idx]
                    if torsion.atom1.idx == force_field_term.atoms[0] and torsion.atom2.idx == force_field_term.atoms[
                        1] and torsion.atom3.idx == force_field_term.atoms[2] and torsion.atom4.idx == \
                            force_field_term.atoms[3] and int(torsion.type.per) == int(force_field_term.parameters['torsion_periodicity'].value):
                        force_field_term.parameters['torsion_phase'].symmetry_group = "T{}".format(torsion.type.idx)
                        force_field_term.parameters['torsion_periodicity'].symmetry_group = "T{}".format(torsion.type.idx)
                        force_field_term.parameters['torsion_k'].symmetry_group = "T{}".format(torsion.type.idx)
                        force_field_term.symmetry_group = "T{}".format(torsion.type.idx)

                        # Pop this torsion out and break the loop
                        amber_torsions.pop(torsion_idx)
                        break

        # Iterate over scaling factors; this also
        for sub_force in force_field_instance.force_field["Scaling14"]:
            for force_field_term in sub_force:
                # Iterate over all amber torsions
                periodicities = []
                types = []
                for torsion in self._top_file.dihedrals:
                    if torsion.atom1.idx == force_field_term.atoms[0] and torsion.atom4.idx == force_field_term.atoms[1]:
                        if torsion.type.idx not in types:
                            periodicities.append(torsion.type.per)
                            types.append(torsion.type.idx)

                min_per_idx = periodicities.index(min(periodicities))
                force_field_term.parameters['scee'].symmetry_group = "SC{}".format(types[min_per_idx])
                force_field_term.parameters['scnb'].symmetry_group = "SC{}".format(types[min_per_idx])
                force_field_term.symmetry_group = "SC{}".format(types[min_per_idx])

        return force_field_instance

    def update_term_types_parameters(self, optimizable_parameters):
        """
        Method that updates the term type parameters in the Parmed AMBER topology object.

        Notes
        -----
        This method should be run before writing the AMBER topology file or AMBER .frcmod file.

        Parameters
        ----------
        optimizable_parameters : list of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter`
            List that contains instances of :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` that are optimizable.

        Returns
        -------
        :obj:`parmed.amber._amberparm.AmberParm`
            Instance of Parmed AMBER topology object.
        """
        # Conversions required:
        # Bond equilibrium value * nanometers_to_angstrom
        # Angle equilibrium value * radians_to_degrees
        # Bond force constant / (kcal_mol_to_kj_mol * nanometers_to_angstrom ** 2 * 2)
        # Angle force const / (kcal_mol_to_kj_mol * 2)  # 2 is because
        # Torsion force const / (kcal_mol_to_kj_mol)
        # Torsion phase * radians_to_degrees

        kcal_mol_to_kj_mol = 4.184
        nanometers_to_angstrom = 10.0
        radians_to_degrees = 180.0 / np.pi

        # Iterate over all optimizable parameters
        atom_idx = 0
        for parameter in optimizable_parameters:
            if parameter.symmetry_group[0] == "B":
                idx = self._bond_types[parameter.symmetry_group]["idx"]
                bond_type = self._top_file.bond_types[idx]
                if parameter.param_key == "bond_eq":
                    bond_type.req = parameter.value * nanometers_to_angstrom
                elif parameter.param_key == "bond_k":
                    bond_type.k = parameter.value / (kcal_mol_to_kj_mol * nanometers_to_angstrom ** 2 * 2.0)

            elif parameter.symmetry_group[0] == "A":
                idx = self._angle_types[parameter.symmetry_group]["idx"]
                angle_type = self._top_file.angle_types[idx]
                if parameter.param_key == "angle_eq":
                    angle_type.theteq = parameter.value * radians_to_degrees
                elif parameter.param_key == "angle_k":
                    angle_type.k = parameter.value / (kcal_mol_to_kj_mol * 2.0)

            elif parameter.symmetry_group[0] == "T":
                idx = self._torsion_types[parameter.symmetry_group]["idx"]
                dihedral_type = self._top_file.dihedral_types[idx]
                if parameter.param_key == "torsion_k":
                    dihedral_type.phi_k = parameter.value / kcal_mol_to_kj_mol
                elif parameter.param_key == "torsion_phase":
                    dihedral_type.phase = parameter.value * radians_to_degrees

            elif parameter.symmetry_group[:2] == "SC":
                idx = self._sc_types[parameter.symmetry_group]["idx"]
                dihedral_type = self._top_file.dihedral_types[idx]
                if parameter.param_key == "scnb":
                    dihedral_type.scnb = 1.0 / parameter.value
                elif parameter.param_key == "scee":
                    dihedral_type.scee = 1.0 / parameter.value

            else:
                # Charges
                if parameter.param_key == "charge":
                    if parameter.symmetry_group == "X":
                        self._top_file.atoms[atom_idx].charge = parameter.value
                    else:
                        assert self._charge_types is not None, "Charge symmetries were not set in AmberSymmetrizer. Use the amber_symmetrizer.get_charge_symmetries method to set them."
                        for atom_sub_idx in self._charge_types[parameter.symmetry_group]["atoms_idx"]:
                            self._top_file.atoms[atom_sub_idx].charge = parameter.value

                    atom_idx += 1
                else:
                    return UserWarning("Trying to update parameter key {} that is not charge. Currently not not implemented.".format(parameter.param_key))

        # Re-do values of amber symmetries
        self.get_symmetries()

        return self._top_file

    def save(self, output_file, format=None):
        """
        Method that writes the AMBER, GROMACS, CHARMM, and .mol2 files with the current force field parameters of the self._top_file instance.

        Notes
        -----
        In order to update the self._top_file instance with the optimal parameters, the method update_term_types_parameters should be run before this one.

        Parameters
        ----------
        output_file : str
            Name of the output file (with the correct PARMED suffix)
        format : str
            Format of the file as in Parmed. E.g. 'amber', 'rst7', etc...

        Returns
        -------
        None
        """
        if format is not None:
            return self._top_file.save(output_file, format=format, overwrite=True)
        else:
            return self._top_file.save(output_file, overwrite=True)

