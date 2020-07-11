# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Utils.amber_symmetrizer.AmberSymmetrizer` used to handle AMBER atom types.
"""
import parmed as pmd
import numpy as np


class AmberSymmetrizer:
    """
    ParaMol class that implements methods to symmetrize the ParaMol Force Field so that it respects AMBER atom-types.
    """
    def __init__(self, prmtop_file):
        self._amber_prmtop = pmd.amber.AmberParm(prmtop_file)
        self._bond_types = None
        self._angle_types = None
        self._torsion_types = None
        self._lj_types = None
        self._sc_types = None

    def __str__(self):
        return "AmberParametrization module. Amber .prmtop file in use is {}".format(self._amber_prmtop)

    def get_amber_symmetries(self):
        """
        Method that gets AMBER symmetries with the aid of Parmed.

        Notes
        -----
        Currently the implemented term types are: bonds, angles, torsions, LJ.
        Note that, in order to save memory, AMBER considers parameters having the sames values to be the same, i.e., having the samme symmetry.
        Hence, a workaround for this issue is to attribute slightly different parameters to each force field term in the .frcmod so that AMBER does not assume that they are the same.

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
        for i in range(len(self._amber_prmtop.bond_types)):
            bond_type = self._amber_prmtop.bond_types[i]
            self._bond_types["B{}".format(i)] = {"idx": bond_type.idx,
                                                 "bond_eq": bond_type.req,
                                                 "bond_k": bond_type.k}
        self._angle_types = {}
        for i in range(len(self._amber_prmtop.angle_types)):
            angle_type = self._amber_prmtop.angle_types[i]
            self._angle_types["A{}".format(i)] = {"idx": angle_type.idx,
                                                  "angle_eq": angle_type.theteq,
                                                  "angle_k": angle_type.k}

        self._torsion_types = {}
        for i in range(len(self._amber_prmtop.dihedral_types)):
            dihedral_type = self._amber_prmtop.dihedral_types[i]
            self._torsion_types["T{}".format(i)] = {"idx": dihedral_type.idx,
                                                    "torsion_periodicity": dihedral_type.per,
                                                    "torsion_phase": dihedral_type.phase,
                                                    "torsion_k": dihedral_type.phi_k}
        self._sc_types = {}
        for i in range(len(self._amber_prmtop.dihedral_types)):
            dihedral_type = self._amber_prmtop.dihedral_types[i]
            self._sc_types["SC{}".format(i)] = {"idx": dihedral_type.idx,
                                                "scee": dihedral_type.scee,
                                                "scnb": dihedral_type.scnb}

        self._lj_types = {}
        for atom_type in self._amber_prmtop.LJ_types:
            lj_type = self._amber_prmtop.LJ_types[atom_type] - 1  # LJ type starts at 1
            self._lj_types["{}".format(atom_type)] = {"idx": lj_type,
                                                      "eps": self._amber_prmtop.LJ_depth[lj_type],
                                                      "sigma": self._amber_prmtop.LJ_radius[lj_type],
                                                      'lj_type_id': lj_type}
        """
        for i in range(len(self._amber_prmtop.LJ_types)):
            print(self._amber_prmtop.LJ_types[i])
            lj_type = self._amber_prmtop.LJ_types[i]
            self._lj_types["{}".format(lj_type)] = {"eps": lj_type.LJ_depth[i],
                                                    "sigma": lj_type.LJ_radius[i],
                                                    'id': lj_type[i]}
        """
        return self._bond_types, self._angle_types, self._torsion_types, self._sc_types, self._lj_types

    def set_force_field_to_amber_format(self, force_field_instance):
        """
        Method that symmetrizes the ParaMol force field so that it respect AMBER atom-types.

        Notes
        -----
        This is done to only allow optimization of parameters respecting AMBER atom types.

        Parameters
        ----------
        force_field_instance : :obj:`ParaMol.Force_field.force_field.ForceField`
            Instance of the ParaMol ForceField (unsymmetrized) to respect AMBER symmetries.

        Returns
        -------
        force_field_instance : :obj:`ParaMol.Force_field.force_field.ForceField`
            Instance of the ParaMol ForceField symmetrized to respect AMBER symmetries.
        """
        # Set bonds to AMBER format
        # Iterate over terms in harmonic bond force
        for force_field_term in force_field_instance.force_field["HarmonicBondForce"]:
            amber_bonds = [bond for bond in self._amber_prmtop.bonds]
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
        for force_field_term in force_field_instance.force_field["HarmonicAngleForce"]:
            amber_angles = [angle for angle in self._amber_prmtop.angles]
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
        for force_field_term in force_field_instance.force_field["PeriodicTorsionForce"]:
            amber_torsions = [torsion for torsion in self._amber_prmtop.dihedrals]
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
        for force_field_term in force_field_instance.force_field["Scaling14"]:
            # Iterate over all amber torsions
            periodicities = []
            types = []
            for torsion in self._amber_prmtop.dihedrals:
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
        for parameter in optimizable_parameters:
            if parameter.symmetry_group[0] == "B":
                idx = self._bond_types[parameter.symmetry_group]["idx"]
                bond_type = self._amber_prmtop.bond_types[idx]
                if parameter.param_key == "bond_eq":
                    bond_type.req = parameter.value * nanometers_to_angstrom
                elif parameter.param_key == "bond_k":
                    bond_type.k = parameter.value / (kcal_mol_to_kj_mol * nanometers_to_angstrom ** 2 * 2.0)
            
            elif parameter.symmetry_group[0] == "A":
                idx = self._angle_types[parameter.symmetry_group]["idx"]
                angle_type = self._amber_prmtop.angle_types[idx]
                if parameter.param_key == "angle_eq":
                    angle_type.theteq = parameter.value * radians_to_degrees
                elif parameter.param_key == "angle_k":
                    angle_type.k = parameter.value / (kcal_mol_to_kj_mol * 2.0)

            elif parameter.symmetry_group[0] == "T":
                idx = self._torsion_types[parameter.symmetry_group]["idx"]
                dihedral_type = self._amber_prmtop.dihedral_types[idx]
                if parameter.param_key == "torsion_k":
                    dihedral_type.phi_k = parameter.value / kcal_mol_to_kj_mol
                elif parameter.param_key == "torsion_phase":
                    dihedral_type.phase = parameter.value * radians_to_degrees

            elif parameter.symmetry_group[:2] == "SC":
                idx = self._sc_types[parameter.symmetry_group]["idx"]
                dihedral_type = self._amber_prmtop.dihedral_types[idx]
                if parameter.param_key == "scnb":
                    dihedral_type.scnb = 1.0 / parameter.value
                elif parameter.param_key == "scee":
                    dihedral_type.scee = 1.0 / parameter.value

        # REDO amber symmetries
        self.get_amber_symmetries()

        return self._amber_prmtop

    def save_prmtop(self, output_seed):
        """
        Method that writes the .prmtop AMBER topology file with the current force field parameters of the self._amber_prmtop instance.

        Notes
        -----
        In order to update the self._amber_prmtop instance with the optimal parameters, the method update_term_types_parameters should be run before this one.

        Parameters
        ----------
        output_seed : str
            Name of the output file (without the .prmtop suffix)

        Returns
        -------
        None
        """
        return self._amber_prmtop.save(output_seed + ".prmtop", overwrite=True)

    def save_mol2(self, output_seed):
        """
        Method that writes a .mol2 file with the current parameters of the self._amber_prmtop instance.

        Notes
        -----
        In order to update the self._amber_prmtop instance with the optimal parameters, the method update_term_types_parameters should be run before this one.

        Parameters
        ----------
        output_seed : str
            Name of the output file (without the .prmtop suffix)

        Returns
        -------
        None
        """
        return self._amber_prmtop.save(output_seed + ".mol2")

    def save_frcmod(self, output_seed):
        """
        Method that saves the .frcmod AMBER file with the current force field parameters of the self._amber_prmtop instance.

        Notes
        -----
        In order to update the self._amber_prmtop instance with the optimal parameters, the method update_term_types_parameters should be run before this one.

        Parameters
        ----------
        output_seed : str
            Name of the output file (without the .prmtop suffix)

        Returns
        -------
        None
        """
        frcmod = pmd.tools.writeFrcmod(self._amber_prmtop, output_seed + ".frcmod")

        return frcmod.execute()


