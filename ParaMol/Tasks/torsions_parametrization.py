# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Objective_function.Tasks.task.TorsionsParametrization` class used to perform parametrization of rotatable (soft) dihedrals.
"""

import simtk.unit as unit
import logging

# ParaMol modules
from .task import *
from .torsions_scan import *
from .parametrization import *
from .torsions_scan import *
from ..Utils.interface import *


class TorsionsParametrization(Task):
    pdb_file = "dihedral_scan.pdb"

    def __init__(self):
        pass

    # ------------------------------------------------------------ #
    #                                                              #
    #                       PUBLIC  METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_task(self, settings, systems, parameter_space=None, objective_function=None, optimizer=None, scan_settings=[-180.0,180.0,10], torsions_to_freeze=None, ase_constraints=None, optimize_mm=False, optimize_mm_type="FREEZE_ATOMS", optimize_qm_before_scan=False, parametrization_type="SIMULTANEOUS"):
        """
        Method that can be used to perform automatic parametrization of the rotatable (soft) torsions using 1-D scans.

        Notes
        -----
        This method only samples equivalent bonds - defined as bonds having exactly the same dihedral types - once. Furthermore, it has available two parametrization schemes:
            - "SEQUENTIAL": parametrization of torsions of unique bonds is done sequentially, i.e., just one torsion is optimized at a time. Moreover, all other soft torsions are kept frozen during the 1D scan.
            - "SIMULTANEOUS": parametrization of torsions of unique bonds is done concomitantly, i.e., all torsions are optimized at the same time. Moreover, all soft torsions are allowed to relax during the 1D scans.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instance of the parameter space.
        objective_function : :obj:`ParaMol.Objective_function.objective_function.ObjectiveFunction`
            Instance of the objective function.
        optimizer : one of the optimizers defined in the subpackage :obj:`ParaMol.Optimizers`
            Instance of the optimizer.
        scan_settings: list of list of float
            List of lists wherein each the most inner list should contain 3 floats defining the settings of the scan in the
            to be performed and in the following order: lower torsion angle, upper torsion angle, angle step (in degrees).
            Example: 1D-Scan  [ [180.0,-180.0,-10.0] ].
        torsions_to_freeze : list of list of int
            List of lists of wherein each inner list should contain 4 integers defining a torsion to be kept fixed (default is None)
        ase_constraints : list of ASE constraints.
            List of ASE constraints to be applied during the scans (default is None)
        optimize_mm : bool
            Flag that controls whether a MM geometry optimization is performed before every QM optimization (default is False).
        optimize_mm_type : str
            Constraint to be used when performing MM optimization. Available options are 'freeze_atoms' or 'freeze_dihedral'. 'freeze_atoms' is recommended.
        optimize_qm_before_scan : bool
            Flag that controls whether a QM geometry optimization is performed before the scan (default is False).
        parametrization_type : str
            Available options are "SIMULTANEOUS" (default) or "SEQUENTIAL".

        Returns
        -------
        systems, parameter_space, objective_function, optimizer
        """
        print("!=================================================================================!")
        print("!                            TORSIONS PARAMETRIZATION                             !")
        print("!=================================================================================!")

        assert parametrization_type.upper() in ["SIMULTANEOUS", "SEQUENTIAL"], \
            "Parametrization type {} not recognized. Available options are 'simultanenous' (default) or 'sequential'.".format(parametrization_type)

        if torsions_to_freeze is None:
            torsions_to_freeze = []
        if ase_constraints is None:
            ase_constraints = []

        # Create QM Engines
        for system in systems:
            if system.interface is None:
                system.interface = ParaMolInterface()

            system.create_qm_engines(settings.qm_engine["qm_engine"], settings.qm_engine[settings.qm_engine["qm_engine"].lower()])

        for system in systems:
            # Create TorsionScan instance
            torsions_scan = TorsionScan()

            # Get RDKit mol and conf
            rdkit_mol, rdkit_conf = torsions_scan.get_rdkit_mol_conf(system)

            # Get rotatable bonds
            rotatable_bonds = self.get_rotatable_bonds(rdkit_mol, methyl=False)
            rotatable_bonds = rotatable_bonds[2:]

            # Get rotatable dihedral
            rotatable_dihedrals = self.get_rotatable_torsions(system, rotatable_bonds)

            if rotatable_dihedrals is []:
                print("No rotatable dihedrals were found.")
                return systems, parameter_space, objective_function, optimizer
            else:
                print("Found {} rotatable bonds and {} rotatable torsions.".format(len(rotatable_bonds), len(rotatable_bonds)))

                if optimize_qm_before_scan:
                    logging.info("Performing QM optimization before starting torsions parametrization.")
                    # ----------------------------------------------------------- #
                    #                   Perform QM optimization                   #
                    # ----------------------------------------------------------- #
                    positions = system.engine.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(
                        asNumpy=True)

                    # Get optimized geometry
                    positions, _, _ = system.qm_engine.qm_engine.run_calculation(
                        coords=positions.in_units_of(unit.angstrom)._value,
                        label=0, calc_type="optimization")
                    positions = positions * unit.nanometers

                    # Set optimized geometry in RDKit conf
                    torsions_scan.set_positions_rdkit_conf(rdkit_conf, positions.in_units_of(unit.angstrom)._value)

                # Perform sampling of dihedral
                bond_symm_global = []
                bond_id = 0
                dihedrals_to_optimize = []
                for bond in rotatable_dihedrals:
                    print("\nPerform sampling of torsions of bond with atoms {} {}.".format(*rotatable_bonds[bond_id]))

                    # Assert whether to optimize this bond
                    bond_symm = []
                    for rot_dihedral in bond:
                        bond_symm.append(rot_dihedral.parameters["torsion_phase"].symmetry_group)

                    bond_symm = set(bond_symm)

                    if bond_symm in bond_symm_global:
                        # If bond has the same number of dihedral types
                        print("This bond is equivalent. No sampling will be performed.")
                        continue

                    else:
                        bond_symm_global.append(bond_symm)
                        dihedral_types_sampled = []
                        for rot_dihedral in bond:
                            if rot_dihedral.parameters["torsion_phase"].symmetry_group not in dihedral_types_sampled:
                                # If this dihedral type was not yet sampled for this rotatable bond
                                print("Sampling torsion with atoms {} {} {} {} belonging to symmetry group {}".format(*rot_dihedral.atoms, rot_dihedral.parameters["torsion_phase"].symmetry_group))

                                if parametrization_type.upper() == "SEQUENTIAL":
                                    # Define torsions to freeze as the ones given by the user plus all the other soft torsions not being scanned
                                    torsions_to_freeze_mod = torsions_to_freeze

                                    # Iterate over all torsions and pick the ones that will not be scanned
                                    for bond in rotatable_dihedrals:
                                        for torsion in bond:
                                            if torsion.atoms is not rot_dihedral.atoms:
                                                torsions_to_freeze_mod.append(torsion.atoms)

                                else:
                                    torsions_to_freeze_mod = torsions_to_freeze

                                # Perform 1D scan and append data to system instance
                                systems, qm_energies_list, qm_forces_list, \
                                mm_energies_list, conformations_list, scan_angles \
                                    = torsions_scan.run_task(settings, systems, [rot_dihedral.atoms], scan_settings,
                                                             torsions_to_freeze=torsions_to_freeze_mod,
                                                             ase_constraints=None,
                                                             optimize_mm=optimize_mm,
                                                             optimize_mm_type=optimize_mm_type,
                                                             optimize_qm_before_scan=False,
                                                             rdkit_conf=[rdkit_conf])

                                # Add to sampled torsions
                                dihedral_types_sampled.append(rot_dihedral.parameters["torsion_phase"].symmetry_group)
                                dihedrals_to_optimize.append(rot_dihedral.atoms)

                                if parametrization_type.upper() == "SEQUENTIAL":
                                    # Save system reference data in a buffer
                                    tmp_coordinates = copy.deepcopy(system.ref_coordinates)
                                    tmp_energies = copy.deepcopy(system.ref_energies)
                                    tmp_forces = copy.deepcopy(system.ref_forces)

                                    # Change system data
                                    system.ref_coordinates = np.asarray(conformations_list)
                                    system.ref_energies = np.asarray(qm_energies_list)
                                    system.ref_forces = np.asarray(qm_forces_list)
                                    system.n_structures = len(system.ref_coordinates)

                                    # Only optimize dihedral types scanned
                                    system.force_field.optimize_torsions(rot_dihedral.atoms, change_other_torsions=True, change_other_parameters=True)
                                    # Perform parametrization of all rotatable dihedral simultaneously
                                    # self.set_zero(system, rotatable_bonds)
                                    parametrization = Parametrization()
                                    systems, parameter_space, objective_function, optimizer = parametrization.run_task(
                                        settings, systems, None, None, None)

                                    # Once it is done, set the data in the buffer back in the system
                                    system.ref_coordinates = np.asarray(tmp_coordinates)
                                    system.ref_energies = np.asarray(tmp_energies)
                                    system.ref_forces = np.asarray(tmp_forces)
                                    system.n_structures = len(system.ref_coordinates)

                    bond_id += 1
                if parametrization_type.upper() == "SIMULTANEOUS":
                    # Only optimize dihedral types scanned
                    system.force_field.optimize_torsions_by_symmetry(dihedrals_to_optimize, change_other_torsions=True, change_other_parameters=True)

                    # Perform parametrization of all rotatable dihedral simultaneously
                    parametrization = Parametrization()
                    systems, parameter_space, objective_function, optimizer = parametrization.run_task(settings, systems, None, None, None)

        print("!=================================================================================!")
        print("!             TORSIONS PARAMETRIZATION TERMINATED SUCCESSFULLY :)                 !")
        print("!=================================================================================!")
        return systems, parameter_space, objective_function, optimizer

    # ------------------------------------------------------------ #
    #                                                              #
    #                        STATIC METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    @staticmethod
    def get_rotatable_bonds(rdkit_mol, methyl=True):
        """
        Method that determines the indices of the atoms which form rotatable (soft) bonds.

        Parameters
        ----------
        rdkit_mol:
            RDKit Molecule
        methyl : bool
            If true, also includes rotatable bonds of methyl groups.

        Returns
        -------
        rotatable_bonds: tuple of tuples
            Tuple of tuples containing the indexes of the atoms forming rotatable (soft) bonds.
        """
        if methyl:
            smart_to_add = ""
        else:
            smart_to_add = ""

        rotatable_bond_mol = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]" + smart_to_add)
        rotatable_bonds = rdkit_mol.GetSubstructMatches(rotatable_bond_mol)
        assert len(rotatable_bonds) == rdmd.CalcNumRotatableBonds(
            rdkit_mol), "Incosistency in the number of rotatable bonds."

        return rotatable_bonds

    @staticmethod
    def get_rotatable_torsions(system, rotatable_bonds):
        """
        Method that determines the quartets that form rotatable (soft) torsions.

        Parameters
        ----------
        system: :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of ParaMol System.
        rotatable_bonds: tuple of tuples
            Tuple of tuples containing the indexes of the atoms forming rotatable (soft) bonds.

        Returns
        -------
        rotatable_torsions : list of lists
            List of lists containing quartet of indices that correspond to rotatable (soft) torsions.
        """

        rotatable_torsions = []

        for bond in rotatable_bonds:
            tmp = []
            for force_field_term in system.force_field.force_field['PeriodicTorsionForce']:
                if (force_field_term.atoms[1] == bond[0] or force_field_term.atoms[1] == bond[1]) and (
                        force_field_term.atoms[2] == bond[0] or force_field_term.atoms[2] == bond[1]):
                    if force_field_term.atoms not in rotatable_torsions:
                        tmp.append(force_field_term)

            # Remove duplicates
            torsion_atoms = []
            rotatable_torsions_bond = []
            for torsion in tmp:
                if torsion.atoms not in torsion_atoms:
                    torsion_atoms.append(torsion.atoms)
                    rotatable_torsions_bond.append(torsion)

            # Append unique rotatable torsion
            rotatable_torsions.append(rotatable_torsions_bond)

        return rotatable_torsions


    # ------------------------------------------------------------ #
    #                                                              #
    #                       PRIVATE METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
