# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Tasks.torsions_scan.TorsionScan` class, which is a ParaMol task that performs torsions scans.
"""
# ParaMol modules
from .task import *
from ..Utils.interface import *

# RDKit modules
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem import rdMolTransforms as rdmt

# SimTK modules
from simtk.openmm import *
from simtk.openmm.app import *
import simtk.unit as unit

import numpy as np
import copy

class TorsionScan(Task):
    """
    ParaMol implementation of torsion scans.
    """
    def __init__(self):
        pass

    # ------------------------------------------------------------ #
    #                                                              #
    #                       PUBLIC  METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_task(self, settings, systems, torsions_to_scan, scan_settings, torsions_to_freeze=None,
                 ase_constraints=None, optimize_mm=False, optimize_mm_type="freeze_atoms",
                 optimize_qm_before_scan=False, rdkit_conf=None):
        """
        Method that performs 1D or 2D torsional scans. Only a scan at a time.

        Notes
        -----
        Initially a QM engine is initiated for every `system` in `systems`.
        Then for every `system` in `systems` a RDKit molecule is created.
        The 1D or 2D torsional scan is performed for the torsion(s) in torsions_to_scan with
        the correspondent settings defined in scan_settings. The torsions defined in torsions_to_freeze are frozen.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        torsions_to_scan : list of list of int
            List of lists wherein the most inner list should contain 4 integers defining the torsions to
            be scanned in a 1D or 2D torsion scan
            Example: 1D-Scan [ [1,2,3,4] ]; 2D-Scan  [ [1,2,3,4],[5,6,7,8] ]
        scan_settings: list of list of float
            List of lists wherein each the most inner list should contain 3 floats defining the settings of the scan in the
            to be performed and in the following order: lower torsion angle, upper torsion angle, angle step (in degrees).
            Example: 1D-Scan  [ [180.0,-180.0,-10.0] ]; 2D-Scan  [ [180.0,-180.0,-10.0],[60.0,-180.0,-10.0] ]
        torsions_to_freeze : list of list of int
            List of lists of wherein each inner list should contain 4 integers defining a torsion to be kept fixed (default is `None`)
        ase_constraints : list of ASE constraints.
            List of ASE constraints to be applied during the scans (default is `None`)
        optimize_mm : bool
            Flag that controls whether a MM geometry optimization is performed before every QM optimization (default is `False`).
        optimize_mm_type : str
            Constraint to be used when performing MM optimization. Available options are 'freeze_atoms' or 'freeze_dihedral'. 'freeze_atoms' is recommended.
        optimize_qm_before_scan : bool
            Flag that controls whether a QM geometry optimization is performed before the scan (default is False).
        rdkit_conf : list of :obj:`rdkit.Chem.rdchem.Conformer`
            List of RDKit conformer for each system. It should be provided with the desired starting conformation.

        Returns
        -------
        systems : list
            List with the updated instances of ParaMol System.
        """
        print("!=================================================================================!")
        print("!                                TORSIONAL SCAN                                   !")
        print("!=================================================================================!")

        # Assert that number of torsional scans to performed has an equal number of scan settings
        assert len(torsions_to_scan) == len(scan_settings), "Number of scan to perform does not match number of scan settings provided."

        if torsions_to_freeze is None:
            torsions_to_freeze = []
        if ase_constraints is None:
            ase_constraints = []

        # Create QM Engines
        for system in systems:
            if system.interface is None:
                system.interface = ParaMolInterface()

            system.create_qm_engines(settings.qm_engine["qm_engine"], settings.qm_engine[settings.qm_engine["qm_engine"].lower()])

        # Iterate over all systems and perform
        for system in systems:
            # Get RDKit mol and conf
            if rdkit_conf is None:
                mol, conf = self.get_rdkit_mol_conf(system)
            else:
                conf = rdkit_conf.pop(0)

            if len(torsions_to_scan) == 0:
                print("No torsions to scan were provided.")
            else:
                torsional_scan_dim = len(torsions_to_scan)

                if torsional_scan_dim == 1:
                    # Perform 1D Scan
                    qm_energies_list, qm_forces_list, mm_energies_list, conformations_list, scan_angles = self.scan_1d(
                        system, conf, torsions_to_scan[0], torsions_to_freeze, scan_settings[0], optimize_mm, optimize_mm_type, optimize_qm_before_scan, ase_constraints)

                    # File name buffer
                    file_name = "scan_{}d_torsion_{}_{}_{}_{}.dat".format(torsional_scan_dim, *torsions_to_scan[0])
                elif torsional_scan_dim == 2:
                    # Perform 2D Scan
                    qm_energies_list, qm_forces_list, mm_energies_list, conformations_list, scan_angles =  self.scan_2d(
                        system, conf, torsions_to_scan[0], torsions_to_scan[1], torsions_to_freeze, scan_settings[0], scan_settings[1], optimize_mm, optimize_mm_type, optimize_qm_before_scan, ase_constraints)

                    # File name buffer
                    file_name = "scan_{}d_torsion_{}_{}_{}_{}_{}_{}_{}_{}.dat".format(torsional_scan_dim, *torsions_to_scan[0], *torsions_to_scan[1])
                else:
                    raise NotImplementedError("{}-d scan type is not implemented.".format(torsional_scan_dim))

                # Append data to system instance
                system.append_data_to_system(conformations_list, qm_energies_list, qm_forces_list)

                # Write readable scan data to file
                self.write_scan_data(scan_angles, qm_energies_list, file_name, torsional_scan_dim)

        print("!=================================================================================!")
        print("!                      TORSIONAL SCAN TERMINATED SUCCESSFULLY :)                  !")
        print("!=================================================================================!")

        return systems, qm_energies_list, qm_forces_list, mm_energies_list, conformations_list, scan_angles

    def scan_1d(self, system, rdkit_conf, torsion_to_scan, torsions_to_freeze, scan_settings, optimize_mm, optimize_mm_type, optimize_qm_before_scan, ase_constraints, force_constant=999999999.0, threshold=1e-2):
        """
        Method that performs 1-dimensional torsional scans.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of a ParaMol System.
        rdkit_conf : :obj:`rdkit.Chem.rdchem.Conformer`
            RDKit conformer.
        torsion_to_scan : list of int
            List containing the indices of the quartet of atoms that define the torsion to be scanned.
        torsions_to_freeze : list of list of int
            List containing lists of the indices of the quartets of atoms to be frozen.
        scan_settings : list
            List containing the settings of the scan in the following order: lower torsion angle, upper torsion angle, angle step (in degrees).
        optimize_mm : bool
            Flag that controls whether a MM geometry optimization is performed before every QM optimization.
        optimize_mm_type : str
            Constraint to be used when performing MM optimization. Available options are 'freeze_atoms' or 'freeze_dihedral'. 'freeze_atoms' is recommended.
        optimize_qm_before_scan : bool
            Flag that controls whether a QM geometry optimization is performed before the scan.
        ase_constraints : list of ASE constraints.
            List of ASE constraints to be applied during the scans.
        threshold : float
            Conservation angle threshold.
        force_constant : float
            Force constant for the dihedral restrain (kJ/mol).

        Returns
        -------
        qm_energies_list, qm_forces_list, mm_energies_list, conformations_list, scan_angles
        """
        assert optimize_mm_type.upper() in ["FREEZE_ATOMS", "FREEZE_DIHEDRAL"], "Optimize MM type {} is unknown.".format(optimize_mm_type)

        print("ParaMol will perform 1-dimensional torsional scan.")
        print("ParaMol will sample the torsion formed by the quartet of atoms with indices {} {} {} {}.\n".format(*torsion_to_scan))

        # Define range of angles to be scanned
        min_angle = scan_settings[0]
        max_angle = scan_settings[1]
        step = scan_settings[2]
        torsion_scan_values = np.arange(min_angle, max_angle, step)

        # Temporary buffers
        qm_energies_list = []
        mm_energies_list = []
        qm_forces_list = []
        conformations_list = []
        scan_angles = []

        # Create temporary OpenMM System, Context and Platform
        dummy_system = copy.deepcopy(system.engine.system)
        dummy_integrator = copy.deepcopy(system.engine.integrator)
        dummy_platform = Platform.getPlatformByName(system.engine.platform_name)

        # Get RDKit positions and define the initial positions variable
        positions = unit.Quantity(rdkit_conf.GetPositions(), unit.angstrom)
        positions_initial = copy.deepcopy(positions)

        if optimize_mm and optimize_mm_type.upper() == "FREEZE_ATOMS":
            dummy_system = self.freeze_atoms(dummy_system, torsion_to_scan)

        # Create OpenMM Context
        dummy_context = Context(dummy_system, dummy_integrator, dummy_platform)
        dummy_context.setPositions(positions)

        if optimize_mm:
            # ----------------------------------------------------------- #
            #                   Perform MM optimization                   #
            # ----------------------------------------------------------- #
            logging.info("Performing MM optimization.")
            LocalEnergyMinimizer.minimize(dummy_context)
            positions = dummy_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)


        if optimize_qm_before_scan:
            # ----------------------------------------------------------- #
            #                   Perform QM optimization                   #
            # ----------------------------------------------------------- #
            # Get optimized geometry
            positions, _, _ = system.qm_engine.qm_engine.run_calculation(coords=positions.in_units_of(unit.angstrom)._value,
                                                                         label=0, calc_type="optimization")
            positions = positions * unit.nanometers

            # Set optimized geometry in RDKit conf
            self.set_positions_rdkit_conf(rdkit_conf, positions.in_units_of(unit.angstrom)._value, )

        # ----------------------------------------------------------- #
        #                       Perform 1D Scan                       #
        # ----------------------------------------------------------- #
        print("ParaMol will now start the torsional scan.")

        for torsion_value in torsion_scan_values:
            print("Step for torsion angle with value {}.".format(torsion_value))

            # Set positions in OpenMM context
            dummy_context.setPositions(positions)

            # Set RDKit geometry to the current in the OpenMM context
            positions = dummy_context.getState(getPositions=True).getPositions()
            self.set_positions_rdkit_conf(rdkit_conf, positions.in_units_of(unit.angstrom)._value)

            # Rotate torsion i-j-k-l; all atoms bonded to atom l are moved
            rdmt.SetDihedralDeg(rdkit_conf, *torsion_to_scan, torsion_value)

            # Get position with new torsion angle
            positions = unit.Quantity(rdkit_conf.GetPositions(), unit.angstrom)

            # Get old value of torsion angle
            old_torsion = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan)

            # Set positions in OpenMM context (new dihedral angle)
            dummy_context.setPositions(positions)
            # ------------------------------------------------------------- #
            #                     MM geometry optimization                  #
            # ------------------------------------------------------------- #
            if optimize_mm:
                logging.info("Performing MM optimization with torsion {} frozen.".format(torsion_to_scan))
                if optimize_mm_type.upper() == "FREEZE_DIHEDRAL":
                    # Freeze torsion
                    # We have to create temporary systems and context so that they do not affect they main ones
                    tmp_system = copy.deepcopy(dummy_system)
                    tmp_system = self.freeze_torsion(tmp_system, torsion_to_scan, torsion_value, force_constant)
                    tmp_context = Context(tmp_system, copy.deepcopy(dummy_integrator), dummy_platform)
                    tmp_context.setPositions(positions)
                    LocalEnergyMinimizer.minimize(tmp_context)
                    positions = tmp_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(
                        asNumpy=True)

                    del tmp_system, tmp_context
                elif optimize_mm_type.upper() == "FREEZE_ATOMS":
                    logging.info("Performing MM optimization with torsion {} frozen.".format(torsion_to_scan))
                    LocalEnergyMinimizer.minimize(dummy_context)
                    positions = dummy_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(
                        asNumpy=True)
                else:
                    raise NotImplementedError("Optimize MM type {} is unknown.".format(optimize_mm_type.upper()))

            # ------------------------------------------------------------- #
            #                       Relaxed QM Scan                         #
            # ------------------------------------------------------------- #
            # Perform QM optimization and get positions, QM forces and energies.
            positions, qm_energy, qm_force = system.qm_engine.qm_engine.run_calculation(
                coords=positions.in_units_of(unit.angstrom)._value,
                label=0, calc_type="optimization", dihedral_freeze=[torsion_to_scan] + torsions_to_freeze, ase_constraints=ase_constraints)

            # Check if torsion angle was conserved during QM optimization
            self.set_positions_rdkit_conf(rdkit_conf, positions)

            new_torsion = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan)
            assert (abs(old_torsion - new_torsion) < threshold) or (abs(abs(old_torsion - new_torsion) - 360) < threshold), \
                "Not conserving torsion angle; old={} new={}".format(old_torsion,new_torsion)

            # Get new MM energy
            mm_energy = dummy_context.getState(getEnergy=True).getPotentialEnergy()

            # Append to list
            qm_energies_list.append(qm_energy)
            qm_forces_list.append(qm_force)
            mm_energies_list.append(mm_energy._value)
            conformations_list.append(positions)
            scan_angles.append(torsion_value)

            # Attribute units to the positions array (useful for next iteration)
            positions = positions * unit.nanometers

        # Set positions of context to last position
        dummy_context.setPositions(positions * unit.nanometers)

        # Set RDKit geometry to the current in the OpenMM context
        self.set_positions_rdkit_conf(rdkit_conf, positions_initial.in_units_of(unit.angstrom)._value)

        del dummy_system, dummy_integrator, dummy_platform, dummy_context

        return qm_energies_list, qm_forces_list, mm_energies_list, conformations_list, scan_angles

    def scan_2d(self, system, rdkit_conf, torsion_to_scan_1, torsion_to_scan_2, torsions_to_freeze, scan_settings_1, scan_settings_2, optimize_mm, optimize_qm_before_scan, ase_constraints, force_constant=9999999.0, threshold=1e-3):
        """
        Method that performs 2-dimensional torsional scans.

        Parameters
        ----------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of a ParaMol System.
        rdkit_conf : :obj:`rdkit.Chem.rdchem.Conformer`
            RDKit conformer
        torsion_to_scan_1 :list of int
            List containing the indices of the quartet of atoms that define the torsion 2 to be scanned.
        torsion_to_scan_2 : list of int
            List containing the indices of the quartet of atoms that define the torsion 1 to be scanned.
        torsions_to_freeze : list of list of int
            List containing lists of the indices of the quartets of atoms to be frozen.
        scan_settings_1 : list of float
            List containing the settings of the scan of torsion 1 in the following order: lower torsion angle, upper torsion angle, angle step (in degrees).
        scan_settings_2 : list of float
            List containing the settings of the scan of torsion 1 in the following order: lower torsion angle, upper torsion angle, angle step (in degrees).
        optimize_mm : bool
            Flag that controls whether a MM geometry optimization is performed before the scan. In case this is argument
            is set to True and optimize_qm_before_scan is also set to True, the MM optimization precedes the QM
            optimization.
        optimize_qm_before_scan : bool
            Flag that controls whether a QM geometry optimization is performed before the scan.
        ase_constraints : list of ASE constraints.
            List of ASE constraints to be applied during the scans.
        threshold : float
            Conservation angle threshold.
        force_constant : float
            Force constant for the dihedral restrain (kJ/mol).

        Returns
        -------
        qm_energies_list, qm_forces_list, mm_energies_list, conformations_list, scan_angles
        """
        print("Performing 2-dimensional torsional scan.")
        print("Sampling torsion 1 formed by the quartet of atoms with indices {} {} {} {}.".format(*torsion_to_scan_1))
        print("Sampling torsion 2 formed by the quartet of atoms with indices {} {} {} {}.".format(*torsion_to_scan_2))

        # Define range of angles to be scanned for torsion 1
        min_angle_1 = scan_settings_1[0]
        max_angle_1 = scan_settings_1[1]
        step_1 = scan_settings_1[2]
        torsion_scan_values_1 = np.arange(min_angle_1, max_angle_1, step_1)

        # Define range of angles to be scanned for torsion 2
        min_angle_2 = scan_settings_2[0]
        max_angle_2 = scan_settings_2[1]
        step_2 = scan_settings_2[2]
        torsion_scan_values_2 = np.arange(min_angle_2, max_angle_2, step_2)

        # Temporary buffers
        qm_energies_list = []
        mm_energies_list = []
        qm_forces_list = []
        conformations_list = []
        scan_angles = []

        # Create temporary OpenMM System, Context and Platform
        dummy_system = copy.deepcopy(system.engine.system)
        dummy_integrator = copy.deepcopy(system.engine.integrator)
        dummy_platform = Platform.getPlatformByName(system.engine.platform_name)

        # Get RDKit positions and define the initial positions variable
        positions = unit.Quantity(rdkit_conf.GetPositions(), unit.angstrom)
        positions_initial = copy.deepcopy(positions)

        # Create OpenMM Context
        dummy_context = Context(dummy_system, dummy_integrator, dummy_platform)
        dummy_context.setPositions(positions)

        if optimize_mm:
            LocalEnergyMinimizer.minimize(dummy_context)
            positions = dummy_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)

        if optimize_qm_before_scan:
            # ----------------------------------------------------------- #
            #                   Perform QM optimization                   #
            # ----------------------------------------------------------- #
            # Get optimized geometry
            positions, _, _ = system.qm_engine.qm_engine.run_calculation(coords=positions.in_units_of(unit.angstrom)._value,
                                                                         label=0, calc_type="optimization", dihedral_freeze=None)
            positions = positions * unit.nanometers

            # Set optimized geometry in RDKit conf
            self.set_positions_rdkit_conf(rdkit_conf, positions.in_units_of(unit.angstrom)._value, )

        # ----------------------------------------------------------- #
        #                       Perform 2D Scan                       #
        # ----------------------------------------------------------- #
        for torsion_value_1 in torsion_scan_values_1:
            for torsion_value_2 in torsion_scan_values_2:
                print("Step for torsion angle 1 with value {}.".format(torsion_value_1))
                print("step for torsion angle 2 with value {}.".format(torsion_value_2))

                # Set positions in OpenMM context
                dummy_context.setPositions(positions)

                # Set RDKit geometry to the current in the OpenMM context
                positions = dummy_context.getState(getPositions=True).getPositions()
                self.set_positions_rdkit_conf(rdkit_conf, positions.in_units_of(unit.angstrom)._value)

                # Rotate torsion i-j-k-l; all atoms bonded to atom l are moved
                rdmt.SetDihedralDeg(rdkit_conf, *torsion_to_scan_1, torsion_value_1)
                rdmt.SetDihedralDeg(rdkit_conf, *torsion_to_scan_2, torsion_value_2)

                # Get position with new torsion angle
                positions = unit.Quantity(rdkit_conf.GetPositions(), unit.angstrom)

                # Get old value of torsion angle
                old_torsion_1 = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan_1)
                old_torsion_2 = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan_2)

                # Perform MM geometry optimization
                if optimize_mm:
                    # Freeze torsions
                    logging.info("Performing MM optimization with torsions {} and {} frozen.".format(torsion_to_scan_1,torsion_to_scan_2))
                    # We have to create temporary systems and context so that they do not affect they main ones
                    tmp_system = copy.deepcopy(dummy_system)
                    self.freeze_atoms(tmp_system)
                    #dummy_system = self.freeze_torsion(dummy_system, torsion_to_scan_1, torsion_value_1, force_constant)
                    #dummy_system = self.freeze_torsion(dummy_system, torsion_to_scan_2, torsion_value_2, force_constant)
                    tmp_context = Context(tmp_system, copy.deepcopy(dummy_integrator), dummy_platform)
                    tmp_context.setPositions(positions)
                    LocalEnergyMinimizer.minimize(tmp_context)
                    positions = tmp_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(
                        asNumpy=True)

                    del tmp_system, tmp_context

                # ------------------------------------------------------------- #
                #                       Relaxed QM Scan                         #
                # ------------------------------------------------------------- #
                # Perform QM optimization and get positions, QM forces and energies.
                positions, qm_energy, qm_force = system.qm_engine.qm_engine.run_calculation(
                    coords=positions.in_units_of(unit.angstrom)._value,
                    label=0, calc_type="optimization", dihedral_freeze=[torsion_to_scan_1, torsion_to_scan_2] + torsions_to_freeze, ase_constraints=ase_constraints)

                # Check if torsion angle was conserved during QM optimization
                self.set_positions_rdkit_conf(rdkit_conf, positions)

                new_torsion_1 = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan_1)
                new_torsion_2 = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan_2)

                assert (abs(old_torsion_1 - new_torsion_1) < threshold) or (
                            abs(abs(old_torsion_1 - new_torsion_1) - 360) < threshold), \
                    "Not conserving torsion angle 1 ; old={} new={}".format(old_torsion_1, new_torsion_1)
                assert (abs(old_torsion_2 - new_torsion_2) < threshold) or (
                            abs(abs(old_torsion_2 - new_torsion_2) - 360) < threshold), \
                    "Not conserving torsion angle 2; old={} new={}".format(old_torsion_2, new_torsion_2)

                # Get new MM energy
                dummy_context.setPositions(positions * unit.nanometers)
                mm_energy = dummy_context.getState(getEnergy=True).getPotentialEnergy()

                # Append to list
                qm_energies_list.append(qm_energy)
                qm_forces_list.append(qm_force)
                mm_energies_list.append(mm_energy._value)
                conformations_list.append(positions)
                scan_angles.append([torsion_value_1, torsion_value_2])

        print("!=================================================================================!\n")

        # Set RDKit geometry to the current in the OpenMM context
        self.set_positions_rdkit_conf(rdkit_conf, positions_initial.in_units_of(unit.angstrom)._value)

        del dummy_system, dummy_integrator, dummy_platform, dummy_context

        return qm_energies_list, qm_forces_list, mm_energies_list, conformations_list, scan_angles

    # ------------------------------------------------------------ #
    #                                                              #
    #                        STATIC METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    @staticmethod
    def get_mm_relaxed_conformations(system, torsions_to_freeze, tolerance=0.01, max_iter=0, force_constant=999999.0, threshold=1e-2):
        """
        Method that creates and returns a RDKit Conformer instance and a RDKit Molecule instance of the ParaMol system passed
        as an argument.

        Parameters
        ----------
        system: :obj:`ParaMol.System.system.ParaMolSystem`
            ParaMol system instance.
        torsions_to_freeze : list of list of int
            List of lists of wherein each inner list should contain 4 integers defining a torsion to be kept fixed.
        tolerance : float
            Specifies how precisely the energy minimum must be located. Minimization will be halted once the root-mean-square value of all force components reaches this tolerance.
        max_iter : int
            Maximum number of iterations to perform. If this is 0, minimation is continued until the results converge without regard to how many iterations it takes. The default value is 0.
        force_constant : float
            Force constant for the dihedral restrain (kJ/mol).
        threshold : float
            Conservation angle threshold.

        Notes
        -----
        This method overwrites the ref_coordinates attribute of the system object. Hence, use this with care.

        Returns
        -------
        mm_relaxed_conformations: np.array, shape=(n_conformations,n_atoms,3)
            MM-relaxed conformations.
        """
        mm_relaxed_conformations = []

        # Create RDKit mol and conf
        rdkit_mol, rdkit_conf = TorsionScan.get_rdkit_mol_conf(system)

        for conf in system.ref_coordinates:
            # Set new position in RDKit conformation
            TorsionScan.set_positions_rdkit_conf(rdkit_conf, conf)

            # Fix torsions at the new dihedral angle value
            logging.info("Performing MM optimization with torsion(s) {} frozen.".format(torsions_to_freeze))
            tmp_system = copy.deepcopy(system.engine.system)
            for torsion in torsions_to_freeze:
                old_torsion = rdmt.GetDihedralDeg(rdkit_conf, *torsion)
                tmp_system = TorsionScan.freeze_torsion(tmp_system, torsion, old_torsion, force_constant)

            # Create temporary context and set the positions in it
            tmp_context = Context(tmp_system, copy.deepcopy(system.engine.integrator), Platform.getPlatformByName(system.engine.platform_name))
            tmp_context.setPositions(conf)
            # Perform minimization
            LocalEnergyMinimizer.minimize(tmp_context, tolerance=tolerance, maxIterations=max_iter)

            # Get MM-relaxed conformation and store it
            positions = tmp_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)._value

            # Set new position in RDKit conformation
            TorsionScan.set_positions_rdkit_conf(rdkit_conf, positions)
            for torsion in torsions_to_freeze:
                new_torsion = rdmt.GetDihedralDeg(rdkit_conf, *torsion)

            assert (abs(old_torsion - new_torsion) < threshold) or (abs(abs(old_torsion - new_torsion) - 360) < threshold), \
                "Not conserving torsion angle; old={} new={}".format(old_torsion,new_torsion)

            mm_relaxed_conformations.append(positions)

        mm_relaxed_conformations = np.asarray(mm_relaxed_conformations)

        return mm_relaxed_conformations


    @staticmethod
    def get_rdkit_mol_conf(system, pdb_file_name="temp_file.pdb"):
        """
        Method that creates and returns a RDKit Conformer instance and a RDKit Molecule instance of the ParaMol system passed
        as an argument.

        Parameters
        ----------
        system: :obj:`ParaMol.System.system.ParaMolSystem`
            ParaMol system instance.
        pdb_file_name: str
            Name of the temporary .pdb file to be written (the default is "temp_file.pdb")

        Notes
        -----
        This methods requires writing of a .pdb file from which the RDKit molecule will be constructed.

        Returns
        -------
        mol, conf: :obj:`rdkit.Chem.rdchem.Mol`, :obj:`rdkit.Chem.rdchem.Conformer`
            Tuple containing the RDKit Molecule and the RDKit conformer.
        """
        # Write out PDB file to be read by RDKit
        file_to_write = open(pdb_file_name, 'w')
        pdbfile.PDBFile.writeFile(topology=system.engine.topology,
                                  positions=system.engine.context.getState(getPositions=True,
                                                                           enforcePeriodicBox=True).getPositions(
                                      asNumpy=True),
                                  file=file_to_write)
        file_to_write.close()

        # Construct a RDKit molecule from a PDB file and get the conformer
        mol = Chem.MolFromPDBFile(pdb_file_name, removeHs=False)
        conf = mol.GetConformer()

        return mol, conf

    @staticmethod
    def set_positions_rdkit_conf(rdkit_conf, positions):
        """
        Method that sets the given positions in the given RDKit conformer.

        Parameters
        ----------
        rdkit_conf : :obj:`rdkit.Chem.rdchem.Conformer`
            RDKit conformer.
        positions : list or np.array
            Position array.

        Returns
        -------
        rdkit_conf : :obj:`rdkit.Chem.rdchem.Conformer`
            RDKit conformer.
        """
        from rdkit.Geometry import Point3D

        for i in range(rdkit_conf.GetNumAtoms()):
            x, y, z = positions[i]  # Convert from nanometers to angstrom
            rdkit_conf.SetAtomPosition(i, Point3D(x, y, z))

        return rdkit_conf


    @staticmethod
    def freeze_torsion(system, torsion_to_freeze, torsion_angle, k):
        """
        Method that freezes the torsion_to_freeze torsion of an OpenMM system by adding a restraint to it.

        Parameters
        ----------
        system : simtk.openmm.System
            Instance of a OpenMM System.
        torsion_to_freeze : list of int
            List containing indices of the atoms to be frozen
        torsion_angle : float
            Value of the desired torsion angle in degrees.
        k : float
            Value of the  force constant to be applied in kilojoules/mole.

        Notes
        -----
        This method should be called before creating a Context. If a Context was already created, it should be re-initialized.
        It works by creating a CustomTorsionForce that fixes the desired torsion. The energy expression of the CustomTorsionForce is:

        .. math:: F=-k*\cos(\theta-\theta_0)

        where :math:`k` is the force constant of the restraint and :math:`\theta_0` is the chosen torsion angle value.

        Returns
        -------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Updated instance of OpenMM System with an extra CustomTorsionForce that freezes the desired torsion.
        """
        energy_expression = f'-fc*cos(theta-theta0)'
        fc = unit.Quantity(k, unit.kilojoule_per_mole)
        restraint = CustomTorsionForce(energy_expression)
        restraint.addGlobalParameter('theta0', torsion_angle * np.pi / 180.0)
        restraint.addGlobalParameter('fc', fc)
        restraint.addTorsion(*torsion_to_freeze)
        system.addForce(restraint)

        return system

    @staticmethod
    def freeze_atoms(system, atom_list):
        """
        Method that freezes atoms in atom_list of an OpenMM system.

        Notes
        -----
        This methods sets the mass of the atoms in atom_list to 0 by changing the OpenMM system.

        Parameters
        ----------
        system : simtk.openmm.System
            Instance of a OpenMM System.
        atom_list : list of int
            List containing indices of the atoms to bre frozen

        Returns
        -------
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Updated instance of OpenMM System.
        """
        # Set mass of the particles of the torsion to zero so that they remain fixed during sampling
        for at in atom_list:
            system.setParticleMass(at, 0)

        return system

    @staticmethod
    def set_zero(system, rotatable_bonds):
        """
        Method that modifies the ForceField of a ParaMol System so that the values of the force constants of torsions
        for which the inner atoms form rotatable (soft) bonds are set to 0.0

        Parameters
        ----------
        system: :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of ParaMol System.
        rotatable_bonds: list of lists of ints
            Lists that contains lists with pairs of atoms's indices.

        Returns
        -------
        system: :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of ParaMol System with updated ForceField.
        """
        for bond in rotatable_bonds:
            for force_field_term in system.force_field.force_field['PeriodicTorsionForce']:
                if (force_field_term.atoms[1] == bond[0] or force_field_term.atoms[1] == bond[1]) and (
                        force_field_term.atoms[2] == bond[0] or force_field_term.atoms[2] == bond[1]):
                    force_field_term.parameters['torsion_k'].value = 0.0
        return system

    @staticmethod
    def write_scan_data(scan_angles, qm_energies, file_name, scan_dim):
        """
        Method that writes human-readable scan data to .dat file.

        Parameters
        ----------
        scan_angles: list
            List containing the set of angles for which the scan was performed.
        qm_energies: list
            List containing the QM energy value for a given angle.
        file_name: str
            Name of the file to be written.
        scan_dim: int
            Dimension of the torsional scan.

        Returns
        -------
            None
        """

        if scan_dim == 1:
            with open(file_name, 'w') as f:
                for angle, energy in zip(scan_angles, qm_energies):
                    line = "{:.3f} {:.6f}\n".format(angle, energy)
                    f.write(line)
        elif scan_dim == 2:
            with open(file_name, 'w') as f:
                for angle, energy in zip(scan_angles, qm_energies):
                    line = "{:.3f} {:.3f} {:.6f}\n".format(*angle, energy)
                    f.write(line)
        else:
            raise NotImplementedError("{}-d scan type is not implemented.".format(scan_dim))

        return
    # ------------------------------------------------------------ #
    #                                                              #
    #                       PRIVATE METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    pass

