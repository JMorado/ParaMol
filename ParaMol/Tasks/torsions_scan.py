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
import logging


class TorsionScan(Task):
    """
    ParaMol implementation of torsion scans.

    Attributes
    ----------
    qm_energies_list : list
        Array containing the QM energies of the scan.
    mm_energies_list : list
        Array containing the MM energies of the scan.
    qm_forces_list : list
        List containing the QM Forces of the scan.
    conformations_list : list
        List containing the conformations of the scan.
    scan_angles : list of float
        Array containing the dihedral angles scanned so far.
    """
    def __init__(self):
        self.qm_energies_list = None
        self.mm_energies_list = None
        self.qm_forces_list = None
        self.qm_conformations_list = None
        self.mm_conformations_list = None
        self.scan_angles = None

    # ------------------------------------------------------------ #
    #                                                              #
    #                       PUBLIC  METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_task(self, settings, systems, torsions_to_scan, scan_settings, interface=None, torsions_to_freeze=None,
                 ase_constraints=None, optimize_qm=True, optimize_qm_before_scan=False, optimize_mm=False, optimize_mm_before_scan=False,
                 optimize_mm_type="freeze_atoms", sampling=False, rotate_from_initial=False, n_structures_to_sample=1, dihedral_conservation_threshold=1e-2,
                 mm_opt_force_constant=99999.0, mm_opt_tolerance=1.0, mm_opt_max_iter=0, rdkit_conf=None, restart=False):
        """
        Method that performs 1D or 2D torsional scans. Only a scan at a time.

        Notes
        -----
        Only one ParaMol system is supported at once.
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
        interface: :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol system instance.
        torsions_to_freeze : list of list of int
            List of lists of wherein each inner list should contain 4 integers defining a torsion to be kept fixed (default is `None`)
        ase_constraints : list of ASE constraints.
            List of ASE constraints to be applied during the scans (default is `None`)
        optimize_qm : bool
            Flag that controls whether a QM geometry optimization is performed (default is `True`).
        optimize_qm_before_scan : bool
            Flag that controls whether a QM geometry optimization is performed before the scan (default is False).
        optimize_mm : bool
            Flag that controls whether a MM geometry optimization is performed (before every QM optimization, default is `False`).
        optimize_mm_before_scan : bool
            Flag that controls whether a MM geometry optimization is performed before the scan (default is False).
        optimize_mm_type : str
            Constraint to be used when performing MM optimization. Available options are 'freeze_atoms' or 'freeze_dihedral'. 'freeze_atoms' is recommended.
        sampling : bool
            Indicates whether to perform sampling at each dihedral angle value using an ASE integrator (default is `False`).
        rotate_from_initial : bool
            Flag whether to perform rotation using initial structure or the last structure.
        n_structures_to_sample : int
            If sampling is `True`, sets the number of structures to sample for each dihedral angle value (default is 1).
        dihedral_conservation_threshold : float
            Threshold that control how much the dihedrals are allow to vary when applying constraints.
        mm_opt_force_constant : float
            Only used if optimize_mm is True and optimize_mm_type is 'freeze_dihedral'. Force constant for the dihedral harmonic restrain (kJ/mol).
        mm_opt_tolerance : float
            Only used if optimize_mm is True. Argument for LocalEnergyMinimizer. This specifies how precisely the energy minimum must be located.
            Minimization will be halted once the root-mean-square value of all force components reaches this tolerance.
        mm_opt_max_iter : int
            Only used if optimize_mm is True. Argument for LocalEnergyMinimizer. The maximum number of iterations to perform.
            If this is 0, minimation is continued until the results converge without regard to how many iterations it takes.
        rdkit_conf : list of :obj:`rdkit.Chem.rdchem.Conformer`
            List of RDKit conformer for each system. It should be provided with the desired starting conformation.
        restart : bool
            Flag that controls whether or not to perform a restart.

        Returns
        -------
        systems : list
            List with the updated instances of ParaMol System.
        """
        print("!=================================================================================!")
        print("!                                TORSIONAL SCAN                                   !")
        print("!=================================================================================!")
        assert len(systems) == 1, "TorsionScan task currently only supports one system at once."
        # Assert that number of torsional scans to performed has an equal number of scan settings
        assert len(torsions_to_scan) == len(scan_settings), "Number of scan to perform does not match number of scan settings provided."

        if len(torsions_to_scan) == 2 and sampling:
            raise NotImplementedError("Sampling still not implemented for 2D scans.")

        if torsions_to_freeze is None:
            torsions_to_freeze = []
        if ase_constraints is None:
            ase_constraints = []

        if optimize_qm or sampling:
            # Create QM Engines
            for system in systems:
                if system.interface is None:
                    system.interface = ParaMolInterface()

                system.create_qm_engines(settings.qm_engine["qm_engine"], settings.qm_engine[settings.qm_engine["qm_engine"].lower()])

        # Create IO Interface
        if interface is None:
            interface = ParaMolInterface()
        else:
            assert type(interface) is ParaMolInterface

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
                    scan_angles, qm_energies_list, qm_forces_list, qm_conformations_list, mm_energies_list, mm_conformations_list = self.scan_1d(
                        interface, settings.restart, system, conf, torsions_to_scan[0], torsions_to_freeze, scan_settings[0], optimize_qm, optimize_qm_before_scan,
                        optimize_mm, optimize_mm_before_scan, optimize_mm_type, ase_constraints, sampling, rotate_from_initial, n_structures_to_sample, dihedral_conservation_threshold,
                        mm_opt_force_constant, mm_opt_tolerance, mm_opt_max_iter, restart)

                    # File name buffer
                    file_name = "{}_scan_{}d_torsion_{}_{}_{}_{}".format(system.name, torsional_scan_dim, *torsions_to_scan[0])
                elif torsional_scan_dim == 2:
                    # Perform 2D Scan
                    scan_angles, qm_energies_list, qm_forces_list, qm_conformations_list, mm_energies_list, mm_conformations_list =  self.scan_2d(
                        interface, settings.restart, system, conf, torsions_to_scan[0], torsions_to_scan[1], torsions_to_freeze, scan_settings[0],
                        scan_settings[1], optimize_qm, optimize_qm_before_scan, optimize_mm, optimize_mm_before_scan, optimize_mm_type, ase_constraints, rotate_from_initial,
                        dihedral_conservation_threshold, mm_opt_force_constant, mm_opt_tolerance, mm_opt_max_iter, restart)

                    # File name buffer
                    file_name = "scan_{}d_torsion_{}_{}_{}_{}_{}_{}_{}_{}.dat".format(torsional_scan_dim, *torsions_to_scan[0], *torsions_to_scan[1])
                else:
                    raise NotImplementedError("{}-d scan type is not implemented.".format(torsional_scan_dim))

                # Append reference data to system instance
                system.append_data_to_system(qm_conformations_list, qm_energies_list, qm_forces_list)

                # Write readable scan data to file
                self.write_scan_data(scan_angles, qm_energies_list, file_name + ".dat", torsional_scan_dim)

                # Write system data
                system.write_data(file_name + ".nc")

        print("!=================================================================================!")
        print("!                      TORSIONAL SCAN TERMINATED SUCCESSFULLY :)                  !")
        print("!=================================================================================!")

        return scan_angles, qm_energies_list, qm_forces_list, qm_conformations_list, mm_energies_list, mm_conformations_list

    def scan_1d(self, interface, restart_settings, system, rdkit_conf, torsion_to_scan, torsions_to_freeze, scan_settings, optimize_qm, optimize_qm_before_scan, optimize_mm,
                optimize_mm_before_scan, optimize_mm_type, ase_constraints, sampling, rotate_from_initial, n_structures_to_sample, threshold, mm_opt_force_constant, mm_opt_tolerance, mm_opt_max_iter, restart,):
        """
        Method that performs 1-dimensional torsional scans.

        Parameters
        ----------
        interface: :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol system instance.
        restart_settings : dict
            Dictionary containing restart ParaMol settings.
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
        optimize_qm : bool
            Flag that controls whether a QM geometry optimization is performed (default is `True`).
            Note that the QM optimization is done after the MM optimization and therefore it inherits any structure obtained in the MM optimization.
        optimize_qm_before_scan : bool
            Flag that controls whether a QM geometry optimization is performed before the scan.
        optimize_mm : bool
            Flag that controls whether a MM geometry optimization is performed before every QM optimization.
        optimize_mm_before_scan : bool
            Flag that controls whether a MM geometry optimization is performed before the scan.
        optimize_mm_type : str
            Constraint to be used when performing MM optimization. Available options are 'freeze_atoms' or 'freeze_dihedral'.
        ase_constraints : list of ASE constraints.
            List of ASE constraints to be applied during the scans.
        rotate_from_initial : bool
            Flag whether to perform rotation using initial structure or the last structure.
        sampling : bool
            Indicates whether to perform sampling at each dihedral angle value using an ASE integrator.
        n_structures_to_sample : int
            If sampling is `True`, sets the number of structures to sample for each dihedral angle value.
        threshold : float
            Conservation angle threshold.
        mm_opt_force_constant : float
            Only used if optimize_mm is True and optimize_mm_type is 'freeze_dihedral'. Force constant for the dihedral restrain (kJ/mol).
        mm_opt_tolerance : float
            Only used if optimize_mm is True. Argument for LocalEnergyMinimizer. This specifies how precisely the energy minimum must be located.
            Minimization will be halted once the root-mean-square value of all force components reaches this tolerance.
        mm_opt_max_iter : int
            Only used if optimize_mm is True. Argument for LocalEnergyMinimizer. The maximum number of iterations to perform.
            If this is 0, minimation is continued until the results converge without regard to how many iterations it takes.
        restart : bool
            Flag that controls whether or not to perform a restart.

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

        # Create temporary OpenMM System, Context and Platform
        dummy_system = copy.deepcopy(system.engine.system)
        dummy_integrator = copy.deepcopy(system.engine.integrator)
        dummy_platform = Platform.getPlatformByName(system.engine.platform_name)

        # Get RDKit positions and define the initial positions variable
        positions = unit.Quantity(rdkit_conf.GetPositions(), unit.angstrom)

        # Create OpenMM Context
        dummy_context = Context(dummy_system, dummy_integrator, dummy_platform)

        if (optimize_mm or optimize_mm_before_scan) and optimize_mm_type.upper() == "FREEZE_ATOMS":
            dummy_system = self.freeze_atoms(dummy_system, torsion_to_scan)

        # ----------------------------------------------------------- #
        #                           Restart                           #
        # ----------------------------------------------------------- #
        if restart:
            self.__dict__ = self.read_restart_pickle(restart_settings, interface, "restart_scan_file")
            # Set positions
            if sampling or optimize_qm:
                positions = self.qm_conformations_list[-1] * unit.nanometers
            elif optimize_mm:
                positions = self.mm_conformations_list[-1] * unit.nanometers

            dummy_context.setPositions(positions)
            # Get new list of torsion scan values
            torsion_scan_values = [item for item in torsion_scan_values if item not in self.scan_angles]
        else:
            self.qm_energies_list = []
            self.mm_energies_list = []
            self.qm_forces_list = []
            self.qm_conformations_list = []
            self.mm_conformations_list = []
            self.scan_angles = []

            # Set positions
            dummy_context.setPositions(positions)

        if optimize_mm_before_scan and (not restart):
            # ----------------------------------------------------------- #
            #                   Perform MM optimization                   #
            # ----------------------------------------------------------- #
            logging.info("Performing MM optimization.")
            LocalEnergyMinimizer.minimize(dummy_context, tolerance=mm_opt_tolerance, maxIterations=mm_opt_max_iter)
            positions = dummy_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)

        if optimize_qm_before_scan and (not restart):
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
        positions_initial = copy.deepcopy(positions)

        for torsion_value in torsion_scan_values:
            print("Step for torsion angle with value {}.".format(torsion_value))

            # Set positions in OpenMM context
            if rotate_from_initial:
                # Set context to initial positions
                dummy_context.setPositions(positions_initial)
            else:
                # Set context to last positions
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

            if sampling:
                positions_before_sampling = copy.deepcopy(positions)
            else:
                n_structures_to_sample = 1

            for n_sample in range(n_structures_to_sample):
                if sampling:
                    print("Sampling structure no. {}".format(n_sample+1))
                    positions, _, _, _ , qm_energy, _, qm_force = system.qm_engine.qm_engine.run_md(coords=positions.in_units_of(unit.angstrom)._value,
                                                                                                    label=0,
                                                                                                    dihedral_freeze=[torsion_to_scan] + torsions_to_freeze,
                                                                                                    ase_constraints=ase_constraints)
                    # Check if torsion angle was conserved during QM optimization
                    self.set_positions_rdkit_conf(rdkit_conf, positions)

                    new_torsion = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan)

                    assert (abs(old_torsion - new_torsion) < threshold) or (abs(abs(old_torsion - new_torsion) - 360) < threshold), \
                        "Not conserving torsion angle; old={} new={}".format(old_torsion, new_torsion)

                    positions = positions * unit.nanometers

                # ------------------------------------------------------------- #
                #                     MM geometry optimization                  #
                # ------------------------------------------------------------- #
                if optimize_mm:
                    logging.info("Performing MM optimization with torsion {} frozen.".format(torsion_to_scan))
                    if optimize_mm_type.upper() == "FREEZE_DIHEDRAL":
                        # Freeze torsion
                        # We have to create temporary systems and context so that they do not affect they main ones
                        tmp_system = copy.deepcopy(dummy_system)
                        tmp_system = TorsionScan.freeze_torsions(tmp_system, [torsion_to_scan], [torsion_value], mm_opt_force_constant)
                        integ = copy.deepcopy(dummy_integrator)
                        tmp_context = Context(tmp_system, integ, dummy_platform)
                        tmp_context.setPositions(positions)
                        LocalEnergyMinimizer.minimize(tmp_context, tolerance=mm_opt_tolerance, maxIterations=mm_opt_max_iter)

                        del tmp_system, tmp_context
                    elif optimize_mm_type.upper() == "FREEZE_ATOMS":
                        logging.info("Performing MM optimization with torsion {} frozen.".format(torsion_to_scan))
                        LocalEnergyMinimizer.minimize(dummy_context, tolerance=mm_opt_tolerance, maxIterations=mm_opt_max_iter)
                        positions = dummy_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(
                            asNumpy=True)

                        for i in range(100):
                            dummy_integrator.setTemperature(3 * (100 - i) * unit.kelvin)
                            dummy_integrator.step(1000)
                        LocalEnergyMinimizer.minimize(dummy_context, tolerance=mm_opt_tolerance, maxIterations=mm_opt_max_iter)
                        positions = dummy_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
                    else:
                        raise NotImplementedError("Optimize MM type {} is unknown.".format(optimize_mm_type.upper()))

                    # Get new MM energy
                    dummy_context.setPositions(positions)
                    mm_energy = dummy_context.getState(getEnergy=True).getPotentialEnergy()

                    # Check if torsion angle was conserved during MM optimization
                    self.set_positions_rdkit_conf(rdkit_conf, positions._value)

                    new_torsion = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan)
                    assert (abs(old_torsion - new_torsion) < threshold) or (abs(abs(old_torsion - new_torsion) - 360) < threshold), \
                        "Not conserving torsion angle in MM optimization; old={} new={}".format(old_torsion, new_torsion)

                    # Append to list
                    self.mm_conformations_list.append(positions.in_units_of(unit.nanometers)._value)
                    self.mm_energies_list.append(mm_energy._value)

                # ------------------------------------------------------------- #
                #                       Relaxed QM Scan                         #
                # ------------------------------------------------------------- #
                if optimize_qm:
                    # Perform QM optimization and get positions, QM forces and energies.
                    positions, qm_energy, qm_force = system.qm_engine.qm_engine.run_calculation(
                        coords=positions.in_units_of(unit.angstrom)._value,
                        label=0,
                        calc_type="optimization",
                        dihedral_freeze=[torsion_to_scan] + torsions_to_freeze,
                        ase_constraints=ase_constraints)

                    # Check if torsion angle was conserved during QM optimization
                    self.set_positions_rdkit_conf(rdkit_conf, positions)

                    new_torsion = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan)
                    assert (abs(old_torsion - new_torsion) < threshold) or (abs(abs(old_torsion - new_torsion) - 360) < threshold), \
                        "Not conserving torsion angle in QM optimization; old={} new={}".format(old_torsion,new_torsion)

                    # Attribute units to the positions array (useful for next iteration)
                    positions = positions * unit.nanometers

                if sampling or optimize_qm:
                    # Append to list
                    self.qm_energies_list.append(qm_energy)
                    self.qm_forces_list.append(qm_force)
                    self.qm_conformations_list.append(positions.in_units_of(unit.nanometers)._value)

            self.scan_angles.append(torsion_value)

            #if sampling:
            #    positions = positions_before_sampling

            # Write scan restart
            self.write_restart_pickle(restart_settings, interface, "restart_scan_file", self.__dict__)

        # Set positions of context to last position
        dummy_context.setPositions(positions * unit.nanometers)

        # Set RDKit geometry to the current in the OpenMM context
        self.set_positions_rdkit_conf(rdkit_conf, positions_initial.in_units_of(unit.angstrom)._value)

        del dummy_system, dummy_integrator, dummy_platform, dummy_context

        return self.scan_angles, self.qm_energies_list, self.qm_forces_list, self.qm_conformations_list, self.mm_energies_list, self.mm_conformations_list

    def scan_2d(self, interface, restart_settings, system, rdkit_conf, torsion_to_scan_1, torsion_to_scan_2, torsions_to_freeze, scan_settings_1, scan_settings_2, optimize_qm,
                optimize_qm_before_scan, optimize_mm, optimize_mm_before_scan, optimize_mm_type, ase_constraints, rotate_from_initial,
                threshold, mm_opt_force_constant, mm_opt_tolerance, mm_opt_max_iter, restart,):
        """
        Method that performs 2-dimensional torsional scans.

        Parameters
        ----------
        interface: :obj:`ParaMol.Utils.interface.ParaMolInterface`
            ParaMol system instance.
        restart_settings : dict
            Dictionary containing restart ParaMol settings.
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
        optimize_qm : bool
            Flag that controls whether a QM geometry optimization is performed (default is `True`).
            Note that the QM optimization is done after the MM optimization and therefore it inherits any structure obtained in the MM optimization.
        optimize_qm_before_scan : bool
            Flag that controls whether a QM geometry optimization is performed before the scan.
        optimize_mm : bool
            Flag that controls whether a MM geometry optimization is performed before the scan. In case this is argument
            is set to True and optimize_qm_before_scan is also set to True, the MM optimization precedes the QM
            optimization.
        optimize_mm_before_scan : bool
            Flag that controls whether a QM geometry optimization is performed before the scan.
        optimize_mm_type : str
            Constraint to be used when performing MM optimization. Available options are 'freeze_atoms' or 'freeze_dihedral'.
        ase_constraints : list of ASE constraints.
            List of ASE constraints to be applied during the scans.
        rotate_from_initial : bool
            Flag whether to perform rotation using initial structure or the last structure.
        threshold : float
            Conservation angle threshold.
        mm_opt_force_constant : float
            Only used if optimize_mm is True and optimize_mm_type is 'freeze_dihedral'. Force constant for the dihedral restrain (kJ/mol).
        mm_opt_tolerance : float
            Only used if optimize_mm is True. Argument for LocalEnergyMinimizer. This specifies how precisely the energy minimum must be located.
            Minimization will be halted once the root-mean-square value of all force components reaches this tolerance.
        mm_opt_max_iter : int
            Only used if optimize_mm is True. Argument for LocalEnergyMinimizer. The maximum number of iterations to perform.
            If this is 0, minimation is continued until the results converge without regard to how many iterations it takes.
        restart : bool
            Flag that controls whether or not to perform a restart.

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

        # Merge both ranges into a list
        torsion_scan_values = [[i, j] for i in torsion_scan_values_1 for j in torsion_scan_values_2]

        # Create temporary OpenMM System, Context and Platform
        dummy_system = copy.deepcopy(system.engine.system)
        dummy_integrator = copy.deepcopy(system.engine.integrator)
        dummy_platform = Platform.getPlatformByName(system.engine.platform_name)

        # Get RDKit positions and define the initial positions variable
        positions = unit.Quantity(rdkit_conf.GetPositions(), unit.angstrom)

        # Create OpenMM Context
        dummy_context = Context(dummy_system, dummy_integrator, dummy_platform)

        if (optimize_mm or optimize_mm_before_scan) and optimize_mm_type.upper() == "FREEZE_ATOMS":
            dummy_system = self.freeze_atoms(dummy_system, torsion_to_scan_1)
            dummy_system = self.freeze_atoms(dummy_system, torsion_to_scan_2)

        # ----------------------------------------------------------- #
        #                           Restart                           #
        # ----------------------------------------------------------- #
        if restart:
            self.__dict__ = self.read_restart_pickle(restart_settings, interface, "restart_scan_file")
            # Set positions
            if optimize_qm:
                positions = self.qm_conformations_list[-1] * unit.nanometers
            elif optimize_mm:
                positions = self.mm_conformations_list[-1] * unit.nanometers

            dummy_context.setPositions(positions)
            # Get new list of torsion scan values
            torsion_scan_values = [item for item in torsion_scan_values if item not in self.scan_angles]
        else:
            self.qm_energies_list = []
            self.mm_energies_list = []
            self.qm_forces_list = []
            self.mm_conformations_list = []
            self.qm_conformations_list = []
            self.scan_angles = []
            # Set positions
            dummy_context.setPositions(positions)

        if optimize_mm_before_scan and (not restart):
            LocalEnergyMinimizer.minimize(dummy_context, tolerance=mm_opt_tolerance, maxIterations=mm_opt_max_iter)
            positions = dummy_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)

        if optimize_qm_before_scan and (not restart):
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
        positions_initial = copy.deepcopy(positions)

        for torsion_value_1, torsion_value_2 in torsion_scan_values:
            print("Step for torsion angle 1 with value {}.".format(torsion_value_1))
            print("step for torsion angle 2 with value {}.".format(torsion_value_2))

            # Set positions in OpenMM context
            if rotate_from_initial:
                # Set context to initial positions
                dummy_context.setPositions(positions_initial)
            else:
                # Set context to last positions
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

            # Set positions in OpenMM context
            dummy_context.setPositions(positions)
            # ------------------------------------------------------------- #
            #                     MM geometry optimization                  #
            # ------------------------------------------------------------- #
            if optimize_mm:
                logging.info("Performing MM optimization with torsion {} frozen.".format(torsion_to_scan_1))
                logging.info("Performing MM optimization with torsion {} frozen.".format(torsion_to_scan_2))

                if optimize_mm_type.upper() == "FREEZE_DIHEDRAL":
                    # Freeze torsion
                    # We have to create temporary systems and context so that they do not affect they main ones
                    tmp_system = copy.deepcopy(dummy_system)
                    tmp_system = TorsionScan.freeze_torsions(tmp_system, [torsion_to_scan_1, torsion_to_scan_2], [torsion_value_1, torsion_value_2], mm_opt_force_constant)
                    tmp_context = Context(tmp_system, copy.deepcopy(dummy_integrator), dummy_platform)
                    tmp_context.setPositions(positions)
                    LocalEnergyMinimizer.minimize(tmp_context, tolerance=mm_opt_tolerance, maxIterations=mm_opt_max_iter)
                    positions = tmp_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(
                        asNumpy=True)

                    del tmp_system, tmp_context
                elif optimize_mm_type.upper() == "FREEZE_ATOMS":
                    logging.info("Performing MM optimization with torsion {} frozen.".format(torsion_to_scan_1))
                    logging.info("Performing MM optimization with torsion {} frozen.".format(torsion_to_scan_2))
                    LocalEnergyMinimizer.minimize(dummy_context, tolerance=mm_opt_tolerance, maxIterations=mm_opt_max_iter)
                    positions = dummy_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(
                        asNumpy=True)
                else:
                    raise NotImplementedError("Optimize MM type {} is unknown.".format(optimize_mm_type.upper()))

                # Get new MM energy
                dummy_context.setPositions(positions)
                mm_energy = dummy_context.getState(getEnergy=True).getPotentialEnergy()

                # Check if torsion angle was conserved during MM optimization
                self.set_positions_rdkit_conf(rdkit_conf, positions._value)

                new_torsion_1 = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan_1)
                new_torsion_2 = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan_2)

                assert (abs(old_torsion_1 - new_torsion_1) < threshold) or (
                        abs(abs(old_torsion_1 - new_torsion_1) - 360) < threshold), \
                    "Not conserving torsion angle 1 after MM optimization; old={} new={}".format(old_torsion_1, new_torsion_1)
                assert (abs(old_torsion_2 - new_torsion_2) < threshold) or (
                        abs(abs(old_torsion_2 - new_torsion_2) - 360) < threshold), \
                    "Not conserving torsion angle 2 after MM optimization; old={} new={}".format(old_torsion_2, new_torsion_2)

                # Append to list
                self.mm_conformations_list.append(positions.in_units_of(unit.nanometers)._value)
                self.mm_energies_list.append(mm_energy._value)

            # ------------------------------------------------------------- #
            #                       Relaxed QM Scan                         #
            # ------------------------------------------------------------- #
            if optimize_qm:
                # Perform QM optimization and get positions, QM forces and energies.
                positions, qm_energy, qm_force = system.qm_engine.qm_engine.run_calculation(
                    coords=positions.in_units_of(unit.angstrom)._value,
                    label=0,
                    calc_type="optimization",
                    dihedral_freeze=[torsion_to_scan_1, torsion_to_scan_2] + torsions_to_freeze,
                    ase_constraints=ase_constraints)

                # Check if torsion angle was conserved during QM optimization
                self.set_positions_rdkit_conf(rdkit_conf, positions)

                new_torsion_1 = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan_1)
                new_torsion_2 = rdmt.GetDihedralDeg(rdkit_conf, *torsion_to_scan_2)

                assert (abs(old_torsion_1 - new_torsion_1) < threshold) or (
                        abs(abs(old_torsion_1 - new_torsion_1) - 360) < threshold), \
                    "Not conserving torsion angle 1 after QM optimization; old={} new={}".format(old_torsion_1, new_torsion_1)
                assert (abs(old_torsion_2 - new_torsion_2) < threshold) or (
                        abs(abs(old_torsion_2 - new_torsion_2) - 360) < threshold), \
                    "Not conserving torsion angle 2 after QM optimization; old={} new={}".format(old_torsion_2, new_torsion_2)

                # Append to list
                self.qm_energies_list.append(qm_energy)
                self.qm_forces_list.append(qm_force)
                self.qm_conformations_list.append(positions.in_units_of(unit.nanometers)._value)

            self.scan_angles.append([torsion_value_1, torsion_value_2])

            # Write scan restart
            self.write_restart_pickle(restart_settings, interface, "restart_scan_file", self.__dict__)

        print("!=================================================================================!\n")

        # Set positions of context to last position
        dummy_context.setPositions(positions * unit.nanometers)

        # Set RDKit geometry to the current in the OpenMM context
        self.set_positions_rdkit_conf(rdkit_conf, positions_initial.in_units_of(unit.angstrom)._value)

        del dummy_system, dummy_integrator, dummy_platform, dummy_context

        return self.scan_angles, self.qm_energies_list, self.qm_forces_list, self.qm_conformations_list, self.mm_energies_list, self.mm_conformations_list

    # ------------------------------------------------------------ #
    #                                                              #
    #                        STATIC METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    @staticmethod
    def get_mm_relaxed_conformations(system, torsions_to_freeze, tolerance=0.01, max_iter=0, force_constant=9999999.0, threshold=1e-2):
        """

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

            tmp_system = copy.deepcopy(system.engine.system)
            old_torsion_list = []

            for torsion in torsions_to_freeze:
                old_torsion = rdmt.GetDihedralDeg(rdkit_conf, *torsion)
                old_torsion_list.append(old_torsion)
                # energy_expression = f'fc*(theta-theta0)*(theta-theta0)'

            TorsionScan.freeze_torsions(tmp_system, torsions_to_freeze, old_torsion_list, force_constant)

            # Create temporary context and set the positions in it
            tmp_context = Context(tmp_system, copy.deepcopy(system.engine.integrator), Platform.getPlatformByName(system.engine.platform_name))
            tmp_context.setPositions(conf)
            # Perform minimization
            LocalEnergyMinimizer.minimize(tmp_context, tolerance=tolerance, maxIterations=max_iter)

            # Get MM-relaxed conformation and store it
            positions = tmp_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)._value

            # Set new position in RDKit conformation
            TorsionScan.set_positions_rdkit_conf(rdkit_conf, positions)
            new_torsion_list = []
            for torsion in torsions_to_freeze:
                new_torsion = rdmt.GetDihedralDeg(rdkit_conf, *torsion)
                new_torsion_list.append(new_torsion)

            for new_torsion, old_torsion in zip(new_torsion_list, old_torsion_list):
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
                                  positions=system.engine.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True),
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
    def freeze_torsions(system, torsions_to_freeze, torsions_angles, k):
        """
        Method that freezes the torsion_to_freeze torsion of an OpenMM system by adding a restraint to it.

        Parameters
        ----------
        system : simtk.openmm.System
            Instance of a OpenMM System.
        torsions_to_freeze : list of lists of int
            List of lists containing indices of the atoms to be frozen
        torsions_angles : list of float
            List of values of the desired torsion angle in degrees.
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
        restraint.addGlobalParameter('fc', fc)
        restraint.addPerTorsionParameter('theta0')

        for torsion, angle in zip(torsions_to_freeze, torsions_angles):
            torsion_id = restraint.addTorsion(*torsion)
            restraint.setTorsionParameters(torsion_id, *torsion, [angle * np.pi / 180.0])

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
            for sub_force in system.force_field.force_field['PeriodicTorsionForce']:
                for force_field_term in sub_force:
                    if (force_field_term.atoms[1] == bond[0] or force_field_term.atoms[1] == bond[1]) and (force_field_term.atoms[2] == bond[0] or force_field_term.atoms[2] == bond[1]):
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

