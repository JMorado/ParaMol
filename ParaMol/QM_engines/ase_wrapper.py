# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.QM_engines.ase_wrapper.ASEWrapper` class, which is a ParaMol implementation of a ASE wrapper.
"""
import ase
import copy
import logging
import os
import shutil
import simtk.unit as unit

import numpy as np

# ASE imports
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
import ase.units as ase_unit
from ase.constraints import FixInternals


# ParaMol imports
from ..Utils.interface import *


class ASEWrapper:
    """
    ParaMol implementation of an ASE wrapper.

    Parameters
    ----------
    system_name : str
        Name of the system to which this wrapper is associated.
    interface : :obj:`ParaMol.Utils.interface.ParaMolInterface`
        ParaMol interface object instance.
    calculator : Any ASE calculator defined in the modules of the subpackage :obj:`ase.calculators`.
        ASE calculator instance.
    n_atoms : int
        Number of atoms of the system.
    atom_list : list of str
        Atom list symbols.
    n_calculations : int
        Number of calculations
    cell : np.ndarray, shape=(3,3), dtype=float
        Box cell vectors.
    work_dir_prefix : str, default="ASEWorkDir"
        Path to the working directory.
    optimizer : any ase.optimizer, default=:obj:`ase.optimize.bfgs.BFGS*`
        ASE optimizer instance. For more info see: https://wiki.fysik.dtu.dk/ase/ase/optimize.html
    opt_fmax : float, default=1e-2
        The convergence criterion to stop the optimization is that the force on all individual atoms should be less than `fmax`.
    opt_log_file : str, default="-"
        File where optimization log will be stored. Use '-' for stdout.
    opt_traj_prefix : str, default="traj\_"
        Prefix given to the pickle file used to store the trajectory during optimization.
    view_atoms : bool, default=`False`
        Whether or not to view atoms after a calculation.
    calc_dir_prefix : str, default="ase\_"
        Prefix given to the directories where the calculations will be performed.
    """

    def __init__(self, system_name, interface, calculator, n_atoms, atom_list, n_calculations, cell, calc_dir_prefix="ase_",
                 work_dir_prefix="ASEWorkDir_", view_atoms=False, optimizer=BFGS,
                 opt_fmax=1e-2, opt_log_file="-", opt_traj_prefix="traj_",
                 md_steps=100, md_dt=1.0*ase_unit.fs, md_initial_temperature=300.0*ase_unit.kB,
                 md_integrator=VelocityVerlet, md_integrator_args={}):

        # Name of the system to which this wrapper is associated to
        self._system_name = system_name

        # General variables
        self._atom_list = atom_list
        self._n_calculations = n_calculations
        self._n_atoms = n_atoms
        self._cell = cell
        self._view_atoms = view_atoms

        # ASE optimizer variables
        self._optimizer = optimizer
        self._opt_fmax = opt_fmax
        self._opt_logfile = opt_log_file
        self._opt_traj_prefix = opt_traj_prefix

        # ASE MD variables
        self._md_steps = md_steps
        self._md_dt = md_dt
        self._md_initial_temperature = md_initial_temperature
        self._md_integrator = md_integrator
        self._md_integrator_args = md_integrator_args

        # ParaMol interface
        self._interface = interface
        self._calculation_dirs = []
        self._calc_dir_prefix = calc_dir_prefix
        self._work_dir = "{}{}".format(work_dir_prefix, self._system_name)

        # Initialize calculators
        assert calculator is not None, "ASE calculator instance is None."
        if self._n_calculations > 1:
            self._ase_calculator = [copy.deepcopy(calculator) for i in range(self._n_calculations)]
        else:
            self._ase_calculator = [calculator]

        # Create Symbols Strings
        self._symbols_string = ''
        for symbol in self._atom_list:
            self._symbols_string += str(symbol)

        # Prepare working directory
        self._prepare_workdir()

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_calculation(self, coords, label, calc_type="single_point", dihedral_freeze=None, ase_constraints=None):
        """
        Method that runs an ASE calculation.

        Parameters
        ----------
        coords : np.ndarray, shape=(n_atoms,3), dtype=float
            Coordinates array.
        label : str or int
            Label of the calculation.
        calc_type : str, default="single_point"
            Available options are "single_point" and "optimization".
        dihedral_freeze : list of list of int, default=None
            List of lists of wherein each inner list should contain 4 integers defining a torsion to be kept fixed.
        ase_constraints : list of ASE constraints, default=None
            List of ASE constraints to be applied during the scans.
            More information: https://wiki.fysik.dtu.dk/ase/ase/constraints.html

        Returns
        -------
        coord : np.ndarray, shape=(n_atoms,3), dtype=float
             Coordinate array in nanometers.
        energy : float
            Energy value in kJ/mol.
        forces : np.ndarray, shape=(n_atoms,3), dtype=float
            Forces array, kJ/mol/nm.
        """
        from ase.visualize import view

        # Change to calculation directory
        self._interface.chdir(self._calculation_dirs[label], absolute=True)

        # Reset ase calculator
        self._ase_calculator[int(label)].reset()

        # Set calculator in Atoms object
        atoms = ase.Atoms(self._symbols_string, coords)
        atoms.set_calculator(self._ase_calculator[label])

        if self._cell is not None:
            # Set the cell and center the atoms
            atoms.set_cell(self._cell)
            atoms.center()

        if calc_type.lower() == "single_point":
            # Run calculation and extract potential energy and forces
            # Set calculator in Atoms object

            energy = atoms.get_potential_energy() * 96.48530749925794  # eV to kJ/mol
            forces = atoms.get_forces() * 96.48530749925794 * 10.0  # eV/A to kJ/mol/nm

            # View molecule after sp calculation
            if self._view_atoms:
                view(atoms)

            # Go back to main folder
            self._interface.chdir_base()

            return energy, forces
        elif calc_type.lower() == "optimization":
            # Run optimization
            from ase.io import write, read
            logging.info("Performing QM optimization using ASE optimizer.")

            # Apply necessary dihedral constraints
            if dihedral_freeze is not None:
                dihedrals_to_fix = []
                for dihedral in dihedral_freeze:
                    dihedrals_to_fix.append([atoms.get_dihedral(*dihedral) * np.pi / 180.0, dihedral])

                constraint = FixInternals(bonds=[], angles=[], dihedrals=dihedrals_to_fix)
                atoms.set_constraint(constraint)

            # Apply any ASE constraints
            # More information: https://wiki.fysik.dtu.dk/ase/ase/constraints.html
            if ase_constraints is not None:
                for constraint in ase_constraints:
                    atoms.set_constraint(constraint)

            opt = self._optimizer(atoms, trajectory=self._opt_traj_prefix+".traj", logfile=self._opt_logfile)
            opt.run(fmax=self._opt_fmax)

            if dihedral_freeze is not None:
                del atoms.constraints

            # View molecule after optimization
            if self._view_atoms:
                view(atoms)

            # Get data
            coords = atoms.get_positions() * 0.1 # Angstrom to nm
            energy = atoms.get_potential_energy() * 96.48530749925794  # eV to kJ/mol
            forces = atoms.get_forces() * 96.48530749925794 * 10.0 # eV/A to kJ/mol/nm

            # Go back to main folder
            self._interface.chdir_base()

            return coords, energy, forces
        else:
            raise NotImplementedError("Calculation of type {} is not implemented.".format(calc_type))

    def run_md(self, coords, label, steps=None, dt=None, initial_temperature=None, integrator=None, integrator_args=None, dihedral_freeze=None, ase_constraints=None):
        """
        Method that runs an ASE calculation.

        Parameters
        ----------
        coords : np.ndarray, shape=(n_atoms,3), dtype=float
            Coordinates array.
        label : str or int
            Label of the calculation.
        steps: int
            Number of steps to take.
        dt : unit.Quantity
            MD Time step.
        initial_temperature : float with ase.unit
            Temperature of MD.
        integrator : ase.md integrator
            ASE integrator.
        integrator_args : dict
            Additional integrator options.
        dihedral_freeze : list of list of int, default=None
            List of lists of wherein each inner list should contain 4 integers defining a torsion to be kept fixed.
        ase_constraints : list of ASE constraints, default=None
            List of ASE constraints to be applied during the scans.
            More information: https://wiki.fysik.dtu.dk/ase/ase/constraints.html

        Returns
        -------
        coord : np.ndarray, shape=(n_atoms,3), dtype=float
             Coordinate array in nanometers.
        energy : float
            Energy value in kJ/mol.
        forces : np.ndarray, shape=(n_atoms,3), dtype=float
            Forces array, kJ/mol/nm.
        """
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        if steps is None:
            assert self._md_steps is not None
            steps = self._md_steps
        if dt is None:
            assert self._md_dt is not None
            dt = self._md_dt
        if initial_temperature is None:
            assert self._md_initial_temperature is not None
            initial_temperature = self._md_initial_temperature
        if integrator is None:
            assert self._md_integrator is not None
            integrator = self._md_integrator
        if integrator_args is None:
            assert self._md_integrator_args is not None
            integrator_args = self._md_integrator_args

        # Change to calculation directory
        self._interface.chdir(self._calculation_dirs[label], absolute=True)

        # Reset ase calculator
        self._ase_calculator[label].reset()

        # Set calculator in Atoms object
        atoms = ase.Atoms(self._symbols_string, coords)
        atoms.set_calculator(self._ase_calculator[label])

        if self._cell is not None:
            # Set the cell and center the atoms
            atoms.set_cell(self._cell)
            atoms.center()

        # Apply necessary dihedral constraints
        if dihedral_freeze is not None:
            dihedrals_to_fix = []
            for dihedral in dihedral_freeze:
                dihedrals_to_fix.append([atoms.get_dihedral(*dihedral) * np.pi / 180.0, dihedral])

            constraint = FixInternals(bonds=[], angles=[], dihedrals=dihedrals_to_fix)
            atoms.set_constraint(constraint)

        # Apply any ASE constraints
        # More information: https://wiki.fysik.dtu.dk/ase/ase/constraints.html
        if ase_constraints is not None:
            for constraint in ase_constraints:
                atoms.set_constraint(constraint)

        # Randomize velocities
        MaxwellBoltzmannDistribution(atoms, initial_temperature, force_temp=False, rng=np.random)

        # Get data
        kinetic_intial = atoms.get_kinetic_energy() * 96.48530749925794  # eV to kJ/mol
        potential_inital = atoms.get_potential_energy() * 96.48530749925794  # eV to kJ/mol
        forces_initial = atoms.get_forces() * 96.48530749925794 * 10.0  # eV/A to kJ/mol/nm

        dyn = integrator(atoms, dt, *integrator_args, trajectory=self._opt_traj_prefix+".traj", logfile=self._opt_logfile, )
        dyn.run(steps)

        # Get data
        coords = atoms.get_positions() * 0.1  # Angstrom to nm
        kinetic_final = atoms.get_kinetic_energy() * 96.48530749925794  # eV to kJ/mol
        potential_final = atoms.get_potential_energy() * 96.48530749925794  # eV to kJ/mol
        forces_final = atoms.get_forces() * 96.48530749925794 * 10.0  # eV/A to kJ/mol/nm

        # View molecule after sp calculation

        self._interface.chdir_base()

        return coords, potential_inital, kinetic_intial, forces_initial, potential_final, kinetic_final, forces_final

    # ------------------------------------------------------------ #
    #                                                              #
    #                        PRIVATE METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def _prepare_workdir(self):
        """
        Method that prepares the working directory.

        Returns
        -------
        None
        """
        # Go to base directory
        self._interface.chdir_base()
        # Create working directory
        self._interface.create_dir(self._work_dir)
        # Change to working directory
        self._interface.chdir(self._work_dir, relative_to_base=True)

        for i in range(self._n_calculations):
            self._interface.create_dir(self._calc_dir_prefix+"{}".format(i))
            self._interface.chdir(self._calc_dir_prefix+"{}".format(i))
            self._calculation_dirs.append(os.getcwd())
            self._interface.chdir_previous()

        self._interface.chdir_base()

        return 1

