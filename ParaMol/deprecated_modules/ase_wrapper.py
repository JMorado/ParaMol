# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.QM_engines.ase_wrapper.ASEWrapper` class, which is a ParaMol implementation of a ASE wrapper.
"""


import ase
import copy
import os
import numpy as np
from ..Utils.interface import *
from ase.optimize import BFGS


class ASEWrapper:
    """
    ParaMol implementation of an ASE wrapper.

    Parameters
    ----------
    interface : :obj:`ParaMol.Utils.interface.ParaMolInterface`
        ParaMol interface object instance.
    calculator_name : str
        Name of the ASE calculator.
    parameters_file : str
        Parameters file.
    n_atoms : int
        Number of atoms of the system.
    atom_list : list of str
        Atom list symbols.
    n_calculations : int
        Number of calculations
    cell : np.ndarray, shape=(3,3), dtype=float
        Box cell vectors.
    calculator_instance : Any ASE calculator defined in the modules of the subpackage :obj:`ase.calculators`.
        ASE calculator instance.
    qm_wrapper_name : str
        Name of the QM wrapper. Should be "ase".
    work_dir : str
        Path to the working directory.
    optimizer : any ase.optimizer, default=:obj:`ase.optimize.bfgs.BFGS*
        ASE optimizer instance. For more info see: https://wiki.fysik.dtu.dk/ase/ase/optimize.html
    fmax : float, default=1e-2
        The convergence criterion to stop the optimization is that the force on all individual atoms should be less than `fmax`.
    log_file : str, default="-"
        File where optimization log will be stored. Use '-' for stdout.
    traj_prefix : str
        Prefix given to the pickle file used to store the trajectory during optimization.
    view_atoms : bool, default=`False`
        Whether or not to view atoms after a calculation.
    """

    def __init__(self, interface, calculator_name, parameters_file, n_atoms, atom_list, n_calculations, cell,
                 qm_wrapper_name="ase", calc_dir_prefix="ase_", work_dir="ASEWorkDir", view_atoms=False, optimizer=BFGS,
                 fmax=1e-2, log_file="-", traj_prefix="traj_", calculator_instance=None):
        self._calculator_name = calculator_name
        self._atom_list = atom_list
        self._parameters_file = parameters_file
        self._n_calculations = n_calculations
        self._n_atoms = n_atoms
        self._cell = cell

        self._view_atoms = view_atoms

        # ASE optimizer variables
        self._optimizer = optimizer
        self._opt_fmax = fmax
        self._opt_logfile = log_file
        self._opt_traj_prefix = traj_prefix

        # Set wrapper variables
        self._symbols_string = None

        # ParaMol interface
        self._interface = interface
        self._calculation_dirs = []

        # Params
        self._qm_wrapper_name = qm_wrapper_name
        self._calc_dir_prefix = calc_dir_prefix
        self._work_dir = work_dir

        # Prepare working directory
        self._prepare_workdir()

        # Initialize calculators
        if calculator_instance is None:
            self._ase_calculator = [None for i in range(self._n_calculations)]
            self._init_ase_calculators()
        else:
            self._ase_calculator = [copy.deepcopy(calculator_instance) for i in range(self._n_calculations)]
            # Create Symbols Strings
            self._symbols_string = ''
            for symbol in self._atom_list:
                self._symbols_string += str(symbol)

    def __str__(self):
        return "ASE wrapper of calculator {}.".format(self._calculator_name)

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_calculation(self, coords, label, type="single_point", dihedral_freeze=None, ase_constraints=None):
        """
        Method that runs an ASE calculation.

        Parameters
        ----------
        coords : np.ndarray, shape=(n_atoms,3), dtype=float
            Coordinates array
        label : int
            Label of the calculation.
        type : str, default="single_point"
            Available options are "single_point" and "optimization".
        dihedral_freeze : list of list of int, default=None
            List of lists of wherein each inner list should contain 4 integers defining a torsion to be kept fixed.
        ase_constraints : list of ASE constraints, default=None
            List of ASE constraints to be applied during the scans.

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

        if type.lower() == "single_point":
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
        elif type.upper() == "optimization":
            # Run optimization
            from ase.constraints import FixInternals
            from ase.io import write, read
            print("Performing QM optimization using ASE BFGS optimizer.")

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

            # Perform Optimization
            opt = self._optimizer(atoms, trajectory='opt_{}.traj'.format(label), logfile=self._opt_logfile)
            opt.run(fmax=self._opt_fmax)

            if dihedral_freeze is not None:
                del atoms.constraints

            # View molecule after optimization
            if self._view_atoms:
                view(atoms)

            # Get data
            coord = atoms.get_positions() * 0.1
            energy = atoms.get_potential_energy() * 96.48530749925794  # eV to kJ/mol
            forces = atoms.get_forces() * 96.48530749925794 * 10.0 # eV/A to kJ/mol/nm

            # Go back to main folder
            self._interface.chdir_base()

            return coord, energy, forces
        else:
            raise NotImplementedError("Calculation of type {} is not implemented.".format(type))

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

    def _init_ase_calculators(self):
        """
        Method that initializes calculators if an ASE calculator instance is not provided.

        Notes
        -----
        It doest not work for all calculators. Currently tested: DFTB+.

        Returns
        -------
        None
        """
        import importlib

        # Dynamically obtain the ASE calculator module
        ase_calculator_module = importlib.import_module('ase.calculators.{}'.format(self._calculator_name))

        # Create Symbols Strings
        self._symbols_string = ''
        for symbol in self._atom_list:
            self._symbols_string += str(symbol)

        # Create all necessary calculators
        for i in range(self._n_calculations):
            # self._ase_calculator[i] = ase.calculators.zzz.Zzz(parameters=self._parameters_file)
            if self._parameters_file is not None:
                exec("{}[{}] = {}.{}(label='{}', parameters={})".format('self._ase_calculator',
                                                                        i,
                                                                        'ase_calculator_module',
                                                                        self._calculator_name.lower().title(),
                                                                        "ase" + str(i),
                                                                        'self._parameters_file'))
            else:
                exec("{}[{}] = {}.{}(label='{}')".format('self._ase_calculator',
                                                         i,
                                                         'ase_calculator_module',
                                                         self._calculator_name.lower().title(),
                                                         "ase" + str(i)))

        return self._ase_calculator