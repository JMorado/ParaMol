# -*- coding: utf-8 -*-
"""
Description
-----------
This module defines the :obj:`ParaMol.Utils.settings.Settings` class used to define ParaMol global settings.
"""
import ase.units as ase_unit
import numpy as np
import simtk.unit as unit
from ase.optimize import BFGS as BFGS_ase
from scipy.optimize import BFGS as BFGS_scipy
from ase.md.verlet import VelocityVerlet

# ---------------------------------------------------------- #
#                                                            #
#                      ParaMol SETTINGS                      #
#                                                            #
# ---------------------------------------------------------- #
class Settings:
    """
    ParaMol global settings.

    Attributes
    ----------
    optimizer : dict
        Dictionary that contains the optimizer settings.
    parameter_space : dict
        Dictionary that contains the parameter space settings.
    objective_function : dict
        Dictionary that contains the objective function settings.
    qm_engine : dict
        Dictionary that contains the QM engine settings.
    restart: dict
        Dictionary that contains the restart settings.
    """
    def __init__(self):
        # ---------------------------------------------------------- #
        #                                                            #
        #                     OPTIMIZER SETTINGS                     #
        #                                                            #
        # ---------------------------------------------------------- #
        self.optimizer = {"method": "scipy",
                          "monte_carlo":
                              {"n_blocks": 100,
                               "max_iter": 1000000000,
                               "f_tol": 1e-8,
                               "prob": 0.25, },
                          "gradient_descent":
                              {"max_iter": 1000000000,
                               "derivative_calculation": "f_increase",
                               "derivative_type": "1-point",
                               "g_tol": 1e-3,
                               "f_tol": 1e-8,
                               "dx": 1e-2,
                               "derivative_h": 1e-4, },
                          "simulated_annealing":
                              {"n_iter": 10000,
                               "p_init": 0.2,
                               "p_final": 0.001,
                               "avg_acceptance_rate": 0.25, },
                          "scipy":
                              {"method": "SLSQP",
                               "options": {"disp": True, "iprint": 2, "verbosity": True, "maxiter": 10000, "ftol": 1e-6, },
                               "jac": "2-point",
                               "hess": BFGS_scipy(), }
                          }
        # ---------------------------------------------------------- #
        #                                                            #
        #                    PARAMETER SPACE SETTINGS                #
        #                                                            #
        # ---------------------------------------------------------- #
        self.parameter_space = {"parameters_magnitudes": {"charge": 0.5,
                                                          "lj_sigma": 0.30,
                                                          "lj_eps": 0.20,
                                                          "torsion_phase": np.pi, # rad
                                                          "torsion_k": 4 * 4.184, # kJ mol^{-1}
                                                          "bond_eq": 0.05, # nm
                                                          "bond_k":  100000, # kJ mol^{-1} nm^{-2}
                                                          "angle_eq": np.pi / 16.0, # rad
                                                          "angle_k": 100.0, # kJ mol^{-1} rad^{-2}
                                                          "scee": 1.0,
                                                          "scnb": 1.0},
                                "prior_widths_method": "default",
                                "scaling_constants_method": "arithmetic",}

        # ---------------------------------------------------------- #
        #                                                            #
        #                 OBJECTIVE FUNCTION SETTINGS                #
        #                                                            #
        # ---------------------------------------------------------- #
        self.objective_function = {"parallel": False,
                                   "platform_name": "Reference",
                                   "weighting_method": "uniform",
                                   "weighting_temperature": 300.0*unit.kelvin,
                                   "checkpoint_freq": 100}

        # ---------------------------------------------------------- #
        #                                                            #
        #                     PROPERTIES SETTINGS                    #
        #                                                            #
        # ---------------------------------------------------------- #
        self.properties = {"include_energies": True,
                           "include_forces": True,
                           "include_esp": False,
                           "include_regularization": False,
                           "energies": {"weight": 1.0},
                           "forces": {"term_type": "components",
                                      "weight": 1.0},
                           "esp": {"weight": 1.0},
                           "regularization": {"method": "L2",
                                              "weight": 1.0,
                                              "scaling_factor": 1.0,
                                              "hyperbolic_beta": 0.1},}

        # ---------------------------------------------------------- #
        #                                                            #
        #                      QM ENGINE SETTINGS                    #
        #                                                            #
        # ---------------------------------------------------------- #
        self.qm_engine = {"qm_engine": "ase",
                          "dftb+": {"work_dir_prefix": "DFTBWorkDir_",
                                    "calc_file": "dftb_in.hsd",
                                    "calc_file_output": "dftb_output.out",
                                    "detailed_file_output": "detailed.out",
                                    "calc_dir_prefix": "dftb_",
                                    "geometry_file": "geometry.gen",
                                    "slater_koster_files_prefix": "../slakos/mio-ext/",
                                    "max_ang_mom": {"H": "s",
                                                    "C": "p",
                                                    "N": "p",
                                                    "O": "p",
                                                    "F": "p",
                                                    "S": "p"}, },
                          "amber": {"sqm_params": {"maxcyc": "0",
                                                   "qm_theory": "'AM1'",
                                                   "dftb_disper": "0",
                                                   "qmcharge": "0",
                                                   "scfconv": "1.0d-8",
                                                   "pseudo_diag": "0",
                                                   "verbosity": "5"},
                                    "work_dir_prefix": "AMBERWorkDir_",
                                    "calc_file_prefix": "sqm_", },
                          "ase": {"calculator": None,
                                  "optimizer": BFGS_ase,
                                  "opt_log_file": "-",
                                  "opt_fmax": 1e-2,
                                  "opt_traj_prefix": "traj_",
                                  "calc_dir_prefix": "ase_",
                                  "work_dir_prefix": "ASEWorkDir_",
                                  "view_atoms": False,
                                  "md_dt": 1.0 * ase_unit.fs,
                                  "md_steps": 100,
                                  "md_initial_temperature": 300 * ase_unit.kB,
                                  "md_integrator": VelocityVerlet,
                                  "md_integrator_args": {}, },
                          }

        # ---------------------------------------------------------- #
        #                                                            #
        #                      RESTART SETTINGS                      #
        #                                                            #
        # ---------------------------------------------------------- #
        self.restart = {"restart_dir": "restart_paramol",
                        "restart_scan_file": "restart_scan.pickle",
                        "restart_soft_torsions_file": "restart_soft_torsions.pickle",
                        "restart_adaptive_parametrization_file": "restart_adaptive_parametrization.pickle",
                        "restart_parameter_space_file": "restart_parameter_space.pickle"}
