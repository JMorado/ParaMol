ParaMol Settings
**********************************

Examples of how to change ParaMol settings
###########################################

.. code-block:: python
   :caption: Example of usage

   from ParaMol.Util.settings import *

   # First create the settings instance
   settings = Settings()

   # Change the optimizer
   settings.optimizer["method"] = "scipy"

   # Include regularization in the objective function
   settings.properties["include_regularization"] = True
   # Change regularization calculation method
   settings.properties["regularization"]["method"] = "L1"

   # Add a mapping to the maximum angular momentum DFTB+ dict
   settings.qm_engine["dftb+"]["max_ang_mom"]["Mg"] = "p"

   # Change the method used to calculate the Hessian when using SciPy optimizers (default is BFGS)
   from scipy.optimize import SR1
   settings.optimizer["scipy"]["options"]["hess"] = SR1



Input Parameters
###########################################

The list of all ParaMol settings is here presented using the syntax **input_parameter** (*type*, default value).

Parameter Space Parameters (parameter_space)
--------------------------------------------
* **prior_widths_method** (`str`, "default"): Method used to calculate the prior_widths. Available methods are 'default', 'arithmetic' and 'geometric'.
* **scaling_constants** (`str`, "default"): Method used to calculate the scaling constants. Available methods are 'default', 'arithmetic' and 'geometric'.
* **parameters_magnitude** (`dict`, see below): Dictionary containing the mapping between parameter keys and their natural magnitude values. This is used to calculate the scaling constants and prior widths if the 'default' option is chosen.
   * **charge** (`float`, 0.2): Charge (in elementary charge units).
   * **lj_sigma** (`float`, 0.30): Lennard-Jones 12-6 sigma parameter (in nanometers).
   * **lj_eps** (`float`, 0.20): Lennard-Jones 12-6 epsilon parameter (in kJ/mol).
   * **torsion_phase** (`float`, pi): Torsion phase (in radians).
   * **bond_eq** (`float`, 0.05): Bond equilibrium length (in nanometers) .
   * **bond_k** (`float`, 100000): Bond force constant (in :math:`kJ mol^{-1} nm^{-2}`).
   * **angle_eq** (`float`, pi/16.0): Angle equilibrium value (in radians).
   * **angle_k** (`float`, 100.0): Angle force constant (:math:`kJ mol^{-1} rad^{-2}`).
   * **scee** (`float`, 1.0): 1-4 Scaling Factor of electrostatic interactions.
   * **scnb** (`float`, 1.0): 1-4 Scaling Factor of Lennard-Jones interactions.


Optimizer Parameters (optimizer)
---------------------------------
* **method** (`str`, "scipy"): Optimizer. Available optimizers: "scipy", "monte_carlo", "gradient_descent", "simulated_annealing.
* **scipy** (`dict`, see below): SciPy optimizer settings.
   * **method** (`str`, "SLSQP"): SciPy optimization method.
   * **options** (`dict`, ): Keyword arguments passed to :obj:`scipy.optimize.minimize` function (except the objective function callable, method name and x0, i.e., the parameter's array).
* **monte_carlo** (`dict`): Monte Carlo optimizer settings.
   * **n_blocks** (`int`, 100): SciPy optimization method.
   * **max_iter** (`int`, 1000000000): Maximum number of iterations allowed in the optimization.
   * **f_tol** (`float`, 1e-8): If the change in the objective function between two successive iterations is lower than this threshold then we assume convergence has been reached.
   * **avg_acceptance_rate** (`float`, 0.25): If at the start of a new MC block the acceptance rate of a given parameter is larger (lower) than `avg_acceptance_rate`, the maximum displacement for that parameter is increased (decreased).
* **simulated_annealing** (`dict`): Simulated Annealing optimizer settings.
   * **n_iter** (`int`, 10000): Number of iterations to perform in total.
   * **p_init** (`float`, 0.2): Probability of accepting worse solution at the beginning. The initial temperature is given by :math:`-1/log(p_{init})`.
   * **p_final** (`float`, 0.001): Probability of accepting worse solution at the end. The final temperature is given by :math:`-1/log(p_{final})`.
   * **avg_acceptance_rate** (`float`, 0.25): If at the start of a new MC block the acceptance rate of a given parameter is larger (lower) than `avg_acceptance_rate`, the maximum displacement for that parameter is increased (decreased).
* **gradient_descent** (`dict`): Gradient Descent optimizer settings.
   * **max_iter** (`int`, 1000000000): Maximum number of iterations allowed in the optimization.
   * **derivative_calculation** (`str`, "f_increase"): When to calculate the derivatives, which are the most computational expensive part of the algorithm. A fair approximation is to re-compute the derivative only if the value of the objective function has increased in successive iterations. Available options are "f-increase" and "always".
   * **derivative_type** (`str`, "1-point"): Type of numerical differentiation to perform. Available options are "1-point" or "2-point".
   * **g_tol** (`float`, 1e-3): If the change in the gradient between two successive iterations is lower than this threshold then we assume convergence has been reached.
   * **f_tol** (`float`, 1e-8): If the change in the objective function between two successive iterations is lower than this threshold then we assume convergence has been reached.
   * **dx** (`float`, 1e-2): Change in x for the numerical differentiation (denominator).
   * **derivative_h** (`float`, 1e-4): Scaling factor that multiplies the step size of the descent.


Objective function parameters (objective_function)
---------------------------------------------------
* **parallel** (`bool`, True): Flag that signals if the objective function calculation is to be performed in parallel.
* **platform_name** (`str`, "Reference"): Name of the OpenMM platform to be used to calculate the objective function. Only options are 'Reference', 'CPU' and 'OpenCL'.
* **weighing_method** (`str`, "uniform"): Method used to weigh the conformations. Available methods are "uniform, "boltzmann" and "non-boltzmann".
* **weighing_temperature** (`simtk.unit.Quantity`, 300.0*unit.kelvin): Temperature used in the weighing. Only relevant if `weighing_method` is "boltzmann" or "non_boltzmann".
* **checkpoint_freq** (`int`, 1000): Frequency at which checkpoint files are saved. Useful for restarts.


Properties parameters (properties)
--------------------------------------
* **include_energies** (`bool`, True): Flag that signals if the objective function calculation includes an energy term.
* **include_forces** (`bool`, True): Flag that signals if the objective function calculation includes a forces term.
* **include_esp** (`bool`, False): Flag that signals if the objective function calculation includes a electrostatic potential term.
* **include_regularization** (`bool`, False): Flag that signals if the objective function calculation includes a regularization term.
* **energies** (`dict`): Energy property settings.
   * **weight** (`int`, 1.0): Weight of this property in the objective function.
* **forces** (`dict`): Force property settings.
   * **weight** (`int`, 1.0): Weight of this property in the objective function.
   * **term_type** (`str`, "components"): Forces term type. Available options are "norm" or "components".
* **esp** (`dict`): Electrostatic potential property settings.
   * **weight** (`int`, 1.0): Weight of this property in the objective function.
* **regularization** (`dict`): Regularization property settings.
   * **weight** (`int`, 1.0): Weight of this property in the objective function.
   * **method** (`str`, "L2"): Type of regularization. Options are 'L1', 'L2' or 'hyperbolic' ('hyperbolic' only for RESP calculations)
   * **scaling_factor** (`float`, 0.1): Scaling factor of the regularization value.
   * **hyperbolic_beta** (`float`, 0.01): Hyperbolic beta value. Only used if `method` is `hyperbolic`.

QM Engine parameters (qm_engine)
---------------------------------
* **qm_engine** (`str`, "ase"): QM engine wrapper to be used. Available QM engines are: "ase", "amber", "dftb+".
* **ase** (`dict`): ASE QM engine settings.
   * **calculator** (`str`, None): ASE calculator instance.
   * **optimizer** (`str`, BFGS): ASE optimizer instance. For more info see: https://wiki.fysik.dtu.dk/ase/ase/optimize.html
   * **opt_log_file** (`str`, "-"): File where optimization log will be stored. Use '-' for stdout.
   * **opt_fmax** (*float*, 1e-2): The convergence criterion to stop the optimization is that the force on all individual atoms should be less than `fmax`.
   * **opt_traj_prefix** (`str`, "traj\_"): Prefix given to the pickle file used to store the trajectory during optimization.
   * **calc_dir_prefix** (`str`, "ase\_"): Prefix given to the directories where the calculations will be performed.
   * **work_dir** (`str`, "ASEWorkDir"): Relative path to the working directory (relatively to the base directory).
   * **view_atoms** (`bool`, `False`): Whether or not to view the molecule after a calculation.
* **amber** (`dict`): AMBER QM engine settings.
   * **sqm_params** (`dict`, {"maxcyc": "0", "qm_theory": "'AM1'", "dftb_disper": "0", "qmcharge": "0", "scfconv": "1.0d-8", "pseudo_diag": "0","verbosity": "5"}): SQM parameters to be used in the input file. See AMBER manual for more information.
   * **calc_file_prefix** (`str`, "sqm\_"): Prefix given to the SQM calculation files.
   * **work_dir** (`str`, "AMBERWorkDir"): Relative path to the working directory (relatively to the base directory).
* **dftb+** (`dict`): DFTB+ QM engine settings.
   * **calc_file** (`str`, "dftb_in.hsd"): Name given to the DFTB+ calculation file.
   * **calc_file_output** (`str`, "dftb_output.out"): Name given to the DFTB+ stdout file.
   * **detailed_file_output** (`str`, "detailed.out"): Name of the detailed output file written by DFTB+. By default DFTB+ writes this to 'detailed.out'. Don't change this unless you know what you're doing.
   * **geometry_file** (`str`, "geometry.gen"): Name given to the DFTB+ .gen geometry file.
   * **slater_koster_files_prefix** (`str`, ""): Path to the Slater-Koster files.
   * **max_ang_mom** (`dict`, {"H": "s", "C": "p", "N": "p", "O": "p", "F": "p", "S": "p"}): Dictionary that defines the maximum angular momentum for each chemical element.
   * **calc_dir_prefix** (`str`, "dftb\_"): Prefix given to each subdirectory.
   * **work_dir** (`str`, "DFTBWorkDir"): Relative path to the working directory (relatively to the base directory).

Restart parameters (qm_engine)
---------------------------------
* **restart_dir_prefix** (`str`, "restart_"): Prefix given to the restart directory.
* **restart_file** (`str`, None): Name of the NETCDF restart file.
* **restart_scan_file** (`str`, "restart_scan.pickle"): Name of the file used to restart a dihedral scan.
* **parameters_generation_pickle** (`str`, "Reference"): Pickle file with the parameters of every generation.



