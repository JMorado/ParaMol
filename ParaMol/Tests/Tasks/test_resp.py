# ParaMol imports
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *

# ParaMol Tasks imports
from ParaMol.Tasks.resp_fitting import *
from ParaMol.Utils.settings import *
from ParaMol.Utils.gaussian_esp import *

import numpy as np


class TestRESPTask:
    # Kwargs dictionary for AMBER topology system. These are shared between all instances.
    kwargs_dict = {"topology_format": "AMBER",
                   "top_file": "ParaMol/Tests/aniline.prmtop",
                   "crd_file": "ParaMol/Tests/aniline.inpcrd"}

    def test_resp(self):
        """
        Test RESP Task.
         Fitting point charges to electrostatic potential
         Charges from ESP fit, RMS=   0.00123 RRMS=   0.10456:
         ESP charges:
                       1
             1  C   -0.062863
             2  C   -0.369134
             3  C    0.503195
             4  C   -0.369134
             5  C   -0.062859
             6  C   -0.239133
             7  H    0.136280
             8  H    0.181401
             9  H    0.181401
            10  H    0.136279
            11  H    0.145494
            12  N   -0.920057
            13  H    0.369564
            14  H    0.369565
        """
        # --------------------------------------------------------- #
        #                         Preparation                       #
        # --------------------------------------------------------- #
        # Create the OpenMM engine for caffeine
        openmm_system = OpenMMEngine(init_openmm=True, **self.kwargs_dict)

        # Create Molecular System
        aniline = ParaMolSystem(name="aniline", engine=openmm_system, n_atoms=14)

        # Create ParaMol's force field representation and ask to optimize charges
        aniline.force_field.create_force_field(opt_charges=True)

        # Create ParaMol settings instance
        paramol_settings = Settings()
        paramol_settings.properties["include_energies"] = False
        paramol_settings.properties["include_forces"] = False
        paramol_settings.properties["include_esp"] = True
        # --------------------------------------------------------- #
        #                 Read ESP Data into ParaMol                #
        # --------------------------------------------------------- #
        gaussian_esp = GaussianESP()
        aniline.ref_coordinates, aniline.ref_esp_grid, aniline.ref_esp = gaussian_esp.read_log_files(["ParaMol/Tests/Tasks/aniline_opt.log"])

        # Set number of structures
        aniline.n_structures = len(aniline.ref_coordinates)

        # --------------------------------------------------------- #
        #                      RESP Charge Fitting                  #
        # --------------------------------------------------------- #
        charges_to_compare = [-0.062863, -0.369134, 0.503195, -0.369134, -0.062859, -0.239133,
                              0.136280, 0.181401, 0.181401, 0.136279, 0.145494, -0.920057,
                              0.369564, 0.369565]
        charges_to_compare = np.asarray(charges_to_compare)

        # Test EXPLICIT solver
        resp_fitting = RESPFitting()
        systems, parameter_space, objective_function, optimizer = resp_fitting.run_task(paramol_settings, [aniline], solver="explicit", total_charge=0)
        charges = np.asarray(parameter_space.optimizable_parameters_values)[:14]

        diff = charges-charges_to_compare
        for diff_charge in diff:
            assert abs(diff_charge) < 1e-4

        # Test SciPy solver
        resp_fitting = RESPFitting()
        systems, parameter_space, objective_function, optimizer = resp_fitting.run_task(paramol_settings, [aniline], solver="scipy", total_charge=0)
        charges = np.asarray(parameter_space.optimizable_parameters_values)

        print(parameter_space.optimizable_parameters_values_scaled)
        diff = charges-charges_to_compare
        for diff_charge in diff:
            assert abs(diff_charge) < 1e-4
