# Import ParaMol modules
from ParaMol.System.system import *
from ParaMol.Utils.interface import *
from ParaMol.QM_engines.amber_wrapper import *
from ParaMol.QM_engines.dftb_wrapper import *
from ParaMol.QM_engines.ase_wrapper import *

import numpy as np

class TestSystem:
    # Kwargs dictionary for AMBER topology system. These are shared between all instances.
    kwargs_dict = {"topology_format": "AMBER",
                   "top_file": "ParaMol/Tests/aniline.prmtop",
                   "crd_file": "ParaMol/Tests/aniline.inpcrd"}

    def test_create_qm_engines(self):
        """
        Test the creation of the QM engines.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        interface = ParaMolInterface()

        # Test AMBER engine
        qm_engine_settings = {"sqm_params": {"maxcyc": "0",
                                             "qm_theory": "'AM1'",
                                             "dftb_disper": "0",
                                             "qmcharge": "0",
                                             "scfconv": "1.0d-8",
                                             "pseudo_diag": "0",
                                             "verbosity": "5"},
                              "work_dir_prefix": "AMBERWorkDir_",
                              "calc_file_prefix": "sqm_", }

        qm_engine = system.create_qm_engines("amber", qm_engine_settings, interface=interface, overwrite_qm_engine=True)
        assert type(qm_engine) is QMEngine
        assert type(qm_engine.qm_engine) is AmberWrapper

        # Test AMBER engine
        qm_engine_settings = {"work_dir_prefix": "DFTBWorkDir_",
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
                                              "S": "p"}}

        qm_engine = system.create_qm_engines("dftb+", qm_engine_settings, interface=interface, overwrite_qm_engine=True)
        assert type(qm_engine) is QMEngine
        assert type(qm_engine.qm_engine) is DFTBWrapper

        # Test ASE Engine
        import ase.calculators.dftb as dftb_ase
        from ase.optimize import BFGS as BFGS_ase

        calc = dftb_ase.Dftb(Hamiltonian_='DFTB',  # line is included by default
                             Hamiltonian_MaxAngularMomentum_='',
                             Hamiltonian_MaxAngularMomentum_H='s',
                             Hamiltonian_MaxAngularMomentum_O='p',
                             Hamiltonian_MaxAngularMomentum_C='p',
                             Hamiltonian_MaxAngularMomentum_N="p",
                             Hamiltonian_Dispersion="DftD3 { \n s6=1.000 \n s8=0.5883 \n Damping = BeckeJohnson { \n a1=0.5719 \n a2=3.6017 \n } \n }",
                             Hamiltonian_SCC='Yes',
                             Hamiltonian_SCCTolerance=1e-8, )

        qm_engine_settings = {"calculator": calc,
                              "optimizer": BFGS_ase,
                              "opt_log_file": "-",
                              "opt_fmax": 1e-2,
                              "opt_traj_prefix": "traj_",
                              "calc_dir_prefix": "ase_",
                              "work_dir_prefix": "ASEWorkDir_",
                              "view_atoms": False, }

        qm_engine = system.create_qm_engines("ase", qm_engine_settings, interface=interface, overwrite_qm_engine=True)
        assert type(qm_engine) is QMEngine
        assert type(qm_engine.qm_engine) is ASEWrapper

        # Remove all created dirs
        interface.remove_all_created_dirs()

    def test_get_energies_ensemble(self):
        """
        Test the function used to get energies of an ensemble of structures.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        # Read data into system
        system.read_data("ParaMol/Tests/aniline_10_struct.nc")

        # Get energies
        energies = system.get_energies_ensemble()
        energies_to_compare = np.asarray([-61.29511275, -14.29133038, -46.76993321, -43.97566967, -51.64154255, -37.39727507, -18.60248536, -52.02555262, -25.79225498, -46.27102486])
        np.testing.assert_almost_equal(energies, energies_to_compare)

    def test_get_forces_ensemble(self):
        """
        Test the function used to get forces of an ensemble of structures.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        # Read data into system
        system.read_data("ParaMol/Tests/aniline_10_struct.nc")

        # Get energies
        forces = system.get_forces_ensemble()[0]
        forces_to_compare = np.asarray([[226.69279709,  -221.50920045,  -155.4342322],
                                        [-1116.02388518,   594.75325055, -1328.57469076],
                                        [652.56273853, -1030.99559853, 1911.77606109],
                                        [-427.9030738,    226.57480277,  -418.60829341],
                                        [1398.64920847,  -558.21611627,   424.17273528],
                                        [-583.48863777,   646.54982489, -1214.44929892],
                                        [-970.36555575,  -505.02849595,  1188.10300875],
                                        [-2211.10528239,   179.16499916,  -638.93207479],
                                        [506.69872438,   103.62439112,    44.16050723],
                                        [1826.87889647,   366.13799034,   309.17267574],
                                        [910.80512382,  -705.24333107,  -271.95571324],
                                        [-632.38102157,  -286.31023169,  -440.47115881],
                                        [453.73793472,   938.54210868,   888.41665337],
                                        [-34.75796701,   251.95560644,  -297.37617932]])

        np.testing.assert_almost_equal(forces, forces_to_compare)

    def test_weighting_methods(self):
        """
        Test the creation of the weighting methods.
        """
        import simtk.unit as unit

        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        # Read data into system
        system.read_data("ParaMol/Tests/aniline_10_struct.nc")

        uniform = system.compute_conformations_weights(weighting_method="uniform")
        uniform_to_compare = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        np.testing.assert_almost_equal(uniform, uniform_to_compare)

        boltzmann = system.compute_conformations_weights(temperature=300 * unit.kelvin, weighting_method="boltzmann")
        boltzmann_to_compare = np.asarray([9.22184537e-01, 7.06265178e-07, 4.15510475e-03, 1.07451351e-03, 3.53751796e-02, 2.78547996e-05, 3.70800366e-07, 9.58279141e-03, 3.51432746e-05, 2.75637985e-02])
        np.testing.assert_almost_equal(boltzmann, boltzmann_to_compare)

        non_boltzmann = system.compute_conformations_weights(temperature=300 * unit.kelvin, emm=system.get_energies_ensemble(), weighting_method="non_boltzmann")
        non_boltzmann_to_compare = [0.12733302, 0.00108862, 0.08359318, 0.10544513, 0.06922439, 0.29104615, 0.01167686, 0.29807564, 0.00220013, 0.01031687]
        np.testing.assert_almost_equal(non_boltzmann, non_boltzmann_to_compare)

    def test_filter_conformations(self):
        """
        Test the filtering function.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        # Read data into system
        system.read_data("ParaMol/Tests/aniline_10_struct.nc")

        # Filter conformations
        system.filter_conformations(2.0)

        # Test energies
        energies = system.ref_energies
        energies_to_compare = np.asarray([-40208.87850801699])
        np.testing.assert_almost_equal(energies, energies_to_compare)

        # Test coordinates
        coordinates = system.ref_coordinates
        coordinates_to_compare = np.asarray([[[2.03194237, 2.13566232, 1.99741292],
                                              [1.99560583, 2.03283095, 2.0001924 ],
                                              [2.03304195, 1.9479599 , 2.1009388 ],
                                              [2.10952282, 1.97587705, 2.17562103],
                                              [1.97260213, 1.82135832, 2.11615801],
                                              [2.00655174, 1.75605106, 2.20266533],
                                              [1.88766515, 1.77413332, 2.0154779 ],
                                              [1.83345687, 1.64601481, 2.02857399],
                                              [1.83921933, 1.60234344, 2.1192987 ],
                                              [1.73651922, 1.63113403, 1.98739231],
                                              [1.83786952, 1.86467838, 1.92140806],
                                              [1.76575589, 1.83405972, 1.85046351],
                                              [1.8971535 , 1.98924279, 1.9072274 ],
                                              [1.86143994, 2.05605388, 1.83113885]]])
        np.testing.assert_almost_equal(coordinates, coordinates_to_compare)

        # Test forces
        forces = system.ref_forces
        forces_to_compare = np.asarray([[[267.32297443,   122.00786739,  -191.12547107],
                                         [-1151.70591569,   273.61948031, -1476.23560797],
                                         [179.51103706, -1302.9103854 ,  1456.27437138],
                                         [-191.52434024,   207.05999987,  -107.5261387 ],
                                         [1526.58254078,    24.26372681,   396.49462983],
                                         [-391.71477366,   427.60240657,  -883.8872929 ],
                                         [-993.67593737,  -642.56024148,  1590.12862914],
                                         [-1566.73343688,   140.16724027, -1492.06135627],
                                         [310.58053356,  -237.84897807,   763.35537676],
                                         [1405.42539371,   298.44662042,   340.91772283],
                                         [1205.79456984,   450.86302534,  -314.97016191],
                                         [-1040.67861363,  -509.59397003,  -924.48436103],
                                         [675.93355237,   206.12506943,  1410.25416854],
                                         [-235.11758427,   542.75813857,  -567.13450863]]])
        np.testing.assert_almost_equal(forces, forces_to_compare)

    def test_convert_system_ref_arrays_to_list(self):
        """
        Test the function used to convert the type of the arrays.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        # Read data into system
        system.read_data("ParaMol/Tests/aniline_10_struct.nc")

        # Read data into system
        system.convert_system_ref_arrays_to_list()

        assert type(system.ref_energies) is list
        assert type(system.ref_coordinates) is list
        assert type(system.ref_forces) is list





