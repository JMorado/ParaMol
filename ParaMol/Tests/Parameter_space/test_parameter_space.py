# Import ParaMol modules
from ParaMol.System.system import *
from ParaMol.Force_field.force_field import *
from ParaMol.Parameter_space.parameter_space import *

import numpy as np


class TestParameterSpace:
    # Kwargs dictionary for AMBER topology system. These are shared between all instances.
    kwargs_dict = {"topology_format": "AMBER",
                   "top_file": "ParaMol/Tests/aniline.prmtop",
                   "crd_file": "ParaMol/Tests/aniline.inpcrd"}

    def test_get_optimizable_parameters(self):
        """
        Test the function that obtains the optimizable parameters.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        assert type(system.force_field) is ForceField
        system.force_field.create_force_field(opt_bonds=True, opt_angles=True, opt_torsions=True, opt_charges=True, opt_lj=True, opt_sc=True, ff_file=None)

        parameter_space = ParameterSpace()

        assert type(parameter_space) is ParameterSpace

        optimizable_parameters, optimizable_parameters_values = parameter_space.get_optimizable_parameters([system])

        assert len(optimizable_parameters_values) == 232

        # Assert force groups
        optimizable_parameters_values_to_compare = np.asarray([0.10860000000000002, 289365.44000000006, 0.10860000000000002, 289365.44000000006, 0.10860000000000002, 289365.44000000006, 0.10120000000000001,
                                                               338569.28, 0.10120000000000001, 338569.28, 0.10860000000000002, 289365.44000000006, 0.10860000000000002, 289365.44000000006, 0.1398,
                                                               385848.48000000004, 0.1398, 385848.48000000004, 0.1398, 385848.48000000004, 0.1398, 385848.48000000004, 0.1386, 349698.72000000003, 0.1398,
                                                               385848.48000000004, 0.1398, 385848.48000000004, 2.0923016, 403.33760000000007, 2.0923016, 403.33760000000007, 2.0923016, 403.33760000000007,
                                                               2.0923016, 403.33760000000007, 2.0923016, 403.33760000000007, 2.0923016, 403.33760000000007, 2.0923016, 403.33760000000007, 2.02580453,
                                                               405.01120000000003, 2.02580453, 405.01120000000003, 2.0923016, 403.33760000000007, 2.0092239, 335.5568, 2.0923016, 403.33760000000007,
                                                               2.0923016, 403.33760000000007, 2.09474507, 557.3088, 2.09474507, 557.3088, 2.09474507, 557.3088, 2.09474507, 557.3088, 2.11097664, 571.5344,
                                                               2.09474507, 557.3088, 2.09474507, 557.3088, 2.11097664, 571.5344, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167,
                                                               3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 4.3932, 3.141594,
                                                               4.3932, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 4.3932, 3.141594, 4.3932, 3.141594,
                                                               15.167, 3.141594, 4.6024, 3.141594, 4.6024, 3.141594, 4.6024, 3.141594, 4.6024, 3.141594, 4.6024, 3.141594, 4.6024, 3.141594, 15.167, 3.141594,
                                                               15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 15.167, 3.141594, 4.6024, 0.131,
                                                               0.2599642458735085, 0.06276000026869928, -0.173, 0.33996695079448314, 0.35982400053705343, -0.093, 0.33996695079448314, 0.35982400053705343,
                                                               0.129, 0.2599642458735085, 0.06276000026869928, -0.191, 0.33996695079448314, 0.35982400053705343, 0.12999999999999998, 0.2599642458735085,
                                                               0.06276000026869928, 0.13659999999999997, 0.33996695079448314, 0.35982400053705343, -0.8182000021951126, 0.3249998524031036, 0.7112799996555186,
                                                               0.3868, 0.10690784617205229, 0.06568880001765333, 0.3868, 0.10690784617205229, 0.06568880001765333, -0.191, 0.33996695079448314, 0.35982400053705343,
                                                               0.12999999999999998, 0.2599642458735085, 0.06276000026869928, -0.093, 0.33996695079448314, 0.35982400053705343, 0.129, 0.2599642458735085,
                                                               0.06276000026869928, 0.8333333333333334, 0.5, 0.8333333333333333, 0.5, 0.8333333333333333, 0.5, 0.8333333333333334, 0.5, 0.8333333333333333, 0.5,
                                                               0.8333333333333333, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5,
                                                               0.8333333333333334, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5,
                                                               0.8333333333333334, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5, 0.8333333333333335, 0.5, 0.8333333333333334, 0.5, 0.8333333333333334, 0.5,
                                                               0.8333333333333334, 0.5, 0.8333333333333334, 0.5])

        optimizable_parameters_values = np.asarray(optimizable_parameters_values)
        np.testing.assert_almost_equal(optimizable_parameters_values_to_compare, optimizable_parameters_values)

    def test_calculate_parameters_magnitudes(self):
        """
        Test the function that calculates the magnitudes of the parameters.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        assert type(system.force_field) is ForceField
        system.force_field.create_force_field(opt_bonds=True, opt_angles=True, opt_torsions=True, opt_charges=True, opt_lj=True, opt_sc=True, ff_file=None)

        parameter_space = ParameterSpace()

        assert type(parameter_space) is ParameterSpace

        _, _ = parameter_space.get_optimizable_parameters([system])

        # Geometric
        parameters_magnitudes_dict, prior_widths = parameter_space.calculate_parameters_magnitudes(method="geometric")
        dict_geometric = {'bond_eq': 0.05,
                          'bond_k': 339330.6478320703,
                          'angle_eq': 2.08431266115259,
                          'angle_k': 453.49623302587844,
                          'torsion_phase': 3.141594,
                          'torsion_k': 10.370921339254062,
                          'charge': 0.5,
                          'lj_sigma': 0.3,
                          'lj_eps': 0.2,
                          'scee': 0.8333333333333334,
                          'scnb': 0.5}

        for param_type in dict_geometric.keys():
            assert abs((dict_geometric[param_type]-parameters_magnitudes_dict[param_type])) < 1e-8
        # Arithmetic
        parameters_magnitudes_dict, prior_widths = parameter_space.calculate_parameters_magnitudes(method="arithmetic")
        dict_arithmetic = {'bond_eq': 0.12305714285714285,
                           'bond_k': 342053.95428571437,
                           'angle_eq': 2.084489174285714,
                           'angle_k': 460.2798476190477,
                           'torsion_phase': 3.1415939999999987,
                           'torsion_k': 11.822788571428562,
                           'charge': 0.22274285729965088,
                           'lj_sigma': 0.27703131992011787,
                           'lj_eps': 0.23681440030404585,
                           'scee': 0.8333333333333333,
                           'scnb': 0.5}

        for param_type in dict_arithmetic.keys():
            assert abs((dict_arithmetic[param_type]-parameters_magnitudes_dict[param_type])) < 1e-8

    def test_jacobi_preconditioning(self):
        """
        Test the jacobi preconditioning function.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        assert type(system.force_field) is ForceField

        system.force_field.create_force_field(opt_bonds=True, opt_angles=True, opt_torsions=True, opt_charges=True, opt_lj=True, opt_sc=True, ff_file=None)

        parameter_space = ParameterSpace()

        assert type(parameter_space) is ParameterSpace

        _, _ = parameter_space.get_optimizable_parameters([system])

        # Perform jacobi preconditioning
        parameter_space.calculate_scaling_constants("arithmetic")

        assert parameter_space.preconditioned is False

        parameter_space.jacobi_preconditioning()

        assert parameter_space.preconditioned is True

        optimizable_parameters_values_scaled_to_compare = np.asarray([ 0.88251683,  0.84596432,  0.88251683,  0.84596432,  0.88251683,
                                                                       0.84596432,  0.82238217,  0.9898125 ,  0.82238217,  0.9898125 ,
                                                                       0.88251683,  0.84596432,  0.88251683,  0.84596432,  1.13605758,
                                                                       1.12803397,  1.13605758,  1.12803397,  1.13605758,  1.12803397,
                                                                       1.13605758,  1.12803397,  1.12630601,  1.02234959,  1.13605758,
                                                                       1.12803397,  1.13605758,  1.12803397,  1.00374789,  0.87628777,
                                                                       1.00374789,  0.87628777,  1.00374789,  0.87628777,  1.00374789,
                                                                       0.87628777,  1.00374789,  0.87628777,  1.00374789,  0.87628777,
                                                                       1.00374789,  0.87628777,  0.97184699,  0.87992382,  0.97184699,
                                                                       0.87992382,  1.00374789,  0.87628777,  0.9638927 ,  0.72902779,
                                                                       1.00374789,  0.87628777,  1.00374789,  0.87628777,  1.0049201 ,
                                                                       1.21080426,  1.0049201 ,  1.21080426,  1.0049201 ,  1.21080426,
                                                                       1.0049201 ,  1.21080426,  1.01270693,  1.24171067,  1.0049201 ,
                                                                       1.21080426,  1.0049201 ,  1.21080426,  1.01270693,  1.24171067,
                                                                       1.        ,  1.28286148,  1.        ,  1.28286148,  1.        ,
                                                                       1.28286148,  1.        ,  1.28286148,  1.        ,  1.28286148,
                                                                       1.        ,  1.28286148,  1.        ,  1.28286148,  1.        ,
                                                                       1.28286148,  1.        ,  1.28286148,  1.        ,  1.28286148,
                                                                       1.        ,  0.37158746,  1.        ,  0.37158746,  1.        ,
                                                                       1.28286148,  1.        ,  1.28286148,  1.        ,  1.28286148,
                                                                       1.        ,  1.28286148,  1.        ,  1.28286148,  1.        ,
                                                                       0.37158746,  1.        ,  0.37158746,  1.        ,  1.28286148,
                                                                       1.        ,  0.3892821 ,  1.        ,  0.3892821 ,  1.        ,
                                                                       0.3892821 ,  1.        ,  0.3892821 ,  1.        ,  0.3892821 ,
                                                                       1.        ,  0.3892821 ,  1.        ,  1.28286148,  1.        ,
                                                                       1.28286148,  1.        ,  1.28286148,  1.        ,  1.28286148,
                                                                       1.        ,  1.28286148,  1.        ,  1.28286148,  1.        ,
                                                                       1.28286148,  1.        ,  1.28286148,  1.        ,  0.3892821 ,
                                                                       0.58812211,  0.93839298,  0.26501767, -0.77668035,  1.22717876,
                                                                       1.51943463, -0.41752181,  1.22717876,  1.51943463,  0.57914315,
                                                                       0.93839298,  0.26501767, -0.85749102,  1.22717876,  1.51943463,
                                                                       0.58363263,  0.93839298,  0.26501767,  0.61326321,  1.22717876,
                                                                       1.51943463, -3.673294  ,  1.17315202,  3.00353356,  1.73653155,
                                                                       0.38590527,  0.27738516,  1.73653155,  0.38590527,  0.27738516,
                                                                       -0.85749102,  1.22717876,  1.51943463,  0.58363263,  0.93839298,
                                                                       0.26501767, -0.41752181,  1.22717876,  1.51943463,  0.57914315,
                                                                       0.93839298,  0.26501767,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                                                                       1.        ,  1.        ])

        np.testing.assert_almost_equal(optimizable_parameters_values_scaled_to_compare, parameter_space.optimizable_parameters_values_scaled)

    def test_update_systems(self):
        """
        Test the update systems function.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        assert type(system.force_field) is ForceField

        system.force_field.create_force_field(opt_bonds=True, opt_angles=True, opt_torsions=True, opt_charges=True, opt_lj=True, opt_sc=True, ff_file=None)

        parameter_space = ParameterSpace()

        assert type(parameter_space) is ParameterSpace

        _, old_values = parameter_space.get_optimizable_parameters([system])

        parameters = np.ones(len(old_values))
        parameter_space.update_systems([system], parameters)
        _, new_param = parameter_space.get_optimizable_parameters([system])

        np.testing.assert_almost_equal(parameters, new_param)

