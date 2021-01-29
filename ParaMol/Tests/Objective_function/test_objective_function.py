# Import ParaMol modules
from ParaMol.System.system import *
from ParaMol.Objective_function.objective_function import *
from ParaMol.Objective_function.Properties.regularization import *
from ParaMol.Objective_function.Properties.esp_property import *
from ParaMol.Objective_function.Properties.force_property import *
from ParaMol.Objective_function.Properties.energy_property import *
from ParaMol.Parameter_space.parameter_space import *

import numpy as np
import simtk.unit as unit


class TestObjectiveFunction:
    # Kwargs dictionary for AMBER topology system. These are shared between all instances.
    kwargs_dict = {"topology_format": "AMBER",
                   "crd_format": "AMBER",
                   "top_file": "ParaMol/Tests/aniline.prmtop",
                   "crd_file": "ParaMol/Tests/aniline.inpcrd"}

    objective_function_settings = {"parallel": False,
                                   "platform_name": "Reference",
                                   "weighting_method": "uniform",
                                   "weighting_temperature": 300.0 * unit.kelvin,
                                   "checkpoint_freq": 100}

    def test_objective_function_and_properties(self):
        """
        Test the calculation of the objective function and creation of properties.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        assert type(system.force_field) is ForceField
        system.force_field.create_force_field(opt_bonds=True, opt_angles=True, opt_torsions=True, opt_charges=True, opt_lj=True, opt_sc=True, ff_file=None)

        # Read data into system
        system.read_data("ParaMol/Tests/aniline_10_struct.nc")

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
        np.testing.assert_almost_equal(optimizable_parameters_values_to_compare, optimizable_parameters_values, decimal=4)

        # Energy property
        energies_dict = {"weight": 1.0}
        prop_energy = EnergyProperty([system], **energies_dict)
        assert type(prop_energy) is EnergyProperty

        prop_energy.calculate_variance()

        objective_function = ObjectiveFunction(restart_settings=None,
                                               parameter_space=parameter_space,
                                               properties=[prop_energy],
                                               systems=[system],
                                               **self.objective_function_settings)
        f_val = objective_function.f(optimizable_parameters_values, opt_mode=False)

        assert abs(f_val - 0.16587870225911971) < 1e-8

        # Forces components property
        forces_dict = {"term_type": "components", "weight": 1.0}
        prop_forces_components = ForceProperty([system], **forces_dict)
        assert type(prop_forces_components) is ForceProperty
        prop_forces_components.calculate_variance()

        objective_function = ObjectiveFunction(restart_settings=None,
                                               parameter_space=parameter_space,
                                               properties=[prop_forces_components],
                                               systems=[system],
                                               **self.objective_function_settings)

        f_val = objective_function.f(optimizable_parameters_values, opt_mode=False)

        assert abs(f_val - 0.3022524295062905) < 1e-8

        # Forces norm property

        forces_dict = {"term_type": "norm", "weight": 1.0}
        prop_forces_norm = ForceProperty([system], **forces_dict)
        assert type(prop_forces_norm) is ForceProperty

        prop_forces_norm.calculate_variance()

        objective_function = ObjectiveFunction(restart_settings=None,
                                               parameter_space=parameter_space,
                                               properties=[prop_forces_norm],
                                               systems=[system],
                                               **self.objective_function_settings)

        f_val = objective_function.f(optimizable_parameters_values, opt_mode=False)
        assert abs(f_val - 0.498237598004599) < 1e-8

        # Regularization property
        reg_dict = {"method": "L2", "weight": 1.0, "scaling_factor": 1.0}
        parameter_space.calculate_prior_widths("arithmetic")

        prop_reg = Regularization(initial_parameters_values=parameter_space.optimizable_parameters_values,
                                  prior_widths=parameter_space.prior_widths,
                                  **reg_dict)

        assert type(prop_reg) is Regularization

        objective_function = ObjectiveFunction(restart_settings=None,
                                               parameter_space=parameter_space,
                                               properties=[prop_reg],
                                               systems=[system],
                                               **self.objective_function_settings)

        f_val = objective_function.f(optimizable_parameters_values, opt_mode=False)

        assert abs(f_val) < 1e-8









