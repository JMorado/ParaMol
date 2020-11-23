# Import ParaMol modules
from ParaMol.System.system import *
from ParaMol.Force_field import *

import numpy as np


class TestForceField:
    # Kwargs dictionary for AMBER topology system. These are shared between all instances.
    kwargs_dict = {"topology_format": "AMBER",
                   "top_file": "ParaMol/Tests/aniline.prmtop",
                   "crd_file": "ParaMol/Tests/aniline.inpcrd"}

    def test_create_force_field(self):
        """
        Test the creation of the ParaMol force field.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        assert system.force_field is ForceField
        system.force_field.create_force_field(opt_bonds=True, opt_angles=True, opt_torsions=True, opt_charges=True, opt_lj=True, opt_sc=True, ff_file=None)

        # Assert force groups
        force_groups = system.force_field.force_groups
        force_groups_to_compare = {'HarmonicBondForce': 0, 'HarmonicAngleForce': 1, 'PeriodicTorsionForce': 2, 'NonbondedForce': 3, 'CMMotionRemover': 4, 'Scaling14': 3}
        assert force_groups == force_groups_to_compare

        # Assert force field
        assert len(system.force_field.force_field['HarmonicBondForce']) == 14
        assert len(system.force_field.force_field['HarmonicAngleForce']) == 21
        assert len(system.force_field.force_field['PeriodicTorsionForce']) == 35
        assert len(system.force_field.force_field['NonbondedForce']) == 14
        assert len(system.force_field.force_field['Scaling14']) == 25

        # Assert force field optimizable
        assert len(system.force_field.force_field_optimizable['HarmonicBondForce']) == 14
        assert len(system.force_field.force_field_optimizable['HarmonicAngleForce']) == 21
        assert len(system.force_field.force_field_optimizable['PeriodicTorsionForce']) == 35
        assert len(system.force_field.force_field_optimizable['NonbondedForce']) == 14
        assert len(system.force_field.force_field_optimizable['Scaling14']) == 25

    def test_create_force_field(self):
        """
        Test the function that gets the optimizable parameter values.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        # Create ParaMol force field
        system.force_field.create_force_field(opt_bonds=True, opt_angles=True, opt_torsions=True, opt_charges=True, opt_lj=True, opt_sc=True, ff_file=None)

        # Get optimizable parameters
        system.force_field.get_optimizable_parameters()

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

        optimizable_parameters_values = np.asarray(system.force_field.optimizable_parameters_values)
        np.testing.assert_almost_equal(optimizable_parameters_values_to_compare, optimizable_parameters_values)

