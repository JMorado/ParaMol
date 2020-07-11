import sys
import unittest
import numpy as np
import simtk.unit as unit

# Append src folder to PATH
src_dir = "../"
sys.path.append(src_dir)

# Import ParaMol modules
from openmm.openmm import *


class TestOpenMMWrapper(unittest.TestCase):
    def test_openmm_amber(self):
        """
        Tests the initialization of the OpenMMWrapper by creating systems from AMBER topology format,
        XML file.
        Also tests the initialization by instantiating the wrapper from previously created system, integrator, platform
        and context instances.
        :return:
        """

        # ------------------------------------------------------------------------------------------------- #
        #                                              AMBER                                                #
        # ------------------------------------------------------------------------------------------------- #
        # Kwargs dictionary for AMBER topology system
        kwargs_dict = {"verbose": True,
                       "topology_format": "AMBER",
                       "topology_file": None,
                       "prmtop_file": "aniline.prmtop",
                       "inpcrd_file": "aniline.inpcrd",
                       "platform_name": "CPU",
                       "temperature": unit.Quantity(300.0, unit.kelvin),
                       "dt": unit.Quantity(0.001, unit.picoseconds)}

        self._openmm_instance = OpenMMWrapper(True, **kwargs_dict)

        self.assertIsInstance(self._openmm_instance, OpenMMWrapper)

        # Test get_potential_energy
        positions = self._openmm_instance.context.getState(getPositions=True).getPositions()
        pot_energy = self._openmm_instance.get_potential_energy(positions)
        self.assertAlmostEqual(pot_energy, -94.4336492887833, 1e-8)

        # Test get_forces
        forces = self._openmm_instance.get_forces(positions)
        forces_to_compare = np.asarray([[-108.20637122, 14.14975219, -203.01049775],
                                        [285.23819821, -44.26763491, 622.12727234],
                                        [-370.49318137, -165.53371961, 862.01433178],
                                        [126.58599999, 33.44856972, -197.69047989],
                                        [-581.70982215, -67.64550267, -522.34967916],
                                        [315.1541032, 101.17451063, -65.19818332],
                                        [-984.3807304, 822.37994851, -1481.98729613],
                                        [1353.7231787, -1122.81542249, 1921.23187906],
                                        [-602.7141258, 145.84354165, 27.17423844],
                                        [228.1849988, 273.6632624, -501.45893709],
                                        [-219.24294377, -22.30555889, -788.64576361],
                                        [-224.02270718, 34.6457255, 272.69745263],
                                        [1041.98078492, 11.56852192, 38.56452241],
                                        [-260.09738479, -14.30599467, 16.53114005]])

        np.testing.assert_array_equal(forces,forces_to_compare)



        # ------------------------------------------------------------------------------------------------- #
        #                                            GROMACS                                                #
        # ------------------------------------------------------------------------------------------------- #
        pass

        # ------------------------------------------------------------------------------------------------- #
        #                                             OPLS                                                  #
        # ------------------------------------------------------------------------------------------------- #
        pass

        # ------------------------------------------------------------------------------------------------- #
        #                                               XML                                                 #
        # ------------------------------------------------------------------------------------------------- #
        # Kwargs dictionary for XML system
        kwargs_dict = {"verbose": True,
                       "topology_format": "XML",
                       "topology_file": "aniline.xml",
                       "prmtop_file": "aniline.prmtop",
                       "inpcrd_file": "aniline.inpcrd",
                       "platform_name": "CPU",
                       "temperature": unit.Quantity(300.0, unit.kelvin),
                       "dt": unit.Quantity(0.001, unit.picoseconds)}

        self._openmm_instance_xml = OpenMMWrapper(True,
                                                  **kwargs_dict)

        # ------------------------------------------------------------------------------------------------- #
        #                                   From OpenMMWrapper Instance                                     #
        # ------------------------------------------------------------------------------------------------- #
        # Kwargs dictionary for OpenMMWrapper instance created from
        kwargs_dict = {"verbose": True,
                       "system": self._openmm_instance_amber.system,
                       'integrator': self._openmm_instance_amber.integrator,
                       'platform': self._openmm_instance_amber.platform,
                       'context': self._openmm_instance_amber.context,
                       'topology': self._openmm_instance_amber.topology}

        self._openmm_instance_create_system_false = OpenMMWrapper(False,
                                                                  **kwargs_dict)

        self.assertIsInstance(self._openmm_instance_create_system_false, OpenMMWrapper)

    def test_get_potential_energy(self):


        """
                print(pot_energy)
                pot_energy = self._openmm_instance_xml.getPotentialEnergy(positions)
                pot_energy = self._openmm_instance_create_system_false.getPotentialEnergy(positions)
        """
    def test_get_forces(self):
        pass

    def test_add_all_torsions(self):
        pass

    def test_set_initial_guess(self):
        pass

    def test_set_bonded_parameters(self):
        pass

    def test_set_non_bonded_parameters_to_zero(self):
        pass

    def test_set_non_bonded_parameters(self):
        pass


if __name__ == '__main__':
    unittest.main()
