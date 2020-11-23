# Import ParaMol modules
from ParaMol.MM_engines.openmm import *


class TestOpenMMEngine:
    # Kwargs dictionary for AMBER topology system. These are shared between all instances.
    kwargs_dict = {"topology_format": "AMBER",
                   "top_file": "ParaMol/Tests/aniline.prmtop",
                   "crd_file": "ParaMol/Tests/aniline.inpcrd"}

    def test_create_OpenMMEngine_amber(self):
        """
        Tests the initialization of the OpenMMWrapper by creating systems from AMBER topology format.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

    def test_create_OpenMMEngine_xml(self):
        """
        Tests the initialization of the OpenMMWrapper by creating systems from XML topology format.
        """
        self.kwargs_dict["topology_format"] = "XML"
        self.kwargs_dict["xml_file"] = "ParaMol/Tests/aniline.xml"

        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

    def _test_OpenMMEngine_get_forces(self):
        """
        Test the calculation of MM forces.
        """
        # Test create OpenMM engine
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)
        assert type(openmm_engine) is OpenMMEngine

        # Test get_forces
        positions = openmm_engine.context.getState(getPositions=True).getPositions()
        forces = openmm_engine.get_forces(positions)
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

        diff = forces-forces_to_compare
        for atom in diff:
            for component in atom:
                assert abs(component) < 1e-4

    def _test_OpenMMEngine_get_potential_energy(self):
        """
        Test the calculation of MM energies.
        """
        # Test create OpenMM engine
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)
        assert type(openmm_engine) is OpenMMEngine

        # Test get_potential_energy
        positions = openmm_engine.context.getState(getPositions=True).getPositions()
        pot_energy = openmm_engine.get_potential_energy(positions)
        energy_to_comapare = -94.4336492887833
        assert abs(pot_energy - energy_to_comapare) < 1e-4

    def test_OpenMMEngine_get_atom_list(self):
        """
        Test get_atom_list.
        """
        # Test create OpenMM engine
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)
        assert type(openmm_engine) is OpenMMEngine

        atom_list_to_compare = ['H', 'C', 'C', 'H', 'C', 'H', 'C', 'N', 'H', 'H', 'C', 'H', 'C', 'H']
        atom_list = openmm_engine.get_atom_list()

        assert atom_list == atom_list_to_compare

    def test_OpenMMEngine_get_atomic_numbers(self):
        """
        Test get_atomic_numbers
        """
        # Test create OpenMM engine
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)
        assert type(openmm_engine) is OpenMMEngine

        atomic_numbers_to_compare = [1, 6, 6, 1, 6, 1, 6, 7, 1, 1, 6, 1, 6, 1]
        atomic_numbers = openmm_engine.get_atomic_numbers()

        assert atomic_numbers == atomic_numbers_to_compare

    def test_OpenMMEngine_get_cell(self):
        """
        Test get_cell.
        """
        # Test create OpenMM engine
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)
        assert type(openmm_engine) is OpenMMEngine

        cell_to_compare = [[39.883, 0.0, 0.0],
                           [0.0, 36.135, 0.0],
                           [0.0, 0.0, 39.848]]

        cell_to_compare = np.asarray(cell_to_compare)
        cell = openmm_engine.get_cell()

        diff = cell-cell_to_compare
        for dir in diff:
            for component in dir:
                assert abs(component) < 1e-4

    def test_OpenMMEngine_add_torsion_terms(self):
        """
        Test add_torsion_terms.
        """
        # Test create OpenMM engine
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)
        assert type(openmm_engine) is OpenMMEngine

        openmm_engine.add_torsion_terms()

        force = openmm_engine.system.getForce(openmm_engine.force_groups_dict["PeriodicTorsionForce"])

        assert force.getNumTorsions() == 140






