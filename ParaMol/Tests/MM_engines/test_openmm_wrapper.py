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

    def test_OpenMMEngine_get_forces(self):
        """
        Test the calculation of MM forces.
        """
        # Test create OpenMM engine
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)
        assert type(openmm_engine) is OpenMMEngine

        # Test get_forces
        positions = openmm_engine.context.getState(getPositions=True).getPositions()
        forces = openmm_engine.get_forces(positions)

        forces_to_compare = np.asarray([[  63.35630106,  162.8967182 , -258.24033288],
                                        [ -59.00944446, -208.77385948,  355.58821171],
                                        [-776.22546605, -504.12792941,  166.35451611],
                                        [ 308.2616859 ,  165.84950054,   -6.51344357],
                                        [ 298.18912045,  411.24031931, -721.17666783],
                                        [ 173.36177528,  -18.60772402,  335.44688256],
                                        [-154.47362429,  159.35849737,  161.19346389],
                                        [ 391.65340945, -360.28848608, -570.75865292],
                                        [ 103.39422437,  135.17881342,  490.66165364],
                                        [-458.56499782,  -23.61221643,  210.89896885],
                                        [ 166.3213486 ,  370.13451591, -766.39734703],
                                        [-314.8255014 , -170.20584891,  100.11854383],
                                        [ 478.93689136, -122.96590207,  736.30926713],
                                        [-220.37572244,    3.92360164, -233.4850635 ]])

        np.testing.assert_almost_equal(forces, forces_to_compare)

    def test_OpenMMEngine_get_potential_energy(self):
        """
        Test the calculation of MM energies.
        """
        # Test create OpenMM engine
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)
        assert type(openmm_engine) is OpenMMEngine

        # Test get_potential_energy
        positions = openmm_engine.context.getState(getPositions=True).getPositions()
        pot_energy = openmm_engine.get_potential_energy(positions)
        energy_to_compare = -109.4067856056198
        assert abs(pot_energy - energy_to_compare) < 1e-4

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






