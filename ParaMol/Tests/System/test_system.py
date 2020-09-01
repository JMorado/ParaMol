# Import ParaMol modules
from ParaMol.System.system import *


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






