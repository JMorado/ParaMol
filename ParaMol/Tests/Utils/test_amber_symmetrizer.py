# Import ParaMol modules
from ParaMol.System.system import *
from ParaMol.Tasks.torsions_scan import *
from ParaMol.Tasks.torsions_parametrization import *

import rdkit
import numpy as np


class TestSystem:
    # Kwargs dictionary for AMBER topology system. These are shared between all instances.
    kwargs_dict = {"topology_format": "AMBER",
                   "top_file": "ParaMol/Tests/aniline.prmtop",
                   "crd_file": "ParaMol/Tests/aniline.inpcrd"}

    def test_get_rdkit_mol_conf(self):
        """
        Test get_rdkit_mol_conf
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        torsion_scan = TorsionScan()
        mol, conf = torsion_scan.get_rdkit_mol_conf(system)

        assert type(mol) is rdkit.Chem.rdchem.Mol
        assert type(conf) is rdkit.Chem.rdchem.Conformer
