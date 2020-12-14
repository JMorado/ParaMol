# Import ParaMol modules
from ..System.system import *
from ..Tasks.torsions_scan import *
from ..Tasks.torsions_parametrization import *

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

    def test_get_rotatable_bonds(self):
        """
        Test get rotatable bonds method.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        torsion_scan = TorsionScan()

        assert type(torsion_scan) is TorsionScan

        mol, conf = torsion_scan.get_rdkit_mol_conf(system)

        assert type(mol) is rdkit.Chem.rdchem.Mol
        assert type(conf) is rdkit.Chem.rdchem.Conformer

        torsion_param = TorsionsParametrization()

        rot_bonds = torsion_param.get_rotatable_bonds(mol)

        assert rot_bonds == ((6, 7),)
        assert len(rot_bonds) == 1

    def test_get_rotatable_torsions(self):
        """
        Test get rotatable torsions.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        system.force_field.create_force_field()
        system.force_field.create_force_field_optimizable()

        torsion_scan = TorsionScan()

        assert type(torsion_scan) is TorsionScan

        mol, conf = torsion_scan.get_rdkit_mol_conf(system)

        assert type(mol) is rdkit.Chem.rdchem.Mol
        assert type(conf) is rdkit.Chem.rdchem.Conformer

        torsion_param = TorsionsParametrization()

        rot_bonds = torsion_param.get_rotatable_bonds(mol)

        assert rot_bonds == ((6, 7),)
        assert len(rot_bonds) == 1

        rot_dihedrals = torsion_param.get_rotatable_torsions(system, rot_bonds)

        assert len(rot_dihedrals) == 1

    def test_dihedral_rotation(self):
        """
        Test get rotatable torsions.
        """
        openmm_engine = OpenMMEngine(True, **self.kwargs_dict)

        assert type(openmm_engine) is OpenMMEngine

        system = ParaMolSystem(name="aniline", engine=openmm_engine, n_atoms=14)

        system.force_field.create_force_field()
        system.force_field.create_force_field_optimizable()

        torsion_scan = TorsionScan()

        assert type(torsion_scan) is TorsionScan

        mol, conf = torsion_scan.get_rdkit_mol_conf(system)

        assert type(mol) is rdkit.Chem.rdchem.Mol
        assert type(conf) is rdkit.Chem.rdchem.Conformer

        torsion_param = TorsionsParametrization()

        rot_bonds = torsion_param.get_rotatable_bonds(mol)

        assert rot_bonds == ((6, 7),)
        assert len(rot_bonds) == 1

        rot_dihedrals = torsion_param.get_rotatable_torsions(system, rot_bonds)

        assert len(rot_dihedrals) == 1
