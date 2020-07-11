import parmed as pmd
import numpy as np
#from Force_field.force_field import *
import simtk.unit as unit
import copy

class AmberParametrization:
    def __init__(self, prmtop_file):
        self._amber_prmtop = pmd.amber.AmberParm(prmtop_file)

        #
        self._bond_types = None
        self._angle_types = None
        self._torsion_types = None
        self._lj_types = None

    def __str__(self):
        return "AmberParametrization module. Amber .prmtop file in use is {}".format(self._amber_prmtop)

    def get_amber_symmetries(self):
        """
        Implemented term types:
        - bond_types
        - angle_types
        - torsion_types
        - LJ_types
        :return:
        """
        # Types to implement:
        # - improper_types;
        # - cmap_types;
        # - pi_torsion_types;
        # - out_of_plane_bend_types;
        # - torsion_torsion_types;
        # - trigonal_angle_types;
        # - urey_bradley_types;
        # - rb_torsion_types;
        self._bond_types = {}
        for i in range(len(self._amber_prmtop.bond_types)):
            bond_type = self._amber_prmtop.bond_types[i]
            self._bond_types["B{}".format(i)] = {"bond_eq": bond_type.req,
                                                 "bond_k": bond_type.k}

        self._angle_types = {}
        for i in range(len(self._amber_prmtop.angle_types)):
            angle_type = self._amber_prmtop.angle_types[i]
            self._angle_types["A{}".format(i)] = {"angle_eq": angle_type.theteq,
                                                  "angle_k": angle_type.k}

        self._torsion_types = {}
        for i in range(len(self._amber_prmtop.dihedral_types)):
            dihedral_type = self._amber_prmtop.dihedral_types[i]
            self._torsion_types["T{}".format(i)] = {"torsion_periodicity": dihedral_type.per,
                                                    "torsion_phase": dihedral_type.phase,
                                                    "torsion_k": dihedral_type.phi_k}
        self._lj_types = {}
        for atom_type in self._amber_prmtop.LJ_types:
            lj_type = self._amber_prmtop.LJ_types[atom_type] - 1 # LJ type starts at 1
            self._lj_types["{}".format(atom_type)] = {"eps": self._amber_prmtop.LJ_depth[lj_type],
                                                      "sigma": self._amber_prmtop.LJ_radius[lj_type],
                                                      'lj_type_id': lj_type}
        """
        for i in range(len(self._amber_prmtop.LJ_types)):
            print(self._amber_prmtop.LJ_types[i])
            lj_type = self._amber_prmtop.LJ_types[i]
            self._lj_types["{}".format(lj_type)] = {"eps": lj_type.LJ_depth[i],
                                                    "sigma": lj_type.LJ_radius[i],
                                                    'id': lj_type[i]}
        """
        return self._bond_types, self._angle_types, self._torsion_types

    def set_force_field_to_amber_format(self, force_field_instance):
        """
        This method symmetrizes the force field so that it respect AMBER atom-types.
        This is done to only allow optimization of parameters that can be exported to the AMBER topology format.
        :return:
        """
        #assert self._bond_types is not None, "Bond types were not set."
        #assert self._angle_types is not None, "Angle types were not set."
        #assert self._torsion_types is not None, "Torsion types were not set."

        # Set bonds to AMBER format
        # Iterate over terms in harmonic bond force
        for force_field_term in force_field_instance.force_field["HarmonicBondForce"]:
            amber_bonds = [bond for bond in self._amber_prmtop.bonds]
            # Iterate over all AMBER bonds
            for bond_idx in range(len(amber_bonds)):
                bond = amber_bonds[bond_idx]
                if bond.atom1.idx == force_field_term.atoms[0] and bond.atom2.idx == force_field_term.atoms[1]:
                    force_field_term.parameters['bond_eq'].symmetry_group = "B{}".format(bond.type.idx)
                    force_field_term.parameters['bond_k'].symmetry_group = "B{}".format(bond.type.idx)

                    # Pop this bond out and break the loop
                    amber_bonds.pop(bond_idx)
                    break

        # Iterate over terms in harmonic angle force
        for force_field_term in force_field_instance.force_field["HarmonicAngleForce"]:
            amber_angles = [angle for angle in self._amber_prmtop.angles]
            # Iterate over all amber angles
            for angle_idx in range(len(amber_angles)):
                angle = amber_angles[angle_idx]
                if angle.atom1.idx == force_field_term.atoms[0] and angle.atom2.idx == force_field_term.atoms[1] and angle.atom3.idx == force_field_term.atoms[2]:
                    force_field_term.parameters['angle_eq'].symmetry_group = "A{}".format(angle.type.idx)
                    force_field_term.parameters['angle_k'].symmetry_group = "A{}".format(angle.type.idx)

                    # Pop this angle out and break the loop
                    amber_angles.pop(angle_idx)
                    break

        # Iterate over terms in torsions
        for force_field_term in force_field_instance.force_field["PeriodicTorsionForce"]:
            amber_torsions = [torsion for torsion in self._amber_prmtop.dihedrals]
            # Iterate over all amber torsions
            for torsion_idx in range(len(amber_torsions)):
                torsion = amber_torsions[torsion_idx]
                if torsion.atom1.idx == force_field_term.atoms[0] and torsion.atom2.idx == force_field_term.atoms[1] and torsion.atom3.idx == force_field_term.atoms[2] and torsion.atom4.idx == force_field_term.atoms[3]:
                    force_field_term.parameters['torsion_phase'].symmetry_group = "T{}".format(torsion.type.idx)
                    force_field_term.parameters['torsion_periodicity'].symmetry_group = "T{}".format(torsion.type.idx)
                    force_field_term.parameters['torsion_k'].symmetry_group = "T{}".format(torsion.type.idx)

                    # Pop this torsion out and break the loop
                    amber_torsions.pop(torsion_idx)
                    break

        # Iterate over scaling factors
        for force_field_term in force_field_instance.force_field["Scaling14"]:
            amber_sc = [torsion for torsion in self._amber_prmtop.dihedrals]
            # Iterate over all amber torsions
            for torsion_idx in range(len(amber_torsions)):
                torsion = amber_torsions[torsion_idx]
                if torsion.atom1.idx == force_field_term.atoms[0] and torsion.atom4.idx == force_field_term.atoms[1]:
                    force_field_term.parameters['scee'].symmetry_group = "SC{}".format(torsion.type.idx)
                    force_field_term.parameters['scnb'].symmetry_group = "SC{}".format(torsion.type.idx)

                    # Pop this torsion out and break the loop
                    amber_torsions.pop(torsion_idx)
                    break

        return force_field_instance

    def update_term_types_parameters(self):
        """
        This 
        :return:
        """
        # TODO: check conversion factors and what units AMBER prmtop file uses.
        kcal_mol_to_kj_mol = 4.184
        nanometers_to_angstrom = 10.0
        radians_to_degrees = 180.0 / np.pi
        # Check if terms of the same type share the same parameters
        pass
        # If so update the the term types parameters
        pass
        # Remember to update angles in degrees and not in radians
        pass

        # Bond equilibrium value * nanometers_to_angstrom
        # Angle equilibrium value * radians_to_degrees
        # Bond force constant / (kcal_mol_to_kj_mol * nanometers_to_angstrom ** 2 * 2)
        # Angle force const / (kcal_mol_to_kj_mol * 2)  # 2 is because
        # Torsion force const / (kcal_mol_to_kj_mol)
        # Torsion phase * radians_to_degrees
        return
    
    def save_prmtop(self, output_seed):
        """
        Saves the .prmtop AMBER topology file with the current force field parameters of the self._amber_prmtop instance.
        In order to update the self._amber_prmtop instance with the optimal parameters, the method
        update_term_types_parameters should be run before this one.

        :param output_file: 
        :return: 
        """
        return self._amber_prmtop.save(output_seed)

    def save_mol2(self, output_seed):
        return self._amber_prmtop.save(output_seed + ".mol2")

    def save_frcmod(self, output_seed):
        return pmd.tools.writeFrcmod(self._amber_prmtop, output_seed + ".frcmod")





        


    """
    def create_openmm_system(self):
        pass

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
        
        return

    """





'''
a = AmberParametrization(prmtop_file="aniline.prmtop")
a.get_term_types()


x = a._amber_param.dihedrals












a._amber_param.dihedral_types[0].phase = 1.00000000 / np.pi * 180.0
#a._amber_param.remake_parm()
#pmd.from_structure(struct, copy=False)
"""

j = 0
for i in a._amber_param.dihedrals:
    i.type = pmd.topologyobjects.DihedralType(j, 2, 180.0, 1.2, 2.0)
    
    j+=1
"""
a._amber_param.load_atom_info()
a._amber_param.load_structure()
a._amber_param.remake_parm()
t = a._amber_param

#print(a._amber_param.dihedral_types)
print(a._amber_param.dihedrals)
#a._amber_param.save("a.prmtop")
kkk = a._amber_param.atoms
a._amber_param.write_parm("a")
#pmd.from_structure(struct, copy=False)
'''
