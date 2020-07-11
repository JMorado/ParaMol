# ParaMol imports
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *
from ParaMol.Utils.amber_symmetrizer import *

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create the OpenMM engine for norfloxacin
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='norfloxacin.prmtop', crd_file='norfloxacin.inpcrd')

# Create ParaMol System
norfloxacin = ParaMolSystem(name="norfloxacin", engine=openmm_system, n_atoms=41)

# Create ParaMol's force field representation and ask to parametrize bonds, angles and torsions
norfloxacin.force_field.create_force_field()

# --------------------------------------------------------- #
#                Symmetrize ParaMol ForceField              #
# --------------------------------------------------------- #
# Symmetry ParaMol ForceField so that it respects atom-type symmetries
amber_symmetrizer = AmberSymmetrizer(prmtop_file="norfloxacin.prmtop")
amber_symmetrizer.get_amber_symmetries()
amber_symmetrizer.set_force_field_to_amber_format(norfloxacin.force_field)

# Write symmetrized Force-Field to file
norfloxacin.force_field.write_ff_file("norfloxacin_symm.ff")
