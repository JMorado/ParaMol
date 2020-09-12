# ParaMol imports
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create the OpenMM engine for ethane
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='ethane.prmtop', crd_file='ethane.inpcrd')

# Create ethane ParaMol System
ethane = ParaMolSystem(name="ethane", engine=openmm_system, n_atoms=8)

# Create ParaMol's force field representation
ethane.force_field.create_force_field()

# Create the OpenMM engine for propane
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='propane.prmtop', crd_file='propane.inpcrd')

# Create propane ParaMol System
propane = ParaMolSystem(name="propane", engine=openmm_system, n_atoms=11)

# Create ParaMol's force field representation
propane.force_field.create_force_field()

# --------------------------------------------------------- #
#                  Save ParaMol ForceField                  #
# --------------------------------------------------------- #
ethane.force_field.write_ff_file("ethane.ff")
propane.force_field.write_ff_file("propane.ff")
