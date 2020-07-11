# ParaMol imports
from ParaMol.Utils.settings import *
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *
from ParaMol.QM_engines.qm_engine import *
from ParaMol.Tasks.parametrization import *
from ParaMol.Utils.amber_symmetrizer import *

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create the OpenMM engine for aspirin
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='aspirin.prmtop', crd_file='aspirin.inpcrd')

# Create ParaMol System
aspirin = ParaMolSystem(name="aspirin", engine=openmm_system, n_atoms=21)

# Create ParaMol's force field representation and ask to parametrize bonds, angles and torsions
aspirin.force_field.create_force_field(opt_bonds=True, opt_angles=True, opt_torsions=True)

# Create ParaMol settings instance
paramol_settings = Settings()

# The objective function will contain a energy, force and regularization term
paramol_settings.properties["include_energies"] = True
paramol_settings.properties["include_forces"] = True
paramol_settings.properties["include_regularization"] = True

# Read conformations, energies and forces from NETCDF ParaMol file format
aspirin.read_data("aspirin_reference_data.nc")
# --------------------------------------------------------- #
#                Symmetrize ParaMol ForceField              #
# --------------------------------------------------------- #
# Symmetry ParaMol ForceField so that it respects atom-type symmetries
amber_symmetrizer = AmberSymmetrizer(prmtop_file="aspirin.prmtop")
amber_symmetrizer.get_amber_symmetries()
amber_symmetrizer.set_force_field_to_amber_format(aspirin.force_field)

# Write symmetrized Force-Field to file
aspirin.force_field.write_ff_file("aspirin_sym.ff")
# --------------------------------------------------------- #
#                       Parametrization                     #
# --------------------------------------------------------- #
parametrization = Parametrization()
systems, parameter_space, objective_function, optimizer = parametrization.run_task(paramol_settings, [aspirin])

# Write ParaMol Force Field file with final parameters
aspirin.force_field.write_ff_file("aspirin_symm_opt.ff")

# Update AMBER symmetrizer with new parameters
amber_symmetrizer.update_term_types_parameters(parameter_space.optimizable_parameters)

# Write AMBER topology file (.prmtop)
amber_symmetrizer.save_prmtop("aspirin_opt.prmtop")
amber_symmetrizer.save_frcmod("aspirin_opt.frcmod")