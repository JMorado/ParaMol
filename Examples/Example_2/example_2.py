# ParaMol imports
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *

# ParaMol Tasks imports
from ParaMol.Tasks.resp_fitting import *
from ParaMol.Utils.settings import *
from ParaMol.Utils.gaussian_esp import *
from ParaMol.Utils.amber_symmetrizer import *

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create the OpenMM engine for caffeine
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='aniline.prmtop', crd_file='aniline.inpcrd')

# Create Molecular System
aniline = ParaMolSystem(name="aniline", engine=openmm_system, n_atoms=14)

# Create ParaMol's force field representation and ask to optimize charges
aniline.force_field.create_force_field(opt_charges=True)

# Create ParaMol settings instance
paramol_settings = Settings()
paramol_settings.properties["include_energies"] = False
paramol_settings.properties["include_forces"] = False
paramol_settings.properties["include_esp"] = True

# --------------------------------------------------------- #
#                 Read ESP Data into ParaMol                #
# --------------------------------------------------------- #
gaussian_esp = GaussianESP()
aniline.ref_coordinates, aniline.ref_esp_grid, aniline.ref_esp = gaussian_esp.read_log_files(["path_to_guassian_log_file"])

# --------------------------------------------------------- #
#                Symmetrize ParaMol ForceField              #
# --------------------------------------------------------- #
# Symmetry ParaMol ForceField so that it respects symmetries
# In this example, we are not setting any symmetry, but we still need to do this step as we want to save a .mol2 file
amber_symmetrizer = AmberSymmetrizer(prmtop_file="aniline.prmtop")
amber_symmetrizer.get_amber_symmetries(aniline.force_field)
amber_symmetrizer.set_force_field_to_amber_format(aniline.force_field)

# Set number of structures
aniline.n_structures = len(aniline.ref_coordinates)

# --------------------------------------------------------- #
#                      RESP Charge Fitting                  #
# --------------------------------------------------------- #
resp_fitting = RESPFitting()
systems, parameter_space, objective_function, optimizer = resp_fitting.run_task(paramol_settings, [aniline], solver="scipy", total_charge=0)
#systems, parameter_space, objective_function, optimizer = resp_fitting.run_task(paramol_settings, [aniline], solver="explicit", total_charge=0)

# Write ParaMol Force Field file with final parameters
aniline.force_field.write_ff_file("aniline_resp.ff")

# Update amber symmetrizer and save .mol2 file
amber_symmetrizer.update_term_types_parameters(parameter_space.optimizable_parameters)
amber_symmetrizer.save_mol2("aniline_resp.mol2")
