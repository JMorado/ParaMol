# ParaMol imports
from ParaMol.Utils.settings import *
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *
from ParaMol.Tasks.adaptive_parametrization import *

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create the OpenMM engine for caffeine
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='caffeine.prmtop', crd_file='caffeine.inpcrd')

# Create ParaMol System
caffeine = ParaMolSystem(name="caffeine", engine=openmm_system, n_atoms=24)

# Create ParaMol's force field representation and ask to parametrize bonds, angles and torsions
caffeine.force_field.create_force_field(opt_bonds=True, opt_angles=True, opt_torsions=True)

# Create ParaMol settings instance
paramol_settings = Settings()

# The objective function will contain a energy, force and regularization term
paramol_settings.properties["include_energies"] = True
paramol_settings.properties["include_forces"] = True
paramol_settings.properties["include_regularization"] = True

# The QM engine used will be ParaMol's AMBER sqm wrapper
paramol_settings.qm_engine["qm_engine"] = "amber"

# --------------------------------------------------------- #
#                  Adaptive Parametrization                 #
# --------------------------------------------------------- #
adaptive_parametrization = AdaptiveParametrization()
adaptive_parametrization.run_task(paramol_settings, [caffeine], rmsd_tol=0.0001, max_iter=100, structures_per_iter=100, )

# Write ParaMol Force Field file with final parameters
caffeine.force_field.write_ff_file("aniline_resp.ff")