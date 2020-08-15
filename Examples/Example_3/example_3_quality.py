import matplotlib.pyplot as plt
import numpy as np

# ParaMol imports
from ParaMol.Utils.settings import *
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *
from ParaMol.QM_engines.qm_engine import *
from ParaMol.Tasks.adaptive_parametrization import *
from ParaMol.Utils.conformational_sampling import conformational_sampling

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create OpenMM system engine
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

# Make a copy of the original force_field dict.
# It will be useful later to assess the quality of the re-parametrized force field.
force_field_original = copy.deepcopy(caffeine.force_field.force_field)
# --------------------------------------------------------- #
#                  Adaptive Parametrization                 #
# --------------------------------------------------------- #
adaptive_parametrization = AdaptiveParametrization()
adaptive_parametrization.run_task(paramol_settings, [caffeine], rmsd_tol=0.0001, max_iter=100, structures_per_iter=100, )

# Write ParaMol Force Field file with final parameters
caffeine.force_field.write_ff_file("caffeine_adaptive_param.ff")

# --------------------------------------------------------- #
#               Quality of the Parametrization              #
# --------------------------------------------------------- #
# Perform conformational sampling
conformational_sampling([caffeine], n_conf=1000, steps_integrator=1000)

# Calculate MM energies and forces after the re-parametrization
mm_energies_after = caffeine.get_energies_ensemble()
mm_forces_after = caffeine.get_forces_ensemble()

# Re-set original parameters
caffeine.engine.set_nonbonded_parameters(force_field_original)
caffeine.engine.set_bonded_parameters(force_field_original)

# Calculate MM energies and forces before the re-parametrization
mm_energies_before = caffeine.get_energies_ensemble()
mm_forces_before = caffeine.get_forces_ensemble()

# Plot the data
plt.title("Quality of the energies")
plt.scatter(mm_energies_before-np.mean(mm_energies_before), caffeine.ref_energies-np.mean(caffeine.ref_energies), s=1, label="Before re-parametrization", color="red")
plt.scatter(mm_energies_after-np.mean(mm_energies_after), caffeine.ref_energies-np.mean(caffeine.ref_energies), s=1, label="After re-parametrization", color="green")
plt.plot(np.arange(-200,200), np.arange(-200,200), color="black")
plt.ylabel("QM Energy (kJ/mol)")
plt.xlabel("MM Energy (kJ/mol)")
plt.legend()
plt.show()

# Quality of the forces
atom_idx = 10
direction = 0 #x=0, y=1, z=2
plt.title("Quality of the atom {} force along direction {}".format(atom_idx, direction))
plt.scatter(mm_forces_before[:,atom_idx,0], caffeine.ref_forces[:, atom_idx, 0], s=1, color="red", label="Before re-parametrization")
plt.scatter(mm_forces_after[:,atom_idx,0], caffeine.ref_forces[:, atom_idx, 0], s=1, color="green", label="After re-parametrization")
plt.plot(np.arange(-5000,5000), np.arange(-5000,5000), color="black")
plt.ylabel("QM Force (kJ/mol/nm)")
plt.xlabel("MM Force (kJ/mol/nm)")
plt.legend()
plt.show()