# ParaMol imports
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *
from ParaMol.QM_engines.qm_engine import *

# ParaMol Tasks imports
from ParaMol.Tasks.parametrization import *
from ParaMol.Tasks.torsions_scan import *
from ParaMol.Utils.settings import *
from ParaMol.Utils.conformational_sampling import *

# --------------------------------------------------------- #
#                          Settings                         #
# --------------------------------------------------------- #
# Create ParaMol settings instance
paramol_settings = Settings()

# The objective function will contain an energy and a regularization term
paramol_settings.properties["include_energies"] = True
paramol_settings.properties["include_forces"] = False
paramol_settings.properties["include_regularization"] = True

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create the OpenMM engine for ethane
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='ethane.prmtop', crd_file='ethane.inpcrd')

# Create ethane ParaMol System
ethane = ParaMolSystem(name="ethane", engine=openmm_system, n_atoms=8)

# Create ParaMol's force field representation
ethane.force_field.create_force_field(ff_file="ethane_sym.ff")

# Create the OpenMM engine for propane
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='propane.prmtop', crd_file='propane.inpcrd')

# Create propane ParaMol System
propane = ParaMolSystem(name="propane", engine=openmm_system, n_atoms=11)

# Create ParaMol's force field representation
propane.force_field.create_force_field(ff_file="propane_sym.ff")

# --------------------------------------------------------- #
#                     Set the QM Engine                     #
# --------------------------------------------------------- #
# Create the ASE calculator
from ase.calculators.dftb import *

calc = Dftb(Hamiltonian_='DFTB',  # line is included by default
			Hamiltonian_MaxAngularMomentum_='',
			Hamiltonian_MaxAngularMomentum_H='s',
			Hamiltonian_MaxAngularMomentum_O='p',
			Hamiltonian_MaxAngularMomentum_C='p',
			Hamiltonian_MaxAngularMomentum_N="p",
			Hamiltonian_Dispersion="DftD3 { \n s6=1.000 \n s8=0.5883 \n Damping = BeckeJohnson { \n a1=0.5719 \n a2=3.6017 \n } \n }",
			Hamiltonian_SCC='Yes',
			Hamiltonian_SCCTolerance=1e-8, )

# Alternative, we could set the calculator in the settings
paramol_settings.qm_engine["ase"]["calculator"] = calc

# --------------------------------------------------------- #
#                   Conformational Sampling                 #
# --------------------------------------------------------- #
systems = [ethane, propane]

# Perform conformational sampling
conformational_sampling(paramol_settings, systems, 1000, 1000)

# --------------------------------------------------------- #
#                   Parameter Optimization                  #
# --------------------------------------------------------- #
parametrization = Parametrization()
systems, parameter_space, objective_function, optimizer = parametrization.run_task(paramol_settings, systems)

# Write final ParaMol FFs
ethane.force_field.write_ff_file("ethane_optimized.ff")
propane.force_field.write_ff_file("propane_optimized.ff")