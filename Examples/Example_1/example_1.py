import numpy as np

# ParaMol imports
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *

# ParaMol Tasks imports
from ParaMol.Tasks.parametrization import *
from ParaMol.Tasks.ab_initio_properties import *
from ParaMol.Utils.settings import *

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create the OpenMM engine for carbon monoxide
openmm_engine = OpenMMEngine(True, "AMBER", "co.prmtop", "co.inpcrd")

# Create the ParaMol System
co = ParaMolSystem("carbon_monoxide", openmm_engine, 2)

# Create ParaMol's force field representation and ask to optimize bonds's parameters
co.force_field.create_force_field(opt_bonds=True)

# Create ParaMol settings instance
paramol_settings = Settings()

# --------------------------------------------------------- #
#       Perform the conformational sampling manually        #
# --------------------------------------------------------- #
# Generate conformations; ParaMol uses nanometers for the length
n_atoms = 2
n_conformations = 100
conformations = np.zeros((n_conformations, n_atoms, 3))

# Change the z distance of atom 2
conformations[:, 1, 2] = np.linspace(0.1, 0.12, n_conformations)

# Set this data in the ParaMol system instance
co.ref_coordinates = conformations
co.n_structures = len(co.ref_coordinates)

# --------------------------------------------------------- #
#              Calculate QM energies and forces             #
# --------------------------------------------------------- #
# Create the ASE calculator
from ase.calculators.dftb import *

calc = Dftb(Hamiltonian_='DFTB',
			Hamiltonian_MaxAngularMomentum_='',
			Hamiltonian_MaxAngularMomentum_O='p',
			Hamiltonian_MaxAngularMomentum_C='p',
			Hamiltonian_SCC='Yes',
			Hamiltonian_SCCTolerance=1e-8,
			Hamiltonian_MaxSCCIterations=10000)

# Set the calculator in the settings; alternatively the QM engine could be created manually
paramol_settings.qm_engine["ase"]["calculator"] = calc

# Calculate Ab initio properties
ab_initio = AbInitioProperties()
ab_initio.run_task(paramol_settings, [co])

# Save coordinates, energies and forces into .nc file
co.write_data("co_scan.nc")

# --------------------------------------------------------- #
#                   Parametrize the CO bond                 #
# --------------------------------------------------------- #
parametrization = Parametrization()
optimal_paramters = parametrization.run_task(paramol_settings, [co])
