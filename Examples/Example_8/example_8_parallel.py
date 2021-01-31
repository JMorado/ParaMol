import sys
sys.path.insert(0, "/home/treason/PycharmProjects/ParaMol_git_master")

import simtk.unit as unit
import multiprocessing as mp

# ParaMol imports
from ParaMol.System.system import *

# ParaMol Tasks imports
from ParaMol.HMC.hmc_sampler import *
from ParaMol.Utils.settings import *

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
system_names = ["aniline_{}".format(i) for i in range(4)]
systems = []

# Create four identical aniline systems
for name in system_names:
    # Create the OpenMM engine for aniline
    openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='aniline.prmtop', crd_format="AMBER", crd_file='aniline.inpcrd')

    # Create ParaMol System
    systems.append(ParaMolSystem(name=name, engine=openmm_system, n_atoms=14))

# Create ParaMol settings instance
paramol_settings = Settings()

# --------------------------------------------------------- #
#                     Set the QM Engine                     #
# --------------------------------------------------------- #
# Create the ASE calculator
from ase.calculators.dftb import *

calc = Dftb(Hamiltonian_='DFTB',  # line is included by default
            Hamiltonian_MaxSCCIterations=1000,
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_H='s',
            Hamiltonian_MaxAngularMomentum_C='p',
            Hamiltonian_MaxAngularMomentum_N="p",
            Hamiltonian_Dispersion="DftD3 { \n s6=1.000 \n s8=0.5883 \n Damping = BeckeJohnson { \n a1=0.5719 \n a2=3.6017 \n } \n }",
            Hamiltonian_SCC='Yes',
            Hamiltonian_SCCTolerance=1e-8, )

# Alternative, we could set the calculator in the settings
paramol_settings.qm_engine["ase"]["calculator"] = calc

# -------------------------------------------------------------- #
#                 Perform the HMC Parallel Sampling              #
# -------------------------------------------------------------- #
HMC_samplers = [HMCSampler() for n in range(len(systems))]
output = mp.Queue()
processes_pool = []

for system, sampler in zip(systems, HMC_samplers):
    hmc_kwargs = {"settings": paramol_settings,
                  "systems": [system],
                  "n_sweeps": 10000,
                  "n_steps_per_sweep": 100,
                  "temperature_pot_qm": unit.Quantity(300, unit.kelvin),
                  "temperature_pot_mm": unit.Quantity(300, unit.kelvin),
                  "temperature_kin_mm": unit.Quantity(300, unit.kelvin)}

    processes_pool.append(mp.Process(target=sampler.run_task, kwargs=hmc_kwargs))

# Run processes
for sampler, system in zip(processes_pool, systems):
    print("Starting HMC sampler of system {}".format(system.name))
    sampler.start()

# Exit the completed processes
for sampler in processes_pool:
    sampler.join()

# Write final data into file
for system in systems:
    system.write_data()


