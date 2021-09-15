import simtk.unit as unit

# ParaMol imports
from ParaMol.System.system import *

# ParaMol Tasks imports
from ParaMol.HMC.hmc_sampler import *
from ParaMol.Utils.settings import *

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create the OpenMM engine for aniline
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='aniline.prmtop', crd_format='AMBER', crd_file='aniline.inpcrd')

# Create ParaMol System
aniline = ParaMolSystem(name="aniline", engine=openmm_system, n_atoms=14)

# Create ParaMol's force field representation of aniline; let us optimize all bonded terms
aniline.force_field.create_force_field(opt_angles=True, opt_bonds=True, opt_torsions=True, opt_lj=False, opt_charges=False)

# Create ParaMol settings instance
paramol_settings = Settings()

# The objective function will contain an energy and a regularization term
paramol_settings.properties["include_energies"] = True
paramol_settings.properties["include_forces"] = True
paramol_settings.properties["include_regularization"] = True
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

# --------------------------------------------------------- #
#                  Perform the HMC Sampling                 #
# --------------------------------------------------------- #
hmc_sampler = HMCSampler()

hmc_sampler.run_task(paramol_settings, [aniline], n_sweeps=10000, n_steps_per_sweep=100,
                     temperature_pot_qm=unit.Quantity(300, unit.kelvin),
                     temperature_pot_mm=unit.Quantity(300, unit.kelvin),
                     temperature_kin_mm=unit.Quantity(300, unit.kelvin),
                     parametrization=True,
                     parametrization_freq=1000)


# Write ParaMol Force Field file with final parameters
aniline.force_field.write_ff_file("aniline_hmc.ff")

# Write final data
aniline.write_data("aniline_hmc.nc")
