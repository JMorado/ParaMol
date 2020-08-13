# ParaMol imports
from ParaMol.System.system import *
from ParaMol.MM_engines.openmm import *
from ParaMol.QM_engines.qm_engine import *

# ParaMol Tasks imports
from ParaMol.Tasks.parametrization import *
from ParaMol.Tasks.torsions_scan import *
from ParaMol.Utils.settings import *

# --------------------------------------------------------- #
#                         Preparation                       #
# --------------------------------------------------------- #
# Create the OpenMM engine for norfloxacin
openmm_system = OpenMMEngine(init_openmm=True, topology_format='AMBER', top_file='norfloxacin.prmtop', crd_file='norfloxacin.inpcrd')

# Create ParaMol System
norfloxacin = ParaMolSystem(name="norfloxacin", engine=openmm_system, n_atoms=41)

# Create ParaMol's force field representation
norfloxacin.force_field.create_force_field()

# Create ParaMol settings instance
paramol_settings = Settings()

# The objective function will contain a energy, force and regularization term
paramol_settings.properties["include_energies"] = True
paramol_settings.properties["include_forces"] = False # One should not include forces when a torsional scan
paramol_settings.properties["include_regularization"] = True

# --------------------------------------------------------- #
#                Symmetrize ParaMol ForceField              #
# --------------------------------------------------------- #
# Symmetry ParaMol ForceField so that it respects atom-type symmetries
amber_symmetrizer = AmberSymmetrizer(prmtop_file="norfloxacin.prmtop")
amber_symmetrizer.get_amber_symmetries()
amber_symmetrizer.set_force_field_to_amber_format(norfloxacin.force_field)

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
#                  Perform the Torsional Scan               #
# --------------------------------------------------------- #
torsion_to_scan = [[7, 4, 11, 15]]
scan_settings = [[-180.0, 180.0, 90.0]]
torsion_scan = TorsionScan()
torsion_scan.run_task(paramol_settings, [norfloxacin], torsion_to_scan, scan_settings, optimize_qm_before_scan=True)

# --------------------------------------------------------- #
#                   Parametrize the Torsion                 #
# --------------------------------------------------------- #
norfloxacin.force_field.optimize_torsions_by_symmetry(torsion_to_scan)
norfloxacin.force_field.write_ff_file("norfloxacin_symm.ff")

parametrization = Parametrization()
systems, parameter_space, objective_function, optimizer = parametrization.run_task(paramol_settings, [norfloxacin])

# Update AMBER symmetrizer with new parameters
amber_symmetrizer.update_term_types_parameters(parameter_space.optimizable_parameters)

# Write AMBER topology file (.prmtop) and .frcmod file
amber_symmetrizer.save_prmtop("norfloxacin_opt.prmtop")
amber_symmetrizer.save_frcmod("norfloxacin_opt.frcmod")