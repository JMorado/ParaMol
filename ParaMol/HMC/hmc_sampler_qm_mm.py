import simtk.unit as unit
import random
import numpy as np
import logging
import os

# from ParaMol
from ..Utils.interface import *
from ..Tasks.task import *
from ..QM_engines.ase_wrapper import *
from ..Tasks.parametrization import *
from .integrators_adapted import *

# SimTK modules
from simtk.openmm import *
from simtk.openmm.app import *


class HMCSampler(Task):
    """
    ParaMol HMC task.
    """
    def __init__(self):
        self._n = None
        self._label = None

        # QM chain
        self._last_accepted_qm = None
        self._n_total_qm = None
        self._n_accepted_qm = None

        # MM chain
        self._last_accepted_mm = None
        self._n_total_mm = None
        self._n_accepted_mm = None
        self._last_accepted_mm_energy = None

        # Data used for the parametrization
        self._param_coordinates = None
        self._param_energies = None
        self._param_forces = None
        self._param_n_structures = None
        self._param_sampling_freq_qm_id = None
        self._param_sampling_freq_mm_id = None

        # Parametrization variables
        self._parametrize = None
        self._old_param = None
        self._new_param = None

    def __str__(self):
        return "Hybrid Monte Carlo sampler - " + str(self._label)

    def __repr__(self):
        return "Hybrid Monte Carlo sampler - " + str(self._label)

    # ------------------------------------------------------------ #
    #                                                              #
    #                       PUBLIC  METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_task(self, settings, systems, n_sweeps, n_steps_per_sweep=100, verbose_freq=10,
                 temperature_pot_qm=unit.Quantity(300, unit.kelvin), temperature_pot_mm=unit.Quantity(300, unit.kelvin),
                 temperature_kin_mm=unit.Quantity(300, unit.kelvin), label="0", optimize_mm=False, ase_calculator_low_level=None, calculate_forces=False, checkpoint_freq=100,
                 parametrization=False, sampling_mm=True, sampling_freq_mm=10, sampling_freq_qm=50,
                 parametrization_freq=100, restart=False, seed=np.random.randint(2**32-1), system_mm_solute=None):
        """
        Method that runs a HMC sampler.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        n_sweeps : int
            Number of MC sweeps to perform.
        n_steps_per_sweep : int
            Number of MD steps per MC sweep.
        verbose_freq : int
            Verbose frequency.
        temperature_pot_qm : unit.Quantity
            Temperature used for the QM potential part.
        temperature_pot_mm : unit.Quantity
            Temperature used for the MM potential part.
        temperature_kin_mm : unit.Quantity
            Temperature used for the MM kinetic part.
        label : int
            HMC sampler label. It has to be an integer number.
        optimize_mm : bool
            Flag that controls whether an MM optimization is performed before the HMC run. Only used if ase_calculator_low_level is None.
        ase_calculator_low_level : ase.calculator.*.*
            ASE calculator used for the low level chain.
        calculate_forces : bool
            Flag that signals whether or not to calculate forces when a structure is accepted. Only relevant for parametrization purposes (does not affect the HMC in itself).
        checkpoint_freq : int
            Frequency at which checkpoint restart files are written.
        parametrization : bool
            Flag that controls whether parametrization is to be performed.
        sampling_mm : bool
            Flag that controls whether or not to sample rejected MM structures.
        sampling_freq_mm : int
            Frequency at which MM structures rejected in the QM ensemble are sampled.
        sampling_freq_qm : int
            Frequency at which structures accepted in sample
        parametrization_freq : int
            Number of structures that has to be collected before performing parametrization.
        restart : bool
            Flag that controls whether or not to perform a restart.
        seed : int
            Numpy random seed.

        Returns
        -------
        systems : listg
            List with the updated instances of ParaMol System.
        """
        assert len(systems) == 1, "HMC task currently only supports one system at once."

        # Create QM Engines and initiate OpenMM
        for system in systems:
            system.convert_system_ref_arrays_to_list()

            # Reset any previously created OpenMM Context\
            system.engine.context = None

            # HMC has to use a sympletic integrator such VelocityVerletIntegrator
            system.engine.integrator = VelocityVerletIntegratorAdapted(system.engine._integrator_params["stepSize"])
            system.engine.init_openmm(create_system_params=system.engine._create_system_params)

            system.engine.get_masses()

            if system_mm_solute.interface is None:
                system_mm_solute.interface = ParaMolInterface()

            if system.interface is None:
                system.interface = ParaMolInterface()

            system_mm_solute.create_qm_engines(settings.qm_engine["qm_engine"], settings.qm_engine[settings.qm_engine["qm_engine"].lower()])

            if ase_calculator_low_level is not None:
                from ase import units as ase_units

                mm_ase_engine = ASEWrapper(system_name=system.name,
                                           interface=system.interface,
                                           calculator=ase_calculator_low_level,
                                           n_atoms=system.n_atoms,
                                           atom_list=system.engine.atom_list,
                                           n_calculations=1,
                                           cell=None,
                                           work_dir_prefix="HMC_LL_ASEWorkDir_",
                                           opt_log_file="ase_md.log")

                temperature_kin_mm_units = unit.Quantity(temperature_kin_mm / ase_units.kB, unit.kelvin)

            else:
                mm_ase_engine = None
                temperature_kin_mm_units = temperature_kin_mm

        system = systems[0]

        self._label = label

        parameter_space, objective_function, optimizer = (None, None, None)
        # TODO: Once this is included in the main ParaMol version, add this line to the default dictionary
        settings.restart["restart_hmc_file_{}".format(self._label)] = "restart_hmc_{}.pickle".format(self._label)

        if restart:
            logging.info("Starting HMC sampler parametrization from a previous restart.")

            # Read HMCSampler pickle
            self.__dict__ = self.read_restart_pickle(settings.restart, system.interface, "restart_hmc_file_{}".format(self._label))

            # Read data into system
            system.read_data(os.path.join(settings.restart["restart_dir"], "{}_hmc_{}.nc".format(system.name, self._label)))

            # Convert np.arrays to list
            system.convert_system_ref_arrays_to_list()

            # Calculate last accepted mm energy if not available
            self._last_accepted_mm_energy = None
            if self._last_accepted_mm_energy is None:
                if mm_ase_engine is None:
                    potential_final_mm = unit.Quantity(system.engine.get_potential_energy(system.ref_coordinates[-1]), unit.kilojoules_per_mole)
                else:
                    coord_to_run = np.asarray(system.ref_coordinates[-1]) * 10
                    potential_final_mm = mm_ase_engine.run_calculation(coords=coord_to_run, label=int(self._label))

                self._last_accepted_mm_energy = potential_final_mm
        else:
            self._n = 1
            self._n_total_qm = 0
            self._n_accepted_qm = 0

            # MM chain
            self._n_total_mm = 0
            self._n_accepted_mm = 0

            if parametrization:
                # Data used for the parametrization
                self._param_coordinates = []
                self._param_energies = []
                self._param_forces = []
                self._param_n_structures = 0
                self._param_sampling_freq_qm_id = 0
                self._param_sampling_freq_mm_id = 0
                self._parametrize = True
            else:
                self._parametrize = False

            if optimize_mm:
                logging.info("Performing MM optimization...")
                system.engine.minimize_system(tolerance=0.1, max_iter=0)

        # Set numpy seed
        np.random.seed(seed)
        mask_atoms = 14

        while self._n <= n_sweeps:
            if len(system.ref_coordinates) > 0 and self._n != 1:
                # Do not need to compute the QM energy if there are structures in the top ensemble or if we are not in the first ever iteration.
                system.engine.set_positions(system.ref_coordinates[-1])
                potential_initial_qm = system.ref_energies[-1]
                potential_initial_qm = unit.Quantity(potential_initial_qm, unit.kilojoules_per_mole)
                potential_initial_mm = self._last_accepted_mm_energy

                mm_solute_initial = unit.Quantity(system_mm_solute.engine.get_potential_energy(system.ref_coordinates[-1][:mask_atoms]), unit.kilojoules_per_mole)


                if mm_ase_engine is not None:
                    coord_to_run = np.asarray(system.ref_coordinates[-1]) * 10
            else:
                # Compute MM initial kinetic and potential energy
                coord_to_run = system.engine.get_positions().in_units_of(unit.angstrom)._value

                if mm_ase_engine is None:
                    potential_initial_mm = unit.Quantity(system.engine.get_potential_energy(system.engine.get_positions()), unit.kilojoules_per_mole)
                else:
                    potential_initial_mm = mm_ase_engine.run_calculation(coords=system.engine.get_positions().in_units_of(unit.angstrom)._value, label=int(self._label))

                potential_initial_qm, _ = system_mm_solute.qm_engine.qm_engine.run_calculation(coords=coord_to_run[:mask_atoms], label=int(self._label))
                potential_initial_qm = unit.Quantity(potential_initial_qm, unit.kilojoules_per_mole)

                mm_solute_initial = unit.Quantity(system_mm_solute.engine.get_potential_energy(coord_to_run[:mask_atoms]*0.1), unit.kilojoules_per_mole)

            if mm_ase_engine is None:
                # ---------------------------------------------------------------- #
                #                             OpenMM HMC                           #
                # ---------------------------------------------------------------- #
                # Use OpenMM as the low level
                # Sample new velocities and calculate kinetic energy
                new_velocities = system.engine.generate_maxwell_boltzmann_velocities(temperature_kin_mm)
                system.engine.set_velocities(new_velocities)
                system.engine.context.applyVelocityConstraints(1e-8)

                # Calculate kinetic energy
                kinetic_initial = unit.Quantity(system.engine.get_kinetic_energy(system.engine.get_velocities()), unit.kilojoules_per_mole)

                # Run classical MD simulation
                system.engine.integrator.step(n_steps_per_sweep)

                # Compute MM final kinetic and potential energy
                kinetic_final = unit.Quantity(system.engine.get_kinetic_energy(system.engine.get_velocities()), unit.kilojoules_per_mole)
                potential_final_mm = unit.Quantity(system.engine.get_potential_energy(system.engine.get_positions()), unit.kilojoules_per_mole)
                coords = system.engine.get_positions()
            else:
                # ---------------------------------------------------------------- #
                #                             ASE HMC                              #
                # ---------------------------------------------------------------- #
                # Use ASE as the low level
                # Run short MD using ASE
                coords, potential_initial_mm, kinetic_initial, forces_initial, potential_final_mm, kinetic_final, forces_final = mm_ase_engine.run_md(coords=coord_to_run,
                                                                                                                                                      label=int(self._label),
                                                                                                                                                      steps=n_steps_per_sweep,
                                                                                                                                                      dt=1.0 * ase_units.fs,
                                                                                                                                                      initial_temperature=temperature_kin_mm, )

                # Compute MM final kinetic and potential energy
                kinetic_initial = unit.Quantity(kinetic_initial, unit.kilojoules_per_mole)
                potential_initial_mm = unit.Quantity(potential_initial_mm, unit.kilojoules_per_mole)
                kinetic_final = unit.Quantity(kinetic_final, unit.kilojoules_per_mole)
                potential_final_mm = unit.Quantity(potential_final_mm, unit.kilojoules_per_mole)
                coords = unit.Quantity(coords, unit.nanometers)

            # ---------------------------------------------------------------- #
            #                       Low Level->High Level                      #
            # ---------------------------------------------------------------- #
            if self._hmc_acceptance_criterion_mm(potential_final_mm, potential_initial_mm, kinetic_final, kinetic_initial, temperature_pot_mm, temperature_kin_mm_units):
                potential_final_qm, forces_final_qm = system_mm_solute.qm_engine.qm_engine.run_calculation(coords=coords[:mask_atoms].in_units_of(unit.angstrom)._value, label=int(self._label))
                potential_final_qm = unit.Quantity(potential_final_qm, unit.kilojoules_per_mole)

                mm_solute_final = unit.Quantity(system_mm_solute.engine.get_potential_energy(coords[:mask_atoms]), unit.kilojoules_per_mole)


                potential_final_qm_mm = potential_final_mm - mm_solute_final + potential_final_qm
                potential_initial_qm_mm = potential_initial_mm - mm_solute_initial + potential_initial_qm

                #potential_initial_mm = mm_solute_initial
                #potential_final_mm = mm_solute_final

                # Nested Markov chain acceptance criterion
                qm_accepted = self._hmc_acceptance_criterion_qm(potential_final_qm_mm, potential_initial_qm_mm, potential_final_mm, potential_initial_mm, temperature_pot_qm, temperature_pot_mm)

                if qm_accepted:
                    self._last_accepted_mm_energy = potential_final_mm

                    # Append energies, forces and conformations
                    system.ref_energies.append(potential_final_qm._value)
                    system.ref_coordinates.append(system.engine.get_positions()._value)

                    if calculate_forces:
                        system.ref_forces.append(forces_final_qm)

                    system.n_structures += 1
                elif len(system.ref_coordinates) > 0:
                    # Append last accepted structure
                    system.ref_energies.append(system.ref_energies[-1])
                    system.ref_coordinates.append(system.ref_coordinates[-1])

                    if calculate_forces:
                        system.ref_forces.append(system.ref_forces[-1])

                    system.n_structures += 1
                else:
                    # No structures have been accepted yet.
                    pass

                if self._parametrize:
                    # TODO: include code related to partial momentum refreshment
                    if sampling_freq_qm == self._param_sampling_freq_qm_id and len(system.ref_coordinates) > 0:
                        self._param_coordinates.append(system.ref_coordinates[-1])
                        self._param_energies.append(system.ref_energies[-1])
                        if calculate_forces:
                            self._param_forces.append(system.ref_forces[-1])
                        self._param_n_structures += 1

                        # Reset sampling counter
                        self._param_sampling_freq_qm_id = 1
                    else:
                        self._param_sampling_freq_qm_id += 1

            # Write restart files
            if self._n % checkpoint_freq == 0:
                self.write_restart_pickle(settings.restart, system.interface, "restart_hmc_file_{}".format(self._label), self.__dict__)
                system.write_data(os.path.join(settings.restart["restart_dir"], "{}_hmc_{}.nc".format(system.name, self._label)))

                system_mm_solute.ref_coordinates = np.asarray(system.ref_coordinates)[:,:mask_atoms,:]
                system.write_coordinates_xyz("{}_hmc_{}.xyz".format(system_mm_solute.name, self._label))

            if self._parametrize and (self._param_n_structures % parametrization_freq == 0 and self._param_n_structures > 0):
                system, parameter_space, objective_function, optimizer = self._run_parametrization(settings, system, parameter_space, objective_function, optimizer, calculate_forces)

            if self._n % verbose_freq == 0:
                self._print_output(system.name)

            self._n += 1

        return systems

    def run_task_ase_one_layer(self, settings, systems, n_sweeps, n_steps_per_sweep=100, verbose_freq=1, temperature_pot_mm=unit.Quantity(300, unit.kelvin),
                               temperature_kin_mm=unit.Quantity(300, unit.kelvin), label="0", checkpoint_freq=100, restart=False, ):
        """
        Method that runs a HMC sampler.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        systems : list of :obj:`ParaMol.System.system.ParaMolSystem`
            List containing instances of ParaMol systems.
        n_sweeps : int
            Number of MC sweeps to perform.
        n_steps_per_sweep : int
            Number of MD steps per MC sweep.
        verbose_freq : int
            Verbose frequency.
        temperature_pot_mm : unit.Quantity
            Temperature used for the MM potential part.
        temperature_kin_mm : unit.Quantity
            Temperature used for the MM kinetic part.
        label : int
            HMC sampler label. It has to be an integer number.
        checkpoint_freq : int
            Frequency at which checkpoint restart files are written.
        restart : bool
            Flag that controls whether or not to perform a restart.

        Returns
        -------
        systems : list
            List with the updated instances of ParaMol System.
        """
        from ase_interface import ANIENS
        from ase_interface import aniensloader
        from ase import units as ase_unit

        assert len(systems) == 1, "HMC task currently only supports one system at once."

        # Create QM Engines and initiate OpenMM
        for system in systems:
            system.convert_system_ref_arrays_to_list()

            # Create OpenMM system
            system.engine.init_openmm(create_system_params=system.engine._create_system_params)
            system.engine.get_masses()

            # Create QM Engine
            if system.interface is None:
                system.interface = ParaMolInterface()

            system.create_qm_engines(settings.qm_engine["qm_engine"], settings.qm_engine[settings.qm_engine["qm_engine"].lower()])

            # Create ASE NN calculator
            # Get atom list and atomic numbers list
            system.engine.get_atom_list()

            """
            mm_ase_engine = ASEWrapper(system_name=system.name,
                                       interface=system.interface,
                                       calculator=ANIENS(aniensloader('../ani_models/ani-1ccx_8x.info', 0)),
                                       n_atoms=system.n_atoms,
                                       atom_list=system.engine.atom_list,
                                       n_calculations=1,
                                       cell=None,
                                       work_dir_prefix="NN_ASEWorkDir_")
            """
            calc = ANIENS(aniensloader('/home/joao/programs/ASE_ANI/ani_models/ani-2x_8x.info',0))
            #calc = DFTD3(dft=calc, cutoff=np.sqrt(9000) * ase_units.Bohr, damping="bj", a1=0.5719, a2=3.6017, s8=0.5883, s6=1.000, alpha6=1.0)

            mm_ase_engine = ASEWrapper(system_name=system.name,
                                       interface=system.interface,
                                       calculator=calc,
                                       n_atoms=system.n_atoms,
                                       atom_list=system.engine.atom_list,
                                       n_calculations=1,
                                       cell=None,
                                       work_dir_prefix="NN_ASEWorkDir_")

        system = systems[0]

        self._label = label

        parameter_space, objective_function, optimizer = (None, None, None)
        # TODO: Once this is included in the main ParaMol version, add this line to the default dictionary
        settings.restart["restart_hmc_file_{}".format(self._label)] = "restart_hmc_{}.pickle".format(self._label)

        if restart:
            logging.info("Starting HMC sampler parametrization from a previous restart.")

            # Read HMCSampler pickle
            self.__dict__ = self.read_restart_pickle(settings.restart, system.interface, "restart_hmc_file_{}".format(self._label))

            # Read data into system
            system.read_data(os.path.join(settings.restart["restart_dir"], "{}_hmc_{}.nc".format(system.name, self._label)))
        else:
            self._n = 1

            # MM chain
            self._n_total_mm = 0
            self._n_accepted_mm = 0

        while self._n <= n_sweeps:
            if self._n % verbose_freq == 0:
                print("HMC sampler of system {} # Sweep number {}.".format(system.name, self._n))
                print("HMC sampler of system {} # Acceptance rate of MM chain {:.4f}".format(system.name, self._acceptance_rate_mm()))
            if len(system.ref_coordinates) > 0 and self._n != 1:
                # Do not need to compute the QM energy if there are structures in the top ensemble or if we are not in the first ever iteration.
                system.engine.set_positions(system.ref_coordinates[-1])
                potential_initial_mm = mm_ase_engine.run_calculation(coords=system.ref_coordinates[-1] * 10, label=int(self._label))
                coord_to_run = system.ref_coordinates[-1] * 10
            else:
                # Compute MM initial kinetic and potential energy
                potential_initial_mm = mm_ase_engine.run_calculation(coords=system.engine.get_positions().in_units_of(unit.angstrom)._value, label=int(self._label))
                coord_to_run = system.engine.get_positions().in_units_of(unit.angstrom)._value

            # Run short MD using ASE
            coords, potential_initial_mm, kinetic_initial, forces_initial, potential_final_mm, kinetic_final, forces_final = mm_ase_engine.run_md(coords=coord_to_run,
                                                                                                                                                  label=int(self._label),
                                                                                                                                                  steps=n_steps_per_sweep,
                                                                                                                                                  dt=0.5*ase_unit.fs,
                                                                                                                                                  initial_temperature=300.0*ase_unit.kB,)
            # Compute MM final kinetic and potential energy
            kinetic_initial = unit.Quantity(kinetic_initial, unit.kilojoules_per_mole)
            potential_initial_mm = unit.Quantity(potential_initial_mm, unit.kilojoules_per_mole)
            kinetic_final = unit.Quantity(kinetic_final, unit.kilojoules_per_mole)
            potential_final_mm = unit.Quantity(potential_final_mm, unit.kilojoules_per_mole)
            coords = unit.Quantity(coords, unit.nanometers)

            if self._hmc_acceptance_criterion_mm(potential_final_mm, potential_initial_mm, kinetic_final, kinetic_initial, temperature_pot_mm, temperature_kin_mm):
                # Append energies, forces and conformations
                system.ref_energies.append(potential_final_mm._value)
                system.ref_coordinates.append(coords._value)
                system.n_structures += 1

            # TODO: include code related to partial momentum refreshment
            elif len(system.ref_coordinates) > 0:
                    # Append last accepted structure
                system.ref_energies.append(system.ref_energies[-1])
                system.ref_coordinates.append(system.ref_coordinates[-1])
                system.n_structures += 1
            else:
                # No structures have been accepted yet.
                pass

            # Write restart files
            if self._n % checkpoint_freq == 0:
                self.write_restart_pickle(settings.restart, system.interface, "restart_hmc_file_{}".format(self._label), self.__dict__)
                system.write_data(os.path.join(settings.restart["restart_dir"], "{}_hmc_{}.nc".format(system.name, self._label)))
                system.write_coordinates_xyz("{}_hmc_{}.xyz".format(system.name, self._label))

                self._n += 1

        return systems

    # ------------------------------------------------------------ #
    #                                                              #
    #                       PRIVATE METHODS                        #
    #                                                              #
    # ------------------------------------------------------------ #
    def _run_parametrization(self, settings, system, parameter_space, objective_function, optimizer, calculate_forces):
        """
        Method that performs parametrization.

        Parameters
        ----------
        settings : dict
            Dictionary containing global ParaMol settings.
        system : :obj:`ParaMol.System.system.ParaMolSystem`
            Instance of ParaMol system.
        parameter_space : :obj:`ParaMol.Parameter_space.parameter_space.ParameterSpace`
            Instances of ParameterSpace.
        objective_function : :obj:`ParaMol.Objective_function.objective_function.ObjectiveFunction`
            Instance of the objective function.
        optimizer : one of the optimizers defined in the subpackage :obj:`ParaMol.Optimizers`
            Instance of the optimizer.
        calculate_forces : bool
            Flag that signals whether or not to calculate forces when a structure is accepted. Only relevant for parametrization purposes (does not affect the HMC in itself).

        Returns
        -------
        system, parameter_space, objective_function, optimizer
        """
        print("Performing parametrization at HMC sweep {} using {} structures.".format(self._n, len(self._param_energies)))
        self._old_param = self._new_param

        # Create parametrization arrays
        tmp_ref_energies = copy.deepcopy(system.ref_energies)
        tmp_ref_coordinates = copy.deepcopy(system.ref_coordinates)

        if calculate_forces:
            tmp_ref_forces = copy.deepcopy(system.ref_forces)

        system.ref_coordinates = np.asarray(self._param_coordinates)
        if calculate_forces:
            system.ref_forces = np.asarray(self._param_forces)

        system.ref_energies = np.asarray(self._param_energies)
        system.n_structures = len(system.ref_coordinates)

        # Perform parametrization
        parametrization = Parametrization()
        _, parameter_space, objective_function, optimizer = parametrization.run_task(settings=settings,
                                                                                     systems=[system],
                                                                                     parameter_space=parameter_space,
                                                                                     objective_function=objective_function,
                                                                                     optimizer=optimizer,
                                                                                     adaptive_parametrization=False,
                                                                                     restart=False)

        # Restore reference arrays
        system.ref_coordinates = tmp_ref_coordinates
        system.ref_energies = tmp_ref_energies

        if calculate_forces:
            system.ref_forces = tmp_ref_forces

        system.n_structures = len(system.ref_coordinates)

        self._new_param = copy.deepcopy(parameter_space.optimizable_parameters_values_scaled)

        self._param_n_structures = 0

        if self._old_param is not None:
            rmsd = self._get_parameters_rmsd(self._old_param, self._new_param)
            print("\n \n \n RMSD is {} \n \n \n".format(rmsd))
            if rmsd < 1e-4:
                print("RMSD. Will not parametrize anymore...")
                self._parametrize = False

        return system, parameter_space, objective_function, optimizer

    def _print_output(self, system_name):
        print("HMC sampler of system {} # Sweep number {}.".format(system_name, self._n))
        print("HMC sampler of system {} # Acceptance rate of MM chain {:.4f}".format(system_name, self._acceptance_rate_mm()))
        print("HMC sampler of system {} # Acceptance rate of QM chain {:.4f}".format(system_name, self._acceptance_rate_qm()))

    @staticmethod
    def _get_parameters_rmsd(old_params, new_params):
        """
        Method that computes the RMSD between the old and new set of parameters.
        Parameters
        ----------
        old_params: list
            List of the old parameters.
        new_params: list
            List of the new parameters
        Returns
        -------
        rmsd: float
            Value of the RMSD.
        """
        old_params = np.asarray(old_params)
        new_params = np.asarray(new_params)

        rmsd = np.power((new_params - old_params),2)
        rmsd = np.sum(rmsd) / float(len(old_params))
        rmsd = np.sqrt(rmsd)

        return rmsd

    # ------------------------------------------------------------ #
    #                                                              #
    #                           MM CHAIN                           #
    #                                                              #
    # ------------------------------------------------------------ #
    def _hmc_acceptance_criterion_mm(self, pot_final, pot_initial, kin_final, kin_initial, temperature_pot, temperature_kin):
        """
        Method that accepts or rejects a configuration generated by a short MD simulation.

        Parameters
        ----------
        pot_final : unit.Quantity
            Final potential energy.
        pot_initial : unit.Quantity
            Initial potential energy.
        kin_final : unit.Quantity
            Final kinetic energy.
        kin_initial: unit.Quantity
            Initial kinetic energy.
        temperature_pot : unit.Quantity
            Temperature used for the potential part.
        temperature_kin : unit.Quantity
            Temperature used for the kinetic part.

        Returns
        -------
        self._last_accepted_mm : bool
            Whether or not this trial move was accepted.
        """

        delta_kinetic = kin_final - kin_initial
        delta_potential = pot_final - pot_initial
        delta_e = delta_potential + delta_kinetic

        self._n_total_mm += 1
        self._last_accepted_mm = False

        if delta_e._value < 0:
            self._last_accepted_mm = True
            self._n_accepted_mm += 1
        else:
            exp_factor = delta_potential / (unit.BOLTZMANN_CONSTANT_kB * temperature_pot * unit.AVOGADRO_CONSTANT_NA) \
                         + delta_kinetic / (unit.BOLTZMANN_CONSTANT_kB * temperature_kin * unit.AVOGADRO_CONSTANT_NA)

            if min(1.0, np.exp(- exp_factor)) > random.uniform(0, 1):
                self._last_accepted_mm = True
                self._n_accepted_mm += 1

        return self._last_accepted_mm

    def _acceptance_rate_mm(self):
        if self._n_total_mm != 0:
            return float(self._n_accepted_mm) / float(self._n_total_mm)
        else:
            # "No classical Hybrid MC trials were performed yet."
            return 0.0

    def _reset_acceptance_rate_mm(self):
        self._n_total_mm = 0.0
        self._n_accepted_mm = 0.0

    # ------------------------------------------------------------ #
    #                                                              #
    #                           QM CHAIN                           #
    #                                                              #
    # ------------------------------------------------------------ #
    def _hmc_acceptance_criterion_qm(self, pot_final_qm, pot_initial_qm, pot_final_mm, pot_initial_mm, temperature_pot_qm, temperature_pot_mm):
        """
        Method that accepts or rejects a configuration generated into the QM Markov chain.

        Parameters
        ----------
        pot_final_qm : unit.Quantity
            Final QM potential energy.
        pot_initial_qm : unit.Quantity
            Initial QM potential energy.
        pot_final_mm : unit.Quantity
            Final MM potential energy.
        pot_initial_mm : unit.Quantity
            Initial MM potential energy.
        temperature_pot_qm : unit.Quantity
            Temperature used for the QM potential part.
        temperature_pot_mm : unit.Quantity
            Temperature used for the MM potential part.

        Returns
        -------
        self._last_accepted_qm : bool
            Whether or not this trial move was accepted.
        """
        delta_qm = pot_final_qm - pot_initial_qm
        delta_mm = pot_final_mm - pot_initial_mm
        delta_e = delta_qm - delta_mm

        self._n_total_qm += 1
        self._last_accepted_qm = False
        print("QM", delta_qm, delta_mm, delta_e)
        if delta_e._value < 0:
            self._last_accepted_qm = True
            self._n_accepted_qm += 1
        else:
            exp_factor = delta_qm / (unit.BOLTZMANN_CONSTANT_kB * temperature_pot_qm * unit.AVOGADRO_CONSTANT_NA) \
                         - delta_mm / (unit.BOLTZMANN_CONSTANT_kB * temperature_pot_mm * unit.AVOGADRO_CONSTANT_NA)

            if min(1.0, np.exp(- exp_factor)) > random.uniform(0, 1):
                self._last_accepted_qm = True
                self._n_accepted_qm += 1

        return self._last_accepted_qm

    def _acceptance_rate_qm(self):
        if self._n_total_qm != 0:
            return float(self._n_accepted_qm) / float(self._n_total_qm)
        else:
            # No HMC trials were performed to accept structures into the QM chain.
            return 0.0

    def _reset_acceptance_rate_qm(self):
        self._n_total_qm = 0.0
        self._n_accepted_qm = 0.0
