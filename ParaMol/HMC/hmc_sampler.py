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

        # Data used for the parametrization
        self._param_coordinates = None
        self._param_energies = None
        self._param_forces = None
        self._param_n_structures = None
        self._param_sampling_freq_qm_id = None
        self._param_sampling_freq_mm_id = None

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
                 temperature_kin_mm=unit.Quantity(300, unit.kelvin), label="0", optimize_mm=False, checkpoint_freq=100, parametrization=False, sampling_mm=True,
                 sampling_freq_mm=10,sampling_freq_qm=50,
                 parametrization_freq=100, restart=False, seed=np.random.randint(2**32-1)):
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
            Flag that controls whether an MM optimization is performed before the HMC run.
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
        systems : list
            List with the updated instances of ParaMol System.
        """
        assert len(systems) == 1, "HMC task currently only supports one system at once."

        # Create QM Engines and initiate OpenMM
        for system in systems:
            system.convert_system_ref_arrays_to_list()

            # Reset any previously created OpenMM Context
            system.engine.context = None

            # HMC uses VelocityVerletIntegrator mandatorily
            system.engine.integrator = VelocityVerletIntegratorAdapted(system.engine._integrator_params["stepSize"])
            system.engine.init_openmm(create_system_params=system.engine._create_system_params)

            system.engine.get_masses()

            if system.interface is None:
                system.interface = ParaMolInterface()

            system.create_qm_engines(settings.qm_engine["qm_engine"], settings.qm_engine[settings.qm_engine["qm_engine"].lower()])

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
            self._n_total_qm = 0
            self._n_accepted_qm = 0

            # MM chain
            self._n_total_mm = 0
            self._n_accepted_mm = 0

            # Data used for the parametrization
            self._param_coordinates = []
            self._param_energies = []
            self._param_forces = []
            self._param_n_structures = 0
            self._param_sampling_freq_qm_id = 0
            self._param_sampling_freq_mm_id = 0

            if optimize_mm:
                logging.info("Performing MM optimization...")
                LocalEnergyMinimizer.minimize(system.engine.context)

        # Set numpy seed
        np.random.seed(seed)

        new_param = None
        while self._n <= n_sweeps:
            if self._n % verbose_freq == 0:
                self._print_output(system.name)

            if len(system.ref_coordinates) > 0 and self._n != 1:
                # Do not need to compute the QM energy if there are structures in the top ensemble or if we are not in the first ever iteration.
                system.engine.set_positions(system.ref_coordinates[-1])
                potential_initial_mm = unit.Quantity(system.engine.get_potential_energy(system.ref_coordinates[-1]), unit.kilojoules_per_mole)
                potential_initial_qm = system.ref_energies[-1]
                potential_initial_qm = unit.Quantity(potential_initial_qm, unit.kilojoules_per_mole)
            else:
                # Compute MM initial kinetic and potential energy
                potential_initial_qm, _ = system.qm_engine.qm_engine.run_calculation(coords=system.engine.get_positions().in_units_of(unit.angstrom)._value, label=int(self._label))
                potential_initial_qm = unit.Quantity(potential_initial_qm, unit.kilojoules_per_mole)
                potential_initial_mm = unit.Quantity(system.engine.get_potential_energy(system.engine.get_positions()), unit.kilojoules_per_mole)

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

            if self._hmc_acceptance_criterion_mm(potential_final_mm, potential_initial_mm, kinetic_final, kinetic_initial, temperature_pot_mm, temperature_kin_mm):
                potential_final_qm, forces_final_qm = system.qm_engine.qm_engine.run_calculation(coords=system.engine.get_positions().in_units_of(unit.angstrom)._value, label=int(self._label))
                potential_final_qm = unit.Quantity(potential_final_qm, unit.kilojoules_per_mole)

                # Nested Markov chain acceptance criterion
                qm_accepted = self._hmc_acceptance_criterion_qm(potential_final_qm, potential_initial_qm, potential_final_mm, potential_initial_mm, temperature_pot_qm, temperature_pot_mm)

                if qm_accepted:
                    # Append energies, forces and conformations
                    system.ref_energies.append(potential_final_qm._value)
                    system.ref_forces.append(forces_final_qm)
                    system.ref_coordinates.append(system.engine.get_positions()._value)
                    system.n_structures += 1
                elif len(system.ref_coordinates) > 0:
                    # Append last accepted structure
                    system.ref_energies.append(system.ref_energies[-1])
                    system.ref_forces.append(system.ref_forces[-1])
                    system.ref_coordinates.append(system.ref_coordinates[-1])
                    system.n_structures += 1
                else:
                    # No structures have been accepted yet.
                    pass

                # TODO: include code related to partial momentum refreshment
                if sampling_freq_qm == self._param_sampling_freq_qm_id and len(system.ref_coordinates) > 0:
                    self._param_coordinates.append(system.ref_coordinates[-1])
                    self._param_energies.append(system.ref_energies[-1])
                    self._param_forces.append(system.ref_forces[-1])
                    self._param_n_structures += 1

                    # Reset sampling counter
                    self._param_sampling_freq_qm_id = 1
                else:
                    self._param_sampling_freq_qm_id += 1

                # TODO: check best way to sample MM confs
                """ 
                if sampling_mm and sampling_freq_mm == self._param_sampling_freq_mm_id and not qm_accepted:
                    self._param_coordinates.append(system.engine.get_positions().in_units_of(unit.angstrom)._value)
                    self._param_energies.append(potential_final_qm._value)
                    self._param_forces.append(forces_final_qm)
                    self._param_n_structures += 1

                    # Reset sampling counter
                    self._param_sampling_freq_mm_id = 1
                    print("SAMPLING MM", self._n)
                elif not qm_accepted:
                    self._param_sampling_freq_mm_id += 1
                """
            # Write restart files
            if self._n % checkpoint_freq == 0:
                self.write_restart_pickle(settings.restart, system.interface, "restart_hmc_file_{}".format(self._label), self.__dict__)
                system.write_data(os.path.join(settings.restart["restart_dir"], "{}_hmc_{}.nc".format(system.name, self._label)))
                system.write_coordinates_xyz("{}_hmc_{}.xyz".format(system.name, self._label))

            if parametrization and (self._param_n_structures % parametrization_freq == 0 and self._param_n_structures > 0):
                print("PARAMETRIZING", self._n, len(self._param_energies))
                old_param = new_param

                # Create parametrization arrays
                tmp_ref_energies = copy.deepcopy(system.ref_energies)
                tmp_ref_coordinates = copy.deepcopy(system.ref_coordinates)
                tmp_ref_forces = copy.deepcopy(system.ref_forces)

                system.ref_coordinates = np.asarray(self._param_coordinates)
                system.ref_forces = np.asarray(self._param_forces)
                system.ref_energies = np.asarray(self._param_energies)
                system.n_structures = len(system.ref_coordinates)

                # Perform parametrization
                parametrization = Parametrization()
                systems, parameter_space, objective_function, optimizer = parametrization.run_task(settings=settings,
                                                                                                   systems=systems,
                                                                                                   parameter_space=parameter_space,
                                                                                                   objective_function=objective_function,
                                                                                                   optimizer=optimizer,
                                                                                                   adaptive_parametrization=False,
                                                                                                   restart=False)

                # Restore reference arrays
                system.ref_coordinates = tmp_ref_coordinates
                system.ref_energies = tmp_ref_energies
                system.ref_forces = tmp_ref_forces
                system.n_structures = len(system.ref_coordinates)

                new_param = copy.deepcopy(parameter_space.optimizable_parameters_values_scaled)

                self._param_n_structures = 0

                if old_param is not None:
                    rmsd = self._get_parameters_rmsd(old_param, new_param)
                    print("\n \n \n RMSD is {} \n \n \n".format(rmsd))
                    if rmsd < 1e-4:
                        print("RMSD. Will not parametrize anymore...")
                        parametrization=False

                #self._reset_acceptance_rate_mm()
                #self._reset_acceptance_rate_qm()

            self._n += 1

        return systems

    def run_task_ase(self, settings, systems, n_sweeps, n_steps_per_sweep=100, verbose_freq=1,
                     temperature_pot_qm=unit.Quantity(300, unit.kelvin), temperature_pot_mm=unit.Quantity(300, unit.kelvin),
                     temperature_kin_mm=unit.Quantity(300, unit.kelvin), label="0", checkpoint_freq=100, parametrization=False,
                     parametrization_freq=100, restart=False, ):
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
        checkpoint_freq : int
            Frequency at which checkpoint restart files are written.
        parametrization : bool
            Flag that controls whether parametrization is to be performed.
        parametrization_freq : int
            Frequency at which parametrizations are performed.
        restart : bool
            Flag that controls whether or not to perform a restart.

        Returns
        -------
        systems : list
            List with the updated instances of ParaMol System.
        """
        from ase_interface import ANIENS
        from ase_interface import aniensloader
        from ase import units as ase_units
        from ase.calculators.dftd3 import DFTD3

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
            calc = ANIENS(aniensloader('/mainfs/home/jm4g18/programs/ASE_ANI/ani_models/ani-1x_8x.info',0))
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

        new_param = None
        if restart:
            logging.info("Starting HMC sampler parametrization from a previous restart.")

            # Read HMCSampler pickle
            self.__dict__ = self.read_restart_pickle(settings.restart, system.interface, "restart_hmc_file_{}".format(self._label))

            # Read data into system
            system.read_data(os.path.join(settings.restart["restart_dir"], "{}_hmc_{}.nc".format(system.name, self._label)))
        else:
            self._n = 1
            self._n_total_qm = 0
            self._n_accepted_qm = 0

            # MM chain
            self._n_total_mm = 0
            self._n_accepted_mm = 0

        while self._n <= n_sweeps:
            if self._n % verbose_freq == 0:
                self._print_output(system.name)

            if len(system.ref_coordinates) > 0 and self._n != 1:
                # Do not need to compute the QM energy if there are structures in the top ensemble or if we are not in the first ever iteration.
                system.engine.set_positions(system.ref_coordinates[-1])
                potential_initial_mm = mm_ase_engine.run_calculation(coords=system.ref_coordinates[-1] * 10, label=int(self._label))
                potential_initial_qm = system.ref_energies[-1]
                potential_initial_qm = unit.Quantity(potential_initial_qm, unit.kilojoules_per_mole)
                coord_to_run = system.ref_coordinates[-1] * 10
            else:
                # Compute MM initial kinetic and potential energy
                potential_initial_qm, _ = system.qm_engine.qm_engine.run_calculation(coords=system.engine.get_positions().in_units_of(unit.angstrom)._value, label=int(self._label))
                potential_initial_qm = unit.Quantity(potential_initial_qm, unit.kilojoules_per_mole)
                potential_initial_mm = mm_ase_engine.run_calculation(coords=system.engine.get_positions().in_units_of(unit.angstrom)._value, label=int(self._label))
                coord_to_run = system.engine.get_positions().in_units_of(unit.angstrom)._value

            # Run short MD using ASE
            coords, potential_initial_mm, kinetic_initial, forces_initial, potential_final_mm, kinetic_final, forces_final = mm_ase_engine.run_md(coords=coord_to_run,
                                                                                                                                                  label=int(self._label),
                                                                                                                                                  steps=n_steps_per_sweep,
                                                                                                                                                  dt=1.0 * ase_units.fs,
                                                                                                                                                  initial_temperature=temperature_kin_mm,)

            # Compute MM final kinetic and potential energy
            kinetic_initial = unit.Quantity(kinetic_initial, unit.kilojoules_per_mole)
            potential_initial_mm = unit.Quantity(potential_initial_mm, unit.kilojoules_per_mole)
            kinetic_final = unit.Quantity(kinetic_final, unit.kilojoules_per_mole)
            potential_final_mm = unit.Quantity(potential_final_mm, unit.kilojoules_per_mole)
            coords = unit.Quantity(coords, unit.nanometers)

            if self._hmc_acceptance_criterion_mm(potential_final_mm, potential_initial_mm, kinetic_final, kinetic_initial, temperature_pot_mm, temperature_kin_mm):
                potential_final_qm, forces_final_qm = system.qm_engine.qm_engine.run_calculation(coords=coords.in_units_of(unit.angstrom)._value, label=int(self._label))
                potential_final_qm = unit.Quantity(potential_final_qm, unit.kilojoules_per_mole)

                # Nested Markov chain acceptance criterion
                qm_accepted = self._hmc_acceptance_criterion_qm(potential_final_qm, potential_initial_qm, potential_final_mm, potential_initial_mm, temperature_pot_qm, temperature_pot_mm)

                if qm_accepted:
                    # Append energies, forces and conformations
                    system.ref_energies.append(potential_final_qm._value)
                    system.ref_forces.append(forces_final_qm)
                    system.ref_coordinates.append(coords._value)
                    system.n_structures += 1

                    # TODO: include code related to partial momentum refreshment
                elif len(system.ref_coordinates) > 0:
                    # Append last accepted structure
                    system.ref_energies.append(system.ref_energies[-1])
                    system.ref_forces.append(system.ref_forces[-1])
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

            if parametrization and self._n % parametrization_freq == 0:
                # Perform parametrization
                old_param = new_param

                rmsd = self._get_parameters_rmsd(self.old_param, self.new_param)
                parametrization = Parametrization()
                systems, parameter_space, objective_function, optimizer = parametrization.run_task(settings=settings,
                                                                                                   systems=systems,
                                                                                                   parameter_space=parameter_space,
                                                                                                   objective_function=objective_function,
                                                                                                   optimizer=optimizer,
                                                                                                   adaptive_parametrization=False,
                                                                                                   restart=False)

                new_param = copy.deepcopy(parameter_space.optimizable_parameters_values_scaled)

                if old_param is not None:
                    rmsd = self._get_parameters_rmsd(old_param, new_param)
                    print("\n \n \n RMSD is {} \n \n \n".format(rmsd))

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
            return 0.0 # "No classical Hybrid MC trials were performed yet."

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
            return 0.0 #"No HMC trials were performed to accept structures into the QM chain."

    def _reset_acceptance_rate_qm(self):
        self._n_total_qm = 0.0
        self._n_accepted_qm = 0.0

        # MM chain
    def _reset_acceptance_rate_mm(self):
        self._n_total_mm = 0.0
        self._n_accepted_mm = 0.0

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PRIVATE METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
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