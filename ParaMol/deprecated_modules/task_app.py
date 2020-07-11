
class TaskApp:
    def __init__(self, options=None):
        self.options = options
        self.systems = None
        pass

    # ------------------------------------------------------------ #
    #                                                              #
    #                          PUBLIC METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def run_task(self, task_name, *args, **kwargs):
        """
        This method perform the task defined in self._param.task.
        """
        if task_name.upper() == 'ADAPTIVE_PARAMETRIZATION':
            # Perform self-consistent parametrization
            import ParaMol.Tasks.adaptive_parametrization as adaptive_param
            task = adaptive_param.AdaptiveParametrization()

        elif task_name.upper() == 'PARAMETRIZATION':
            # Perform single parametrization
            import ParaMol.Tasks.parametrization as parametrization
            task = parametrization.Parametrization()

        elif task_name.upper() == 'AB_INITIO_PROPERTIES':
            # Calculate QM forces and energies
            import ParaMol.Tasks.ab_initio_properties as ab_initio
            task = ab_initio.AbInitioProperties()

        elif task_name.upper() == "OBJECTIVE_FUNCTION_PLOT":
            # Compute heat map
            import ParaMol.Tasks.objective_function_plot as obj_fun_plot
            task = obj_fun_plot.ObjectiveFunctionPlot()

        elif task_name.upper() == "TORSIONS_PARAMETRIZATION":
            # Perform dihedral scan
            import ParaMol.Tasks.torsions_parametrization as torsions_param
            task = torsions_param.TorsionsParametrization()

        elif task_name.upper() == "RESP":
            # Perform RESP charge fitting
            #import Tasks.resp_fitting as resp_fitting
            #task = resp_task.RESPTask()
            pass
        else:
            import ParaMol.Tasks.task as task_base_class
            task = task_base_class.Task(task_name.upper())

        task.run_task(*args, **kwargs)

        print("{} task was performed successfully!".format(task_name.upper()))

        # Delete task
        del task

        return True

    @staticmethod
    def init_paramol_systems(settings):
        """
        This method should called be after an instance of TaskApp is created.
        It performs all the commands necessary to make a task run, viz.:
            - Reads initial structure guess, if requested.
            - Add extra torsional terms, if requested.
            - Read .param file, if requested.
            - Writes .param file.
            - Reads coordinates, forces and energies, if provided.

            - If conformations were provided but not forces and energies, compute them at the QM level.

        :return: list of Parameter instances.
        :rtype: list of Parameter
        """

        assert type(settings.systems_settings) is dict

        systems = []

        for system in settings.systems_settings:
            # Create system engine
            openmm_system = OpenMMEngine(create_system=False, topology_format=settings.systems_settings['top_format'],
                                         top_file=settings.systems_settings['top_file'],
                                         xml_file=settings.systems_settings['xml_file'],
                                         crd_file=settings.systems_settings['crd_file'])

            # Create Molecular System
            sys = MolSystem(name=settings.systems_settings['name'],
                            engine=openmm_system,
                            n_cpus=settings.systems_settings['n_cpus'])

            # Create OpenMM System
            sys.engine.create_system()

            if settings.systems_settings["extra_torsions"]:
                # Add extra torsions if required
                sys.engine.add_all_torsisons(periodicities=settings.systems_settings["periodicities"],
                                             default_phase=settings.systems_settings["default_phase"],
                                             default_v=settings.systems_settings["default_v"])

            # Create ParaMol Force Field representation
            sys.force_field.create_force_field(opt_bonds=settings.systems_settings['opt_bonds'],
                                               opt_angles=settings.systems_settings['opt_bonds'],
                                               opt_torsions=settings.systems_settings['opt_bonds'],
                                               opt_lj=settings.systems_settings['opt_lj'],
                                               opt_charges=settings.systems_settings['opt_charges'],
                                               opt_sc=settings.systems_settings["opt_sc"])
            # TODO: symmetrization

            # Load data into the system
            system.load_data(coords_file=settings.systems_settings["energies_file"],
                             energies_file=settings.systems_settings["energies_file"],
                             forces_file=settings.systems_settings["energies_file"])

            # Append system to list
            systems.append(sys)

        return systems

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PRIVATE METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    pass
