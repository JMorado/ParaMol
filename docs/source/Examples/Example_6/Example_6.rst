Example 6: Example of how to restart a calculation
==================================================

Restarting a calculation in ParaMol is as simple as passing the argument :obj:`restart=True` into the :obj:`task.run_task` method:


.. code-block:: python

    from ParaMol.Tasks.parametrization import *
    from ParaMol.Utils.settings import *

    [...]

    paramol_settings = Settings()

    parametrization = Parametrization()
    parametrization.run_task(paramol_settings, ..., restart=True)

    [...]

The argument :obj:`restart=True` can be used in the :obj:`run_task` method of the following built-in tasks:


    - :obj:`ParaMol.Tasks.parametrization.Parametrization`

    A checkpoint file of the :obj:`ParaMol.Parameter_space.parameter_space` is written with a frequency defined by the :obj:`paramol_settings.restart["restart_parametrization_checkpoint_freq"]` variable. The name of this file is defined in the variable :obj:`paramol_settings.restart["restart_parametrization_file"]`.
    Furthermore, a checkpoint file is also stored after the optimization procedures finishes, which is useful in order restart an adaptive parametrization with the correct :obj:`ParameterSpace` state. The name of this file is given by the variable :obj:`paramol_settings.restart["restart_parameter_space_file"]`.

    - :obj:`ParaMol.Tasks.adaptive_parametrization.AdaptiveParametrization`

    A checkpoint file with the adaptive parametrization procedure state is stored after every iteration of the algorithm.
    The name of this checkpoint file is defined in the variable :obj:`paramol_settings.restart["restart_adaptive_parametrization_file"]`.
    Additionally, at the end of every iteration, a :doc:`NetCDF file <../../Files_specification>` containing system's information, such as coordinates, energies and forces is saved in a file named :obj:`restart_[system.name]_data.nc`.

    - :obj:`ParaMol.Tasks.torsions_parametrization.TorsionsParametrization`

    A checkpoint file with the state of the procedure used to automatically parametrize soft torsions is stored after the scan of a soft dihedral is completed. Furthermore, as this task resorts to :obj:`ParaMol.Tasks.torsions_scan.TorsionScan` and :obj:`ParaMol.Tasks.Parametrization.parametrization` tasks, their respective checkpoint files are also saved.
    The name of the :obj:`TorsionsParametrization` checkpoint file is defined in the variable :obj:`paramol_settings.restart["restart_soft_torsions_file"]`.

    - :obj:`ParaMol.Tasks.torsions_scan.TorsionScan`

    A checkpoint file with the torsional scan state is stored after every geometry optimization.
    The name of this checkpoint file is defined in the variable :obj:`paramol_settings.restart["restart_scan_file"]`.


The checkpoint files are Python Pickle files that store the instance dictionaries of the previously indicated classes. These are saved into a directory with a name defined by the :obj:`paramol_settings.restart["restart_dir"]` variable.

More information about how to control the names of the checkpoint files can be found at the ParaMol Settings :doc:`documentation page <../../ParaMol_settings>`.
