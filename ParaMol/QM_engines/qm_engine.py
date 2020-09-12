# -*- coding: utf-8 -*-
"""
Description
-----------

This module defines the :obj:`ParaMol.QM_engines.qm_engine.QMEngine` class, which is a ParaMol representation of the parameter space used in the optimization.
"""
import logging


class QMEngine:
    """
    ParaMol wrapper of the QM engines.

    Notes
    -----
    Available QM engines are "amber", "dftb+" and "ase".

    Parameters
    ----------
    system : :obj:`ParaMol.System.system.ParaMolSystem`
        ParaMol system associated with the QM engine.
    qm_engine_name : str
        Name of the QM engine. Available QM engines are "amber", "dftb+" and "ase".
    qm_engine_settings : dict
        Keyword arguments passed to the QM engine wrapper.
    interface : :obj:`ParaMol.Utils.interface.ParaMolInterface`
        ParaMol interface.

    Attributes
    ----------
    qm_engine : any QM engine defined in the subpackage :obj:`ParaMol.QM_engines`
        QM engine instance.
    """
    def __init__(self, system, qm_engine_name, qm_engine_settings, interface):
        # Private variables
        self._qm_engine_name = qm_engine_name
        self._qm_engine_settings = qm_engine_settings
        self._system = system
        self._interface = interface

        # Public variables
        self.qm_engine = None
        self._qm_engine_init(qm_engine_name, qm_engine_settings)

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PRIVATE METHODS                      #
    #                                                              #
    # ------------------------------------------------------------ #
    def _qm_engine_init(self, qm_engine_name, qm_engine_settings):
        """
        Method that initializes the QM engine.
        
        qm_engine_name : str
            Name of the QM engine. Available QM engines are "amber", "dftb+" and "ase".
        qm_engine_settings : dict
            Keyword arguments passed to the QM engine wrapper.
            
        Notes
        -----
        The QM engine instance can be accessed through the self.qm_engine public attribute.

        Returns
        -------
        qm_engine : one of the QM engines defined in the subpackage :obj:`ParaMol.QM_engines`
            QM engine instance.
        """
        logging.info("Initializing QM engine with name {}.".format(qm_engine_name))

        periodic = True

        if periodic:
            cell = self._system.engine.get_cell()
        else:
            cell = None

        if qm_engine_name.lower() == "amber":
            import ParaMol.QM_engines.amber_wrapper as amber

            # Get atom list and atomic numbers list
            self._system.engine.get_atom_list()
            self._system.engine.get_atomic_numbers()

            # Create AmberWrapper
            self.qm_engine = amber.AmberWrapper(system_name=self._system.name,
                                                interface=self._interface,
                                                prmtop_file=self._system.engine.top_file,
                                                inpcrd_file=self._system.engine.crd_file,
                                                cell=cell,
                                                n_atoms=self._system.n_atoms,
                                                atom_list=self._system.engine.atom_list,
                                                atomic_number_list=self._system.engine.atomic_number_list,
                                                **qm_engine_settings)

        elif qm_engine_name.lower() == "dftb+":
            import ParaMol.QM_engines.dftb_wrapper as dftb

            # Get atom list and atomic numbers list
            self._system.engine.get_atom_list()

            self.qm_engine = dftb.DFTBWrapper(system_name=self._system.name,
                                              interface=self._interface,
                                              n_atoms=self._system.n_atoms,
                                              atom_list=self._system.engine.atom_list,
                                              n_calculations=self._system.n_cpus,
                                              **qm_engine_settings)

        elif qm_engine_name.lower() == "ase":
            import ParaMol.QM_engines.ase_wrapper as ase

            # Get atom list and atomic numbers list
            self._system.engine.get_atom_list()

            self.qm_engine = ase.ASEWrapper(system_name=self._system.name,
                                            interface=self._interface,
                                            n_atoms=self._system.n_atoms,
                                            atom_list=self._system.engine.atom_list,
                                            n_calculations=self._system.n_cpus,
                                            cell=cell,
                                            **qm_engine_settings)
        else:
            raise NotImplementedError("{} QM engine is not implemented yet.".format(qm_engine_name))

        logging.info("QM engine {} initialized successfully!".format(qm_engine_name))

        return self.qm_engine

