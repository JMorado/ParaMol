# -*- coding: utf-8 -*-
"""
ParaMol Force_field subpackage.

Contains modules related to the ParaMol representation of a force field.

The param_keys used for each force group in ParaMol are as follows:

    - 'HarmonicBondForce'
        - 'bond_eq' : bond equilibrium value
        - 'bond_k' : bond force constant
    - 'HarmonicAngleForce'
        - 'angle_eq' : angle equilibrium value
        - 'angle_k' : angle force constant
    - 'PeriodicTorsionForce'
        - 'torsion_periodicity' : torsion periodicity
        - 'torsion_phase' : torsion phase
        - 'torsion_k' : torsion barrier height
    - 'NonbondedForce'
        - 'charge' : Coulomb charge
        - 'lj_eps' : Lennard-Jones 12-6 epsilon (depth of the potential well)
        - 'lj_sigma' : Lennard-Jones 12-6 sigma (finite distance at which the inter-particle potential is zero)
        - 'scee' : 1-4 electrostatic scaling factor
        - 'scnb' : 1-4 Lennard-Jones scaling factor
"""

__all__ = ['force_field',
           'force_field_term',
           'force_field_term_parameter']