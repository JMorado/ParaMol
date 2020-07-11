"""
Description
===========
- The :obj:`ParaMol.Force_field.force_field_term.FFTerm` class is the ParaMol representation of a force field term, e.g. a bond, angle, torsion or Couloumb term.
- A :obj:`ParaMol.Force_field.force_field_term.FFTerm` can contain several parameters. For example, if its force_group is "HarmonicBondForce" it should contain a parameter with key "bond_k" that corresponds to the bond force constant and a parameter with key "bond_eq" that corresponds to the bond equilibrium value. These keys are stored as `param_key` attributes in the correspondent :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` instance.
"""

from .force_field_term_parameter import *


class FFTerm:
    """
    ParaMol representation of a force field term.

    Parameters
    ----------
    force_group : str
        Name of the force group as given by OpenMM.
    idx : int
        Index of the force field term within a given force group in OpenMM..
    atoms : list
        Atoms involved in the force field term.
    symmetry_group : str
        Symmetry group of the force field term.

    Attributes
    ----------
    force_group : str
        Name of the force group as given by OpenMM.
    idx : int
        Index of the force group.
    atoms : list
        Atoms involved in the force field term:
    symmetry_group : str
        Symmetry group of the force field term.
    parameters : dict
        Dictionary that contains the mapping between `param_key` and :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter`.
    """
    def __init__(self, force_group, idx, atoms, symmetry_group = "X"):
        self.force_group = force_group
        self.idx = idx
        self.atoms = atoms
        self.parameters = {}
        self.symmetry_group = symmetry_group

    def __str__(self):
        return "Force Field term with id {} belonging to " \
               "force group {}. Contains {} parameters.".format(self.idx, self.force_group, len(self.parameters))

    def __repr__(self):
        return "Force Field term with id {} belonging to " \
               "force group {}. Contains {} parameters.".format(self.idx, self.force_group, len(self.parameters))

    # ------------------------------------------------------------ #
    #                                                              #
    #                         PUBLIC METHODS                       #
    #                                                              #
    # ------------------------------------------------------------ #
    def add_parameter(self, symmetry_group, optimize, param_key, value):
        """
        Method that adds a :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` with key `param_key` to the parameters dictionary.

        Parameters
        ----------
        symmetry_group : str
            Symmetry group of the force field term.
        optimize : bool
            Flags that signals whether or not this is an optimizable parameters.
        param_key : str
            Key of the parameter.
        value : int/float
            Value of the parameter

        Returns
        -------
        parameters : dict
            Dictionary containing the current mapping between `param_key` and :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter`.
        """
        self.parameters[param_key] = Parameter(symmetry_group, optimize, param_key, value)

        return self.parameters
