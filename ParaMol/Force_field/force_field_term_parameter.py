"""
Description
===========
- The :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` class is the ParaMol representation of a force field term parameter, e.g., a force constant or an equilibrium value.
- A :obj:`ParaMol.Force_field.force_field_term_parameter.Parameter` instance has `param_key` and `force_group` attributes that define what is the force field term to which this parameter belongs and what is the role of the parameter in the force field term.
- The attribute `optimize` determines whether or not this parameters is optimizable.
- The `symmetry_group` attribute enables to make this parameter equivalent to others in symmetry-constrained optimizations.
"""


class Parameter:
    """
    ParaMol representation of a force field parameter.

    Parameters
    ----------
    symmetry_group : str
        Symmetry group of the force field term.
    optimize : bool
        Flags that signals whether or not this is an optimizable parameters.
    param_key : str
        Key of the parameter.
    value : float/int
        Value of the parameter.
    ff_term : ParaMol.Force_field.force_field_term.FFTerm
        Force field term to which this parameter original belongs

    Attributes
    ----------
    symmetry_group : str
        Symmetry group of the force field term.
    optimize : bool
        Flags that signals whether or not this is an optimizable parameters.
    param_key : str
        Key of the parameter.
    value : float/int
        Value of the parameter.
    multiplicity :
        Multiplicity of the parameter, i.e., how many parameters with the same `symmetry_group` exist and `param_key` exist. Only relevant for symmetry-constrained optimizations.
    ff_term : ParaMol.Force_field.force_field_term.FFTerm
        Force field term to which this parameter original belongs
    """
    def __init__(self, symmetry_group, optimize, param_key, value, ff_term=None):
        self.symmetry_group = symmetry_group
        self.optimize = optimize
        self.value = value
        self.param_key = param_key
        self.multiplicity = 1.0
        self.ff_term = ff_term

    def __str__(self):
        """
        Defines the print statement to obtain the string representation of a parameter.

        Returns
        -------

        """
        description = "\n"
        description += "Parameter key: {} \n".format(self.param_key)
        description += "Value: {} \n".format(self.value)
        description += "Optimize: {} \n".format(self.optimize)
        description += "Symmetry group:  {} \n".format(self.symmetry_group)

        return description

