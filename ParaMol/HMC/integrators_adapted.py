"""
This module implements an adapted VelocityVerlet integrator that works the Hyrid MC carlo code.
The class implemented here is totally based on the original one, which is part of openmmtools.
Joao Morado 22.05.2019
"""
from openmmtools.integrators import *


class VelocityVerletIntegratorAdapted(mm.CustomIntegrator):
    """Verlocity Verlet integrator.

    Notes
    -----
    This integrator is taken verbatim from Peter Eastman's example appearing in the CustomIntegrator header file documentation.

    References
    ----------
    W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson, J. Chem. Phys. 76, 637 (1982)

    Examples
    --------

    Create a velocity Verlet integrator.

    >>> timestep = 1.0 * unit.femtoseconds
    >>> integrator = VelocityVerletIntegrator(timestep)

    """

    def __init__(self, timestep=1.0 * unit.femtoseconds):
        """Construct a velocity Verlet integrator.

        Parameters
        ----------
        timestep : np.unit.Quantity compatible with femtoseconds, default: 1*unit.femtoseconds
           The integration timestep.

        """

        super(VelocityVerletIntegratorAdapted, self).__init__(timestep)

        self.addPerDofVariable("x1", 0)
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()
