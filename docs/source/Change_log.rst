Change Log
==========

Version 1.1.2
-------------
- Improvements to RESP. Now ParaMol automatically reads the energies from Gaussian .log files
- Small adjustments to examples.
- Reverted scaling of regularization on LLS fitting to the original version. Results from non-linear optimizers and LLS fitting may not coincide as for a similar parameterization situation the number of parameters differs due to the requirement of using 'auxiliary' terms in LLS fitting. General solution will be included in future versions.

Version 1.1.1
-------------
- Several improvements to explicit RESP solution.

Version 1.1.0
-------------
- Implementation of LLS to derive bonded parameters. Bonds, angles and torsions can now be determined directly through LLS fitting to QM energies. Currently, the only type of regularization that can be used with LLS fitting is the L2 regularization.
- Besides the previously implemented AMBER symmetrized, now GROMACS and CHARMM symmetrizes are also available.

Version 1.0.2
-------------
- Fixes and improvements to AmberSymmetrizer.
- Updates to examples.

Version 1.0.1
-------------
- Improvements to the RESP procedure.
- RESP: Multiconformational fitting can be now performed using the explicit solver and with different weighting methods.

Version 1.0.0
-------------
- Stable release of ParaMol (5th July 2020).
- Available tasks: parametrization, adaptive parametrization, dihedral scans, RESP fitting, objective function plots, ab initio properties.
- MM engines: OpenMM, RESP.
- QM engines: ASE, DFTB+, AMBER.
