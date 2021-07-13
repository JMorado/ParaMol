[![Build Status](https://travis-ci.com/JMorado/ParaMol.svg?branch=master)](https://travis-ci.org/JMorado/ParaMol)
[![Documentation Status](https://readthedocs.org/projects/paramol/badge/?version=latest)](https://paramol.readthedocs.io/en/latest/?badge=latest)

# ParaMol 

![ParaMol](docs/source/paramol-white.png)


ParaMol is a Python library that aims to ease the process of force field parametrization of molecules. 

Current Version: 1.1.3


# Quick Installation
The easiest way to install ParaMol it.

The last stable version can be installed via conda:

    conda install paramol -c jmorado -c ambermd -c conda-forge -c omnia -c rdkit -c anaconda

# Available Tasks
- Parametrization.
- LLS fitting.
- Adaptive parametrization.
- RESP charge fitting.
- ab initio properties calculation.
- Objective function plot.
- Torsional scans.
- Automatic parametrization of soft torsions.

# Current features
 - Parallel and serial computation of the objective .</li>
 - LLS fitting of bonded terms. </li>
 - Available symmetrizers: AMBER, GROMACS, CHARMM.
 - Optimization algorithms: Scipy Optimizers, Monte Carlo, Gradient Descent, Simulated Annealing.</li>
 - QM engines: DFTB+, AMBER, or any QM engine available through ASE.</li>
 
# Tests
 ParaMol uses the [pytest](https://docs.pytest.org/en/stable/) framework to test the code. pytest can be install through pip:
    
    pip install -U pytest

 Once pytest is installed, the tests can be run by simply typing:
 
    python -m pytest
    
 in ParaMol's root directory.

# Contact

ParaMol is developed and maintained by João Morado at the University of Southampton.

If you have any question or issue to report please contact j.morado@soton.ac.uk.

# References

Morado, J.; Mortenson, P. N.; Verdonk, M. L.; Ward, R. A.; Essex, J. W.; Skylaris, C.-K. ParaMol: A Package for Automatic Parameterization of Molecular Mechanics Force Fields. J. Chem. Inf. Model. 2021, acs.jcim.0c01444. https://doi.org/10.1021/acs.jcim.0c01444.  
