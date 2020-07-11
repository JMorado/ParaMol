Example 2: RESP charge fitting of aniline
=========================================================================================

Fitting of charges
##################


In order to generate the electrostatic potential of aniline, first we will optimise the geometry of aniline at the B3LYP/6-31G* level. This is the usual level of theory used for small molecules and transition-metal complexes accurate calculations), whereas calculations at the AM1 level of theory are usually performed for large organic molecules (relatively crude calculations). We will use Gaussian to perform the optimization. The Gaussian input file reads:

.. literalinclude:: gaussian_calculations/aniline.com
    :language: text
    :caption: Geometry optimization Gaussian input file.

Furthermore, in order too calculate the electrostatic potential, we will run a Gaussian single-point energy calculation, at the HF/6-31G* level (without transition metals) or at the B3LYP/DZpdf/6-31G* level. The following keywods have to be used:

    - IOp(6/33=2) makes Gaussian write out the potential points and potentials (do not change)
    - IOp(6/41=10) specifies that 10 concentric layers of points are used for each atom (do not change)
    - IOp(6/42) gives the density of points in each layer. A value of 17 gives about 2500 points/atom. Lower values may be needed for large molecules, since the programs cannot normally handle more than 100 000 potential points. A value of 10 gives about 1000 points/atom.

The Gaussian input file for this calculation reads:

.. literalinclude:: gaussian_calculations/aniline_opt.com
    :language: text
    :caption: ESP calculation Gaussian input file.

Finally, in order to read the ESP data into ParaMol and perform RESP charge fitting, the following

.. literalinclude:: ../../../../Examples/Example_2/example_2.py
    :language: python


Comparision of the ESP charges
###############################

We may now check the results obtained using the solvers available in ParaMol ('scipy' and 'explicit').

.. literalinclude:: aniline_resp_scipy.ff
    :language: text
    :caption: ESP charges obtained when using the 'SLSQP' SciPy optimizer ('scipy' solver).

.. literalinclude:: aniline_resp_explicit.ff
    :language: text
    :caption: ESP charges obtained when using the 'explicit' solver.

.. literalinclude:: gaussian_calculations/esp_charges_gaussian.dat
    :language: text
    :caption: ESP charges as calculated by Gaussian.
