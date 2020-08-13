Installation
============

Quick Install
-------------


ParaMol has been extensively tested both on Linux and OS X platforms and all its features work as expected.
Nevertheless, it has never been tested on Windows. It is expected that most of ParaMol features work correctly
on this platform except the parallel capabilities (Windows does not support forks).
Therefore, please remember that Window is not natively suported by ParaMol.

The easiest way to install ParaMol is via the conda package manager:

.. code-block::

    conda install paramol -c jmorado -c ambermd -c conda-forge -c omnia -c rdkit -c anaconda

For the development version use:

.. code-block::

    conda install paramol-dev -c jmorado -c ambermd -c conda-forge -c omnia -c rdkit -c anaconda

Source Code
------------

ParaMol is an open source code and it has a github repository:

    https://github.com/JMorado/ParaMol