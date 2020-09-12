Example 7: Example of multiple system parametrization
=========================================================================

Mapping of different topologies
##################################

The mapping.

ParaMol is able to parametrize multiple systems at the same time.
Even though this is possible to do

In this example we are going to use ParaMol's torsional scan Task (:obj:`ParaMol.Tasks.torsions_scan.TorsionScan`) to parametrize a torsion of the norfloxacin analog represented above.
First of all, we are going to symmetrize the ParaMol Force Field so that it respects atom-type symmetries and write it to a file in order to choose what torsions we want to parametrize.


.. literalinclude:: ../../../../Examples/Example_7/example_7_write_ff.py
    :language: python



.. literalinclude:: ethane_symm_original.ff
    :language: text
    :caption: Original ethane ParaMol Force Field file.

.. literalinclude:: propane_symm_original.ff
    :language: text
    :caption: Original propane ParaMol Force Field file.

.. literalinclude:: ethane_symm_mod.ff
    :language: text
    :caption: Modified ethane ParaMol Force Field file.

.. literalinclude:: propane_symm_mod.ff
    :language: text
    :caption: Modified propane ParaMol Force Field file.



Simultaneous parametrization of multiple systems
##################################################

Now that we have done the necessary changes in the ParaMol Force Field file, we are ready to perform the torsional scan and subsequent parameters's optimization.
Luckily, as all the torsions around the C11-N4 are of the same type and, therefore, they share the same set of parameters, we only need to perform the torsional scan of one of the torsions with symmetry T8. Furthermore, when asking ParaMol to create its Force Field representation, we need to provide the modified ParaMol Force Field file so that ParaMol creates its internal representation of the Force Field from this file. The same procedure can be done to set special constraints or to optimize other parameters.


.. literalinclude:: ../../../../Examples/Example_7/example_7.py
    :language: python




.. literalinclude:: ethane_symm_opt.ff
    :language: text
    :caption: Final ethane ParaMol Force Field file.

.. literalinclude:: propane_symm_opt.ff
    :language: text
    :caption: Final propane ParaMol Force Field file.





