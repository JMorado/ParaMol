Example 7: Example of multiple system parametrization
=========================================================================

Mapping of different topologies
##################################

ParaMol is able to parametrize multiple systems at the same time.
Even though this is possible to do without applying any symmetry constraint, in the context of force field development it is often desirable to derive the parameters for a given term type and for a set of molecules.
Therefore, in this example we are going to concomitantly optimize the barrier height of the dihedral type hc-c3-c3-hc for ethane and propane.

**NOTE:** When optimizing more than once system at once, the mapping of the symmetry groups of different systems has to be performed manually so that there is a one-to-one correspondence between symmetry group and term type.
The automatic mapping of symmetries for different systems is a feature that will likely be implemented in future ParaMol versions.


The first step is to write out the ParaMol Force Field files so that the correct symmetries are set, which can be done using the following code:


.. literalinclude:: ../../../../Examples/Example_7/example_7_write_ff.py
    :language: python



.. literalinclude:: ethane_symm_original.ff
    :language: text
    :caption: Original ethane ParaMol Force Field file.

.. literalinclude:: propane_symm_original.ff
    :language: text
    :caption: Original propane ParaMol Force Field file.

As can be seen in the ParaMol Force Field files, all term types belong to the default symmetry group, *i.e.*, "X", which means that they do not possess any symmetry.
We will set the symmetry label of the dihedral types hc-c3-c3-hc to "T0" (see modified ParaMol Force Fields below). In this way, we will be able to find the best barrier height that minimizes the objective function for both ethane and propane.


.. literalinclude:: ethane_symm_mod.ff
    :language: text
    :caption: Modified ethane ParaMol Force Field file.

.. literalinclude:: propane_symm_mod.ff
    :language: text
    :caption: Modified propane ParaMol Force Field file.



Simultaneous parametrization of multiple systems
##################################################

Now that we have done the necessary changes in the ParaMol Force Field files, we are ready to perform the conformational sampling, *ab initio* properties calculation and subsequent parameters' optimization.
This can be done using the following script:

.. literalinclude:: ../../../../Examples/Example_7/example_7.py
    :language: python


Finally, as can be seen in the final ParaMol Force Field files, a new barrier height value was found that best suits both systems.
Specifically, GAFF slightly overestimates the barrier height of this torsions with respect to the SCC-DFTB-D3 level of theory, since, after re-parametrization, its values has decreased from 0.6276 kj/mol to 0.4690 kj/mol.

.. literalinclude:: ethane_symm_opt.ff
    :language: text
    :caption: Final ethane ParaMol Force Field file.

.. literalinclude:: propane_symm_opt.ff
    :language: text
    :caption: Final propane ParaMol Force Field file.





