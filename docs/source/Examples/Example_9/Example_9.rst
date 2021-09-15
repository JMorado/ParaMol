Example 9: Example of self-parameterizing nMC-MC
=========================================================================================

Self-parameterizing methodology that iteratively couples the nMC-MC algorithm with a parameterization step.
This algorithm allows on-the-fly derivation of bespoke FFs owing to its capability of performing sampling of
relevant configurations and subsequent optimization of the FF parameters, all in one scheme.

Self-parameterizing nMC-MC
###############################

.. literalinclude:: ../../../../Examples/Example_9/example_9_serial_param.py
    :language: python

