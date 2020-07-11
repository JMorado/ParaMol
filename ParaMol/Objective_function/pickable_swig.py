# -*- coding: utf-8 -*-

"""
Description
-----------

This module defines the auxiliary classes used to make an OpenMM context pickalable.
"""

from simtk.openmm import *


class PickalableSWIG(object):
    """
    Defines the __setstate__ and __getstate__ methods necessary to convert an object instance into a Pickle.
    """
    def __setstate__(self, state):
        self.__init__(*state['args'])

    def __getstate__(self):
        return {'args': self.args}


class PickalableContext(Context, PickalableSWIG):
    """
    Wrapper around the Context OpenMM class.
    This is necessary to make an OpenMM context pickalable, i.e., serializable.

    Notes
    -----
    For more information see: https://stackoverflow.com/questions/9310053/how-to-make-my-swig-extension-module-work-with-pickle
    """
    def __init__(self, *args):
        self.args = args
        Context.__init__(self, *args)


