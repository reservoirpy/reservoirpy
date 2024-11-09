"""
=======================================================
Experimental features (:mod:`reservoirpy.experimental`)
=======================================================

.. currentmodule:: reservoirpy.experimental

You can find in this module all features in development and testing for
future releases.

.. warning::

    All features in the experimental module may still be under heavy
    development. You can still report any bug by opening an issue
    as explained in :ref:`opening_an_issue`.

Nodes
=====

.. autosummary::
    :toctree: generated/
    :template: autosummary/class.rst

    LIF - Leaky Integrate and Fire RNN (Liquid State Machine)
    Add - Add two vectors.
    BatchFORCE - Fast implementation of FORCE algorithm.
    RandomChoice - Randomly select features in a vector of data.
    AsabukiNorm - Normalization as defined in Asabuki et al. (2018)
"""

# Author: Nathan Trouvain at 03/02/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from .add import Add
from .batchforce import BatchFORCE
from .norm import AsabukiNorm
from .randomchoice import RandomChoice
from .sklearn import from_sklearn
from .spiking.lif import LIF

__all__ = [
    "Add",
    "BatchFORCE",
    "AsabukiNorm",
    "RandomChoice",
    "from_sklearn",
    "LIF",
]
