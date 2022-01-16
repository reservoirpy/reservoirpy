"""
============================================
ReservoirPy Nodes (:mod:`reservoirpy.nodes`)
============================================

.. currentmodule:: reservoirpy.nodes

Reservoirs
==========

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Reservoir - Recurrent pool of leaky integrator neurons
   NVAR - Non-linear Vector Autoregressive machine (NG-RC)

Readouts
========

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Ridge - Layer of neurons connected through offline linear regression.
   FORCE - Layer of neurons connected through online FORCE learning.

Activation functions
====================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Tanh - Hyperbolic tangent node.
   Sigmoid - Logistic function node.
   Softmax - Softmax node.
   Softplus - Softplus node.
   ReLU - Rectified Linear Unit node.
   Identity - Identity function node.

Input and Output
================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Input - Input node, used to distribute input data to other nodes.
   Output - Output node, used to gather stated from hidden nodes.

Operators
=========

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Concat - Concatenate vector of data along feature axis.


Optimized ESN
=============

.. autosummary
   :toctree: generated/
   :template: autosummary/class.rst

   ESN - Echo State Network model with distributed offline learning.

Miscellaneous
=============

.. autosummary
   :toctree: generated/
   :template: autosummary/class.rst

   RandomChoice - Randomly select features in a vector of data.
   AsabukiNorm - Normalization as defined in Asabuki et al. (2018)
"""
# Author: Nathan Trouvain at 16/12/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from .reservoir import Reservoir
from .ridge import Ridge
from .force import FORCE
from .norm import AsabukiNorm
from .nvar import NVAR
from .esn import ESN
from .activations import Tanh, Sigmoid, Softmax, Softplus, Identity, ReLU
from .randomchoice import RandomChoice
from .io import Input, Output
from .concat import Concat

__all__ = ["Reservoir",
           "Input",
           "Output",
           "Ridge",
           "FORCE",
           "Tanh",
           "Softmax",
           "Softplus",
           "Identity",
           "Sigmoid",
           "ReLU",
           "RandomChoice",
           "AsabukiNorm",
           "NVAR",
           "ESN",
           "Concat"]
