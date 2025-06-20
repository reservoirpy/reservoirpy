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
   IPReservoir - Reservoir with intrinsic plasticity learning rule
   LocalPlasticityReservoir - Reservoir with weight plasticity

Offline readouts
================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   Ridge - Layer of neurons connected through offline linear regression.
   ScikitLearnNode - Interface for linear models from the scikit-learn library.

Online readouts
===============

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   LMS - Layer of neurons connected through least mean squares learning rule.
   RLS - Layer of neurons connected through recursive least squares learning rule.

Optimized ESN
=============

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ESN - Echo State Network model with distributed offline learning.

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
   Delay - Adds a discrete delay between input and output.
"""


# Author: Nathan Trouvain at 16/12/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

# from .esn import ESN
# from .readouts import LMS, RLS, Ridge, ScikitLearnNode
# from .reservoirs import NVAR, IPReservoir, LocalPlasticityReservoir
from .activations import Identity, ReLU, Sigmoid, Softmax, Softplus, Tanh
from .io import Input, Output
from .lms import LMS
from .reservoir import Reservoir
from .ridge import Ridge
from .rls import RLS

__all__ = [
    "Reservoir",
    "Input",
    "Output",
    "Ridge",
    "LMS",
    "RLS",
    "Tanh",
    "Softmax",
    "Softplus",
    "Identity",
    "Sigmoid",
    "ReLU",
    #  "NVAR",
    #  "ESN",
    #  "IPReservoir",
    #  "ScikitLearnNode",
    #  "LocalPlasticityReservoir",
]
