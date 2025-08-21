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
   ES2N - Edge of Stability Echo State Network

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

"""


# Author: Nathan Trouvain at 16/12/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from .activations import Identity, ReLU, Sigmoid, Softmax, Softplus, Tanh
from .es2n import ES2N
from .intrinsic_plasticity import IPReservoir
from .io import Input, Output
from .lif import LIF
from .lms import LMS
from .local_plasticity_reservoir import LocalPlasticityReservoir
from .nvar import NVAR
from .reservoir import Reservoir
from .ridge import Ridge
from .rls import RLS
from .sklearn_node import ScikitLearnNode

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
    "NVAR",
    "IPReservoir",
    "ScikitLearnNode",
    "LIF",
    "LocalPlasticityReservoir",
    "ES2N",
]
