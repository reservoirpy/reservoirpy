"""
====================================
Jax backend (:mod:`reservoirpy.jax`)
====================================

Jax implementations of the ReservoirPy nodes, models and tools, mirroring the
:mod:`reservoirpy` API: ``reservoirpy.jax.nodes.Reservoir`` is the Jax version
of :class:`reservoirpy.nodes.Reservoir`, and so on. Requires Jax
(``pip install reservoirpy[jax]``).

See :doc:`/user_guide/jax_backend` for an introduction.

.. currentmodule:: reservoirpy.jax

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ESN - Simple Echo State Network.

.. toctree::
   :maxdepth: 1

   reservoirpy.jax.nodes
   reservoirpy.jax.mat_gen
   reservoirpy.jax.activationsfunc
   reservoirpy.jax.node
   reservoirpy.jax.model
   reservoirpy.jax.ops
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from . import activationsfunc, mat_gen, nodes, type, utils
from .esn import ESN
from .model import Model
from .node import Node

__all__ = [
    "mat_gen",
    "activationsfunc",
    # "observables",  # NOT PLANNED FOR NOW
    # "hyper",  # NOT PLANNED
    # "datasets",  # NOT PLANNED
    "nodes",
    "Node",
    "Model",
    "ESN",
    "type",
    "utils",
]


def __getattr__(attr):
    if attr in ["observables", "hyper", "datasets", "set_seed", "__version__"]:
        raise AttributeError(f"{attr} does not have a Jax implementation. Please use reservoir.{attr} instead.")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attr}")
