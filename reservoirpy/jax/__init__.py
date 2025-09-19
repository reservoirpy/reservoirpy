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
