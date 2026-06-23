# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import os
import tempfile
import uuid

from . import activationsfunc, hyper, mat_gen, nodes, observables, type
from ._version import __version__
from .esn import ESN
from .model import Model
from .node import Node
from .utils.random import set_seed
from .utils.save_load import load, save

# Generate a unique temporary directory to prevent permission errors
_TEMPDIR = os.path.join(tempfile.gettempdir(), f"reservoirpy-temp-{uuid.uuid4()}")
if not os.path.exists(_TEMPDIR):  # pragma: no cover
    try:
        os.mkdir(_TEMPDIR)
    except OSError:
        _TEMPDIR = tempfile.gettempdir()

__all__ = [
    "__version__",
    "mat_gen",
    "activationsfunc",
    "observables",
    "hyper",
    "nodes",
    "Node",
    "Model",
    "set_seed",
    "ESN",
    "type",
    "save",
    "load",
]


def __getattr__(attr):
    if attr in ["experimental", "verbosity", "compat"]:
        raise type.DeprecatedError(f"reservoirpy.{attr} has been removed in ReservoirPy v0.4.0.")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
