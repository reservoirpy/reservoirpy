import os
import tempfile
import uuid

from . import activationsfunc, hyper, mat_gen, nodes, observables, type
from ._version import __version__
from .model import Model
from .node import Node
from .utils.random import set_seed

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
    "type",
]
