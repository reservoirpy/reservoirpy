import os
import tempfile
import uuid

from . import activationsfunc, compat, hyper, mat_gen, nodes, observables, type
from ._version import __version__
from .compat import load_compat
from .compat.utils.save import load
from .model import Model
from .node import Node
from .ops import link, link_feedback, merge
from .utils import verbosity
from .utils.random import set_seed

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
    "load",
    "Node",
    "Model",
    "link",
    "link_feedback",
    "merge",
    "set_seed",
    "verbosity",
    "load_compat",
]
