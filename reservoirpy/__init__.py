import os
import logging
import tempfile

from ._version import __version__

from .utils import verbosity
from reservoirpy.compat.utils.save import load
from .utils.random import set_seed

from . import mat_gen
from . import observables
from . import activationsfunc
from . import hyper
from . import nodes
from . import compat
from . import types

from .node import Node
from .model import Model
from .ops import link, merge, link_feedback
from .compat import load_compat


logger = logging.getLogger(__name__)

_TEMPDIR = os.path.join(tempfile.gettempdir(), "reservoirpy-temp")
if not os.path.exists(_TEMPDIR):
    try:
        os.mkdir(_TEMPDIR)
    except OSError:
        _TEMPDIR = tempfile.gettempdir()

tempfile.tempdir = _TEMPDIR

__all__ = [
    "__version__",
    "mat_gen", "activationsfunc",
    "observables",
    "hyper", "nodes", "load", "Node", "Model",
    "link", "link_feedback", "merge", "set_seed",
    "verbosity", "load_compat"
    ]
