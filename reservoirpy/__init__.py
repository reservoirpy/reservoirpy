import os
import logging
import tempfile

from ._version import __version__

from .utils.save import load
from .utils.random import set_seed

from . import mat_gen
from . import observables
from . import regression_models
from . import activationsfunc
from . import hyper

from .node import Node, Model, link, combine

from ._esn import ESN
from ._esn_online import ESNOnline


logger = logging.getLogger(__name__)

_TEMPDIR = os.path.join(tempfile.gettempdir(), "reservoirpy-temp")
if not os.path.exists(_TEMPDIR):
    try:
        os.mkdir(_TEMPDIR)
    except OSError:
        _TEMPDIR = tempfile.gettempdir()

tempfile.tempdir = _TEMPDIR

VERBOSITY = 1


def verbosity(level=None):
    global VERBOSITY
    if isinstance(level, int) and VERBOSITY != level:
        VERBOSITY = level
    return VERBOSITY


def silence_pbar(it, *args, **kwargs):
    return it


__all__ = [
    "__version__",
    "ESN", "ESNOnline",
    "mat_gen", "activationsfunc",
    "observables", "regression_models",
    "hyper", "load", "Node", "Model",
    "link", "combine", "set_seed",
    "verbosity"
]
