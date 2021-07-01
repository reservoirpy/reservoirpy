import os
import tempfile

from ._version import __version__

from .utils.save import load
from .utils.parallel import get_joblib_backend, set_joblib_backend

from . import mat_gen
from . import observables
from . import regression_models
from . import activationsfunc
from . import hyper

from ._esn import ESN
from ._esn_online import ESNOnline

_TEMPDIR = os.path.join(tempfile.gettempdir(), "reservoirpy-temp")
if not os.path.exists(_TEMPDIR):
    try:
        os.mkdir(_TEMPDIR)
    except OSError:
        _TEMPDIR = tempfile.gettempdir()

tempfile.tempdir = _TEMPDIR

__all__ = [
    "__version__",
    "ESN", "ESNOnline",
    "mat_gen", "activationsfunc",
    "observables", "regression_models",
    "hyper", "load"
]
