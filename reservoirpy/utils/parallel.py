# Author: Nathan Trouvain at 19/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import gc
import os
import sys
import tempfile
import uuid
from collections import defaultdict

import numpy as np

from ..type import global_dtype

_AVAILABLE_BACKENDS = ("loky", "multiprocessing", "threading", "sequential")

# FIX waiting for a workaround to avoid crashing with multiprocessing
# activated with Python < 3.8. Seems to be due to compatibility issues
# with pickle5 protocol and loky library.
if sys.version_info < (3, 8):
    _BACKEND = "sequential"
else:
    _BACKEND = "loky"


def get_joblib_backend(workers=-1, backend=None):
    if backend is not None:
        if sys.version_info < (3, 8):
            return "sequential"
        if backend in _AVAILABLE_BACKENDS:
            return backend
        else:
            raise ValueError(
                f"'{backend}' is not a Joblib backend. Available "
                f"backends are {_AVAILABLE_BACKENDS}."
            )
    return _BACKEND if workers > 1 or workers == -1 else "sequential"


def set_joblib_backend(backend):
    global _BACKEND
    if backend in _AVAILABLE_BACKENDS:
        _BACKEND = backend
    else:
        raise ValueError(
            f"'{backend}' is not a valid joblib "
            f"backend value. Available backends are "
            f"{_AVAILABLE_BACKENDS}."
        )
