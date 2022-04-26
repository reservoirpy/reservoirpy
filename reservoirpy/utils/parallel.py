# Author: Nathan Trouvain at 19/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import sys
import warnings

_AVAILABLE_BACKENDS = ("loky", "multiprocessing", "threading", "sequential")

# FIX waiting for a workaround to avoid crashing with multiprocessing
# activated with Python < 3.8. Seems to be due to compatibility issues
# with pickle5 protocol and loky library.
if sys.version_info < (3, 8):
    _BACKEND = "sequential"
    _JOBS = 1
else:
    _BACKEND = "loky"
    _JOBS = -1


def get_joblib_backend(workers=-1, backend=None):
    if backend is not None:
        if sys.version_info < (3, 8):
            warnings.warn(
                "joblbib multiprocessing/loky backend deactivated with "
                "Python<3.8 due to compatibility issues. Backend set to "
                "'sequential' by default."
            )
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


def set_joblib_n_jobs(n_jobs):
    global _JOBS
    if n_jobs not in [1, 0]:
        if sys.version_info < (3, 8):
            _JOBS = 1
            warnings.warn(
                "joblbib multiprocessing/loky backend deactivated with "
                "Python<3.8 due to compatibility issues. n_jobs set to "
                "1 by default."
            )
            return
    _JOBS = n_jobs
    return


def get_joblib_n_jobs():
    return _JOBS
