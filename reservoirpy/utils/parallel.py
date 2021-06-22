# Author: Nathan Trouvain at 19/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import os
import tempfile
import uuid
import warnings

from collections import defaultdict
from typing import Tuple
from multiprocessing import Manager, Process

import numpy as np
import joblib

from tqdm import tqdm

manager = Manager()
lock = manager.Lock()

_BACKEND = "loky"
_AVAILABLE_BACKENDS = ["loky", "multiprocessing", "threading"]

temp_registry = defaultdict(list)


def get_joblib_backend(workers=-1):
    return _BACKEND if workers > 1 or workers == -1 else "sequential"


def set_joblib_backend(backend):
    if backend in _AVAILABLE_BACKENDS:
        _BACKEND = backend
    else:
        raise ValueError(f"'{backend}' is not a valid joblib "
                         f"backend value. Available backends are "
                         f"{_AVAILABLE_BACKENDS}.")


def as_memmap(data, caller=None):
    if caller is not None:
        caller_name = caller.__class__.__name__
    else:
        caller_name = ""
    filename = os.path.join(tempfile.gettempdir(), f"{caller_name + str(uuid.uuid4())}.dat")
    joblib.dump(data, filename)
    temp_registry[caller].append(filename)
    return joblib.load(filename, mmap_mode="r+")


def memmap(shape: Tuple, dtype: np.dtype,
           mode: str = "w+", caller=None) -> np.memmap:
    """Create a new numpy.memmap object, stored in a temporary
    folder on disk.

    Parameters
    ----------
    shape : tuple of int
        Shape of the memmaped array.
    directory: str:
        Directory where the memmap will be stored.
    dtype : numpy.dtype
        Data type of the array.
    mode : {‘r+’, ‘r’, ‘w+’, ‘c’}, optional
        Mode in which to open the memmap file. See `Numpy documentation
        <https://numpy.org/doc/stable/reference/generated/numpy.memmap.html>`_
        for more information.

    Returns
    -------
        numpy.memmap
            An empty memory-mapped array.

    """
    if caller is not None:
        caller_name = caller.__class__.__name__
    else:
        caller_name = ""
    filename = os.path.join(tempfile.gettempdir(),  f"{caller_name + str(uuid.uuid4())}.dat")
    if caller is not None:
        temp_registry[caller].append(filename)
    return np.memmap(filename, shape=shape, mode=mode, dtype=dtype)


def clean_tempfile(caller):
    for file in temp_registry[caller]:
        try:
            os.remove(file)
        except OSError:
            pass


class ParallelProgressQueue:

    def __init__(self, total, text, verbose):
        self._verbose = verbose
        if verbose is True:
            self._queue = manager.Queue()
            self._process = Process(target=self._listen)
        else:
            self._queue = None
            self._process = None

        self._total = total
        self._text = text

    def __enter__(self):
        if self._verbose:
            self._process.start()
        return _ProgressBarQueue(self._queue, self._verbose)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._verbose:
            self._queue.put(None)
            self._process.join()

    def _listen(self):
        with tqdm(total=self._total, desc=self._text) as pbar:
            for i in iter(self._queue.get, None):
                pbar.update(i)


class _ProgressBarQueue:

    def __init__(self, queue, verbose):
        self._queue = queue
        self._verbose = verbose

    def update(self, value):
        if self._verbose:
            self._queue.put(value)
