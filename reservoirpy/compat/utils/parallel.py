# Author: Nathan Trouvain at 24/02/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import os
import uuid
from functools import partial
from multiprocessing import Manager, Process
from typing import Tuple

import joblib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ...utils.parallel import clean_tempfile, get_joblib_backend, temp_registry


def parallelize(
    esn,
    func,
    workers,
    lengths,
    return_states,
    pbar_text=None,
    verbose=False,
    **func_kwargs,
):
    workers = min(len(lengths), workers)
    backend = get_joblib_backend() if workers > 1 or workers == -1 else "sequential"

    steps = np.sum(lengths)
    ends = np.cumsum(lengths)
    starts = ends - np.asarray(lengths)

    fn_kwargs = (
        {k: func_kwargs[k][i] for k in func_kwargs.keys()} for i in range(len(lengths))
    )

    states = None
    if return_states:
        shape = (steps, esn.N)
        states = memmap(shape, dtype=esn.typefloat, caller=esn)

    with ParallelProgressQueue(total=steps, text=pbar_text, verbose=verbose) as pbar:

        func = partial(func, pbar=pbar)

        with Parallel(backend=backend, n_jobs=workers) as parallel:

            def func_wrapper(states, start_pos, end_pos, *args, **kwargs):
                s = func(*args, **kwargs)

                out = None
                # if function returns states and outputs
                if hasattr(s, "__len__") and len(s) == 2:
                    out = s[0]  # outputs are always returned first
                    s = s[1]

                if return_states:
                    states[start_pos:end_pos] = s[:]

                return out

            outputs = parallel(
                delayed(func_wrapper)(states, start, end, **kwargs)
                for start, end, kwargs in zip(starts, ends, fn_kwargs)
            )

    if return_states:
        states = [np.array(states[start:end]) for start, end in zip(starts, ends)]

    clean_tempfile(esn)

    return outputs, states


class ParallelProgressQueue:
    def __init__(self, total, text, verbose):
        self._verbose = verbose
        if verbose is True:
            self._queue = Manager().Queue()
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


def memmap(shape: Tuple, dtype: np.dtype, mode: str = "w+", caller=None) -> np.memmap:
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
    global temp_registry
    from ... import _TEMPDIR

    if caller is not None:
        caller_name = caller.__class__.__name__
    else:
        caller_name = ""
    filename = os.path.join(_TEMPDIR, f"{caller_name + str(uuid.uuid4())}.dat")
    if caller is not None:
        temp_registry[caller].append(filename)
    return np.memmap(filename, shape=shape, mode=mode, dtype=dtype)


def as_memmap(data, caller=None):
    global temp_registry
    from ... import _TEMPDIR

    if caller is not None:
        caller_name = caller.__class__.__name__
    else:
        caller_name = ""
    filename = os.path.join(_TEMPDIR, f"{caller_name + str(uuid.uuid4())}.dat")
    joblib.dump(data, filename)
    temp_registry[caller].append(filename)
    return joblib.load(filename, mmap_mode="r+")
