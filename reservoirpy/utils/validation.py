# Author: Nathan Trouvain at 21/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Any, Mapping

import numpy as np
from scipy.sparse import issparse


def is_sequence_set(seq: Any) -> bool:
    return isinstance(seq, list) or (isinstance(seq, np.ndarray) and seq.ndim > 2)


def is_mapping(obj):
    return isinstance(obj, Mapping) or (
        (hasattr(obj, "items") and hasattr(obj, "get"))
        or (
            not (isinstance(obj, list) or isinstance(obj, tuple))
            and hasattr(obj, "__getitem__")
            and not hasattr(obj, "__array__")
        )
    )


def add_bias(X):
    if isinstance(X, np.ndarray):
        X = np.atleast_2d(X)
        return np.hstack([np.ones((X.shape[0], 1)), X])
    elif isinstance(X, list):
        new_X = []
        for x in X:
            x = np.atleast_2d(x)
            new_X.append(np.hstack([np.ones((x.shape[0], 1)), x]))
        return new_X


def check_vector(array, allow_reshape=True, allow_timespans=True, caller=None):
    if caller is not None:
        msg = f" in {caller.name if hasattr(caller, 'name') else caller}."

    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"Data type '{type(array)}' not understood. All vectors "
            f"should be Numpy arrays."
        )

    if allow_reshape:
        array = np.atleast_2d(array)

    if not allow_timespans:
        if array.shape[0] > 1:
            msg = (
                f"Impossible to operate on multiple timesteps. Data should"
                f" have shape (1, n) but is {array.shape}."
            )
            raise ValueError(msg)

    return array
