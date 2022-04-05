# Author: Nathan Trouvain at 21/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numbers
from typing import Any, Mapping

import numpy as np
from scipy.sparse import issparse


def is_sequence_set(seq: Any) -> bool:
    return isinstance(seq, list) or (isinstance(seq, np.ndarray) and seq.ndim > 2)


def is_array(obj: Any) -> bool:
    return obj is not None and isinstance(obj, np.ndarray) or issparse(obj)


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
    msg = "."
    if caller is not None:
        if hasattr(caller, "name"):
            msg = f" in {caller.name}."
        else:
            msg = f"in {caller}."

    if not isinstance(array, np.ndarray):
        # maybe a single number, make it an array
        if isinstance(array, numbers.Number):
            array = np.asarray(array)
        else:
            msg = (
                f"Data type '{type(array)}' not understood. All vectors "
                f"should be Numpy arrays" + msg
            )
            raise TypeError(msg)

    if not (np.issubdtype(array.dtype, np.number)):
        msg = f"Impossible to operate on non-numerical data, in array: {array}" + msg
        raise TypeError(msg)

    if allow_reshape:
        array = np.atleast_2d(array)

    if not allow_timespans:
        if array.shape[0] > 1:
            msg = (
                f"Impossible to operate on multiple timesteps. Data should"
                f" have shape (1, n) but is {array.shape}" + msg
            )
            raise ValueError(msg)

    # TODO: choose axis to expand and/or np.atleast_2d

    return array
