# Author: Nathan Trouvain at 01/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Any
from collections.abc import Sequence, Mapping

import numpy as np
from scipy.sparse import issparse

from .._types import Weights


DTYPES = (np.float32, np.float64)


def check_matrix(matrix: Any) -> Weights:

    if not isinstance(matrix, np.ndarray):
        if issparse(matrix):
            checked_matrix = matrix
        else:
            checked_matrix = np.asanyarray(matrix)
    else:
        checked_matrix = matrix

    if checked_matrix.dtype not in DTYPES:
        if not issparse(checked_matrix):
            try:
                checked_matrix = checked_matrix.astype(float)
            except Exception:
                raise ValueError("Trying to use non numeric array (of dtype "
                                 f"{matrix.dtype}) : {matrix}")
        else:
            raise ValueError("Trying to use non numeric array (of dtype "
                             f"{matrix.dtype}) : {matrix}")

    if checked_matrix.ndim == 1:
        checked_matrix = checked_matrix.reshape(1, -1)
    elif not is_2d(checked_matrix):
        raise ValueError("Trying to use array that is more than "
                         f"2-dimensional ({matrix.shape}): {matrix}")
    return checked_matrix


def is_square(array: Weights) -> bool:
    return array.shape[0] == array.shape[1] and is_2d(array)


def is_2d(array: Weights) -> bool:
    return array.ndim == 2


def is_object_sequence(seq: Any) -> bool:
    return isinstance(seq, Sequence) and not isinstance(seq, str)


def is_array(obj: Any) -> bool:
    return obj is not None and isinstance(obj, np.ndarray) or issparse(obj)


def is_mapping(obj):
    return isinstance(obj, Mapping) or hasattr(obj, "__getitem__")


def is_numerical(obj):
    return (hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.number)) \
           or isinstance(obj, int) \
           or isinstance(obj, float)
