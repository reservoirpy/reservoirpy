# Author: Nathan Trouvain at 22/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from scipy.sparse import issparse, sparray, spmatrix

global_dtype = np.float64

Shape1D = Tuple[int]
Shape2D = Tuple[int, int]
Shape3D = Tuple[int, int, int]
NodeName = str

Timestep = np.ndarray[Shape1D, np.dtype[np.floating]]
Timeseries = np.ndarray[Shape2D, np.dtype[np.floating]]
MultiTimeseries = Union[
    np.ndarray[Shape3D, np.dtype[np.floating]],
    Sequence[Timeseries],
]

NodeInput = Timeseries | MultiTimeseries

Weights = TypeVar("Weights", np.ndarray, spmatrix, sparray)
Shape = TypeVar("Shape", int, Tuple[int, ...])
Data = TypeVar("Data", Iterable[np.ndarray], np.ndarray)
MappedData = TypeVar(
    "MappedData",
    Iterable[np.ndarray],
    np.ndarray,
    Dict[str, Iterable[np.ndarray]],
    Dict[str, np.ndarray],
)


def is_array(obj: Any) -> bool:
    return obj is not None and isinstance(obj, np.ndarray) or issparse(obj)


def is_multiseries(x: NodeInput):
    return (isinstance(x, np.ndarray) and len(x.shape) == 3) or isinstance(x, Sequence)
