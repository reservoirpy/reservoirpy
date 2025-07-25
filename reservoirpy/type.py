# Author: Nathan Trouvain at 22/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Any, Iterable, Sequence, TypeVar, Union

import numpy as np
from scipy.sparse import issparse, sparray

global_dtype = np.float64

Shape1D = tuple[int]
Shape2D = tuple[int, int]
Shape3D = tuple[int, int, int]
NodeName = str

Timestep = np.ndarray[Shape1D, np.dtype[np.floating]]
Timeseries = np.ndarray[Shape2D, np.dtype[np.floating]]
MultiTimeseries = Union[
    np.ndarray[Shape3D, np.dtype[np.floating]],
    Sequence[Timeseries],
]

NodeInput = Union[Timeseries, MultiTimeseries]
ModelInput = Union[NodeInput, dict[str, NodeInput]]
MappedTimestep = dict[str, Timestep]
ModelTimestep = Union[Timestep, MappedTimestep]
State = dict[str, np.ndarray]

Weights = Union[np.ndarray, sparray]
Shape = TypeVar("Shape", int, tuple[int, ...])
Data = TypeVar("Data", Iterable[np.ndarray], np.ndarray)


def is_array(obj: Any) -> bool:
    return obj is not None and isinstance(obj, np.ndarray) or issparse(obj)


def is_multiseries(x: ModelInput) -> bool:
    if isinstance(x, dict):
        return is_multiseries(x[list(x)[0]])
    return (isinstance(x, np.ndarray) and len(x.shape) == 3) or isinstance(x, Sequence)


def timestep_from_input(x: Union[NodeInput, Timestep]):
    if isinstance(x, Sequence):
        return np.zeros((x[0].shape[-1],))
    else:
        return np.zeros((x.shape[-1],))
