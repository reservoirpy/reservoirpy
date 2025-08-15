# Author: Nathan Trouvain at 22/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeVar, Union

import numpy as np
from scipy.sparse import issparse, sparray

global_dtype = np.float64

# Creating a real type alias (Array1D) and then using it in another alias (Timestep)
# as a str is a trick to both benefits from type checks from pyright and the like
# and also have type aliases in documentation.
# See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_type_aliases
# See also ../docs/source/conf.py:184
Array1D = np.ndarray[tuple[int], np.dtype[np.floating]]
Array2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]
Array3D = np.ndarray[tuple[int, int, int], np.dtype[np.floating]]

Timestep = Union["Array1D"]
Timeseries = Union["Array2D"]
MultiTimeseries = Union["Array3D", "Sequence[Timeseries]"]
NodeInput = Union[Timeseries, MultiTimeseries]
ModelInput = Union[NodeInput, dict[str, NodeInput]]
MappedTimestep = dict[str, Timestep]
ModelTimestep = Union[Timestep, MappedTimestep]
State = dict[str, np.ndarray]
Edge = tuple["Node", int, "Node"]
Buffer = np.ndarray[tuple[int, int], np.dtype[np.floating]]

Weights = Union[np.ndarray, sparray]
Shape = TypeVar("Shape", int, tuple[int, ...])
FeedbackBuffers = dict[Edge, Buffer]


def is_array(obj: Any) -> bool:
    return obj is not None and isinstance(obj, np.ndarray) or issparse(obj)


def is_multiseries(x: Union[NodeInput, Mapping[Any, NodeInput]]) -> bool:
    if isinstance(x, dict):
        return is_multiseries(x[list(x)[0]])
    return (isinstance(x, np.ndarray) and len(x.shape) == 3) or isinstance(x, Sequence)


def get_data_dimension(x: Union[Timestep, Timeseries, MultiTimeseries]) -> int:
    if isinstance(x, Sequence):
        if len(x) == 0:
            raise ValueError("Can't get dimension of an empty list.")
        dim = x[0].shape[-1]
    else:
        dim = x.shape[-1]  # works for both timesteps & timeseries

    return dim


def timestep_from_input(x: Union[NodeInput, Timestep]):
    if isinstance(x, Sequence):
        return np.zeros((x[0].shape[-1],))
    else:
        return np.zeros((x.shape[-1],))
