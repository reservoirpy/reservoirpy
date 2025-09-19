# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from __future__ import annotations

from typing import Sequence, TypeVar, Union

import jax
import numpy as np
from scipy.sparse import sparray

from reservoirpy.type import get_data_dimension, is_array, is_multiseries

global_dtype = jax.numpy.float64

# Creating a real type alias (Array1D) and then using it in another alias (Timestep)
# as a str is a trick to both benefits from type checks from pyright and the like
# and also have type aliases in documentation.
# See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_type_aliases
# See also ../docs/source/conf.py:184
Array1D = Union[np.ndarray[tuple[int], np.dtype[np.floating]], jax.Array]
Array2D = Union[np.ndarray[tuple[int, int], np.dtype[np.floating]], jax.Array]
Array3D = Union[np.ndarray[tuple[int, int, int], np.dtype[np.floating]], jax.Array]

Timestep = Union["Array1D"]
Timeseries = Union["Array2D"]
MultiTimeseries = Union["Array3D", "Sequence[Timeseries]"]
NodeInput = Union[Timeseries, MultiTimeseries]
ModelInput = Union[NodeInput, dict[str, NodeInput]]
MappedTimestep = dict[str, Timestep]
ModelTimestep = Union[Timestep, MappedTimestep]
State = dict[str, jax.Array]
Edge = tuple["Node", int, "Node"]
Buffer = jax.Array

Weights = Union[np.ndarray, sparray, jax.Array]
Shape = TypeVar("Shape", int, tuple[int, ...])
FeedbackBuffers = dict[Edge, Buffer]
