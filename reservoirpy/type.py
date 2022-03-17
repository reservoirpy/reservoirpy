# Author: Nathan Trouvain at 22/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

global_dtype = np.float64
global_ctype = "d"

Weights = TypeVar("Weights", np.ndarray, csr_matrix, csc_matrix, coo_matrix)
Shape = TypeVar("Shape", int, Tuple[int, ...])
Data = TypeVar("Data", Iterable[np.ndarray], np.ndarray)
MappedData = TypeVar(
    "MappedData",
    Iterable[np.ndarray],
    np.ndarray,
    Dict[str, Iterable[np.ndarray]],
    Dict[str, np.ndarray],
)


class NodeType(Protocol):
    """Node base Protocol class for type checking and interface inheritance."""

    name: str
    params: Dict[str, Any]
    hypers: Dict[str, Any]
    is_initialized: bool
    input_dim: Shape
    output_dim: Shape
    is_trained_offline: bool
    is_trained_online: bool
    is_trainable: bool
    fitted: bool

    def __call__(self, *args, **kwargs) -> np.ndarray:
        ...

    def __rshift__(self, other: Union["NodeType", Sequence["NodeType"]]) -> "NodeType":
        ...

    def __rrshift__(self, other: Union["NodeType", Sequence["NodeType"]]) -> "NodeType":
        ...

    def __and__(self, other: Union["NodeType", Sequence["NodeType"]]) -> "NodeType":
        ...

    def get_param(self, name: str) -> Any:
        ...

    def initialize(self, x: MappedData = None, y: MappedData = None):
        ...

    def reset(self, to_state: np.ndarray = None) -> "NodeType":
        ...

    def with_state(
        self, state=None, stateful=False, reset=False
    ) -> Iterator["NodeType"]:
        ...

    def with_feedback(
        self, feedback=None, stateful=False, reset=False
    ) -> Iterator["NodeType"]:
        ...


Activation = Callable[[np.ndarray], np.ndarray]
ForwardFn = Callable[[NodeType, Data], np.ndarray]
BackwardFn = Callable[[NodeType, Optional[Data], Optional[Data]], None]
PartialBackFn = Callable[[NodeType, Data, Optional[Data]], None]
ForwardInitFn = Callable[[NodeType, Optional[Data], Optional[Data]], None]
EmptyInitFn = Callable[[NodeType], None]
