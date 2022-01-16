# Author: Nathan Trouvain at 22/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import sys
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Iterable, Iterator, Optional, Tuple,
                    TypeVar, Union, Sequence, List)
from typing import overload

if sys.version_info < (3, 8):
    from typing_extensions import Protocol, runtime_checkable
else:
    from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

global_dtype = np.float64
global_ctype = "d"

Weights = TypeVar("Weights", np.ndarray, csr_matrix, csc_matrix, coo_matrix)
Shape = Tuple[int, ...]

Data = TypeVar("Data", Iterable[np.ndarray], np.ndarray)
MappedData = TypeVar("MappedData",
                     Iterable[np.ndarray], np.ndarray,
                     Dict[str, Iterable[np.ndarray]],
                     Dict[str, np.ndarray])


@runtime_checkable
class GenericNode(Protocol):
    """Node base Protocol class for type checking and interface inheritance."""

    _factory_id: int = -1
    _registry: List = list()
    _name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._factory_id = -1
        cls._registry = list()

    def __repr__(self):
        klas = type(self).__name__
        hypers = [(str(k), str(v)) for k, v in self._hypers.items()]
        all_params = ["=".join((k, v)) for k, v in hypers]
        all_params += [f"in={self.input_dim}", f"out={self.output_dim}"]
        return f"'{self.name}': {klas}(" + ", ".join(all_params) + ")"

    def __setstate__(self, state):
        curr_name = state.get("name")
        if curr_name in type(self)._registry:
            new_name = curr_name + "-(copy)"
            state["name"] = new_name
        self.__dict__ = state

    def __del__(self):
        try:
            type(self)._registry.remove(self._name)
        except ValueError:
            pass

    def __getattr__(self, item):
        if item in ["_params", "_hypers"]:
            raise AttributeError()
        if item in self._params:
            return self._params.get(item)
        elif item in self._hypers:
            return self._hypers.get(item)
        else:
            raise AttributeError()

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.call(*args, **kwargs)

    def __rshift__(self, other: Union[
        "GenericNode", Sequence["GenericNode"]]) -> "GenericNode":
        return self.link(other)

    def __rrshift__(self, other: Union[
        "GenericNode", Sequence["GenericNode"]]) -> "GenericNode":
        from .ops import link
        return link(other, self)

    def __and__(self, other: Union[
        "GenericNode", Sequence["GenericNode"]]) -> "GenericNode":
        from .ops import merge
        return merge(self, other)

    def _get_name(self, name=None):
        if name is None:
            type(self)._factory_id += 1
            _id = self._factory_id
            name = f"{type(self).__name__}-{_id}"

        if name in type(self)._registry:
            raise NameError(f"Name '{name}' is already taken "
                            f"by another node. Node names should "
                            f"be unique.")

        type(self)._registry.append(name)
        return name

    @property
    def name(self) -> str: return self._name
    """Name of the Node or Model."""

    @property
    def params(self) -> Dict[str, Any]: return self._params
    """Parameters of the Node or Model."""

    @property
    def hypers(self) -> Dict[str, Any]: return self._hypers
    """Hyperparameters of the Node or Model."""

    @property
    def is_initialized(self) -> bool: return self._is_initialized

    @property
    def input_dim(self) -> Shape: ...

    @property
    def output_dim(self) -> Shape: ...

    @property
    def is_trained_offline(self) -> bool: ...

    @property
    def is_trained_online(self) -> bool: ...

    @property
    def is_trainable(self) -> bool: ...

    @property
    def fitted(self) -> bool: ...

    @is_trainable.setter
    def is_trainable(self, value: bool): ...

    def copy(self, name: str = None, copy_feedback: bool = False,
             shallow: bool = False) -> "GenericNode": ...

    @overload
    def state(self) -> Optional[Dict[str, np.ndarray]]: ...

    def state(self) -> Optional[np.ndarray]: ...

    def set_input_dim(self, value: Union[int, Shape]): ...

    def set_output_dim(self, value: Union[int, Shape]): ...

    def set_feedback_dim(self, value: Union[int, Shape]): ...

    def get_param(self, name: str) -> Any:
        if name in self._params:
            return self._params.get(name)
        elif name in self._hypers:
            return self._hypers.get(name)
        else:
            raise NameError(f"No parameter named '{name}' "
                            f"found in node {self}")

    def set_param(self, name: str, value: Any): ...

    def initialize(self, x: MappedData = None, y: MappedData = None): ...

    def reset(self, to_state: np.ndarray = None) -> "GenericNode": ...

    @contextmanager
    def with_state(self, state=None, stateful=False, reset=False) -> Iterator[
        "GenericNode"]: ...

    @contextmanager
    def with_feedback(self, feedback=None, stateful=False, reset=False) -> \
            Iterator["GenericNode"]: ...

    def link(self, other: "GenericNode", name: str = None) -> "GenericNode":
        """Link the Node to another Node or Model.

        Parameters
        ----------
        other: GenericNode
            Other Node or Model to link this Node with.
        name: str, optional
            Name for the created Model.

        Returns
        -------
        Model
            A new Model instance including the two Nodes.
        """
        from .ops import link
        return link(self, other, name=name)

    @overload
    def call(self, x: Data, from_state: np.ndarray = None,
             stateful: bool = True,
             reset: bool = False) -> np.ndarray: ...

    def call(self, x: MappedData, forced_feedback: MappedData = None,
             from_state: MappedData = None,
             stateful: bool = True, reset: bool = False,
             return_states: Iterable[str] = None) -> MappedData: ...

    @overload
    def run(self, X: Data = None, from_state: np.ndarray = None,
            stateful: bool = True,
            reset: bool = False) -> Data: ...

    def run(self, X: MappedData = None, forced_feedbacks: MappedData = None,
            from_state: MappedData = None,
            stateful: bool = True, reset: bool = False, shift_fb: bool = True,
            return_states: Iterable[str] = None) -> MappedData: ...

    @overload
    def train(self, X: Data, Y: Data = None, force_teachers: bool = True,
              call: bool = True,
              learn_every: int = 1, from_state: np.ndarray = None,
              stateful: bool = True, reset: bool = False) -> Data: ...

    def train(self, X: MappedData, Y: MappedData = None,
              force_teachers: bool = True, call: bool = True,
              learn_every: int = 1,
              from_state: MappedData = None, stateful: bool = True,
              reset: bool = False,
              return_states: Iterable[str] = None) -> MappedData: ...

    @overload
    def fit(self, X: Data = None, Y: Data = None) -> "GenericNode": ...

    def fit(self, X: MappedData = None, Y: MappedData = None,
            from_state: MappedData = None, stateful: bool = True,
            reset: bool = False) -> "GenericNode": ...

Activation = Callable[[np.ndarray], np.ndarray]
ForwardFn = Callable[[GenericNode, Data], np.ndarray]
BackwardFn = Callable[[GenericNode, Optional[Data], Optional[Data]], None]
PartialBackFn = Callable[[GenericNode, Data, Optional[Data]], None]
ForwardInitFn = Callable[
    [GenericNode, Optional[Data], Optional[Data]], None]
EmptyInitFn = Callable[[GenericNode], None]
