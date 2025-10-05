"""
====================================
Node API (:mod:`reservoirpy.node`)
====================================

Note
----

See the following guides to:

- **Learn more about how to work with ReservoirPy Nodes**: :ref:`node`
- **Learn more about how to combine nodes within a Model**: :ref:`model`
- **Learn how to subclass Node to make your own**: :ref:`create_new_node`


**Simple tools for complex reservoir computing architectures.**

The Node API features a simple implementation of computational graphs, similar
to what can be found in other popular deep learning and differentiable calculus
libraries. It is however simplified and made the most flexible possible by
discarding the useless "fully differentiable operations" functionalities. If
you wish to use learning rules making use of chain rule and full
differentiability of all operators, ReservoirPy may not be the tool you need
(actually, the whole paradigm of reservoir computing might arguably not be the
tool you need).

The Node API is composed of a base :py:class:`Node` class that can be described
as a stateful recurrent operator able to manipulate streams of data. A
:py:class:`Node` applies a `_step` function on some data, and then stores the
result in its `state` attribute. The :py:meth:`~.Node._step` operation can be a function
depending on the data, on the current ``state`` vector of the Node and on the Node
parameters.

Nodes can also be connected together to form a :py:class:`~.Model`. Models hold
references to the connected nodes and make data flow from one node to
the next, allowing to create *deep* models and other more complex
architectures and computational graphs.
:py:class:`~.Model` is essentially a subclass of :py:class:`Node`,
that can also be connected to other nodes and models.

See the following guides to:

- **Learn more about how to work with ReservoirPy Nodes**: :ref:`node`
- **Learn more about how to combine nodes within a Model**: :ref:`model`
- **Learn how to subclass Node to make your own**: :ref:`create_new_node`

.. currentmodule:: reservoirpy.node


.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

    Node
    TrainableNode
    OnlineNode
    ParallelNode
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from joblib import Parallel, delayed

from reservoirpy.utils import get_non_defaults
from reservoirpy.utils.data_validation import (
    check_node_input,
    check_timeseries,
    check_timestep,
)

from .type import NodeInput, State, Timeseries, Timestep, is_array, is_multiseries


class Node(ABC):
    """Generic Node ABC

    All Nodes should inherit this class.

    """

    #: True if the Node has been initialized
    initialized: bool = False
    #: Expected dimension of the Node input. Can be None before initialization
    input_dim: int = None
    #: Expected dimension of the Node input. Can be None before initialization
    output_dim: int = None
    #: Current state of the Node. Must have "out" as one of the keys.
    state: State
    #: Optional name of the Node.
    name: Optional[str] = None

    @abstractmethod
    def initialize(
        self,
        x: Union[NodeInput, Timestep],
        y: Optional[Union[NodeInput, Timestep]] = None,
    ):
        """Define input and output dimensions, and instantiate variables.

        Only called once, before fitting or running the node.

        Parameters
        ----------
        x : array of shape (input_dim,) or (timestep, input_dim)
            Input data to the node.
        y : None
            Training data to the node. As it is not a trainable node, ``y`` is
            expected to be ``None``.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _step(self, state: State, x: Timestep) -> State:
        ...  # pragma: no cover

    def step(self, x: Optional[Timestep] = None) -> Timestep:
        """Call the Node function on a single step of data and update
        the state of the Node.

        Parameters
        ----------
        x : array of shape (input_dim,), optional
            One single step of input data. If None, an empty array is used
            instead and the Node is assumed to have an input_dim of 0

        Returns
        -------
        array of shape (output_dim,)
            An output vector.
        """
        # TODO: stateful argument (for every step, run, fit, train, ...)
        # Auto-regressive mode
        if x is None:
            x: Timestep = np.empty((0,))
        check_timestep(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x)

        new_state = self._step(self.state, x)

        self.state = new_state
        return new_state["out"]

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        current_state = state
        n_timesteps = x.shape[-2]

        output = np.empty((n_timesteps, self.output_dim))
        for i, x_step in enumerate(x):
            current_state = self._step(state=current_state, x=x_step)
            output[i] = current_state["out"]

        return current_state, output

    def run(self, x: Optional[NodeInput] = None, iters: Optional[int] = None, workers=1) -> NodeInput:
        """Run the Node on a sequence of data.
        Can update the state of the
        Node several times.

        Parameters
        ----------
        x : array-like of shape ([n_inputs,] timesteps, input_dim) or list of \
                arrays of shape (timesteps, input_dim), optional
            A timeseries, array of shape (timesteps, features), or a sequence of
            timeseries. Input of the Node.
        iters : int, optional
            If ``x`` is ``None``, a dimensionless timeseries of length ``iters``
            is used instead.
        workers : int, default to 1
            Number of workers used for parallelization. If set to -1, all available
            workers (threads or processes) are used.

        Returns
        -------
        array of shape ([n_inputs,] timesteps, output_dim) or list of arrays
            A sequence of output vectors.
        """
        # Auto-regressive mode
        if x is None:
            x = np.empty((iters, 0))
        check_node_input(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x)

        initial_state = self.state

        if is_multiseries(x):
            output = Parallel(n_jobs=workers)(delayed(self._run)(initial_state, timeseries) for timeseries in x)
            states, result = zip(*output)
            final_state = states[-1]

            if is_array(x):
                result = np.array(result)
            else:
                result = list(result)
        else:
            final_state, result = self._run(initial_state, x)

        self.state = final_state
        return result

    def predict(self, x: Optional[NodeInput] = None, iters: Optional[int] = None, workers=1) -> NodeInput:
        """Alias for :py:meth:`~.Node.run`

        Run the Node on a sequence of data.
        Can update the state of the
        Node several times.

        Parameters
        ----------
        x : array-like of shape ([n_inputs,] timesteps, input_dim) or list of
                arrays of shape (timesteps, input_dim), optional
            A sequence of data of shape (timesteps, features).
        iters : int, optional
            If ``x`` is ``None``, a dimensionless timeseries of length ``iters``
            is used instead.
        workers : int, default to 1
            Number of workers used for parallelization. If set to -1, all available
            workers (threads or processes) are used.

        Returns
        -------
        array of shape ([n_inputs,] timesteps, output_dim) or list of arrays
            A sequence of output vectors.
        """
        return self.run(x=x, iters=iters, workers=workers)

    def reset(self) -> State:
        """Reset all Node state

        Returns
        -------
        dict[str, np.array]: previous state of the Node.
        """
        previous_state = self.state
        self.state = {key: np.zeros(val.shape) for key, val in self.state.items()}
        return previous_state

    def _set_input_dim(self, x: Optional[Union[NodeInput, Timestep]]):
        if x is None:
            return
        if isinstance(x, Sequence):
            if len(x) == 0:
                return
            input_dim = x[0].shape[-1]
        else:
            input_dim = x.shape[-1]  # works for both timesteps & timeseries
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                f"Trying to set {self} input_dim to {input_dim} while it has already been set to {self.input_dim}"
            )
        self.input_dim = input_dim

    def _set_output_dim(self, y: Optional[Union[NodeInput, Timestep]]):
        if y is None:
            return
        if isinstance(y, Sequence):
            if len(y) == 0:
                return
            output_dim = y[0].shape[-1]
        else:
            output_dim = y.shape[-1]
        if self.output_dim is not None and self.output_dim != output_dim:
            raise ValueError(
                f"Trying to set {self} input_dim to {output_dim} while it has already been set to {self.output_dim}"
            )
        self.output_dim = output_dim

    def __call__(self, x: Optional[Timestep] = None) -> Timestep:
        return self.step(x)

    def __str__(self):
        # base_name = self.name if self.name is not None else self.__class__.__name__
        base_name = self.__class__.__name__
        arguments = get_non_defaults(self)
        arguments_str = ", ".join(f"{arg}={val}" for arg, val in arguments.items())
        return f"{base_name}({arguments_str})"

    def __repr__(self):
        return self.__str__()

    def __rshift__(self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]) -> "Model":
        """self >> other"""
        from .ops import ModelBuilderUtil, link

        if isinstance(other, int):
            return ModelBuilderUtil(node=self, delay=other, node_is_first=True)

        return link(self, other)

    def __rrshift__(self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]) -> "Model":
        """other >> self"""
        from .ops import ModelBuilderUtil, link

        if isinstance(other, int):
            return ModelBuilderUtil(node=self, delay=other, node_is_first=False)

        return link(other, self)

    def __lshift__(self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]) -> "Model":
        """self << other"""
        from .ops import link_feedback

        return link_feedback(sender=other, receiver=self)

    def __rlshift__(self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]) -> "Model":
        """other << self"""
        from .ops import link_feedback

        return link_feedback(sender=self, receiver=other)

    def __and__(self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]) -> "Model":
        """self & other"""
        from .ops import merge

        return merge(self, other)

    def __rand__(self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]) -> "Model":
        """other & self"""
        from .ops import merge

        return merge(other, self)


class TrainableNode(Node):
    """Node that can be trained.

    :py:class:`reservoirpy.node.TrainableNode` implements the stateful method
    :py:meth:`~.TrainableNode.fit`.

    """

    @abstractmethod
    def fit(self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0) -> "TrainableNode":
        """Offline fitting method of a Node.

        Parameters
        ----------
        x : list or array-like of shape ([series, ] timesteps, input_dim), optional
            Input sequences dataset.
        y : list or array-like of shape ([series], timesteps, output_dim), optional
            Teacher signals dataset. If None, the method will try to fit the
            Node in an unsupervised way, if possible.
        warmup : int, default to 0
            Number of timesteps to consider as warmup and
            discard at the beginning of each timeseries before training.

        Returns
        -------
        Node
            Node trained offline.
        """
        ...  # pragma: no cover


class OnlineNode(TrainableNode):
    """Node that can be trained in an online fashion.

    :py:class:`reservoirpy.node.OnlineNode` implements the stateful method
    :py:meth:`~.OnlineNode.partial_fit` and :py:meth:`~.OnlineNode._learning_step`.

    """

    @abstractmethod
    def _learning_step(self, x: Timestep, y: Optional[Timestep]) -> Timestep:
        ...  # pragma: no cover

    def partial_fit(self, x: Timeseries, y: Optional[Timeseries]) -> Timeseries:
        """Fit the Node in an online fashion.

        This method both trains the Node parameters and produce predictions on
        the run. Calling :py:meth:`partial_fit` updates the Node without
        resetting the parameters, unlike :py:meth:`fit`.

        Parameters
        ----------
        x : array-like of shape (timesteps, input_dim)
            Input sequence of data.
        y : array-like of shape (timesteps, output_dim), optional.
            Target sequence of data. If None, the Node will train in an
            unsupervised way, if possible.

        Returns
        -------
        array of shape (timesteps, output_dim)
            All outputs computed during the training.
        """
        check_timeseries(x, expected_dim=self.input_dim)
        if y is not None:
            check_timeseries(y)

        if not self.initialized:
            self.initialize(x, y)

        n_timesteps = x.shape[-2]
        y_pred = np.empty((n_timesteps, self.output_dim))
        if y is not None:
            for i, (x_, y_) in enumerate(zip(x, y)):
                y_pred_ = self._learning_step(x_, y_)
                y_pred[i] = y_pred_
        else:
            for i, x_ in enumerate(x):
                y_pred_ = self._learning_step(x_, None)
                y_pred[i] = y_pred_

        self.state = {"out": y_pred_}
        return y_pred

    def fit(self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0) -> "OnlineNode":
        # Re-initialize in any case
        self.initialize(x, y)
        check_node_input(x, expected_dim=self.input_dim)
        if y is not None:
            check_node_input(y)

        if is_multiseries(x):
            for i in range(len(x)):
                yi = y[i][warmup:] if y is not None else None
                _y_pred_current = self.partial_fit(x[i][warmup:], yi)
        else:
            _y_pred = self.partial_fit(x[warmup:], y[warmup:] if y is not None else None)

        return self


class ParallelNode(TrainableNode, ABC):
    """Node that can be trained in parallel.

    :py:class:`reservoirpy.node.ParallelNode` implements the methods
    :py:meth:`~.ParallelNode.master` and :py:meth:`~.ParallelNode.worker`.

    """

    @abstractmethod
    def worker(self, x: Timeseries, y: Optional[Timeseries]):
        ...  # pragma: no cover

    @abstractmethod
    def master(self, generator: Iterable):
        ...  # pragma: no cover

    def fit(
        self,
        x: NodeInput,
        y: Optional[NodeInput] = None,
        warmup: int = 0,
        workers: int = 1,
    ) -> "ParallelNode":
        check_node_input(x, expected_dim=self.input_dim)
        if y is not None:
            check_node_input(y)

        if not self.initialized:
            self.initialize(x, y)

        # Multi-series
        if is_multiseries(x):
            parallel_operator = Parallel(n_jobs=workers, return_as="generator")
            if y is None:
                results = parallel_operator(delayed(self.worker)(x_ts[warmup:]) for x_ts in x)
            else:
                results = parallel_operator(
                    delayed(self.worker)(x_ts[warmup:], y_ts[warmup:]) for x_ts, y_ts in zip(x, y)
                )

        # Single timeseries
        else:
            results = (self.worker(x[warmup:], y[warmup:]) for _ in range(1))

        self.master(results)

        return self
