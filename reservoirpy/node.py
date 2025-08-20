"""
====================================
Node API (:class:`reservoirpy.Node`)
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
result in its `state` attribute. The `_step` operation can be a function
depending on the data, on the current `state` vector of the Node and on the Node
parameters.

Nodes can also be connected together to form a :py:class:`Model`. Models hold
references to the connected nodes and make data flow from one node to
the next, allowing to create *deep* models and other more complex
architectures and computational graphs.
:py:class:`Model` is essentially a subclass of :py:class:`Node`,
that can also be connected to other nodes and models.

See the following guides to:

- **Learn more about how to work with ReservoirPy Nodes**: :ref:`node`
- **Learn more about how to combine nodes within a Model**: :ref:`model`
- **Learn how to subclass Node to make your own**: :ref:`create_new_node`

.. currentmodule:: reservoirpy.node

.. autoclass:: Node

   .. rubric:: Methods

   .. autosummary::

      ~Node.step
      ~Node.run
      ~Node.predict
      ~Node.fit
      ~Node.initialize
      ~Node.partial_fit
      ~Node.reset

   .. rubric:: Attributes

   .. autosummary::

      ~Node.initialized
      ~Node.input_dim
      ~Node.output_dim
      ~Node.state
      ~Node.name

"""

from abc import ABC, abstractmethod
from itertools import repeat
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from joblib import Parallel, delayed

from reservoirpy.utils.data_validation import check_node_input, check_timestep

from .type import NodeInput, State, Timeseries, Timestep, is_multiseries


class Node(ABC):
    #: True if the Node has been initialized
    initialized: bool
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

    def run(
        self, x: Optional[NodeInput] = None, iters: Optional[int] = None
    ) -> NodeInput:
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
            result = []
            # TODO: parallelization
            for timeseries in x:
                final_state, output = self._run(initial_state, timeseries)
                result.append(output)

            if isinstance(x, np.ndarray):
                result = np.array(result)
        else:
            final_state, result = self._run(initial_state, x)

        self.state = final_state
        return result

    def predict(
        self, x: Optional[NodeInput] = None, iters: Optional[int] = None
    ) -> NodeInput:
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

        Returns
        -------
        array of shape ([n_inputs,] timesteps, output_dim) or list of arrays
            A sequence of output vectors.
        """
        return self.run(x=x, iters=iters)

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

    def __repr__(self):
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__

    def __rshift__(
        self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]
    ) -> "Model":
        from .ops import link

        return link(self, other)

    def __rrshift__(
        self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]
    ) -> "Model":
        from .ops import link

        return link(other, self)

    def __lshift__(
        self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]
    ) -> "Model":
        from .ops import link_feedback

        return link_feedback(sender=other, receiver=self)

    def __rlshift__(
        self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]
    ) -> "Model":
        from .ops import link_feedback

        return link_feedback(sender=self, receiver=other)

    def __and__(
        self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]
    ) -> "Model":
        from .ops import merge

        return merge(self, other)

    def __rand__(
        self, other: Union["Node", "Model", Sequence[Union["Node", "Model"]]]
    ) -> "Model":
        from .ops import merge

        return merge(other, self)


class TrainableNode(Node):
    @abstractmethod
    def fit(
        self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0
    ) -> "TrainableNode":
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
    @abstractmethod
    def _learning_step(self, x: Timestep, y: Optional[Timestep]) -> Timestep:
        ...  # pragma: no cover

    @abstractmethod
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
        ...  # pragma: no cover

    def fit(
        self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0
    ) -> "OnlineNode":
        # Re-initialize in any case
        self.initialize(x, y)
        check_node_input(x, expected_dim=self.input_dim)
        if y is not None:
            check_node_input(y)

        if is_multiseries(x):
            if y is None:
                y = repeat(None)
            for x_ts, y_ts in zip(x, y):
                _y_pred_current = self.partial_fit(
                    x_ts[warmup:], None if y_ts is None else y_ts[warmup:]
                )
        else:
            _y_pred = self.partial_fit(x[warmup:], None if y is None else y[warmup:])

        return self


class ParallelNode(TrainableNode, ABC):
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
                results = parallel_operator(delayed(self.worker)(x_ts) for x_ts in x)
            else:
                results = parallel_operator(
                    delayed(self.worker)(x_ts[warmup:], y_ts[warmup:])
                    for x_ts, y_ts in zip(x, y)
                )

        # Single timeseries
        else:
            results = (self.worker(x[warmup:], y[warmup:]) for _ in range(1))

        self.master(results)

        return self
