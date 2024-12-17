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
:py:class:`Node` applies a `forward` function on some data, and then stores the
result in its `state` attribute. The `forward` operation can be a function
depending on the data, on the current `state` vector of the Node.

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

      ~Node.call
      ~Node.clean_buffers
      ~Node.copy
      ~Node.create_buffer
      ~Node.fit
      ~Node.get_buffer
      ~Node.get_param
      ~Node.initialize
      ~Node.initialize_buffers
      ~Node.partial_fit
      ~Node.reset
      ~Node.run
      ~Node.set_buffer
      ~Node.set_input_dim
      ~Node.set_output_dim
      ~Node.set_param
      ~Node.state
      ~Node.train
      ~Node.with_state
      ~Node.zero_state

   .. rubric:: Attributes

   .. autosummary::

      ~Node.fitted
      ~Node.hypers
      ~Node.input_dim
      ~Node.is_initialized
      ~Node.is_trainable
      ~Node.is_trained_offline
      ~Node.is_trained_online
      ~Node.output_dim
      ~Node.params

References
==========

    ReservoirPy Node API was heavily inspired by Explosion.ai *Thinc*
    functional deep learning library [1]_, and *Nengo* core API [2]_.
    It also follows some *scikit-learn* schemes and guidelines [3]_.

    .. [1] `Thinc <https://thinc.ai/>`_ website
    .. [2] `Nengo <https://www.nengo.ai/>`_ website
    .. [3] `scikit-learn <https://scikit-learn.org/stable/>`_ website

"""

# Author: Nathan Trouvain at 22/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.sparse import issparse

from ._base import _Node, call, check_one_sequence, check_xy, train
from .type import (
    BackwardFn,
    Data,
    EmptyInitFn,
    ForwardFn,
    ForwardInitFn,
    PartialBackFn,
    Shape,
    global_dtype,
)
from .utils import progress
from .utils.model_utils import to_ragged_seq_set
from .utils.parallel import clean_tempfile, memmap_buffer
from .utils.validation import check_vector


def _init_with_sequences(node, X, Y=None):
    """Initialize a Node with a sequence of inputs/targets."""
    X = to_ragged_seq_set(X)

    if Y is not None:
        Y = to_ragged_seq_set(Y)
    else:
        Y = [None for _ in range(len(X))]

    if not node.is_initialized:
        node.initialize(X[0], Y[0])

    return X, Y


def _init_vectors_placeholders(node, x, y):
    msg = f"Impossible to initialize node {type(node).__name__}: "
    in_msg = (
        msg + "input_dim is unknown and no input data x was given "
        "to call/run the node."
    )

    x_init = y_init = None
    if isinstance(x, np.ndarray):
        x_init = np.atleast_2d(check_vector(x, caller=node))
    elif isinstance(x, list):
        x_init = list()
        for i in range(len(x)):
            x_init.append(np.atleast_2d(check_vector(x[i], caller=node)))
    elif x is None:
        if node.input_dim is not None:
            if hasattr(node.input_dim, "__iter__"):
                x_init = [np.empty((1, d)) for d in node.input_dim]
            else:
                x_init = np.empty((1, node.input_dim))
        else:
            raise RuntimeError(in_msg)

    if y is not None:
        y_init = np.atleast_2d(check_vector(y, caller=node))
    elif node.output_dim is not None:
        y_init = np.empty((1, node.output_dim))
    else:
        # check if output dimension can be inferred from a teacher node
        if node._teacher is not None and node._teacher.output_dim is not None:
            y_init = np.empty((1, node._teacher.output_dim))

    return x_init, y_init


def _partial_backward_default(node, X_batch, Y_batch=None):
    """By default, for offline learners, partial_fit simply stores inputs and
    targets, waiting for fit to be called."""

    node._X.append(X_batch)

    if Y_batch is not None:
        node._Y.append(Y_batch)

    return


class Node(_Node):
    """Node base class.

    Parameters
    ----------
    params : dict, optional
        Parameters of the Node. Parameters are mutable, and can be modified
        through learning or by the effect of hyperparameters.
    hypers : dict, optional
        Hyperparameters of the Node. Hyperparameters are immutable, and define
        the architecture and properties of the Node.
    forward : callable, optional
        A function defining the computation performed by the Node on some data
        point :math:`x_t`, and that would update the Node internal state from
        :math:`s_t` to :math:`s_{t+1}`.
    backward : callable, optional
        A function defining an offline learning rule, applied on a whole
        dataset, or on pre-computed values stored in buffers.
    partial_backward : callable, optional
        A function defining an offline learning rule, applied on a single batch
        of data.
    train : callable, optional
        A function defining an online learning, applied on a single step of
        a sequence or of a timeseries.
    initializer : callable, optional
        A function called at first run of the Node, defining the dimensions and
        values of its parameters based on the dimension of input data and its
        hyperparameters.
    buffers_initializer : callable, optional
        A function called at the beginning of an offline training session to
        create buffers used to store intermediate results, for batch or
        multi-sequence offline learning.
    input_dim : int
        Input dimension of the Node.
    output_dim : int
        Output dimension of the Node. Dimension of its state.
    name : str
        Name of the Node. It must be a unique identifier.

    See also
    --------
        Model
            Object used to compose node operations and create computational
            graphs.
    """

    _state: Optional[np.ndarray]

    _params: Dict[str, Any]
    _hypers: Dict[str, Any]
    _buffers: Dict[str, Any]

    input_dim: int
    output_dim: int

    _forward: ForwardFn
    _backward: BackwardFn
    _partial_backward: PartialBackFn
    _train: PartialBackFn

    _initializer: ForwardInitFn
    _buffers_initializer: EmptyInitFn

    _trainable: bool
    _fitted: bool

    _X: List  # For partial_fit default behavior (store first, then fit)
    _Y: List

    def __init__(
        self,
        params: Dict[str, Any] = None,
        hypers: Dict[str, Any] = None,
        forward: ForwardFn = None,
        backward: BackwardFn = None,
        partial_backward: PartialBackFn = _partial_backward_default,
        train: PartialBackFn = None,
        initializer: ForwardInitFn = None,
        buffers_initializer: EmptyInitFn = None,
        input_dim: int = None,
        output_dim: int = None,
        dtype: np.dtype = global_dtype,
    ):
        self._params = dict() if params is None else params
        self._hypers = dict() if hypers is None else hypers
        # buffers are all node state components that should not live
        # outside the node training loop, like partial computations for
        # linear regressions. They can also be shared across multiple processes
        # when needed.
        self._buffers = dict()

        self._forward = forward
        self._backward = backward
        self._partial_backward = partial_backward
        self._train = train

        self._initializer = initializer
        self._buffers_initializer = buffers_initializer

        self.input_dim = input_dim
        self.output_dim = output_dim

        self._dtype = dtype

        self._is_initialized = False
        self._teacher = None

        self._trainable = self._backward is not None or self._train is not None

        self._fitted = False if self.is_trainable and self.is_trained_offline else True

        self._X, self._Y = [], []

    def __iand__(self, other):
        raise TypeError(
            f"Impossible to merge nodes in-place: {self} is not a Model instance."
        )

    @property
    def is_initialized(self):
        """Returns if the Node is initialized or not."""
        return self._is_initialized

    @property
    def is_trained_offline(self):
        """Returns if the Node can be fitted offline or not."""
        return self.is_trainable and self._backward is not None

    @property
    def is_trained_online(self):
        """Returns if the Node can be trained online or not."""
        return self.is_trainable and self._train is not None

    @property
    def is_trainable(self):
        """Returns if the Node can be trained."""
        return self._trainable

    @is_trainable.setter
    def is_trainable(self, value: bool):
        """Freeze or unfreeze the Node. If set to False,
        learning is stopped."""
        if self.is_trained_offline or self.is_trained_online:
            if type(value) is bool:
                self._trainable = value
            else:
                raise TypeError("'is_trainable' must be a boolean.")

    @property
    def fitted(self):
        """Returns if the Node parameters have fitted already, using an
        offline learning rule. If the node is trained online, returns True."""
        return self._fitted

    @property
    def dtype(self):
        """Numpy numerical type of node parameters."""
        return self._dtype

    @property
    def unsupervised(self):
        return False

    def state(self) -> Optional[np.ndarray]:
        """Node current internal state.

        Returns
        -------
        array of shape (1, output_dim), optional
            Internal state of the Node.
        """
        if not self.is_initialized:
            return None
        return self._state

    def get_param(self, name: str):
        """Get one of the parameters or hyperparameters given its name."""
        if name in self._params:
            return self._params.get(name)
        elif name in self._hypers:
            return self._hypers.get(name)
        else:
            raise AttributeError(f"No attribute named '{name}' found in node {self}")

    def set_param(self, name: str, value: Any):
        """Set the value of a parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        value : array-like
            Parameter new value.
        """
        if name in self._params:
            if hasattr(value, "dtype"):
                if issparse(value):
                    value.data = value.data.astype(self.dtype)
                else:
                    value = value.astype(self.dtype)
            self._params[name] = value
        elif name in self._hypers:
            self._hypers[name] = value
        else:
            raise KeyError(
                f"No param named '{name}' "
                f"in {type(self).__name__}. Available params are: "
                f"{list(self._params.keys())}."
            )

    def create_buffer(
        self, name: str, shape: Shape = None, data: np.ndarray = None, as_memmap=True
    ):
        """Create a buffer array on disk, using numpy.memmap. This can be
        used to store transient variables on disk. Typically, called inside
        a `buffers_initializer` function.

        Parameters
        ----------
        name : str
            Name of the buffer array.
        shape : tuple of int, optional
            Shape of the buffer array.
        data : array-like
            Data to store in the buffer array.
        """
        if as_memmap:
            self._buffers[name] = memmap_buffer(self, data=data, shape=shape, name=name)
        else:
            if data is not None:
                self._buffers[name] = data
            else:
                self._buffers[name] = np.empty(shape)

    def set_buffer(self, name: str, value: np.ndarray):
        """Dump data in the buffer array.

        Parameters
        ----------
        name : str
            Name of the buffer array.
        value : array-like
            Data to store in the buffer array.
        """
        self._buffers[name][:] = value.astype(self.dtype)

    def get_buffer(self, name) -> np.memmap:
        """Get data from a buffer array.

        Parameters
        ----------
        name : str
            Name of the buffer array.

        Returns
        -------
            numpy.memmap
                Data as Numpy memory map.
        """
        if self._buffers.get(name) is None:
            raise AttributeError(f"No buffer named '{name}' in {self}.")
        return self._buffers[name]

    def initialize(self, x: Data = None, y: Data = None) -> "Node":
        """Call the Node initializers on some data points.
        Initializers are functions called at first run of the Node,
        defining the dimensions and values of its parameters based on the
        dimension of some input data and its hyperparameters.

        Data point `x` is used to infer the input dimension of the Node.
        Data point `y` is used to infer the output dimension of the Node.

        Parameters
        ----------
        x : array-like of shape ([n_inputs], 1, input_dim)
            Input data.
        y : array-like of shape (1, output_dim)
            Ground truth data. Used to infer output dimension
            of trainable nodes.

        Returns
        -------
        Node
            Initialized Node.
        """
        if not self.is_initialized:
            x_init, y_init = _init_vectors_placeholders(self, x, y)
            self._initializer(self, x=x_init, y=y_init)
            self.reset()
            self._is_initialized = True
        return self

    def initialize_buffers(self) -> "Node":
        """Call the Node buffer initializer. The buffer initializer will create
        buffer array on demand to store transient values of the parameters,
        typically during training.

        Returns
        -------
        Node
            Initialized Node.
        """
        if self._buffers_initializer is not None:
            if len(self._buffers) == 0:
                self._buffers_initializer(self)

        return self

    def clean_buffers(self):
        """Clean Node's buffer arrays."""
        if len(self._buffers) > 0:
            self._buffers = dict()
            clean_tempfile(self)

        # Empty possibly stored inputs and targets in default buffer.
        self._X = self._Y = []

    def reset(self, to_state: np.ndarray = None) -> "Node":
        """Reset the last state saved to zero or to
        another state value `to_state`.

        Parameters
        ----------
        to_state : array of shape (1, output_dim), optional
            New state value.

        Returns
        -------
        Node
            Reset Node.
        """
        if to_state is None:
            self._state = self.zero_state()
        else:
            self._state = check_one_sequence(
                to_state, self.output_dim, allow_timespans=False, caller=self
            ).astype(self.dtype)
        return self

    @contextmanager
    def with_state(
        self, state: np.ndarray = None, stateful: bool = False, reset: bool = False
    ) -> "Node":
        """Modify the state of the Node using a context manager.
        The modification will have effect only within the context defined,
        before the state returns back to its previous value.

        Parameters
        ----------
        state : array of shape (1, output_dim), optional
            New state value.
        stateful : bool, default to False
            If set to True, then all modifications made in the context manager
            will remain after leaving the context.
        reset : bool, default to False
            If True, the Node will be reset using its :py:meth:`Node.reset`
            method.

        Returns
        -------
        Node
            Modified Node.
        """
        if not self._is_initialized:
            raise RuntimeError(
                f"Impossible to set state of node {type(self).__name__}: node"
                f" is not initialized yet."
            )

        current_state = self._state

        if state is None:
            if reset:
                state = self.zero_state()
            else:
                state = current_state

        self.reset(to_state=state)
        yield self

        if not stateful:
            self._state = current_state

    def zero_state(self) -> np.ndarray:
        """A null state vector."""
        if self.output_dim is not None:
            return np.zeros((1, self.output_dim), dtype=self.dtype)
        else:
            raise Exception(
                f"Cannot return a null state vector from {type(self).__name__} as it has no output dimension."
            )

    def call(
        self,
        x: Data,
        from_state: np.ndarray = None,
        stateful: bool = True,
        reset: bool = False,
    ) -> np.ndarray:
        """Call the Node forward function on a single step of data.
        Can update the state of the
        Node.

        Parameters
        ----------
        x : array of shape ([n_inputs], 1, input_dim)
            One single step of input data.
        from_state : array of shape (1, output_dim), optional
            Node state value to use at beginning of computation.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Node state will be reset to zero before this operation.

        Returns
        -------
        array of shape (1, output_dim)
            An output vector.
        """
        x, _ = check_xy(
            self,
            x,
            allow_timespans=False,
            allow_n_sequences=False,
        )

        if not self._is_initialized:
            self.initialize(x)

        return call(self, x, from_state=from_state, stateful=stateful, reset=reset)

    def run(self, X: np.array, from_state=None, stateful=True, reset=False):
        """Run the Node forward function on a sequence of data.
        Can update the state of the
        Node several times.

        Parameters
        ----------
        X : array-like of shape ([n_inputs], timesteps, input_dim)
            A sequence of data of shape (timesteps, features).
        from_state : array of shape (1, output_dim), optional
            Node state value to use at beginning of computation.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Node state will be reset to zero before this operation.

        Returns
        -------
        array of shape (timesteps, output_dim)
            A sequence of output vectors.
        """
        X_, _ = check_xy(
            self,
            X,
            allow_n_sequences=False,
        )

        if isinstance(X_, np.ndarray):
            if not self._is_initialized:
                self.initialize(np.atleast_2d(X_[0]))
            seq_len = X_.shape[0]
        else:  # multiple inputs?
            if not self._is_initialized:
                self.initialize([np.atleast_2d(x[0]) for x in X_])
            seq_len = X_[0].shape[0]

        with self.with_state(from_state, stateful=stateful, reset=reset):
            states = np.zeros((seq_len, self.output_dim))
            for i in progress(range(seq_len), f"Running {type(self).__name__}: "):
                if isinstance(X_, (list, tuple)):
                    x = [np.atleast_2d(Xi[i]) for Xi in X_]
                else:
                    x = np.atleast_2d(X_[i])

                s = call(self, x)
                states[i, :] = s

        return states

    def train(
        self,
        X: np.ndarray,
        Y: Union[_Node, np.ndarray] = None,
        force_teachers: bool = True,
        call: bool = True,
        learn_every: int = 1,
        from_state: np.ndarray = None,
        stateful: bool = True,
        reset: bool = False,
    ) -> np.ndarray:
        """Train the Node parameters using an online learning rule, if
        available.

        Parameters
        ----------
        X : array-like of shape ([n_inputs], timesteps, input_dim)
            Input sequence of data.
        Y : array-like of shape (timesteps, output_dim), optional.
            Target sequence of data. If None, the Node will train in an
            unsupervised way, if possible.
        call : bool, default to True
            It True, call the Node and update its state before applying the
            learning rule. Otherwise, use the train method
            on the current state.
        learn_every : int, default to 1
            Time interval at which training must occur, when dealing with a
            sequence of input data. By default, the training method is called
            every time the Node receive an input.
        from_state : array of shape (1, output_dim), optional
            Node state value to use at beginning of computation.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Node state will be reset to zero before this operation.

        Returns
        -------
        array of shape (timesteps, output_dim)
            All outputs computed during the training. If `call` is False,
            outputs will be the result of :py:meth:`Node.zero_state`.
        """
        if not self.is_trained_online:
            raise TypeError(f"Node {self} has no online learning rule implemented.")

        X_, Y_ = check_xy(
            self,
            X,
            Y,
            allow_n_sequences=False,
            allow_n_inputs=False,
        )

        if not self._is_initialized:
            x_init = np.atleast_2d(X_[0])
            y_init = None
            if hasattr(Y, "__iter__"):
                y_init = np.atleast_2d(Y_[0])

            self.initialize(x=x_init, y=y_init)
            self.initialize_buffers()

        states = train(
            self,
            X_,
            Y_,
            call_node=call,
            force_teachers=force_teachers,
            learn_every=learn_every,
            from_state=from_state,
            stateful=stateful,
            reset=reset,
        )

        self._unregister_teacher()

        return states

    def partial_fit(
        self,
        X_batch: Data,
        Y_batch: Data = None,
        warmup=0,
        **kwargs,
    ) -> "Node":
        """Partial offline fitting method of a Node.
        Can be used to perform batched fitting or to pre-compute some variables
        used by the fitting method.

        Parameters
        ----------
        X_batch : array-like of shape ([n_inputs], [series], timesteps, input_dim)
            A sequence or a batch of sequence of input data.
        Y_batch : array-like of shape ([series], timesteps, output_dim), optional
            A sequence or a batch of sequence of teacher signals.
        warmup : int, default to 0
            Number of timesteps to consider as warmup and
            discard at the beginning of each timeseries before training.

        Returns
        -------
        Node
            Partially fitted Node.
        """
        if not self.is_trained_offline:
            raise TypeError(f"Node {self} has no offline learning rule implemented.")

        X, Y = check_xy(self, X_batch, Y_batch, allow_n_inputs=False)

        X, Y = _init_with_sequences(self, X, Y)

        self.initialize_buffers()

        for i in range(len(X)):
            X_seq = X[i]
            Y_seq = None
            if Y is not None:
                Y_seq = Y[i]

            if X_seq.shape[0] <= warmup:
                raise ValueError(
                    f"Warmup set to {warmup} timesteps, but one timeseries is only "
                    f"{X_seq.shape[0]} long."
                )

            if Y_seq is not None:
                X_nona, Y_nona = _filter_where_na_target(X_seq[warmup:], Y_seq[warmup:])
                self._partial_backward(self, X_nona, Y_nona, **kwargs)
            else:
                self._partial_backward(self, X_seq[warmup:], **kwargs)

        return self

    def fit(self, X: Data = None, Y: Data = None, warmup=0) -> "Node":
        """Offline fitting method of a Node.

        Parameters
        ----------
        X : array-like of shape ([n_inputs], [series], timesteps, input_dim), optional
            Input sequences dataset. If None, the method will try to fit
            the parameters of the Node using the precomputed values returned
            by previous call of :py:meth:`partial_fit`.
        Y : array-like of shape ([series], timesteps, output_dim), optional
            Teacher signals dataset. If None, the method will try to fit
            the parameters of the Node using the precomputed values returned
            by previous call of :py:meth:`partial_fit`, or to fit the Node in
            an unsupervised way, if possible.
        warmup : int, default to 0
            Number of timesteps to consider as warmup and
            discard at the beginning of each timeseries before training.

        Returns
        -------
        Node
            Node trained offline.
        """

        if not self.is_trained_offline:
            raise TypeError(f"Node {self} has no offline learning rule implemented.")

        self._fitted = False

        # Call the partial backward function on the dataset if it is
        # provided all at once.
        if X is not None:
            if self._partial_backward is not None:
                self.partial_fit(X, Y, warmup=warmup)

        elif not self._is_initialized:
            raise RuntimeError(
                f"Impossible to fit node {type(self).__name__}: node"
                f" is not initialized, and fit was called "
                f"without input and teacher data."
            )

        self._backward(self, self._X, self._Y)

        self._fitted = True
        self.clean_buffers()

        return self


class Unsupervised(Node):
    @property
    def unsupervised(self):
        return True
