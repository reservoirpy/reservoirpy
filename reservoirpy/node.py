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
to what can be found in other popular deep learning and differenciable calculus
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
depending on the data, on the current `state` vector of the Node, and
optionally on data coming from other distant nodes `states` though feedback
connections (distant nodes can be reached using the `feedback` attribute of the
node they are connected to).

Nodes can also be connected together to form a :py:class:`Model`. Models hold
references to the connected nodes and make data flow from one node to
the next, allowing to create *deep* models and other more complex
architectures and computational graphs.
:py:class:`Model` is essentialy a subclass of :py:class:`Node`,
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
      ~Node.feedback
      ~Node.fit
      ~Node.get_buffer
      ~Node.get_param
      ~Node.initialize
      ~Node.initialize_buffers
      ~Node.initialize_feedback
      ~Node.link
      ~Node.link_feedback
      ~Node.partial_fit
      ~Node.reset
      ~Node.run
      ~Node.set_buffer
      ~Node.set_feedback_dim
      ~Node.set_input_dim
      ~Node.set_output_dim
      ~Node.set_param
      ~Node.set_state_proxy
      ~Node.state
      ~Node.state_proxy
      ~Node.train
      ~Node.with_feedback
      ~Node.with_state
      ~Node.zero_feedback
      ~Node.zero_state

   .. rubric:: Attributes

   .. autosummary::

      ~Node.feedback_dim
      ~Node.fitted
      ~Node.has_feedback
      ~Node.hypers
      ~Node.input_dim
      ~Node.is_fb_initialized
      ~Node.is_initialized
      ~Node.is_trainable
      ~Node.is_trained_offline
      ~Node.is_trained_online
      ~Node.name
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
from typing import Any, Dict, Optional, Union, List
from uuid import uuid4

import numpy as np

from .model import Model
from .types import (BackwardFn, Data, EmptyInitFn, ForwardFn,
                    ForwardInitFn, GenericNode, PartialBackFn,
                    Shape, )
from reservoirpy.utils import progress
from ._utils import to_ragged_seq_set
from reservoirpy.utils.parallel import clean_tempfile, memmap_buffer
from reservoirpy.utils.validation import (check_node_io, check_node_state,
                                          check_vector, )


def _initialize_with_seq_set(node, X, Y=None):
    """Initialize a Node with a sequence of inputs/targets."""
    X = to_ragged_seq_set(X)

    if Y is not None:
        Y = to_ragged_seq_set(Y)
    else:
        Y = [None for _ in range(len(X))]

    if not node.is_initialized:
        node.initialize(X[0], Y[0])

    return X, Y


def _node_fb_init_general(node, fb):
    """Void feedback initializer. Works in any case."""
    fb_dim = None
    if isinstance(fb, list):
        fb_dim = tuple([fb.shape[1] for fb in fb])
    elif isinstance(fb, np.ndarray):
        fb_dim = fb.shape[1]

    node.set_feedback_dim(fb_dim)


def _remove_input_for_feedback(model: Model) -> Union["Node", Model]:
    """Remove inputs nodes from feedback Model and gather remaining nodes
    into a new Model. Allow to get inputs for feedback model from its input
    nodes states."""
    all_nodes = set(model.nodes)
    input_nodes = set(model.input_nodes)
    filtered_nodes = all_nodes - input_nodes
    filtered_edges = [edge for edge in model.edges
                      if edge[0] not in input_nodes]

    # return a Node if Model - Inputs = 1 Node
    # else return a Model - Inputs
    if len(filtered_nodes) == 1:
        return list(filtered_nodes)[0]
    return Model(filtered_nodes, filtered_edges, name=str(uuid4()))


class Node(GenericNode):
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
    fb_initializer : callable, optional
        A function called at first run of the Node, defining the dimensions and
        values of its parameters based on the dimension of data received as
        a feedback from another Node.
    buffers_initializer : callable, optional
        A function called at the begining of an offline training session to
        create buffers used to store intermediate results, for batch or
        multisequece offline learning.
    input_dim : int
        Input dimension of the Node.
    output_dim : int
        Output dimension of the Node. Dimension of its state.
    feedback_dim :
        Dimension of the feedback signal received by the Node.
    name : str
        Name of the Node. It must be an unique identifier.

    See also
    --------
        Model
            Object used to compose node operations and create computational
            graphs.
    """
    _name: str

    _state: Optional[np.ndarray]
    _state_proxy: Optional[np.ndarray]
    _feedback: Optional[GenericNode]

    _params: Dict[str, Any]
    _hypers: Dict[str, Any]
    _buffers: Dict[str, Any]

    _input_dim: int
    _output_dim: int
    _feedback_dim: int

    _forward: ForwardFn
    _backward: BackwardFn
    _partial_backward: PartialBackFn
    _train: PartialBackFn

    _initializer: ForwardInitFn
    _buffers_initializer: EmptyInitFn
    _feedback_initializer: ForwardInitFn

    _trainable: bool
    _fitted: bool

    def __init__(self,
                 params: Dict[str, Any] = None,
                 hypers: Dict[str, Any] = None,
                 forward: ForwardFn = None,
                 backward: BackwardFn = None,
                 partial_backward: PartialBackFn = None,
                 train: PartialBackFn = None,
                 initializer: ForwardInitFn = None,
                 fb_initializer: ForwardInitFn = _node_fb_init_general,
                 buffers_initializer: EmptyInitFn = None,
                 input_dim: int = None,
                 output_dim: int = None,
                 feedback_dim: int = None,
                 name: str = None,
                 *args, **kwargs):

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
        self._feedback_initializer = fb_initializer
        self._buffers_initializer = buffers_initializer

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._feedback_dim = feedback_dim

        self._name = self._get_name(name)

        self._is_initialized = False
        self._is_fb_initialized = False
        self._state_proxy = None
        self._feedback = None

        # used to store a reduced version of the feedback if needed
        # when feedback is a Model (inputs of the feedback Model are suppressed
        # in the reduced version, as we do not need then to re-run them
        # because we assume they have already run during the forward call)
        self._reduced_fb = None

        self._trainable = self._backward is not None or \
                          self._partial_backward is not None or \
                          self._train is not None

        self._fitted = False if self.is_trainable and self.is_trained_offline \
            else True

    def __lshift__(self, other):
        return self.link_feedback(other)

    def __ilshift__(self, other):
        return self.link_feedback(other, inplace=True)

    def __iand__(self, other):
        raise TypeError(f"Impossible to merge nodes inplace: {self} is not "
                        f"a Model instance.")

    def _call(self, x=None, from_state=None, stateful=True, reset=False):
        """One step call, without input check."""
        with self.with_state(from_state, stateful=stateful, reset=reset):
            state = self._forward(self, x)
            self._state = state

        return state

    def _check_io(self, *args, **kwargs):
        return check_node_io(self, *args, **kwargs)

    @property
    def input_dim(self):
        """Node input dimension."""
        return self._input_dim

    @property
    def output_dim(self):
        """Node output and internal state dimension."""
        return self._output_dim

    @property
    def feedback_dim(self):
        """Node feedback signal dimension."""
        return self._feedback_dim

    @property
    def is_initialized(self):
        """Returns if the Node is initialized or not."""
        return self._is_initialized

    @property
    def has_feedback(self):
        """Returns if the Node receives feedback or not."""
        return self._feedback is not None

    @property
    def is_trained_offline(self):
        """Returns if the Node can be fitted offline or not."""
        return self.is_trainable and (self._backward is not None or
                                      self._partial_backward is not None)

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
    def is_fb_initialized(self):
        """Returns if the Node feedback intializer has been called already."""
        return self._is_fb_initialized

    def state(self) -> Optional[np.ndarray]:
        """Node current internal state.

        Returns
        -------
            numpy.ndarray, optional
                Internal state of the Node.
        """
        if not self.is_initialized:
            return None
        return self._state

    def state_proxy(self) -> Optional[np.ndarray]:
        """Returns the internal state freezed to be sent to other Nodes,
        connected through a feedback connection. This prevents any change
        occuring on the Node before feedback have reached the other Node to
        propagate to the other Node to early.

        Returns
        -------
            numpy.ndarray, optional
                Internal state of the Node.
        """
        if self._state_proxy is None:
            return self._state
        return self._state_proxy

    def feedback(self) -> np.ndarray:
        """State of the Nodes connected to this Node through feedback
        connections.

        Returns
        -------
            numpy.ndarray or list or numpy.ndarray, optional
                State of the feedback Nodes, i.e. the feedback signal.
        """
        if self.has_feedback:
            if not self._feedback.is_initialized:
                raise RuntimeError(f"Impossible to get feedback "
                                   f"from node or model {self._feedback} "
                                   f"to node {self.name}: "
                                   f"{self._feedback.name} "
                                   f"is not initialized.")

            if isinstance(self._feedback, Model):
                # retrieve all state_proxy
                input_data = {c.name: p.state_proxy()
                              for p, c in self._feedback.edges
                              if p in self._feedback.input_nodes}

                # call the feedback Model on it (no need to run the input
                # nodes again)
                if isinstance(self._reduced_fb, Model):
                    return self._reduced_fb.call(input_data)
                else:
                    reduced_name = self._reduced_fb.name
                    return self._reduced_fb.call(input_data[reduced_name])

            # if it is not a Model, then just get the state of the Node
            elif isinstance(self._feedback, Node):
                return self._feedback.state_proxy()
        else:
            raise RuntimeError(f"Node {self} is not connected to any feedback "
                               f"Node or Model.")

    def set_state_proxy(self, value: np.ndarray = None):
        """Change the freezed state of the Node. Used internaly to send
        the current state to feedback receiver Nodes during the next call.

        Parameters
        ----------
        value : np.ndarray
            State to freeze, waiting to be send to feedback receivers.
        """
        if value is not None:
            value = check_node_state(self, value)
        self._state_proxy = value

    def set_input_dim(self, value: int):
        """Set the input dimension of the Node. Can only be called once,
        during Node initialization."""
        if not self._is_initialized:
            if self._input_dim is not None and value != self._input_dim:
                raise ValueError(f"Imposible to use {self.name} with input "
                                 f"data of dimension {value}. Node has input "
                                 f"dimension {self._input_dim}.")
            self._input_dim = value
        else:
            raise TypeError(f"Input dimension of {self.name} is "
                            "immutable after initialization.")

    def set_output_dim(self, value: int):
        """Set the output dimension of the Node. Can only be called once,
        during Node initialization."""
        if not self._is_initialized:
            if self._output_dim is not None and value != self._output_dim:
                raise ValueError(f"Imposible to use {self.name} with target "
                                 f"data of dimension {value}. Node has output "
                                 f"dimension {self._output_dim}.")
            self._output_dim = value
        else:
            raise TypeError(f"Output dimension of {self.name} is "
                            "immutable after initialization.")

    def set_feedback_dim(self, value: int):
        """Set the feedback dimension of the Node. Can only be called once,
        during Node initialization."""
        if not self.is_fb_initialized:
            self._feedback_dim = value
        else:
            raise TypeError(f"Output dimension of {self.name} is "
                            "immutable after initialization.")

    def get_param(self, name: str):
        """Get one of the parameters or hyperparmeters given its name."""
        if name in self._params:
            return self._params.get(name)
        elif name in self._hypers:
            return self._hypers.get(name)
        else:
            raise AttributeError(f"No attribute named '{name}' "
                                 f"found in node {self}")

    def set_param(self, name: str, value: Any):
        """Set the value of a parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        value : np.ndarray
            Parameter new value.
        """
        if name in self._params:
            self._params[name] = value
        elif name in self._hypers:
            self._hypers[name] = value
        else:
            raise KeyError(f"No param named '{name}' "
                           f"in {self.name}. Available params are: "
                           f"{list(self._params.keys())}.")

    def create_buffer(self, name: str, shape: Shape = None,
                      data: np.ndarray = None):
        """Create a buffer array on disk, using numpy.memmap. This can be
        used to store transient variables on disk. Typically called inside
        a `buffers_initializer` function.

        Parameters
        ----------
        name : str
            Name of the buffer array.
        shape : tuple of int, optional
            Shape of the buffer array.
        data : numpy.ndarray
            Data to store in the buffer array.
        """
        self._buffers[name] = memmap_buffer(self, data=data,
                                            shape=shape, name=name)

    def set_buffer(self, name: str, value: np.ndarray):
        """Dump data in the buffer array.

        Parameters
        ----------
        name : str
            Name of the buffer array.
        value : numpy.ndarray
            Data to store in the buffer array.
        """
        self._buffers[name][:] = value

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
        x : numpy.ndarray or list of numpy.ndarray
            Input data.
        y : numpy.ndarray
            Groudn truth data. Used to infer output dimension
            of trainable nodes.

        Returns
        -------
            Node
                Initialized Node.
        """
        if not self._is_initialized:
            if isinstance(x, np.ndarray):
                x = np.atleast_2d(check_vector(x, caller=self))

            elif isinstance(x, list):
                for i in range(len(x)):
                    x[i] = np.atleast_2d(check_vector(x[i], caller=self))

            if y is not None:
                y = np.atleast_2d(check_vector(y, caller=self))
                self._initializer(self, x=x, y=y)

            else:
                self._initializer(self, x=x)

            self.reset()
            self._is_initialized = True

        return self

    def initialize_feedback(self) -> "Node":
        """Call the Node feedback initializer. The feedback initializer will
        determine feedback dimension given some feedback signal, and intialize
        all parameters related to the feedback connection.

        Feedback sender Node must be initialized, as the feedback intializer
        will probably call the :py:meth:`Node.feedback` method to get
        a sample of feedback signal.

        Returns
        -------
            Node
                Initialized Node.
        """
        if self.has_feedback:
            if not self.is_fb_initialized:
                if isinstance(self._feedback, Model):
                    input_data = {c.name: p.state_proxy()
                                  for p, c in self._feedback.edges
                                  if p in self._feedback.input_nodes}
                    # remove input nodes of the feedback Model, as they
                    # will probably be called by a more global Model or
                    # by the user itself.
                    self._reduced_fb = _remove_input_for_feedback(
                        self._feedback)

                    # initialize all the nodes that haven't been reached before
                    if isinstance(self._reduced_fb, Model):
                        self._reduced_fb.initialize(x=input_data)
                    else:
                        reduced_name = self._reduced_fb.name
                        self._reduced_fb.initialize(x=input_data[reduced_name])

                    self._feedback._is_initialized = True

                self._feedback_initializer(self, self.zero_feedback())
                self._is_fb_initialized = True

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

    def reset(self, to_state: np.ndarray = None) -> "Node":
        """Reset the last state saved to zero or to
        another state value `to_state`.

        Parameters
        ----------
        to_state : np.ndarray, optional
            New state value.

        Returns
        -------
            Node
                Reset Node.
        """
        if to_state is None:
            self._state = self.zero_state()
        else:
            self._state = check_node_state(self, to_state)
        return self

    @contextmanager
    def with_state(self, state: np.ndarray = None,
                   stateful: bool = False, reset: bool = False) -> "Node":
        """Modify the state of the Node using a context manager.
        The modification will have effect only within the context defined,
        before the state returns back to its previous value.

        Parameters
        ----------
        state : numpy.ndarray, optional
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
                Modifyed Node.
        """
        if not self._is_initialized:
            raise RuntimeError(
                f"Impossible to set state of node {self.name}: node"
                f"is not initialized yet.")

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

    @contextmanager
    def with_feedback(self, feedback: np.ndarray = None,
                      stateful=False, reset=False) -> "Node":
        """Modify the feedback received or sent by the Node using
        a context manager.
        The modification will have effect only within the context defined,
        before the feedback returns to its previous state.

        If the Node is receiving feedback, then this function will alter the
        state of the Node connected to it through feedback connections.

        If the Node is sending feedback, then this function will alter the
        state (or state proxy, see :py:meth:`Node.state_proxy`) of the Node.

        Parameters
        ----------
        feedback : numpy.ndarray, optional
            New feedback signal.
        stateful : bool, default to False
            If set to True, then all modifications made in the context manager
            will remain after leaving the context.
        reset : bool, default to False
            If True, the feedback  will be reset to zero.

        Returns
        -------
            Node
                Modifyed Node.
        """
        if self.has_feedback:
            current_fb = self.feedback()

            if feedback is None:
                if reset:
                    feedback = self.zero_feedback()
                else:
                    feedback = current_fb

            current_proxy = self._feedback._state_proxy
            self._feedback.set_state_proxy(feedback)

            yield self

            if not stateful:
                self._feedback._state_proxy = current_proxy

        else:  # maybe a feedback sender then ?
            current_state_proxy = self._state_proxy

            if feedback is None:
                if reset:
                    feedback = self.zero_state()
                else:
                    feedback = current_state_proxy

            self.set_state_proxy(feedback)

            yield self

            if not stateful:
                self._state_proxy = current_state_proxy

    def zero_state(self) -> np.ndarray:
        """A null state vector."""
        if self.output_dim is not None:
            return np.zeros((1, self.output_dim))

    def zero_feedback(self) -> Optional[Union[List[np.ndarray], np.ndarray]]:
        """A null feedback vector. Returns None if the Node receives
         no feedback."""
        if self._feedback is not None:
            if isinstance(self._feedback, Node):
                return self._feedback.zero_state()
            elif isinstance(self._feedback, Model):
                zeros = []
                for output in self._feedback.output_nodes:
                    zeros.append(output.zero_state())
                if len(zeros) == 1:
                    return zeros[0]
                else:
                    return zeros
        return None

    def link_feedback(self,
                      node: GenericNode,
                      inplace: bool = False,
                      name: str = None) -> "GenericNode":
        """Create a feedback connection between the Node and another Node or
        Model.

        Parameters
        ----------
        node : Node or Model
            Feedback sender Node or Model.
        inplace : bool, default to False
            If False, then this function returns a copy of the current Node
            with feedback enabled. If True, feedback is directly added to the
            current Node.
        name : str, optional
            Name of the node copy, if `inplace` is False.

        Returns
        -------
            Node
                A Node with a feedback connection.
        """
        from .ops import link_feedback
        return link_feedback(self, node, inplace=inplace, name=name)

    def call(self, x: Data, from_state: np.ndarray = None,
             stateful: bool = True, reset: bool = False) -> np.ndarray:
        """Call the Node forward function on a single step of data.
        Can update the state of the
        Node.

        Parameters
        ----------
        x : numpy.ndarray
            One single step of input data.
        from_state : numpy.ndarray
            Node state value to use at begining of computation.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Node state will be reset to zero before this operation.

        Returns
        -------
            numpy.ndarray
                An output vector.
        """

        x = self._check_io(x, self.input_dim, allow_timespans=False,
                           allow_list=False)

        if not self._is_initialized:
            self.initialize(x)

        state = self._call(x, from_state, stateful, reset)

        return state

    def run(self, X: np.array, from_state=None, stateful=True, reset=False):
        """Run the Node forward function on a sequence of data.
        Can update the state of the
        Node several times.

        Parameters
        ----------
        X : numpy.array
            A sequence of data of shape (timesteps, features).
        from_state : numpy.ndarray
            Node state value to use at begining of computation.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Node state will be reset to zero before this operation.

        Returns
        -------
            numpy.ndarray
                A sequence of output vectors.
        """

        X = self._check_io(X, self.input_dim, allow_timespans=True)

        if not self._is_initialized:
            self.initialize(np.atleast_2d(X[0]))

        seq_len = X.shape[0]

        with self.with_state(from_state, stateful=stateful, reset=reset):
            states = np.zeros((seq_len, self.output_dim))
            for i in progress(range(seq_len), f"Running {self.name}: "):
                x = np.atleast_2d(X[i])
                s = self._call(x)
                states[i, :] = s

        return states

    def train(self, X: np.ndarray, Y: np.ndarray = None,
              force_teachers: bool = True, call: bool = True,
              learn_every: int = 1, from_state: np.ndarray = None,
              stateful: bool = True, reset: bool = False) -> np.ndarray:
        """Train the Node parameters using an online learning rule, if
        available.

        Parameters
        ----------
        X : numpy.ndarray
            Input sequence of data.
        Y : numpy.ndarray, optional.
            Target sequence of data. If None, the Node will search a feedback
            signal, or train in an unsupervised way, if possible.
        force_teachers : bool, default to True
            If True, this Node will broadcast the available ground truth signal
            to all Nodes using this Node as a feedback sender. Otherwise,
            the real state of this Node will be sent to the feedback receivers.
        call : bool, default to True
            It True, call the Node and update its state before applying the
            learning rule. Otherwise, use the train method
            on the current state.
        learn_every : int, default to 1
            Time interval at which training must occur, when dealing with a
            sequence of input data. By default, the training method is called
            every time the Node receive an input.
        from_state : numpy.ndarray
            Node state value to use at begining of computation.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Node state will be reset to zero before this operation.

        Returns
        -------
            numpy.ndarray
                All outputs computed during the training. If `call` is False,
                outputs will be the result of :py:meth:`Node.zero_state`.

        """

        if self._train is None:
            raise TypeError(f"Node {self} has no online learning rule "
                            f"implemented.")

        X = self._check_io(X, self.input_dim, allow_timespans=True,
                           allow_list=False)
        Y = self._check_io(Y, self.output_dim, allow_timespans=True,
                           io_type="output", allow_list=False)

        if not self._is_initialized:
            x_init = np.atleast_2d(X[0])
            y_init = np.atleast_2d(Y[0]) if Y is not None else None
            self.initialize(x=x_init, y=y_init)
            self.initialize_buffers()

        seq_len = X.shape[0]
        seq = progress(range(seq_len), f"Training {self.name}") \
            if seq_len > 1 else range(seq_len)

        with self.with_state(from_state, stateful=stateful, reset=reset):
            states = np.zeros((seq_len, self.output_dim))
            for i in seq:
                x = X[i, :]

                y = Y[i, :] if Y is not None else None
                if y is None and self.has_feedback:
                    y = self.feedback()

                if call:
                    s = self.call(x)
                else:
                    s = self.state()

                if force_teachers:
                    self.set_state_proxy(y)

                if i % learn_every == 0 or seq_len == 1:
                    self._train(self, x=x, y=y)

                states[i, :] = s

            return states

    def partial_fit(self, X_batch: Data,
                    Y_batch: Data = None,
                    **kwargs) -> "Node":
        """Partial offline fitting method of a Node.
        Can be used to perform batched fitting or to precompute some variables
        used by the fitting method.

        Parameters
        ----------
        X_batch : numpy.ndarray
            A sequence of input data.
        Y_batch : numpy.ndarray, optional
            A sequence of teacher signals.

        Returns
        -------
            Node
                Partially fitted Node.
        """
        if self._partial_backward is None:
            raise TypeError(f"Node {self} has no offline learning rule "
                            f"implemented.")

        X_batch, Y_batch = _initialize_with_seq_set(self, X_batch, Y_batch)

        self.initialize_buffers()

        for X, Y in zip(X_batch, Y_batch):
            self._partial_backward(self, X, Y)

        return self

    def fit(self, X: Data = None, Y: Data = None) -> "Node":
        """Offline fitting method of a Node.

        Parameters
        ----------
        X : numpy.ndarray or list of numpy.ndarray, optional
            Input sequences dataset. If None, the method will try to fit
            the parameters of the Node using the precomputed values returned
            by previous call of :py:meth:`partial_fit`.
        Y : numpy.ndarray or list of numpy.ndarray, optional
            Teacher signals dataset. If None, the method will try to fit
            the parameters of the Node using the precomputed values returned
            by previous call of :py:meth:`partial_fit`, or to fit the Node in
            an unsupervised way, if possible.

        Returns
        -------
            Node
                Node trained offline.
        """

        if self._backward is None:
            raise TypeError(f"Node {self} has no offline learning rule "
                            f"implemented.")

        self._fitted = False

        # Call the partial backward function on the dataset if it is
        # provided all at once.
        if X is not None:
            X, Y = _initialize_with_seq_set(self, X, Y)

            if self._partial_backward is not None:
                for X_batch, Y_batch in zip(X, Y):
                    self.partial_fit(X_batch, Y_batch)

        elif not self._is_initialized:
            raise RuntimeError(f"Impossible to fit node {self.name}: node"
                               f"is not initialized, and fit was called "
                               f"without input and teacher data.")

        self._backward(self, X, Y)

        self._fitted = True

        self.clean_buffers()

        return self

    def copy(self, name: str = None, copy_feedback: bool = False,
             shallow: bool = False):
        """Returns a copy of the Node.

        Parameters
        ----------
        name : str
            Name of the Node copy.
        copy_feedback : bool, default to False
            If True, also copy the Node feedback senders.
        shallow : bool, default to False
            If False, performs a deep copy of the Node.

        Returns
        -------
            Node
                A copy of the Node.
        """
        if shallow:
            new_obj = copy(self)
        else:
            if self.has_feedback:
                # store feedback node
                fb = self._feedback
                # temporarily remove it
                self._feedback = None

                # copy and restore feedback, deep copy of feedback depends
                # on the copy_feedback parameter only
                new_obj = deepcopy(self)
                new_obj._feedback = fb
                self._feedback = fb

            else:
                new_obj = deepcopy(self)

        if copy_feedback:
            if self.has_feedback:
                fb_copy = deepcopy(self._feedback)
                new_obj._feedback = fb_copy

        n = self._get_name(name)
        new_obj._name = n

        return new_obj
