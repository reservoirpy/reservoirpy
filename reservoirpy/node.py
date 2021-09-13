"""Node API: simple tools for complex reservoir computing architectures.

The Node API features a simple implementation of computational graphs, similar
to what can be found in other popular deep learning and differenciable calculus
libraries. It is however simplified and made the most flexible possible by
discarding the useless "fully differentiable operations" functionalities. If
you wish to use learning rules making use of chain rule and full
differentiability of all operators, ReservoirPy may not be the tool you need
(actually, the whole paradigm of reservoir computing might arguably not be the
tool you need).

The Node API is composed of a base :py:class:`Node` that can be described as a
stateful recurrent operator able to manipulate streams of data. A
:py:class:`Node` applies a `forward` function on some data, and then stores the
result in its `state` attribute. The `forward` operation can be a function
depending on the data, on the current `state` vector of the Node, and
optionally on data coming from other distant nodes `states` though feedback
connections (distant nodes can be reached using the `feedback` attribute of the
node they are connected to).

Nodes can also be connected together to form a :py:class:`Model`. Models hold
references to the connected nodes and make data flow from one node to
the next. :py:class:`Model` is essentialy a subclass of :py:class:`Node`, that
can also be connected to other nodes and models.

TODO: add link to tutorial

:py:class:`Node` subclassing
============================

    .. highlight:: python

    Subclassing the :py:class:`Node` to create a custom operator takes only a
    few steps to be done and operational. Sublasses of :py:class:`Node` can
    then be used as any other node instances.

    First, one needs to create the `forward` function that will be applied
    by the new node class::

        def forward(node: Node, x: np.ndarray) -> np.ndarray:
        '''Does something to the current state of the node, the input
        data and some feedback.'''

            state = node.state()  # get current node state
            feedback = node.feedback()  # call state of some distant node
            some_param = node.const1
            some_other_param = node.const2

            return x + some_param * state + some_other_param * feedback

    This function **must** take as parameter a vector `x` of shape
    ``(1, dim x)`` (one timestep of data) and the node instance itself. You can
    access any parameter stored in the node through this instance.

    Then, one needs to create the `intialize` function that will be used at
    runtime to infer the input and output dimensions of the node, and optionaly
    initialize some parameters (some neuronal weights, for instance)::

        def intialize(node: Node, x=None):
        '''This function receives a data point x at runtime and uses it to
        infer input and output dimensions.
        '''
            if x is not None:
                node.set_input_dim(x.shape[1])
                node.set_output_dim(x.shape[1])

                # you can initialize parameters here
                node.set_param("const1", 1)

    Additionaly, another function can be created to initialize feedback signal
    dimension, if the node requires feedback::

        def intialize_fb(node: Node):
        '''This function is called at runtime and
        infer feedback dimensions.
        '''
            if node.has_feedback:
                # in our case, feedback dimension is just the dimension of the
                # feedback vector.
                feedback = node.feedback()
                node.set_feedback_dim(feedback.shape[1])

    TODO: section about trainable nodes

    That's it! You can now create a new base :py:class:`Node` instance
    parametrized with the functions you have just written::

        node = Node(forward=forward,
                    initializer=initialize,
                    fb_initializer=initialize_fb,
                    params={"const1": None},
                    hypers={"const2": -1},
                    name="custom_node")

    .. note::
        Do not forget to declare the mutable parameters `params` and immutable
        hyperparameters `hypers` as dictionnaries. `params` should store all
        parameters that need to be initialized and that will evolve during the
        life cycle of the node (for example, neuronal weights whom value will
        change during training). `hypers` should store parameters used to
        define the architecture or the behavior of the node instance, and that
        will not change through learning mechanisms.

    You can also create a new subclass of :py:class:`Node` in a similar way::

        class CustomNode(Node):

            def __init__(self, const2=-1, name=None):
                super().__init__(forward=forward,
                                 initializer=initialize,
                                 fb_initializer=initialize_fb,
                                 params={"const1": None},
                                 hypers={"const2": const2},
                                 name=name)

        node = CustomeNode(const2=-1, name="custom_node")

    This allow more flexibility, as you can redefine the complete behavior of
    the node in the subclass. Be careful to expose the `name` parameter in the
    subclass ``__init__``, and to pass it to the base class as parameter.
    It is a good practice to find meaningful names for your node instances.


References
==========

    ReservoirPy Node API was heavily inspired by Explosion.ai *Thinc*
    functional deep learning library [1]_, and *Nengo* core API [2]_.
    It also follows some *scikit-learn* schemes and guidelines [3]_.

    .. [1] `Thinc <https://thinc.ai/>`_ website
    .. [2] `Nengo <https://www.nengo.ai/>`_ website
    .. [3] `scikit-learn <https://scikit-learn.org/stable/>`_ website

"""
# Author: Nathan Trouvain at 01/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from contextlib import ExitStack, contextmanager
from itertools import product
from typing import Callable, Dict, List, Optional
from multiprocessing import Array
from uuid import uuid4

import numpy as np

from .utils.graphflow import (DataDispatcher, find_entries_and_exits,
                              get_offline_subgraphs, topological_sort, )
from .utils.types import MappedData
from .utils.validation import check_vector, is_mapping, is_sequence_set
from .utils.parallel import memmap_buffer


def _remove_input_for_feedback(model: "Model") -> "Node":
    all_nodes = set(model.nodes)
    input_nodes = set(model.input_nodes)
    filtered_nodes = all_nodes - input_nodes
    filtered_edges = [edge for edge in model.edges
                      if edge[0] not in input_nodes]

    # return a Node if Model - Inputs = 1 Node
    # else return a Model - Inputs
    if len(filtered_nodes) == 1:
        return list(filtered_nodes)[0]
    return Model(filtered_nodes, filtered_edges)


def _to_ragged_array_seq(data):
    if is_mapping(data):
        new_data = {}
        for name, datum in data.items():
            if not is_sequence_set(datum):
                new_datum = [datum]
            else:
                new_datum = datum
            new_data[name] = datum
        return new_data
    else:
        if not is_sequence_set(data):
            return [data]
        else:
            return data


def forward(model: "Model",
            x: MappedData) -> List[np.ndarray]:
    """Function applied by a :py:class:`Model` instance.

    This function if basically a composition of the forward functions of all
    nodes involved in the model architecture. For instance, let :math:`f`
    be the forward function of a first node, and let :math:`g` be the forward
    function of a second node. If first node is connected to second node in a
    model, then the model forward function :math:`h` will compute, at each
    timestep :math:`t` of a timeserie :math:`X`:

    .. math::

        h(X_t) = g(f(X_t)) = (g \\circ f)(X_t)

    Parameters
    ----------
    model : Model
        A :py:class:`Model` instance.
    x : numpy.ndarray or dict of numpy.ndarray
        A vector of shape `(1, ndim)` corresponding to a timestep of data, or
        a dictionnary mapping node names to vector of shapes
        `(1, ndim of corresponding node)`.

    Returns
    -------
        list of numpy.ndarray
            New states of all terminal nodes of the model, i.e. its output.
    """
    data = model.data_dispatcher.load(x)

    for node in model.nodes:
        node(data[node].x)

    return [out_node.state() for out_node in model.output_nodes]


def train(model: "Model", x: MappedData, y: MappedData = None):
    data = model.data_dispatcher.load(x, y)

    for node in model.nodes:
        if node.is_trainable and hasattr(node, "_train"):
            node.train(data[node].x, data[node].y)
        else:
            node(data[node].x)

    return [out_node.state() for out_node in model.output_nodes]


def partial_backward(model: "Model",
                     X_batch: MappedData,
                     Y_batch: MappedData = None):
    data = model.data_dispatcher.load(X_batch, Y_batch)

    for node in model.nodes:
        if node.is_trainable and node not in model.already_trained:
            if hasattr(node, "_partial_backward"):
                node.partial_fit(data[node].x, data[node].y)


def backward(model: "Model", X: MappedData = None, Y: MappedData = None):

    for node in model.nodes:
        if node.is_trainable and hasattr(node, "_partial_backward") and \
                node not in model.already_trained:
            node.fit()


def initializer(model: "Model",
                x: MappedData,
                y: Optional[MappedData] = None):
    """Initializes a :py:class:`Model` instance at runtime, using samples of
    data to infer all :py:class:`Node` dimensions.

    Parameters
    ----------
    model : Model
        A :py:class:`Model` instance.
    x : numpy.ndarray or dict of numpy.ndarray
        A vector of shape `(1, ndim)` corresponding to a timestep of data, or
        a dictionnary mapping node names to vector of shapes
        `(1, ndim of corresponding node)`.
    y : numpy.ndarray or dict of numpy.ndarray
        A vector of shape `(1, ndim)` corresponding to a timestep of target
        data or feedback, or a dictionnary mapping node names to vector of
        shapes `(1, ndim of corresponding node)`.
    """
    data = model.data_dispatcher.load(x, y)

    # first, probe network to init forward flow
    # (no real call, only zero states)
    for node in model.nodes:
        node.initialize(x=data[node].x, y=data[node].y)

    # second, probe feedback demanding nodes to
    # init feedback flow
    for fb_node in model.feedback_nodes:
        fb_node.initialize_feedback()


def link(node1: "Node", node2: "Node") -> "Model":
    """Connects two :py:class:`Node` instances to form a :py:class:`Model`
    instance. `node1` output will be used as input for `node2` in the
    created model.

    `node1` and `node2` can also be :py:class:`Model` instances. In this case,
    the new :py:class:`Model` created will contain all nodes previously
    contained in all the models, and link all `node1` outputs to all `node2`
    inputs.

    Parameters
    ----------
    node1, node2 : Node, Node
        :py:class:`Node` instance to connect. `node1` is connected to `node2`
        such that `node1` output is used as input by `node2`.

    Returns
    -------
        Model
            A :py:class:`Model` instance chaining the nodes.
    """
    # fetch all nodes in the two subgraphs.
    all_nodes = []
    for node in (node1, node2):
        if hasattr(node, "nodes"):  # is a model already ?
            all_nodes += node.nodes
        else:
            all_nodes += [node]

    # fetch all edges in the two subgraphs.
    all_edges = []
    for node in (node1, node2):
        if hasattr(node, "edges"):
            all_edges += node.edges

    # create edges between output nodes of the
    # subgraph 1 and input nodes of the subgraph 2.
    senders = []
    if hasattr(node1, "output_nodes"):
        senders += node1.output_nodes
    else:
        senders += [node1]

    receivers = []
    if hasattr(node2, "input_nodes"):
        receivers += node2.input_nodes
    else:
        receivers += [node2]

    new_edges = list(product(senders, receivers))

    # maybe nodes are already initialized ?
    # check if connected dimensions are ok
    for sender, receiver in new_edges:
        if sender.output_dim is not None and \
                receiver.input_dim is not None and \
                sender.output_dim != receiver.input_dim:
            raise ValueError(f"Dimension mismatch between connected nodes: "
                             f"sender node {sender.name} has output dimension "
                             f"{sender.output_dim} but receiver node "
                             f"{receiver.name} "
                             f"has input dimension {receiver.input_dim}.")

    # all outputs from subgraph 1 are connected to
    # all inputs from subgraph 2.
    all_edges += new_edges

    # pack everything
    return Model(nodes=all_nodes, edges=all_edges)


def combine(*models: "Model"):
    """Merge different :py:class:`Model` instances into a single
    :py:class:`Model` instance.

    :py:class:`Node` instances contained in the models to merge will be
    gathered in a single model, along with all previously defined connections
    between them.

    Parameters
    ----------
    *models : Model instances
        All models to merge.

    Returns
    -------
        Model
            A new :py:class:`Model` instance.

    """
    all_nodes = set()
    all_edges = set()
    for model in models:
        if hasattr(model, "nodes"):
            all_nodes |= set(model.nodes)
            all_edges |= set(model.edges)
        else:
            raise TypeError(f"Impossible to combine models: "
                            f"object of type {type(model)} is not "
                            f"a model.")

    return Model(nodes=list(all_nodes), edges=list(all_edges))


class Node:
    _state: np.ndarray
    _params: Dict
    _hypers: Dict
    _buffers: Dict
    _input_dim: int
    _output_dim: int
    _forward: Callable
    _backward: Callable
    _partial_backward: Callable
    _train: Callable
    _trainable: bool
    _fitted: bool
    _initializer: Callable
    _feedback: "Node"
    _feedback_dim: int
    _feedback_initializer: Callable
    _name: str
    _state_proxy: np.ndarray
    _factory_id: int = -1
    _registry: Dict = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._factory_id = -1
        cls._registry = dict()

    def __init__(self, params=None, hypers=None, forward=None,
                 backward=None, partial_backward=None, train=None,
                 initializer=None, fb_initializer=None, input_dim=None,
                 output_dim=None, feedback_dim=None, name=None):

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

        self._trainable = self.is_trained_offline or self.is_trained_online
        self._fitted = False if self.is_trainable and self.is_trained_offline\
            else True

    def __repr__(self):
        klas = type(self).__name__
        init_params = [k for k in self._params.keys() if
                       self._params[k] is not None]
        hypers = [(str(k), str(v)) for k, v in self._hypers.items()]
        all_params = ["=".join((k, v)) for k, v in hypers] + init_params
        all_params += [f"in={self.input_dim}", f"out={self.output_dim}"]
        return f"'{self.name}': {klas}(" + ", ".join(all_params) + ")"

    def __getattr__(self, item):
        param = self.get_param(item)
        if param is None:
            if item not in self._params:
                raise AttributeError(f"No attribute named '{item}' "
                                     f"found in node {self.name}")
        return param

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def __rshift__(self, other):
        return self.link(other)

    def __lshift__(self, other):
        return self.link_feedback(other)

    def _get_name(self, name=None):
        if name is None:
            type(self)._factory_id += 1
            _id = self._factory_id
            name = f"{type(self).__name__}-{_id}"

        if name in type(self)._registry:
            raise NameError(f"Name '{name}' is already taken "
                            f"by another node. Node names should "
                            f"be unique.")
        return name

    def _check_state(self, s):
        s = check_vector(s)

        if not self._is_initialized:
            raise RuntimeError(
                f"Impossible to set state of node {self.name}: node "
                f"is not initialized yet.")

        if s.shape[1] != self.output_dim:
            raise ValueError(f"Impossible to set state of node {self.name}: "
                             f"dimension mismatch between state vector ("
                             f"{s.shape[1]}) "
                             f"and node output dim ({self.output_dim}).")
        return s

    def _check_input(self, x):
        x = check_vector(x, allow_reshape=True)

        if self._is_initialized:
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"Impossible to call node {self.name}: node input "
                    f"dimension is (1, {self.input_dim}) and input dimension "
                    f"is {x.shape}.")
        return x

    def _check_feedback(self, fb):  # TODO: build this
        fb = check_vector(fb, allow_reshape=True)

        if self._is_initialized:
            if fb.shape[1] != self._feedback_dim:
                raise ValueError(
                    f"Impossible to call node {self.name}: node input "
                    f"dimension is (1, {self.input_dim}) and input dimension "
                    f"is {fb.shape}.")
        return fb

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return self._params

    @property
    def hypers(self):
        return self._hypers

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def feedback_dim(self):
        return self._feedback_dim

    @property
    def is_initialized(self):
        return self._is_initialized

    @property
    def has_feedback(self):
        return self._feedback is not None

    @property
    def is_trained_offline(self):
        return self._backward is not None or \
               self._partial_backward is not None

    @property
    def is_trained_online(self):
        return self._train is not None

    @property
    def is_trainable(self):
        return self._trainable

    @property
    def fitted(self):
        return self._fitted

    @is_trainable.setter
    def is_trainable(self, value: bool):
        if type(value) is bool:
            self._trainable = value
        else:
            raise TypeError("'is_trainable' must be a boolean.")

    @property
    def is_fb_initialized(self):
        return self._is_fb_initialized

    def state(self):
        if not self.is_initialized:
            return None
        return self._state

    def state_proxy(self):
        if self._state_proxy is None:
            return self._state
        return self._state_proxy

    def feedback(self):
        if not self._feedback.is_initialized:
            raise RuntimeError(f"Impossible to get feedback "
                               f"from node or model {self._feedback} "
                               f"to node {self.name}: {self._feedback.name} "
                               f"is not initialized.")

        if isinstance(self._feedback, Model):
            input_data = {c.name: p.state_proxy()
                          for p, c in self._feedback.edges
                          if p in self._feedback.input_nodes}
            if isinstance(self._reduced_fb, Model):
                return self._reduced_fb.call(input_data)
            else:
                reduced_name = self._reduced_fb.name
                return self._reduced_fb.call(input_data[reduced_name])
        elif isinstance(self._feedback, Node):
            return self._feedback.state_proxy()
        else:
            return None

    def set_state_proxy(self, value=None):
        if value is not None:
            value = self._check_state(value)
        self._state_proxy = value

    def set_input_dim(self, value):
        if not self._is_initialized:
            self._input_dim = value
        else:
            raise TypeError(f"Input dimension of {self.name} is "
                            "immutable after initialization.")

    def set_output_dim(self, value):
        if not self._is_initialized:
            self._output_dim = value
        else:
            raise TypeError(f"Output dimension of {self.name} is "
                            "immutable after initialization.")

    def set_feedback_dim(self, value):
        if not self.is_fb_initialized:
            self._feedback_dim = value
        else:
            raise TypeError(f"Output dimension of {self.name} is "
                            "immutable after initialization.")

    def get_param(self, name):
        if name in self._params:
            return self._params.get(name)
        elif name in self._hypers:
            return self._hypers.get(name)
        else:
            return None

    def set_param(self, name, value):
        if name in self._params:
            self._params[name] = value
        elif name in self._hypers:
            self._hypers[name] = value
        else:
            raise KeyError(f"No param or hyperparam named '{name}' "
                           f"in {self.name}. Available params are: "
                           f"{list(self._params.keys())}. Available "
                           f"hyperparams are {list(self._hypers.keys())}.")

    def create_buffer(self, name, shape=None, data=None):
        self._buffers[name] = memmap_buffer(self, data=data,
                                            shape=shape, name=name)

    def set_buffer(self, name, value):
        self._buffers[name][:] = value

    def get_buffer(self, name):
        return self._buffers[name]

    def initialize(self, x=None, y=None):
        if not self._is_initialized:
            x = check_vector(x)
            if y is not None:
                y = check_vector(y)
                self._initializer(self, x=x, y=y)
            else:
                self._initializer(self, x=x)

            self.reset()
            self._is_initialized = True

        return self

    def initialize_feedback(self):
        if isinstance(self._feedback, Model):
            input_data = {c.name: p.state_proxy()
                          for p, c in self._feedback.edges
                          if p in self._feedback.input_nodes}
            self._reduced_fb = _remove_input_for_feedback(self._feedback)

            if isinstance(self._reduced_fb, Model):
                self._reduced_fb.initialize(x=input_data)
            else:
                reduced_name = self._reduced_fb.name
                self._reduced_fb.initialize(x=input_data[reduced_name])

        elif isinstance(self._feedback, Node):
            self._feedback_initializer(self)

        self._is_fb_initialized = True
        self._feedback._is_initialized = True

        return self

    def reset(self, to_state: np.ndarray = None):
        """Reset the last state saved to zero or to
        another state value `from_state`.

        Parameters
        ----------
        to_state : np.ndarray, optional
            New state value for stateful
            computations, by default None.
        """
        if to_state is None:
            self._state = self.zero_state()
        else:
            self._state = self._check_state(to_state)
        return self

    def reset_feedback(self, to_feedback=None):
        if to_feedback is None:
            self._feedback.reset()
        else:
            self._feedback.reset(to_state=to_feedback)

    @contextmanager
    def with_state(self, state=None, stateful=False, reset=False):

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
    def with_feedback(self, feedback=None, stateful=False, reset=False):

        if self.has_feedback:  # if it is a feedback receiver
            current_fb = self._feedback

            if feedback is None:
                if reset:
                    feedback = self.zero_feedback()
                else:
                    feedback = current_fb

            if isinstance(feedback, Node):
                self._feedback = feedback
                yield self
                self._feedback = current_fb
            elif isinstance(feedback, np.ndarray):
                current_proxy = self._feedback._state_proxy
                self._feedback.set_state_proxy(feedback)
                yield self
                if not stateful:
                    self._feedback._state_proxy = current_proxy
            else:
                raise TypeError(f"Impossible to get feedback from {feedback}: "
                                f"it is neither a Node instance nor an array.")

        else:  # maybe a feedback sender then ?
            current_state_proxy = self._state_proxy

            if feedback is None:
                if reset:
                    feedback = self.zero_state()
                else:
                    feedback = current_state_proxy

            self.set_state_proxy(feedback)
            yield self

    def zero_state(self):
        """A null state vector."""
        if self.output_dim is not None:
            return np.zeros((1, self.output_dim))

    def zero_feedback(self):
        """A null state vector."""
        if self._feedback is not None:
            return self._feedback.zero_state()
        return None

    def link(self, other):
        if isinstance(other, Node) or \
                hasattr(other, "__getitem__") and \
                all([isinstance(obj, Node) for obj in other]):
            return link(self, other)
        else:
            raise TypeError(f"Impossible to link node {self.name} with "
                            f"oject of type {type(other)}.")

    def link_feedback(self, node):
        if not isinstance(node, Node):
            raise TypeError(f"Impossible to receive feedback from {node}: "
                            f"it is not a Node instance.")
        self._feedback = node
        return self

    def call(self, x, from_state=None, stateful=True, reset=False):
        x = self._check_input(x)

        if not self._is_initialized:
            self.initialize(x)

        with self.with_state(from_state, stateful=stateful, reset=reset):
            state = self._forward(self, x)
            self._state = state

        return state

    def run(self, X, from_state=None, stateful=True, reset=False):

        if not self._is_initialized:
            self.initialize(X[0])

        with self.with_state(from_state, stateful=stateful, reset=reset):
            states = np.zeros((X.shape[0], self.output_dim))
            for i, x in enumerate(X):
                s = self.call(x)
                states[i, :] = s

        return states

    def train(self, x, y=None):
        if self._train is not None:
            x = self._check_input(x)
            y = self._check_state(y)

            if not self._is_initialized:
                self.initialize(x, y)

            state = self(x)
            self._train(self, x, y)

            return state

    def partial_fit(self, X_batch, Y_batch=None):
        if self._partial_backward is not None:
            if not self._is_initialized:
                self.initialize(X_batch[0], Y_batch[0])

            self._partial_backward(self, X_batch, Y_batch)

            return self

    def fit(self, X=None, Y=None):
        if self._backward is not None:
            if not self._is_initialized:
                self.initialize(X[0], Y[0])

            if X is not None and Y is not None:
                if self._partial_backward is not None:
                    for X_batch, Y_batch in zip(X, Y):
                        self.partial_fit(X_batch, Y_batch)

            self._backward(self, X, Y)

            self._fitted = True

            return self


class Model(Node):
    _nodes: List
    _edges: List
    _inputs: List
    _outputs: List
    _dispatcher: "DataDispatcher"

    def __init__(self, nodes=None, edges=None, name=None):
        params = {n.name: n.params for n in nodes}
        hypers = {n.name: n.hypers for n in nodes}
        super(Model, self).__init__(params=params,
                                    hypers=hypers,
                                    forward=forward,
                                    backward=backward,
                                    train=train,
                                    initializer=initializer,
                                    name=name)

        self._edges = edges
        self._inputs, self._outputs = find_entries_and_exits(nodes, edges)
        self._nodes = topological_sort(nodes, edges, self._inputs)

        self._dispatcher = DataDispatcher(self)

    def __repr__(self):
        klas = self.__class__.__name__
        nodes = [n.name for n in self._nodes]
        return f"'{self.name}': {klas}('" + "', '".join(nodes) + "')"

    def __getitem__(self, item):
        return self.get_node(item)

    def _check_input(self, x):
        if is_mapping(x):
            input_names = [n.name for n in self.input_nodes]

            for input_name in input_names:
                if input_name not in x:
                    raise NameError(
                        f"Missing input data for node '{input_name}' "
                        f"of model {self.name}.")

            for name, value in x.items():
                if name not in input_names:
                    raise NameError(f"Node '{name}' not found in input nodes "
                                    f"of model {self.name}.")
                value = check_vector(value)
                x[name] = value

                if self._is_initialized:
                    if value.shape[1] != self.get_node(name).input_dim:
                        raise ValueError(
                            f"Impossible to call node {name}: node input "
                            f"dimension is (1, "
                            f"{self.get_node(name).input_dim}) and input "
                            f"dimension is {x.shape}.")
        return x

    def _check_if_only_online(self):
        if any([n.is_trained_offline and not n.fitted for n in self.nodes]):
            raise RuntimeError(f"Impossible to train model {self.name} using "
                               f"online method: model contains untrained "
                               f"offline nodes.")

    def _load_proxys(self):
        for node in self._nodes:
            node._state_proxy = node.state()

    def _clean_proxys(self):
        for node in self._nodes:
            node._state_proxy = None

    def get_node(self, name):
        for n in self._nodes:
            if n.name == name:
                return n
        raise KeyError(f"No node named '{name}' found in "
                       f"model {self.name}.")

    @property
    def nodes(self):
        return self._nodes

    @property
    def node_names(self):
        return [n.name for n in self._nodes]

    @property
    def edges(self):
        return self._edges

    @property
    def input_dim(self):
        dims = [n.input_dim for n in self.input_nodes]
        if len(dims) < 2:
            return dims[0]
        return dims

    @property
    def output_dim(self):
        dims = [n.output_dim for n in self.output_nodes]
        if len(dims) < 2:
            return dims[0]
        return dims

    @property
    def input_nodes(self):
        return self._inputs

    @property
    def output_nodes(self):
        return self._outputs

    @property
    def trainable_nodes(self):
        return [n for n in self.nodes if
                getattr(n, "_backward", None) is not None]

    @property
    def feedback_nodes(self):
        return [n for n in self.nodes if n.has_feedback]

    @property
    def data_dispatcher(self):
        return self._dispatcher

    @contextmanager
    def with_state(self, state=None, stateful=False, reset=False):
        current_state = {n.name: n.state() for n in self.nodes}

        if state is None and not reset:
            yield self
            if not stateful:
                self.reset(to_state=current_state)
            return

        with ExitStack() as stack:
            for node in self.nodes:
                value = None
                if state is not None:
                    value = state.get(node.name)
                stack.enter_context(node.with_state(value,
                                                    stateful=stateful,
                                                    reset=reset))
            yield self

    @contextmanager
    def with_feedback(self, feedback=None, stateful=False, reset=False):

        if feedback is None and not reset:
            yield self
            return

        with ExitStack() as stack:
            for node in self.nodes:
                value = None
                if feedback is not None:
                    value = feedback.get(node.name)
                stack.enter_context(node.with_feedback(value,
                                                       stateful=stateful,
                                                       reset=reset))
            yield self

    def reset(self, to_state: Dict = None):
        """Reset the last state saved to zero or to
        another state value `from_state`.

        Parameters
        ----------
        to_state : np.ndarray, optional
            New state value for stateful
            computations, by default None.
        """
        if to_state is None:
            for node in self.nodes:
                node.reset()
        else:
            for node_name, current_state in to_state.items():
                self.get_node(node_name).reset(to_state=current_state)
        return self

    def zero_state(self):
        pass

    def initialize(self, x=None, y=None):
        self._is_initialized = False
        self._initializer(self, x=x)
        self.reset()
        self._is_initialized = True
        return self

    def call(self, x, forced_feedback=None, from_state=None, stateful=True,
             reset=False):
        x = self._check_input(x)

        if not self._is_initialized:
            self.initialize(x)

        # load current states in proxys interfaces accessible
        # through feedback. These proxys are not updated during the graph call.
        self._load_proxys()

        with self.with_state(from_state, stateful=stateful, reset=reset):
            # maybe load forced feedbacks in proxys ?
            with self.with_feedback(forced_feedback,
                                    stateful=stateful,
                                    reset=reset):
                self._forward(self, x)

                # wash states proxys
                self._clean_proxys()

                state = {}
                if len(self.output_nodes) > 1:
                    for out_node in self.output_nodes:
                        state[out_node.name] = out_node.state()
                else:
                    state = self.output_nodes[0].state()

        return state

    def run(self, X, forced_feedbacks=None, from_state=None, stateful=True,
            reset=False, shift_fb=True):

        if not self._is_initialized:
            if is_mapping(X):
                self.initialize({name: x[0] for name, x in X.items()})
            else:
                self.initialize(X[0])

        states = {n.name: np.zeros((X.shape[0], n.output_dim))
                  for n in self.output_nodes}

        with self.with_state(from_state, stateful=stateful, reset=reset):
            for i, (x, forced_feedback) in enumerate(
                    self._dispatcher.dispatch(X, forced_feedbacks,
                                              shift_fb=shift_fb)):
                state = self.call(x, forced_feedback=forced_feedback)

                if is_mapping(state):
                    for name, value in state.items():
                        states[name][i, :] = value
                else:
                    states[self.output_nodes[0].name][i, :] = state

        if len(states) == 1:
            return states[self.output_nodes[0].name]

        return states

    def train(self, x, y=None):
        x = self._check_input(x)

        if not self._is_initialized:
            self.initialize(x)

        self._check_if_only_online()

        with self.with_feedback(feedback=y):
            state = self._train(self, x, y)

        return state

    def partial_fit(self, X_batch, Y_batch=None):
        return self

    def fit(self, X=None, Y=None):

        X, Y = _to_ragged_array_seq(X), _to_ragged_array_seq(Y)
        X, Y = list(self._dispatcher.dispatch(X, Y, shift_fb=False))

        if not self._is_initialized:
            if is_mapping(X):
                self.initialize({name: x[0] for name, x in X.items()},
                                {name: y[0] for name, y in Y.items()})
            else:
                self.initialize(X[0], Y[0])

        offline_subgraphs = get_offline_subgraphs(self.nodes, self.edges)

        already_trained = set()
        for nodes, edges in offline_subgraphs:
            submodel = Model(nodes, edges, name=f"{uuid4()}")
            submodel.already_trained = already_trained

            # for seq/batch in dataset
            for x_seq, y_seq in zip(X, Y):
                states = self.run(x_seq, y_seq)
                self._partial_backward(submodel, states, y_seq)

            self._backward(submodel)

            already_trained |= set([n for n in submodel.nodes
                                    if n.is_trainable])

        self._backward(self, X, Y)

        return self
