# Author: Nathan Trouvain at 07/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from contextlib import ExitStack, contextmanager
from itertools import product
# Author: Nathan Trouvain at 01/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Callable, Dict, List

import numpy as np

from .utils.graphflow import (DataDispatcher,
                              find_entries_and_exits,
                              topological_sort, )
from .utils.validation import check_vector, is_mapping


def forward(model: "Model", x):
    data = model.data_dispatcher.load(x)

    for node in model.nodes:
        node(data[node].x)

    return [out_node.state() for out_node in model.output_nodes]


def initializer(model: "Model", x, y=None):
    data = model.data_dispatcher.load(x, y)

    # first, probe network to init forward flow
    for node in model.nodes:
        node.initialize(x=data[node].x, y=data[node].y)

    # second, probe feedback demanding nodes to
    # init feedback flow
    for fb_node in model.feedback_nodes:
        fb_node.initialize_feedback()


def link(node1: "Node", node2: "Node") -> "Model":
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


def combine(*models):
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
    _input_dim: int
    _output_dim: int
    _forward: Callable
    _initializer: Callable
    _name: str
    _state_proxy: np.ndarray
    _factory_id: int = -1
    _registry: Dict = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._factory_id = -1
        cls._registry = dict()

    def __init__(self, params=None, hypers=None, forward=None,
                 initializer=None, input_dim=None, output_dim=None,
                 name=None):

        self._params = dict() if params is None else params
        self._hypers = dict() if hypers is None else hypers
        self._forward = forward
        self._initializer = initializer
        self._input_dim = input_dim
        self._output_dim = output_dim

        self._name = self._get_name(name)

        self._is_initialized = False
        self._state_proxy = None

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
                f"Impossible to set state of node {self.name}: node"
                f"is not initialized yet.")

        if s.shape[1] != self.output_dim:
            raise ValueError(f"Impossible to set state of node {self.name}: "
                             f"dimension mismatch between state vector ("
                             f"{s.shape[1]}) "
                             f"and node output dim ({self.output_dim}).")
        return s

    def _check_input(self, x):
        x = check_vector(x)

        if self._is_initialized:
            if x.shape[1] != self.input_dim:
                raise ValueError(
                    f"Impossible to call node {self.name}: node input "
                    f"dimension is (1, {self.input_dim}) and input dimension "
                    f"is {x.shape}.")
        return x

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
    def is_initialized(self):
        return self._is_initialized

    def state(self):
        if not self.is_initialized:
            return None
        return self._state

    def state_proxy(self):
        if self._state_proxy is None:
            return self._state
        return self._state_proxy

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
    def with_feedback(self, feedback=None, reset=False):
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

    def link(self, other):
        if isinstance(other, Node) or \
                hasattr(other, "__getitem__") and \
                all([isinstance(obj, Node) for obj in other]):
            return link(self, other)
        else:
            raise TypeError(f"Impossible to link node {self.name} with"
                            f"oject of type {type(other)}.")

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


class Model(Node):
    _nodes: List
    _edges: List
    _inputs: List
    _outputs: List
    _dispatcher: "DataDispatcher"

    def __init__(self, nodes=None, edges=None):
        params = {n.name: n.params for n in nodes}
        hypers = {n.name: n.hypers for n in nodes}
        super(Model, self).__init__(params=params,
                                    hypers=hypers,
                                    forward=forward,
                                    initializer=initializer)

        self._edges = edges
        self._inputs, self._outputs = find_entries_and_exits(nodes, edges)
        self._nodes = topological_sort(nodes, edges, self._inputs)
        self._fb_nodes = [n for n in self.nodes if hasattr(n, "feedback")]
        self._trainables = [n for n in self.nodes if hasattr(n, "fit")]

        self._dispatcher = DataDispatcher(self)

    def __repr__(self):
        klas = self.__class__.__name__
        nodes = [n.name for n in self._nodes]
        return f"'{self.name}': {klas}('" + "', '".join(nodes) + "')"

    def __getitem__(self, item):
        return self.get_node(item)

    def _check_input(self, x):
        # TODO : handle mappings
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
        return self._trainables

    @property
    def feedback_nodes(self):
        return self._fb_nodes

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
    def with_feedback(self, feedback=None, reset=False):

        if feedback is None and not reset:
            yield self
            return

        with ExitStack() as stack:
            for node in self.nodes:
                value = None
                if feedback is not None:
                    value = feedback.get(node.name)
                stack.enter_context(node.with_feedback(value,
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

    def initialize(self, x=None):
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
            with self.with_feedback(forced_feedback, reset=reset):
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
