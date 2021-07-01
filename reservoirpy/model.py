# Author: Nathan Trouvain at 01/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import threading
from typing import Callable, Dict, List, Tuple
from itertools import product
from collections import defaultdict

import numpy as np

from .utils.validation import is_mapping, is_numerical


def forward(model, x):

    data = model.data.init(x)

    y = None
    for node in model.nodes:
        y = node(data[node])

    return y


def find_parents_and_children(edges):
    parents = defaultdict(list)
    children = defaultdict(list)

    for edge in edges:
        parent, child = edge
        parents[child] += [parent]
        children[parent] += [child]

    return parents, children


def topological_sort(nodes, edges, inputs=None):

    if inputs is None:
        inputs, _ = find_entries_and_exits(nodes, edges)

    parents, children = find_parents_and_children(edges)

    # using Kahn's alogorithm
    ordered_nodes = []
    edges = set(edges)
    inputs = set(inputs)
    while len(inputs) > 0:
        n = inputs.pop()
        ordered_nodes.append(n)
        for m in children.get(n, ()):
            edges.remove((n, m))
            parents[m].remove(n)
            if parents.get(m) is None or len(parents[m]) < 1:
                inputs.add(m)
    if len(edges) > 0:
        raise RuntimeError("Model has a cycle: impossible "
                           "to automatically determine nodes order.")
    else:
        return ordered_nodes


def find_entries_and_exits(nodes, edges):
    nodes = set(nodes)
    senders = set([n for n, _ in edges])
    receivers = set([n for _, n in edges])

    lonely = nodes - senders - receivers
    if len(lonely) > 0:
        raise RuntimeError("Model has lonely nodes, connected to "
                           "no inputs and no outputs.")

    entrypoints = senders - receivers
    endpoints = receivers - senders

    return list(entrypoints), list(endpoints)


def link(node1: "Node", node2: "Node") -> "Model":
    # fetch all nodes in the two subgraphs.
    all_nodes = []
    for node in (node1, node2):
        if hasattr(node, "nodes"):
            all_nodes += node.nodes
        else:
            all_nodes += [node]

    # fetch all edges in the two subgraphs.
    all_edges = []
    for node in (node1, node2):
        if hasattr(node, "edges"):
            all_nodes += node.edges
        else:
            all_nodes += [node]

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

    # all outputs from subgraph 1 are connected to
    # all inputs from subgraph 2.
    all_edges += list(product(senders, receivers))

    # pack everything
    return Model(nodes=all_nodes, edges=all_edges)


def merge(*models):

    all_nodes = set()
    all_edges = set()
    for model in models:
        all_nodes |= set(model.nodes)
        all_edges |= set(model.edges)

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
    _factory_id: int = -1

    @classmethod
    def _get_name(cls, name=None):
        if name is None:
            cls._factory_id += 1
            _id = cls._factory_id
            return f"{cls.__name__}-{_id}"
        else:
            return name

    def __init__(self, params=None, hypers=None, forward=None,
                 initializer=None, input_dim=None, output_dim=None,
                 name=None):

        self._params = dict() if params is None else params
        self._hypers = dict() if hypers is None else hypers
        self._forward = forward
        self._initializer = initializer
        self._input_dim = input_dim
        self._output_dim = output_dim

        self._name = Node._get_name(name)

        self._is_initialized = False

    def __repr__(self):
        return f"{self.name}"

    def __getattr__(self, item):
        return self.get_param(item)

    def __call__(self, x, from_state=None, stateful=True):
        return self.call(x, from_state=from_state, stateful=stateful)

    def _check_state(self, s):
        if not is_numerical(s):
            if hasattr(s, "dtype"):
                klas = s.dtype
            else:
                klas = type(s)
            raise TypeError(f"Impossible to set state of node {self.name}: new state "
                            f"is not numeric. State type is {klas}.")

        if not self._is_initialized:
            raise RuntimeError(f"Impossible to set state of node {self.name}: node"
                               f"is not initialized yet.")

        if s.ndim < 2:
            s = s[np.newaxis, :]
        if s.shape[1] != self.output_dim:
            raise ValueError(f"Impossible to set state of node {self.name}: "
                             f"dimension mismatch between state vector ({s.shape[1]}) "
                             f"and node output dim ({self.output_dim}).")
        return s

    def _check_input(self, x):
        if not is_numerical(x):
            if hasattr(x, "dtype"):
                klas = x.dtype
            else:
                klas = type(x)
            raise TypeError(f"Impossible to call node {self.name}: inputs "
                            f"are not numeric. Inputs type is {klas}.")

        if x.ndim < 2:
            x = x[np.newaxis, :]

        if self._is_initialized:
            if x.shape[1] != self.input_dim:
                raise ValueError(f"Impossible to call node {self.name}: node input "
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

    def initialize(self, x=None):
        self._initializer(self, x=x)
        self.reset()
        self._is_initialized = True

    def state(self):
        return self._state

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

    def zero_state(self):
        """A null state vector."""
        return np.zeros((1, self.output_dim))

    def link(self, other):
        if isinstance(other, Node):
            return link(self, other)
        else:
            raise TypeError(f"Impossible to link node {self.name} with"
                            f"oject of type {type(other)}.")

    def call(self, x, from_state=None, stateful=True):
        x = self._check_input(x)

        if not self._is_initialized:
            self.initialize(x)

        if not stateful:
            from_state = self.zero_state()

        if from_state is not None:
            self.reset(to_state=from_state)

        state = self._forward(self, x)

        if stateful:
            self._state = state

        return state

    def run(self, X, from_state=None, stateful=True):

        if not self._is_initialized:
            # send a probe to infer shapes and initialize params
            self.call(X[0])

        if not stateful:
            last_state = self.state().copy()
            self.reset()

        if from_state is not None:
            self.reset(to_state=from_state)

        states = np.zeros((X.shape[0], self.output_dim))
        for i, x in enumerate(X):
            s = self.call(x)
            states[i, :] = s

        if not stateful:
            self.reset(to_state=last_state)

        return states

    def __rshift__(self, other):
        return self.link(other)


class Model(Node):

    _nodes: List
    _edges: List
    _forward: Callable
    _inputs: List
    _outputs: List
    _data: "DataDispatcher"

    def __init__(self, nodes=None, edges=None, feedbacks=None):
        params = {n.name: n.params for n in nodes}
        hypers = {n.name: n.hypers for n in nodes}
        super(Model, self).__init__(params=params,
                                    hypers=hypers,
                                    forward=forward)

        self._edges = edges
        self._inputs, self._outputs = find_entries_and_exits(nodes, edges)
        self._nodes = topological_sort(nodes, edges, self._inputs)
        self._data = DataDispatcher(self)

    def __getitem__(self, item):
        return self.get_node(item)

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
    def edges(self):
        return self._edges

    @property
    def input_nodes(self):
        return self._inputs

    @property
    def output_nodes(self):
        return self._outputs

    @property
    def data(self):
        return self._data


class Input(Node):

    def __init__(self):
        super(Input, self).__init__(forward=lambda x: x)


class DataDispatcher:

    data: Tuple
    inputs: List
    parents: Dict

    def __init__(self, model):
        self.nodes = model.nodes
        self.inputs = model.input_nodes
        self.parents, _ = find_parents_and_children(model.edges)

    def __getitem__(self, item):
        parents = self.parents[item]
        if len(parents) > 1:
            data = []
            for parent in parents:
                data.append(parent.state())
        else:
            data = parents[0].state()
        return data

    def get(self, item):
        parents = self.parents.get(item, ())
        data = []
        for parent in parents:
            data.append(parent.state())
        return data

    def init(self, data, Y=None):
        if is_mapping(data):
            for inp_node in self.inputs:
                if len(self.parents[inp_node]) < 1:
                    self.parents[inp_node] += [Input()]
                self.parents[inp_node][0](data.get(inp_node))
        else:
            for inp_node in self.inputs:
                if len(self.parents[inp_node]) < 1:
                    self.parents[inp_node] += [Input()]
                self.parents[inp_node][0](data)
        return self
