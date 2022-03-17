# Author: Nathan Trouvain at 12/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import defaultdict, deque, namedtuple
from typing import Dict, List

import numpy as np

from .._base import _Node
from . import safe_defaultdict_copy
from .validation import is_mapping

DataPoint = namedtuple("DataPoint", "x, y")


def find_parents_and_children(edges):
    """Returns two dicts linking nodes to their parents and children in the graph."""
    parents = defaultdict(list)
    children = defaultdict(list)

    for edge in edges:
        parent, child = edge
        parents[child] += [parent]
        children[parent] += [child]

    return parents, children


def topological_sort(nodes, edges, inputs=None):
    """Topological sort of nodes in a Model, to determine execution order."""
    if inputs is None:
        inputs, _ = find_entries_and_exits(nodes, edges)

    parents, children = find_parents_and_children(edges)

    # using Kahn's algorithm
    ordered_nodes = []
    edges = set(edges)
    inputs = deque(inputs)
    while len(inputs) > 0:
        n = inputs.pop()
        ordered_nodes.append(n)
        for m in children.get(n, ()):
            edges.remove((n, m))
            parents[m].remove(n)
            if parents.get(m) is None or len(parents[m]) < 1:
                inputs.append(m)
    if len(edges) > 0:
        raise RuntimeError(
            "Model has a cycle: impossible "
            "to automatically determine operations "
            "order in the model."
        )
    else:
        return ordered_nodes


def get_offline_subgraphs(nodes, edges):
    """Cut a graph into several subgraphs where output nodes are untrained offline
    learner nodes."""
    inputs, outputs = find_entries_and_exits(nodes, edges)
    parents, children = find_parents_and_children(edges)

    offlines = set(
        [n for n in nodes if n.is_trained_offline and not n.is_trained_online]
    )
    included, trained = set(), set()
    subgraphs, required = [], []
    _nodes = nodes.copy()
    while trained != offlines:
        subnodes, subedges = [], []
        for node in _nodes:
            if node in inputs or all([p in included for p in parents.get(node)]):

                if node.is_trained_offline and node not in trained:
                    trained.add(node)
                    subnodes.append(node)
                else:
                    if node not in outputs:
                        subnodes.append(node)
                    included.add(node)

        subedges = [
            edge for edge in edges if edge[0] in subnodes and edge[1] in subnodes
        ]
        subgraphs.append((subnodes, subedges))
        _nodes = [n for n in nodes if n not in included]

    required = _get_required_nodes(subgraphs, children)

    return list(zip(subgraphs, required))


def _get_required_nodes(subgraphs, children):
    """Get nodes whose outputs are required to run/fit children nodes."""
    req = []
    fitted = set()
    for i in range(1, len(subgraphs)):
        currs = set(subgraphs[i - 1][0])
        nexts = set(subgraphs[i][0])

        req.append(_get_links(currs, nexts, children))

        fitted |= set([node for node in currs if node.is_trained_offline])

    nexts = set(
        [n for n in subgraphs[-1][0] if n.is_trained_offline and n not in fitted]
    )
    currs = set(
        [n for n in subgraphs[-1][0] if not n.is_trained_offline or n in fitted]
    )

    req.append(_get_links(currs, nexts, children))

    return req


def _get_links(previous, nexts, children):
    """Returns graphs edges between two subgraphs."""
    links = {}
    for n in previous:
        next_children = []
        if n not in nexts:
            next_children = [c.name for c in children.get(n, []) if c in nexts]

        if len(next_children) > 0:
            links[n.name] = next_children

    return links


def find_entries_and_exits(nodes, edges):
    """Find outputs and inputs nodes of a directed acyclic graph."""
    nodes = set(nodes)
    senders = set([n for n, _ in edges])
    receivers = set([n for _, n in edges])

    lonely = nodes - senders - receivers

    entrypoints = senders - receivers | lonely
    endpoints = receivers - senders | lonely

    return list(entrypoints), list(endpoints)


class DataDispatcher:
    """A utility used to feed data to nodes in a Model."""

    _inputs: List
    _parents: Dict

    def __init__(self, model):
        self._nodes = model.nodes
        self._trainables = model.trainable_nodes
        self._inputs = model.input_nodes
        self.__parents, _ = find_parents_and_children(model.edges)

        self._parents = safe_defaultdict_copy(self.__parents)
        self._teachers = dict()

    def __getitem__(self, item):
        return self.get(item)

    def _check_inputs(self, input_mapping):
        if is_mapping(input_mapping):
            for node in self._inputs:
                if input_mapping.get(node.name) is None:
                    raise KeyError(
                        f"Node {node.name} not found "
                        f"in data mapping. This node requires "
                        f"data to run."
                    )

    def _check_targets(self, target_mapping):
        if is_mapping(target_mapping):
            for node in self._nodes:
                if (
                    node in self._trainables
                    and not node.fitted
                    and target_mapping.get(node.name) is None
                ):
                    raise KeyError(
                        f"Trainable node {node.name} not found "
                        f"in target/feedback data mapping. This "
                        f"node requires "
                        f"target values."
                    )

    def _format_xy(self, X, Y):
        if not is_mapping(X):
            X_map = {inp.name: X for inp in self._inputs}
        else:
            X_map = X.copy()
        self._check_inputs(X_map)

        Y_map = None
        if Y is not None:
            if not is_mapping(Y):
                Y_map = {trainable.name: Y for trainable in self._trainables}
            else:
                Y_map = Y.copy()
            self._check_targets(Y_map)

        # check if all sequences have same length,
        # taking the length of the first input sequence
        # as reference
        current_node = list(X_map.keys())[0]
        sequence_length = len(X_map[current_node])
        for node, sequence in X_map.items():
            if sequence_length != len(sequence):
                raise ValueError(
                    f"Impossible to use data with inconsistent "
                    f"number of timesteps: {node} is given "
                    f"a sequence of length {len(sequence)} as "
                    f"input "
                    f"while {current_node} is given a sequence "
                    f"of length {sequence_length}"
                )

        if Y_map is not None:
            # Pad teacher nodes in a sequence
            for node, value in Y_map.items():
                if isinstance(value, _Node):
                    Y_map[node] = [value for _ in range(sequence_length)]

            for node, sequence in Y_map.items():
                # Y_map might be a teacher node (not a sequence)
                if hasattr(Y_map, "__len__"):
                    if sequence_length != len(sequence):
                        raise ValueError(
                            f"Impossible to use data with inconsistent "
                            f"number of timesteps: {node} is given "
                            f"a sequence of length {len(sequence)} as "
                            f"targets/feedbacks while {current_node} is "
                            f"given a sequence of length {sequence_length}."
                        )

        return X_map, Y_map, sequence_length

    def get(self, item):
        parents = self._parents.get(item, ())
        teacher = self._teachers.get(item, None)

        x = []
        for parent in parents:
            if isinstance(parent, _Node):
                x.append(parent.state())
            else:
                x.append(parent)

        # in theory, only operators can support several incoming signal
        # i.e. several operands, so unpack data is the list is unecessary
        if len(x) == 1:
            x = x[0]

        return DataPoint(x=x, y=teacher)

    def load(self, X=None, Y=None):
        """Load input and target data for dispatch."""
        self._parents = safe_defaultdict_copy(self.__parents)
        self._teachers = dict()

        if X is not None:
            self._check_inputs(X)
            if is_mapping(X):
                for node in self._nodes:
                    if X.get(node.name) is not None:
                        self._parents[node] += [X[node.name]]

            else:
                for inp_node in self._inputs:
                    self._parents[inp_node] += [X]

        if Y is not None:
            self._check_targets(Y)
            for node in self._nodes:
                if is_mapping(Y):
                    if Y.get(node.name) is not None:
                        self._teachers[node] = Y.get(node.name)
                else:
                    if node in self._trainables:
                        self._teachers[node] = Y
        return self

    def dispatch(
        self,
        X,
        Y=None,
        shift_fb=True,
        return_targets=False,
        force_teachers=True,
        format_xy=True,
    ):
        """Transform data from a dict of arrays
        ([node], timesteps, dimension) to an iterator yielding
        a node: data mapping for each timestep."""
        if format_xy:
            X_map, Y_map, sequence_length = self._format_xy(X, Y)
        else:
            X_map, Y_map = X, Y
            current_node = list(X_map.keys())[0]
            sequence_length = len(X_map[current_node])

        for i in range(sequence_length):
            x = {node: X_map[node][i] for node in X_map.keys()}
            if Y_map is not None:
                y = None
                if return_targets:
                    y = {node: np.atleast_2d(Y_map[node][i]) for node in Y_map.keys()}
                # if feedbacks vectors are meant to be fed
                # with a delay in time of one timestep w.r.t. 'X_map'
                if shift_fb:
                    if i == 0:
                        if force_teachers:
                            fb = {
                                node: np.zeros_like(Y_map[node][i])
                                for node in Y_map.keys()
                            }
                        else:
                            fb = {node: None for node in Y_map.keys()}
                    else:
                        fb = {node: Y_map[node][i - 1] for node in Y_map.keys()}
                # else assume that all feedback vectors must be instantaneously
                # fed to the network. This means that 'Y_map' already contains
                # data that is delayed by one timestep w.r.t. 'X_map'.
                else:
                    fb = {node: Y_map[node][i] for node in Y_map.keys()}
            else:
                fb = y = None

            yield x, fb, y
