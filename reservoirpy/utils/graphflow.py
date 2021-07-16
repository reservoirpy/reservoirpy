# Author: Nathan Trouvain at 12/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import defaultdict, namedtuple
from typing import Dict, List

import numpy as np

from .validation import is_mapping, is_node

DataPoint = namedtuple("DataPoint", "x, y")


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

    # using Kahn's algorithm
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
                           "to automatically determine operations "
                           "order in the model.")
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


class DataDispatcher:
    _inputs: List
    _parents: Dict

    def __init__(self, model):
        self._nodes = model.nodes
        self._trainables = model.trainable_nodes
        self._inputs = model.input_nodes
        self.__parents, _ = find_parents_and_children(model.edges)

        self._parents = self.__parents.copy()
        self._teachers = dict()

    def __getitem__(self, item):
        return self.get(item)

    def _check_inputs(self, input_mapping):
        if is_mapping(input_mapping):
            for node in self._inputs:
                if input_mapping.get(node.name) is None:
                    raise KeyError(f"Node {node.name} not found "
                                   f"in data mapping. This node requires "
                                   f"data to run.")

    def _check_targets(self, target_mapping):
        for node in self._nodes:
            if is_mapping(target_mapping):
                if node in self._trainables and target_mapping.get(
                        node.name) is None:
                    raise KeyError(f"Trainable node {node.name} not found "
                                   f"in target/feedback data mapping. This "
                                   f"node requires "
                                   f"target values.")

    def get(self, item):
        parents = self._parents.get(item, ())
        teacher = self._teachers.get(item, None)

        x = []
        for parent in parents:
            if is_node(parent):
                x.append(parent.state())
            else:
                x.append(parent)

        if len(x) == 1:
            x = x[0]

        return DataPoint(x=x, y=teacher)

    def load(self, X, Y=None):
        self._parents = self.__parents.copy()
        self._teachers = dict()

        self._check_inputs(X)

        for inp_node in self._inputs:
            if is_mapping(X):
                self._parents[inp_node] += [X[inp_node.name]]
            else:
                self._parents[inp_node] += [X]

        self._check_targets(Y)

        if Y is not None:
            for node in self._nodes:
                if is_mapping(Y):
                    if Y.get(node.name) is not None:
                        self._teachers[node] = Y.get(node.name)
                else:
                    if node in self._trainables:
                        self._teachers[node] = Y
        return self

    def dispatch(self, X, Y=None, shift_fb=True):

        if not is_mapping(X):
            X = {inp.name: X for inp in self._inputs}
        self._check_inputs(X)

        if Y is not None:
            if not is_mapping(Y):
                Y = {trainable.name: Y for trainable in self._trainables}
            self._check_targets(Y)

        # check is all sequences have same length,
        # taking the length of the first input sequence
        # as reference
        current_node = list(X.keys())[0]
        sequence_length = len(X[current_node])
        for node, sequence in X.items():
            if sequence_length != len(sequence):
                raise ValueError(f"Impossible to use data with inconsistent "
                                 f"number of timesteps: {node} is given "
                                 f"a sequence of length {len(sequence)} as "
                                 f"input "
                                 f" while {current_node} is given a sequence "
                                 f"of "
                                 f"length {sequence_length}")

        if Y is not None:
            for node, sequence in Y.items():
                if sequence_length != len(sequence):
                    raise ValueError(
                        f"Impossible to use data with inconsistent "
                        f"number of timesteps: {node} is given "
                        f"a sequence of length {len(sequence)} as "
                        f"targets/feedbacks while {current_node} is "
                        f"given a sequence of length {sequence_length}.")

        for i in range(sequence_length):
            x = {node: X[node][i, :] for node in X.keys()}
            if Y is not None:
                # if feedbacks vectors are meant to be fed
                # with a delay in time of one timestep w.r.t. 'X'
                if shift_fb:
                    if i == 0:
                        y = {node: None for node in
                             Y.keys()}
                    else:
                        y = {node: Y[node][i-1, :] for node in Y.keys()}
                # else assume that all feedback vectors must be instantaneously
                # fed to the network. This means that 'Y' already contains data
                # that is delayed by one timestep w.r.t. 'X'.
                else:
                    y = {node: Y[node][i, :] for node in Y.keys()}
            else:
                y = None

            yield x, y
