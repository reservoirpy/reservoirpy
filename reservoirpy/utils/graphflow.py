# Author: Nathan Trouvain at 12/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import defaultdict, namedtuple, deque
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
        raise RuntimeError("Model has a cycle: impossible "
                           "to automatically determine operations "
                           "order in the model.")
    else:
        return ordered_nodes


def get_offline_subgraphs(nodes, edges):

    global_inputs, global_outputs = find_entries_and_exits(nodes, edges)
    parents, children = find_parents_and_children(edges)

    # cut the graph:
    # all "blocking" nodes are disconnected from their children
    cuts = dict()
    for node in nodes:
        if node.is_trained_offline and not node.is_trained_online:
            cuts[node] = children.get(node, [])
            children[node] = []

    # run Kahn algorithm to create subgraphs going from
    # previous blocking nodes to the next ones
    # (or first from inputs to first blocking nodes).
    inputs = global_inputs.copy()
    prev_inputs = inputs.copy()
    subgraphs = []
    # while all blocking nodes have not been added to a subgraph as output:
    while len(cuts) > 0:
        subnodes = []
        subedges = []
        next_inputs = []
        inputs = deque(inputs)
        ascendants = {n: p.copy() for n, p in parents.items()}
        descendants = {n: c.copy() for n, c in children.items()}
        while len(inputs) > 0:
            n = inputs.popleft()
            subnodes.append(n)
            for m in descendants.get(n, ()):
                ascendants[m].remove(n)
                subedges.append((n, m))
                if ascendants.get(m) is None or len(ascendants[m]) == 0:
                    inputs.append(m)

            if hasattr(n, "stop") and n not in global_outputs:
                next_inputs.append(n)
                # output of the previous subgraph will serve as inputs for
                # the next.

            if n in cuts:
                # current node is blocking: add it to current subgraph
                # and remove the seal diconnecting it from its children.
                # It will be used as a standard node in the next subgraph
                # creation loop.
                child = cuts.pop(n)
                children[n] = child

        # rollback to find minimal requirements for current subgraphs inputs:
        # if current subgraph inputs are not the global inputs of the global
        # graph, it means they have predecessors probably already integrated
        # to previously constructed subgraphs. Add all the predecessors to
        # current subgraph.
        for inp in prev_inputs:
            if inp not in global_inputs:
                pred_nodes, pred_edges = find_predecessors_of(inp, edges)
                subnodes = set(subnodes) | set(pred_nodes)
                subedges = set(subedges) | set(pred_edges)

        # would be quicker to not redo a topo sort...
        subgraphs.append(topological_sort(list(subnodes), list(subedges)))

        # inputs for the next subgraphs are current blocking nodes used as
        # inputs for the current subgraph.
        prev_inputs = next_inputs.copy()
        inputs = deque(next_inputs)

    return subgraphs


def find_predecessors_of(node, edges):
    precedent_nodes = []
    precedent_egdes = []
    for edge in edges:
        if edge[1] is node:
            precedent_nodes.append(edge[0])
            precedent_egdes.append(edge)

            prec_nodes, prec_edges = find_predecessors_of(edge[0], edges)
            precedent_nodes.extend(prec_nodes)
            precedent_egdes.extend(prec_edges)

    return precedent_nodes, precedent_egdes


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
