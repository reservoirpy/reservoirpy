# Author: Nathan Trouvain at 01/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Callable, Dict, List

import numpy as np


def link(input_node, output_node):
    if isinstance(input_node, Node):
        if isinstance(output_node, Node):
            ...


def feedback(sender_node, receiver_node):
    ...


class _Node:

    _params: Dict
    _initializers: Dict
    _feedbacks: List
    _function: Callable
    _output: np.ndarray
    _state: np.ndarray

    def __init__(self):
        ...

    def connect(self, node):
        if isinstance(node, Model):
            ...
        elif isinstance(node, Node):
            ...
        else:
            raise ValueError(f"Can not connect to node of type {type(node)}.")

    def run(self, inputs):
        ...

    def fit(self, inputs, teachers):
        ...


class Node:
    pass


class Model(_Node):

    _nodes: List
    _links: List
    _parents: Dict
    _children: Dict
    _inputs: List
    _outputs: List

    @property
    def input_nodes(self):
        if self._inputs is None:
            self._inputs = self._find_input_nodes()
        return self._inputs

    @property
    def output_nodes(self):
        if self._outputs is None:
            self._outputs = self._find_output_nodes()
        return self._outputs

    def __init__(self):
        ...

    def _find_input_nodes(self):
        inputs = []
        for node in self._nodes:
            if self._parents.get(node) is None:
                inputs.append(node)
        return inputs

    def _find_output_nodes(self):
        inputs = []
        for node in self._nodes:
            if self._children.get(node) is None:
                inputs.append(node)
        return inputs

    def _topological_sort(self):
        # using Kahn's alogorithm
        ordered_nodes = []
        links = set(self._links)
        inputs = set(self.input_nodes)
        while len(inputs) > 0:
            n = inputs.pop()
            ordered_nodes.append(n)
            for m in self._children.get(n, ()):
                links.remove((n, m))
                if self._parents.get(m) is None:
                    inputs.add(m)
        if len(links) > 0:
            raise RuntimeError("Model has a cycle: impossible"
                               "to automatically determine ops order.")
        else:
            return ordered_nodes

    def connect(self, node):
        ...

    def register_feedback(self, sender, receiver):
        ...
