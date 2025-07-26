"""
===================================
Models (:class:`reservoirpy.Model`)
===================================

Note
----

See the following guides to:

- **Learn more about how to work with ReservoirPy Nodes**: :ref:`node`
- **Learn more about how to combine nodes within a Model**: :ref:`model`


Models are an extension of the Node API. They allow to combine nodes into
complex computational graphs, to create complicated Reservoir Computing
architecture like *Deep Echo State Networks*.

See :ref:`model` to learn more about how to create and manipulate
a :py:class:`Model`.

.. currentmodule:: reservoirpy.model

.. autoclass:: Model

   .. rubric:: Methods

   .. autosummary::

      ~Model.call
      ~Model.fit
      ~Model.get_node
      ~Model.initialize
      ~Model.initialize_buffers
      ~Model.reset
      ~Model.run
      ~Model.train
      ~Model.update_graph
      ~Model.with_state


   .. rubric:: Attributes

   .. autosummary::

      ~Model.data_dispatcher
      ~Model.edges
      ~Model.fitted
      ~Model.hypers
      ~Model.input_dim
      ~Model.input_nodes
      ~Model.is_empty
      ~Model.is_initialized
      ~Model.is_trainable
      ~Model.is_trained_offline
      ~Model.is_trained_online
      ~Model.name
      ~Model.node_names
      ~Model.nodes
      ~Model.output_dim
      ~Model.output_nodes
      ~Model.params
      ~Model.trainable_nodes

.. autoclass:: FrozenModel

"""

# Author: Nathan Trouvain at 01/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import defaultdict
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from typing_extensions import assert_type

from .node import Node, OnlineNode, ParallelNode, TrainableNode
from .type import (
    ModelInput,
    ModelTimestep,
    MultiTimeseries,
    NodeInput,
    State,
    Timeseries,
    Timestep,
    is_multiseries,
    timestep_from_input,
)
from .utils.graphflow import (
    find_inputs,
    find_outputs,
    find_parents_and_children,
    topological_sort,
    unique_ordered,
)
from .utils.model_utils import (
    check_input_output_connections,
    check_unnamed_in_out,
    check_unnamed_trainable,
    unfold_mapping,
)


class Model:
    """Model base class.

    Parameters
    ----------
    nodes : list of Node, optional
        Nodes to include in the Model.
    edges : list of (Node, Node), optional
        Edges between Nodes in the graph. An edge between a
        Node A and a Node B is created as a tuple (A, B).
    """

    nodes: list[Node]
    edges: list[tuple[Node, Node]]
    inputs: list[Node]
    outputs: list[Node]
    named_nodes: dict[str, Node]
    trainable_nodes: list[Node]
    execution_order: list[Node]
    parents: dict[Node, list[Node]]
    children: dict[Node, list[Node]]

    is_trainable: bool
    is_multi_input: bool
    is_multi_output: bool
    is_online: bool
    is_parallel: bool
    initialized: bool

    def __init__(
        self,
        nodes: Sequence[Node],
        edges: Sequence[tuple[Node, Node]],
    ):
        # convert to List[Node], removes duplicates, use dict to preserve order
        self.nodes = unique_ordered(nodes)
        self.edges = unique_ordered(edges)

        self.inputs = find_inputs(self.nodes, self.edges)
        self.outputs = find_outputs(self.nodes, self.edges)
        self.named_nodes = {n.name: n for n in self.nodes if n.name is not None}
        self.trainable_nodes = [n for n in nodes if isinstance(n, TrainableNode)]
        self.is_trainable = len(self.trainable_nodes) > 0
        self.is_multi_input = len(self.inputs) > 1
        self.is_multi_output = len(self.outputs) > 1
        self.is_online = all([isinstance(n, OnlineNode) for n in self.trainable_nodes])
        self.is_parallel = all(
            [isinstance(n, ParallelNode) for n in self.trainable_nodes]
        )
        self.parents, self.children = find_parents_and_children(self.nodes, self.edges)

        # execution order / cycle detection
        self.execution_order = topological_sort(
            self.nodes, self.edges, inputs=self.inputs
        )

        self.initialized = False

    def initialize(
        self,
        x: Union[ModelInput, ModelTimestep],
        y: Optional[Union[ModelInput, ModelTimestep]] = None,
    ):
        """Initializes a :py:class:`Model` instance at runtime, using samples of
        data to infer all :py:class:`Node` dimensions.

        Parameters
        ----------
        x : numpy.ndarray or dict of numpy.ndarray
            A vector of shape `(1, ndim)` corresponding to a timestep of data, or
            a dictionary mapping node names to vector of shapes
            `(1, ndim of corresponding node)`.
        y : numpy.ndarray or dict of numpy.ndarray, optional
            A vector of shape `(1, ndim)` corresponding to a timestep of target
            data, or a dictionary mapping node names to vector of
            shapes `(1, ndim of corresponding node)`.
        """
        check_unnamed_in_out(self)
        check_input_output_connections(self.edges)
        check_unnamed_trainable(self)

        # Turn y into a dict[Node, NodeInput]

        if isinstance(y, Mapping):
            y_ = {self.named_nodes[name]: val for name, val in y.items()}
        elif y is None:
            y_ = {}
        else:
            [trainable_node] = self.trainable_nodes
            y_ = {trainable_node: y}

        # Initialize node_inputs from the model input

        node_inputs = {node: np.empty((0,)) for node in self.nodes}

        if isinstance(x, dict):
            for node_name, val in x.items():
                node = self.named_nodes[node_name]
                val = timestep_from_input(val)
                node_inputs[node] = np.concatenate((node_inputs[node], val), axis=-1)
        else:
            [node] = self.inputs
            x = timestep_from_input(x)
            node_inputs[node] = np.concatenate((node_inputs[node], x), axis=-1)

        # Initialize each node in execution_order

        for node in self.execution_order:
            node_input = node_inputs[node]
            if node.initialized:
                if node_input.shape[-1] != node.input_dim:
                    raise ValueError(
                        f"{node} expects input of dimension {node.input_dim}"
                        f"but receives input of dimension {node_input.shape[-1]}"
                    )
            else:
                if node in y_:
                    node.initialize(x=node_input, y=y_[node])
                else:
                    node.initialize(x=node_input)
            out = np.zeros((node.output_dim,))
            for child in self.children[node]:
                node_inputs[child] = np.concatenate((node_inputs[child], out), axis=-1)

        # TODO: Jax compilation
        self.initialized = True

    def _step(self, state: dict[Node, State], x: ModelTimestep) -> dict[Node, State]:

        new_state: dict[Node, State] = {}

        for node in self.execution_order:
            inputs = []
            if isinstance(x, dict):
                if node.name in x:
                    inputs.append(x[node.name])
            else:
                if self.inputs[0] == node:
                    inputs.append(x)
            inputs += [new_state[parent]["out"] for parent in self.parents[node]]
            node_input = np.concatenate(inputs, axis=-1)
            new_state[node] = node._step(state[node], node_input)

        return new_state

    def step(self, x: Optional[ModelTimestep]) -> ModelTimestep:
        # Auto-regressive mode
        if x is None:
            x = np.zeros((0,))

        if not self.initialized:
            self.initialize(x)

        state = {node: node.state for node in self.nodes}
        new_state = self._step(state, x)

        for node in new_state:
            node.state = new_state[node]

        if len(self.outputs) == 1:
            return new_state[self.outputs[0]]["out"]
        else:
            return {node.name: new_state[node]["out"] for node in self.outputs}

    def _run(
        self, state: dict[Node, State], x: Union[Timeseries, dict[str, Timeseries]]
    ) -> tuple[dict[Node, State], dict[Node, Timeseries]]:

        output_timeseries: dict[Node, Timeseries] = {}
        new_state: dict[Node, State] = {}

        for node in self.execution_order:
            inputs = []
            if isinstance(x, dict):
                if node.name in x:
                    inputs += x[node.name]
            else:
                if self.inputs == [node]:
                    inputs += x
            inputs += [output_timeseries[parent] for parent in self.parents[node]]
            node_input = np.concatenate(inputs, axis=-1)
            new_state[node], output_timeseries[node] = node._run(
                state[node], node_input
            )

        return new_state, output_timeseries

    def run(self, x: Optional[ModelInput], iters: Optional[int] = None) -> ModelInput:
        # Auto-regressive mode
        if x is None:
            x_: ModelInput = np.zeros((iters, 0))
        else:
            x_ = x

        if not self.initialized:
            self.initialize(x_)

        previous_states = {node: node.state for node in self.nodes}
        if is_multiseries(x_):
            result: dict[Node, list[Timeseries]] = defaultdict(list)
            iterable_x = unfold_mapping(x_) if isinstance(x_, dict) else x_

            for timeseries in iterable_x:
                new_state, output = self._run(previous_states, timeseries)
                for node in output:
                    result[node].append(output[node])

        else:
            new_state, result = self._run(previous_states, x_)

        for node in new_state:
            node.state = new_state[node]

        if len(self.outputs) == 1:
            return result[self.outputs[0]]
        else:
            return {node.name: result[node] for node in self.outputs}

    def partial_fit(self, x: ModelInput, y: Optional[ModelInput]) -> ModelInput:
        ...

    def fit(
        self, x: ModelInput, y: Optional[ModelInput], warmup: int = 0, workers: int = 1
    ) -> "Model":
        if not self.initialized:
            self.initialize(x, y)
        ...

    def __call__(self, x: Optional[ModelTimestep]) -> ModelTimestep:
        return self.step(x)

    def __rshift__(
        self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]
    ) -> "Model":
        from .ops import link

        return link(self, other)

    def __rrshift__(
        self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]
    ) -> "Model":
        from .ops import link

        return link(other, self)

    def __lshift__(
        self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]
    ) -> "Model":
        raise NotImplementedError()

    def __rlshift__(
        self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]
    ) -> "Model":
        raise NotImplementedError()

    def __and__(
        self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]
    ) -> "Model":
        from .ops import merge

        return merge(self, other)

    def __rand__(
        self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]
    ) -> "Model":
        from .ops import merge

        return merge(other, self)
