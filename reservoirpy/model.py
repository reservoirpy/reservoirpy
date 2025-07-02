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
from typing import Optional, Sequence, Tuple

import numpy as np

from .node import Node, OnlineNode, ParallelNode, TrainableNode
from .type import MappedData, NodeInput, Timestep, timestep_from_input
from .utils.graphflow import (
    DataDispatcher,
    find_inputs,
    find_outputs,
    find_parents_and_children,
    topological_sort,
    unique_ordered,
)
from .utils.model_utils import check_input_output_connections, check_unnamed_in_out
from .utils.validation import is_mapping


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
    edges: list[Tuple[Node, Node]]
    _inputs: list[Node]
    _outputs: list[Node]
    _named_nodes: dict[str, Node]
    _dispatcher: "DataDispatcher"
    initialized: bool
    is_trainable: bool
    is_multi_input: bool
    is_multi_output: bool
    is_online: bool
    is_parallel: bool

    def __init__(
        self,
        nodes: Sequence[Node],
        edges: Sequence[Tuple[Node, int, Node]],
    ):
        # convert to List[Node], removes duplicates, use dict to preserve order
        self.nodes = unique_ordered(nodes)
        self.edges = unique_ordered(edges)

        self._inputs = find_inputs(self.nodes, self.edges)
        self._outputs = find_outputs(self.nodes, self.edges)
        self._named_nodes = {
            n.name: i for i, n in enumerate(self.nodes) if n.name is not None
        }
        self._trainable_nodes = [
            i for i, n in enumerate(nodes) if isinstance(n, TrainableNode)
        ]
        self._is_trainable = len(self._trainable_nodes) > 0
        self._is_multi_input = len(self._inputs) > 1
        self._is_multi_output = len(self._outputs) > 1
        self._is_online = all(
            [isinstance(n, OnlineNode) for n in self._trainable_nodes]
        )
        self._is_parallel = all(
            [isinstance(n, ParallelNode) for n in self._trainable_nodes]
        )

        self.initialized = False

    def initialize(self, x: MappedData, y: Optional[MappedData] = None):
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
        self._parents, self._children = find_parents_and_children(
            self.nodes, self.edges
        )
        check_unnamed_in_out(self)
        check_input_output_connections(self.edges)

        # execution order / cycle detection
        self._execution_order = topological_sort(
            self.nodes, self.edges, inputs=self.inputs
        )

        # TODO: y
        # nodes initialization
        node_inputs = {node: np.empty((0,)) for node in self.nodes}

        if isinstance(x, dict):
            for node_name, val in x.items():
                node = self._named_nodes[node_name]
                val = timestep_from_input(val)
                np.concatenate((node_inputs[node], val), axis=1, out=node_inputs[node])
        elif isinstance(x, Timestep | NodeInput):
            [node] = self._inputs
            x = timestep_from_input(x)
            np.concatenate((node_inputs[node], x), axis=1, out=node_inputs[node])

        for node in self._execution_order:
            node_input = node_inputs[node]
            node.initialize(node_input)
            out = np.zeros((node.output_dim,))
            node_parents = self._parents[node]
            for parent in node_parents:
                np.concatenate(
                    (node_inputs[parent], x), axis=1, out=node_inputs[parent]
                )

        # TODO: Jax compilation
        self.initialized = True

    def _step(
        self, state: dict[Node, tuple], x: Timestep
    ) -> tuple[dict[Node, tuple], dict[Node, Timestep]]:
        node_states: dict[Node, tuple[np.ndarray]] = {}
        output: dict[Node, Timestep] = {}

        for node in self._execution_order:
            # TODO: optimize
            sources: list[Node] = self._children[node]
            node_input: np.ndarray = np.concatenate(
                [node_states[source] for source in sources], axis=-1
            )
            node_states[node], output[node] = node._step(state[node], node_input)

        return node_states, output
