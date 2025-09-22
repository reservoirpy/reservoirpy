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

      ~Model.initialize
      ~Model.step
      ~Model.run
      ~Model.predict
      ~Model.fit
      ~Model.partial_fit


   .. rubric:: Attributes

   .. autosummary::

    ~Model.nodes
    ~Model.edges
    ~Model.inputs
    ~Model.outputs
    ~Model.named_nodes
    ~Model.trainable_nodes
    ~Model.execution_order
    ~Model.parents
    ~Model.children
    ~Model.is_trainable
    ~Model.is_multi_input
    ~Model.is_multi_output
    ~Model.is_parallel
    ~Model.initialized


"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from collections import defaultdict
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from joblib import Parallel, delayed

from reservoirpy.node import Node
from reservoirpy.type import FeedbackBuffers, NodeInput, State, Timestep
from reservoirpy.utils.data_validation import check_model_input, check_model_timestep

from .node import Node, OnlineNode, ParallelNode, TrainableNode
from .type import (
    FeedbackBuffers,
    ModelInput,
    ModelTimestep,
    NodeInput,
    State,
    Timeseries,
    Timestep,
    get_data_dimension,
    is_multiseries,
)
from .utils.graphflow import (
    find_indirect_children,
    find_inputs,
    find_outputs,
    find_parents_and_children,
    find_pseudo_inputs,
    topological_sort,
    unique_ordered,
)
from .utils.model_utils import (
    check_input_output_connections,
    check_unnamed_in_out,
    check_unnamed_trainable,
    data_from_buffer,
    join_data,
    map_input,
    map_teacher,
    mapping_iterator,
    unfold_mapping,
)


class Model:
    """Model base class.

    Parameters
    ----------
    nodes : list of Node, optional
        Nodes to include in the Model.
    edges : list of (Node, int, Node), optional
        Edges between Nodes in the graph. An edge between a
        Node A and a Node B with a delay of :math:`d` is created as a tuple ``(A, d, B)``.
    """

    #: List of Nodes contained in the model, in insertion order.
    nodes: list[Node]
    #: List of ``(NodeA, d, NodeB)`` edges, representing a connection from ``NodeA`` to ``NodeB`` with a delay of ``d``.
    edges: list[tuple[Node, int, Node]]
    #: Dictionary of edges -> np.ndarray where arrays contain the values to be sent to the receiving node.
    feedback_buffers: FeedbackBuffers
    #: List of Nodes that expects an input
    inputs: list[Node]
    #: List of Nodes expected to output their values
    outputs: list[Node]
    #: Dictionary that associates a name to a Node with that name in the model.
    named_nodes: dict[str, Node]
    #: List of nodes that can be trained
    trainable_nodes: list[Node]
    #: List of Nodes contained in the model, in the order they should be run.
    execution_order: list[Node]
    #: Dictionary that associates a list of all Nodes connected to the key Node.
    parents: dict[Node, list[Node]]
    #: Dictionary that associates a list of all Nodes to which the key node is connected.
    children: dict[Node, list[Node]]

    #: If True, the model can be trained (with :py:meth:`~.Model.fit`).
    is_trainable: bool
    #: If True, the model has multiple Node inputs
    is_multi_input: bool
    #: If True, the model has multiple outputs and returns a dictionary that associates a node name to a node output.
    is_multi_output: bool
    #: If True, the model can be trained online (with :py:meth:`~.Model.partial_fit`).
    is_online: bool
    #: If True, the model can be trained in parallel (offline).
    is_parallel: bool
    #: If True, the model and its Nodes has been initialized.
    initialized: bool

    def __init__(
        self,
        nodes: Sequence[Node],
        edges: Sequence[tuple[Node, int, Node]],
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
        self.is_parallel = all([isinstance(n, ParallelNode) for n in self.trainable_nodes])
        self.parents, self.children = find_parents_and_children(self.nodes, self.edges)

        # execution order / cycle detection (without teacher forcing)
        direct_edges = [edge for edge in self.edges if edge[1] == 0]
        self.execution_order = topological_sort(self.nodes, direct_edges, inputs=self.inputs)
        self.feedback_buffers = None

        self.initialized = False

    def initialize(
        self,
        x: Union[ModelInput, ModelTimestep],
        y: Optional[Union[ModelInput, ModelTimestep]] = None,
    ):
        """Initializes a :py:class:`Model` instance at runtime, using samples of
        data to infer all :py:class:`Node` dimensions and instantiate the feedback buffers.

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

        y_ = map_teacher(self, y)

        # Infer node input dimensions from the input they receive

        node_input_dims = {node: 0 for node in self.nodes}

        if isinstance(x, dict):
            for node_name, val in x.items():
                node = self.named_nodes[node_name]
                node_input_dims[node] += get_data_dimension(val)
        else:
            for node in self.inputs:
                node_input_dims[node] += get_data_dimension(x)

        # also use y as forced teachers. Useful for models with feedback
        indirect_children = find_indirect_children(nodes=self.nodes, edges=self.edges)
        for supervised_node, y_teacher in y_.items():
            for child in indirect_children[supervised_node]:
                node_input_dims[child] += get_data_dimension(y_teacher)

        # execution order / cycle detection (with teacher forcing)
        pseudo_inputs = find_pseudo_inputs(self.nodes, self.edges, y_mapping=y_)
        pseudo_edges = [edge for edge in self.edges if edge[0] not in y_]
        self.pseudo_execution_order = topological_sort(self.nodes, pseudo_edges, inputs=pseudo_inputs)
        # Initialize each node in execution_order
        for node in self.pseudo_execution_order:
            node_input_dim = node_input_dims[node]
            if node.initialized:
                if node_input_dim != node.input_dim:
                    raise ValueError(
                        f"{node} expects input of dimension {node.input_dim} "
                        f"but receives input of dimension {node_input_dim}."
                    )
            else:
                if node in y_:
                    node.initialize(x=np.zeros((node_input_dim,)), y=y_[node])
                else:
                    node.initialize(x=np.zeros((node_input_dim,)))
            if node in y_.keys():
                if get_data_dimension(y_[node]) != node.output_dim:
                    raise ValueError(
                        f"{node} expects training data of dimension {node.output_dim} "
                        f"but receives data of dimension {get_data_dimension(y_[node])}."
                    )
            else:
                for child in self.children[node]:
                    node_input_dims[child] += node.output_dim

        self.feedback_buffers = {(p, d, c): np.zeros((d, p.output_dim)) for p, d, c in self.edges if d > 0}

        # TODO: Jax compilation
        self.initialized = True

    def _step(
        self,
        state: tuple[FeedbackBuffers, Mapping[Node, State]],
        x: Mapping[Node, Timestep],
    ) -> tuple[FeedbackBuffers, dict[Node, State]]:
        buffers, node_states = state

        new_state: dict[Node, State] = {}

        for node in self.execution_order:
            inputs = []
            if node in x:
                inputs.append(x[node])
            inputs += [new_state[parent]["out"] for parent in self.parents[node]]
            inputs += [buffer[-1] for (_p, _d, c), buffer in buffers.items() if c == node]
            node_input = np.concatenate(inputs, axis=-1)
            new_state[node] = node._step(node_states[node], node_input)

        new_buffers = {edge: buffer.copy() for edge, buffer in buffers.items()}
        for (p, d, c), buffer in new_buffers.items():
            buffer[-1] = new_state[p]["out"]
            new_buffers[(p, d, c)] = np.roll(buffer, 1, axis=0)

        return new_buffers, new_state

    def step(self, x: Optional[ModelTimestep] = None) -> ModelTimestep:
        """Call the Model function on a single step of data and update
        its state.

        Parameters
        ----------
        x : array of shape (input_dim,), optional
            One single step of input data. If None, an empty array is used
            instead and the Node is assumed to have an input_dim of 0

        Returns
        -------
        array of shape (output_dim,) or dict of 1D arrays.
            An output vector. If the model has multiple outputs, a dictionary is returned instead.
        """
        # Auto-regressive mode
        if x is None:
            x = np.zeros((0,))
        check_model_timestep(x)

        if not self.initialized:
            self.initialize(x)

        x_mapping = map_input(self, x)

        state = {node: node.state for node in self.nodes}
        buffers = self.feedback_buffers
        new_buffers, new_state = self._step((buffers, state), x_mapping)

        for node in new_state:
            node.state = new_state[node]
        self.feedback_buffers = new_buffers

        if not self.is_multi_output:
            return new_state[self.outputs[0]]["out"]
        else:
            return {node.name: new_state[node]["out"] for node in self.outputs}

    def _run(
        self,
        state: tuple[FeedbackBuffers, Mapping[Node, State]],
        x: Mapping[Node, Timeseries],
    ) -> tuple[tuple[FeedbackBuffers, Mapping[Node, State]], Mapping[Node, Timeseries]]:

        buffers, node_states = state
        # can be run offline (can be faster) if there is no (buffer) feedback
        can_be_run_offline = len(buffers) == 0

        new_state: dict[Node, State] = {}

        if can_be_run_offline:
            output_timeseries: dict[Node, Timeseries] = {}
            # "offline" run: Node by Node
            for node in self.execution_order:
                inputs = []
                if node in x:
                    inputs.append(x[node])
                inputs += [output_timeseries[parent] for parent in self.parents[node]]
                node_input = np.concatenate(inputs, axis=-1)
                new_state[node], output_timeseries[node] = node._run(node_states[node], node_input)
        else:
            # "online" run: step by step
            n_timesteps = x[list(x.keys())[0]].shape[0]
            output_timeseries: dict[Node, Timeseries] = {
                node: np.zeros((n_timesteps, node.output_dim)) for node in self.nodes
            }
            for i, (timestep,) in enumerate(mapping_iterator(x)):
                buffers, node_states = self._step((buffers, node_states), timestep)

                for node in self.nodes:
                    output_timeseries[node][i] = node_states[node]["out"]

        return (buffers, new_state), output_timeseries

    def run(self, x: Optional[ModelInput] = None, iters: Optional[int] = None, workers: int = 1) -> ModelInput:
        """Run the Model on a sequence of data.

        Parameters
        ----------
        x : array ([s,] t, d), list of arrays (t, d) or a mapping of them, optional
            A timeseries, multiple timeseries, or a mapping of timeseries or multiple timeseries. Input of the Model.
        iters : int, optional
            If ``x`` is ``None``, a dimensionless timeseries of length ``iters``
            is used instead.
        workers : int, default to 1
            Number of workers used for parallelization. If set to -1, all available
            workers (threads or processes) are used.

        Returns
        -------
        array of shape ([n_inputs,] timesteps, output_dim) or list of arrays, or dict
            A sequence of output vectors. If the model has multiple outputs, a mapping is returned instead.
        """
        # Auto-regressive mode
        if x is None:
            x = np.zeros((iters, 0))
        check_model_input(x)

        if not self.initialized:
            self.initialize(x)

        x_mapping = map_input(self, x)

        previous_states = {node: node.state for node in self.nodes}
        previous_buffers = self.feedback_buffers
        if is_multiseries(x_mapping):
            result: dict[Node, list[Timeseries]] = defaultdict(list)
            iterable_x = unfold_mapping(x_mapping)

            p_output = Parallel(n_jobs=workers, require="sharedmem")(
                delayed(self._run)((previous_buffers, previous_states), timeseries) for timeseries in iterable_x
            )
            new_model_states, output = zip(*p_output)
            new_buffers, new_state = new_model_states[-1]
            for o in output:
                for node in o:
                    result[node].append(o[node])

        else:
            (new_buffers, new_state), result = self._run((previous_buffers, previous_states), x_mapping)

        for node in new_state:
            node.state = new_state[node]
        self.feedback_buffers = new_buffers

        if not self.is_multi_output:
            return result[self.outputs[0]]
        else:
            return {node.name: result[node] for node in self.outputs}

    def predict(self, x: Optional[ModelInput] = None, iters: Optional[int] = None, workers: int = 1) -> ModelInput:
        """Alias for :py:meth:`~.Model.run`.

        Parameters
        ----------
        x : array ([s,] t, d), list of arrays (t, d) or a mapping of them, optional
            A timeseries, multiple timeseries, or a mapping of timeseries or multiple timeseries. Input of the Model.
        iters : int, optional
            If ``x`` is ``None``, a dimensionless timeseries of length ``iters``
            is used instead.
        workers : int, default to 1
            Number of workers used for parallelization. If set to -1, all available
            workers (threads or processes) are used.

        Returns
        -------
        array of shape ([n_inputs,] timesteps, output_dim) or list of arrays, or dict
            A sequence of output vectors. If the model has multiple outputs, a mapping is returned instead.
        """
        return self.run(x=x, iters=iters, workers=workers)

    def _learning_step(
        self,
        state: tuple[FeedbackBuffers, dict[Node, State]],
        x: dict[Node, Timestep],
        y: dict[Node, Timestep],
    ) -> tuple[FeedbackBuffers, dict[Node, State]]:
        buffers, node_states = state
        new_state: dict[Node, State] = {}

        for node in self.execution_order:
            inputs = []
            if node in x:
                inputs.append(x[node])
            inputs += [buffer[-1] for (_p, _d, c), buffer in buffers.items() if c == node]
            inputs += [new_state[parent]["out"] for parent in self.parents[node]]
            node_input = np.concatenate(inputs, axis=-1)
            if isinstance(node, OnlineNode):
                node_target = y.get(node, None)
                new_state[node] = {"out": node._learning_step(node_input, node_target)}
            else:
                new_state[node] = node._step(node_states[node], node_input)

        new_buffers = {edge: buffer.copy() for edge, buffer in buffers.items()}
        for (p, d, c), buffer in new_buffers.items():
            buffer[-1] = new_state[p]["out"]
            new_buffers[(p, d, c)] = np.roll(buffer, 1, axis=0)

        return new_buffers, new_state

    def partial_fit(
        self,
        x: Union[Timeseries, dict[str, Timeseries]],
        y: Optional[Union[Timeseries, dict[str, Timeseries]]] = None,
    ) -> ModelInput:
        """Fit the Model in an online fashion.

        This method both trains the Model parameters and produce predictions on
        the run. Calling :py:meth:`partial_fit` updates the Model without
        resetting the parameters, unlike :py:meth:`fit`.

        Parameters
        ----------
        x : array-like of shape (timesteps, input_dim)
            Input sequence of data.
        y : array-like of shape (timesteps, output_dim), optional.
            Target sequence of data. If None, the Node will train in an
            unsupervised way, if possible.

        Returns
        -------
        array of shape (timesteps, output_dim) or mapping of arrays
            All outputs computed during the training.
        """
        if not self.is_online:
            raise TypeError("Trying to partial_fit a Model that can't be trained in an online manner.")

        check_model_input(x)
        if y is not None:
            check_model_input(y)

        if not self.initialized:
            self.initialize(x, y)

        x_ = map_input(self, x)
        y_ = map_teacher(self, y)

        n_timesteps = x_[list(x_.keys())[0]].shape[0]
        output_timeseries: dict[Node, Timeseries] = {
            node: np.zeros((n_timesteps, node.output_dim)) for node in self.nodes
        }

        states = {node: node.state for node in self.nodes}
        buffers = self.feedback_buffers
        for i, (x_timestep, y_timestep) in enumerate(mapping_iterator(x_, y_)):
            buffers, states = self._learning_step((buffers, states), x_timestep, y_timestep)
            for node in self.nodes:
                output_timeseries[node][i] = states[node]["out"]

        for node in states:
            node.state = states[node]
        self.feedback_buffers = buffers

        if not self.is_multi_output:
            return output_timeseries[self.outputs[0]]
        else:
            return {node.name: output_timeseries[node] for node in self.outputs}

    def fit(
        self,
        x: ModelInput,
        y: Optional[ModelInput] = None,
        warmup: int = 0,
        workers: int = 1,
    ) -> "Model":
        """Offline fitting method of a Model.

        Parameters
        ----------
        x : list or array-like of shape ([series, ] timesteps, input_dim) or a mapping of input, optional
            Input sequences dataset.
        y : list or array-like of shape ([series], timesteps, output_dim), or a mapping of input optional
            Teacher signals dataset. If None, the method will try to fit the
            Node in an unsupervised way, if possible.
        warmup : int, default to 0
            Number of timesteps to consider as warmup and
            discard at the beginning of each timeseries before training.

        Returns
        -------
        Node
            Node trained offline.
        """
        check_model_input(x)
        if y is not None:
            check_model_input(y)

        if not self.initialized:
            self.initialize(x, y)

        # TODO: try removing the default list
        result: dict[Node, NodeInput] = defaultdict(list)
        buffers = self.feedback_buffers

        x_ = map_input(self, x)
        y_ = map_teacher(self, y)

        # forced teaching
        for supervised in y_:
            # TODO: handle Unsupervised has children
            result[supervised] = y_[supervised]

        for node in self.pseudo_execution_order:
            inputs: list[NodeInput] = []
            if node in x_:
                inputs.append(x_[node])
            inputs += [result[parent] for parent in self.parents[node]]
            for (p, _d, c), buffer in buffers.items():
                if c == node:
                    new_buffer, data = data_from_buffer(buffer, result[p])
                    buffers[(p, _d, c)] = new_buffer
                    inputs.append(data)
            node_input = join_data(*inputs)
            if isinstance(node, TrainableNode):
                node_target = y_.get(node, None)
                if isinstance(node, ParallelNode):
                    node.fit(node_input, node_target, warmup=warmup, workers=workers)
                else:
                    node.fit(node_input, node_target, warmup=warmup)
            else:
                result[node] = node.run(node_input, workers=workers)

        return self

    def reset(self) -> tuple[dict[Node, State], FeedbackBuffers]:
        """Reset all Node states and buffers in the Model.

        Returns
        -------
        dict[str, np.array], dict[Edge, array]: previous states of the Nodes and previous feedback buffer values.
        """
        previous_node_states = {node: node.reset() for node in self.nodes}
        previous_buffers = self.feedback_buffers
        self.feedback_buffers = {k: np.zeros(v.shape) for k, v in self.feedback_buffers.items()}

        return previous_node_states, previous_buffers

    def __call__(self, x: Optional[ModelTimestep] = None) -> ModelTimestep:
        return self.step(x)

    def __str__(self):
        return f"Model({', '.join(map(str, self.nodes))})"

    def __repr__(self):
        return self.__str__()

    def __rshift__(self, other: Union[int, Node, "Model", Sequence[Union[Node, "Model"]]]) -> "Model":
        """self >> other"""
        from .ops import ModelBuilderUtil, link

        if isinstance(other, int):
            return ModelBuilderUtil(node=self, delay=other, node_is_first=True)

        return link(self, other)

    def __rrshift__(self, other: Union[int, Node, "Model", Sequence[Union[Node, "Model"]]]) -> "Model":
        """other >> self"""
        from .ops import ModelBuilderUtil, link

        if isinstance(other, int):
            return ModelBuilderUtil(node=self, delay=other, node_is_first=False)

        return link(other, self)

    def __lshift__(self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]) -> "Model":
        """self << other"""
        from .ops import link_feedback

        return link_feedback(sender=other, receiver=self)

    def __rlshift__(self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]) -> "Model":
        """other << self"""
        from .ops import link_feedback

        return link_feedback(sender=self, receiver=other)

    def __and__(self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]) -> "Model":
        """self & other"""
        from .ops import merge

        return merge(self, other)

    def __rand__(self, other: Union[Node, "Model", Sequence[Union[Node, "Model"]]]) -> "Model":
        """other & self"""
        from .ops import merge

        return merge(other, self)
