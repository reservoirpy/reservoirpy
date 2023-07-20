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
      ~Model.link
      ~Model.reset
      ~Model.run
      ~Model.train
      ~Model.update_graph
      ~Model.with_feedback
      ~Model.with_state


   .. rubric:: Attributes

   .. autosummary::

      ~Model.data_dispatcher
      ~Model.edges
      ~Model.feedback_nodes
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
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import _base
from ._base import _Node, check_xy
from .type import MappedData
from .utils import progress, verbosity
from .utils.graphflow import (
    DataDispatcher,
    dispatch,
    find_entries_and_exits,
    get_offline_subgraphs,
    topological_sort,
)
from .utils.model_utils import (
    allocate_returned_states,
    build_forward_sumodels,
    dist_states_to_next_subgraph,
    fold_mapping,
    to_data_mapping,
)
from .utils.validation import is_mapping


def run_and_partial_fit(
    submodel,
    complete_model,
    offlines,
    relations,
    x_seq,
    y_seq,
    warmup,
    stateful=True,
    reset=False,
    return_states=None,
    force_teachers=True,
):
    """Run a submodel and call its partial fit function."""

    if not submodel.is_empty:
        x_seq = {n: x for n, x in x_seq.items() if n in submodel.node_names}

        if force_teachers and y_seq is not None:
            y_seq = {
                n: y
                for n, y in y_seq.items()
                if n in [o.name for o in complete_model.nodes if o.is_trained_offline]
            }
        else:
            y_seq = None

        submodel._is_initialized = True
        states = run_submodel(
            complete_model,
            submodel,
            x_seq,
            y_seq,
            return_states=return_states,
            stateful=stateful,
            reset=reset,
        )

        dist_states = dist_states_to_next_subgraph(states, relations)
    else:
        dist_states = x_seq

    if y_seq is not None:
        for node in offlines:
            node.partial_fit(
                dist_states.get(node.name), y_seq.get(node.name), warmup=warmup
            )
    else:
        for node in offlines:
            node.partial_fit(dist_states.get(node.name), warmup=warmup)

    return dist_states


def run_submodel(
    model,
    submodel,
    X: MappedData,
    forced_feedbacks: Dict[str, np.ndarray] = None,
    from_state: Dict[str, np.ndarray] = None,
    stateful=True,
    reset=False,
    shift_fb=True,
    return_states: Sequence[str] = None,
) -> MappedData:

    X_, forced_feedbacks_ = to_data_mapping(submodel, X, forced_feedbacks)

    submodel._initialize_on_sequence(X_[0], forced_feedbacks_[0])

    states = []
    for X_seq, fb_seq in zip(X_, forced_feedbacks_):
        with model.with_state(reset=reset, stateful=stateful):
            states_seq = model._run(
                X_seq,
                fb_seq,
                from_state=from_state,
                stateful=stateful,
                shift_fb=shift_fb,
                return_states=return_states,
                submodel=submodel,
            )

            states.append(states_seq)

    return fold_mapping(submodel, states, return_states)


def forward(model: "Model", x: MappedData) -> List[np.ndarray]:
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
    x : dict or array-like of shape ([n_inputs], 1, input_dim)
        A vector corresponding to a timestep of data, or
        a dictionnary mapping node names to input vectors.

    Returns
    -------
    array-like of shape (n_outputs, 1, output_dim)
        New states of all terminal nodes of the model, i.e. its output.
    """
    data = model.data_dispatcher.load(x)

    for node in model.nodes:
        _base.call(node, data[node].x)

    return [out_node.state() for out_node in model.output_nodes]


def train(model: "Model", x=None, y: MappedData = None, force_teachers=True):
    """Training function for a Model. Run all train functions of all online
    nodes within the Model. Nodes have already been called. Only training
    is performed."""

    data = model.data_dispatcher.load(X=x, Y=y)

    for node in model.nodes:
        if node.is_trained_online:
            _base.train(
                node,
                data[node].x,
                data[node].y,
                force_teachers=force_teachers,
                call_node=False,
            )


def initializer(model: "Model", x: MappedData, y: Optional[MappedData] = None):
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


class Model(_Node):
    """Model base class.

    Parameters
    ----------
    nodes : list of Node, optional
        Nodes to include in the Model.
    edges : list of (Node, Node), optional
        Edges between Nodes in the graph. An edge between a
        Node A and a Node B is created as a tuple (A, B).
    name : str, optional
        Name of the Model.
    """

    _node_registry: Dict[str, _Node]
    _nodes: List[_Node]
    _edges: List[Tuple[_Node, _Node]]
    _inputs: List[_Node]
    _outputs: List[_Node]
    _dispatcher: "DataDispatcher"

    def __init__(
        self,
        nodes: Sequence[_Node] = None,
        edges: Sequence[Tuple[_Node, _Node]] = None,
        name: str = None,
    ):

        if nodes is None:
            nodes = tuple()
        if edges is None:
            edges = tuple()

        self._forward = forward
        self._train = train
        self._initializer = initializer

        self._name = self._get_name(name)

        nodes, edges = self._concat_multi_inputs(nodes, edges)

        self._edges = edges

        # always maintain nodes in topological order
        if len(nodes) > 0:
            self._inputs, self._outputs = find_entries_and_exits(nodes, edges)
            self._nodes = topological_sort(nodes, edges, self._inputs)
        else:
            self._inputs = list()
            self._outputs = list()
            self._nodes = nodes

        self._is_initialized = False
        self._trainable = any([n.is_trainable for n in nodes])
        self._fitted = all([n.fitted for n in nodes])

        self._params = {n.name: n.params for n in nodes}
        self._hypers = {n.name: n.hypers for n in nodes}

        self._node_registry = {n.name: n for n in self.nodes}

        self._dispatcher = DataDispatcher(self)

    def __repr__(self):
        klas = self.__class__.__name__
        nodes = [n.name for n in self._nodes]
        return f"'{self.name}': {klas}('" + "', '".join(nodes) + "')"

    def __getitem__(self, item):
        return self.get_node(item)

    def __iand__(self, other) -> "Model":
        from .ops import merge

        return merge(self, other, inplace=True)

    @staticmethod
    def _concat_multi_inputs(nodes, edges):
        from .ops import concat_multi_inputs

        return concat_multi_inputs(nodes, edges)

    def _check_if_only_online(self):
        if any([n.is_trained_offline and not n.fitted for n in self.nodes]):
            raise RuntimeError(
                f"Impossible to train model {self.name} using "
                f"online method: model contains untrained "
                f"offline nodes."
            )

    def _load_proxys(self, keep=False):
        """Save states of all nodes into their state_proxy"""
        for node in self._nodes:
            if keep and node._state_proxy is not None:
                continue
            node._state_proxy = node.state()

    def _clean_proxys(self):
        """Destroy state_proxy of all nodes"""
        for node in self._nodes:
            node._state_proxy = None

    def _initialize_on_sequence(self, X=None, Y=None):
        if not self._is_initialized:
            x_init = None
            if X is not None:
                if is_mapping(X):
                    x_init = {name: np.atleast_2d(x[0]) for name, x in X.items()}
                else:
                    x_init = np.atleast_2d(X[0])

            y_init = None
            if Y is not None:
                if is_mapping(Y):
                    y_init = {name: np.atleast_2d(y[0]) for name, y in Y.items()}
                else:
                    y_init = np.atleast_2d(Y[0])

            self.initialize(x_init, y_init)

    def _call(self, x=None, return_states=None, submodel=None, *args, **kwargs):

        if submodel is None:
            submodel = self

        self._forward(submodel, x)

        state = {}
        if return_states == "all":
            for node in submodel.nodes:
                state[node.name] = node.state()

        elif hasattr(return_states, "__iter__"):
            for name in return_states:
                state[name] = submodel[name].state()

        else:
            if len(submodel.output_nodes) > 1:
                for out_node in submodel.output_nodes:
                    state[out_node.name] = out_node.state()
            else:
                state = submodel.output_nodes[0].state()

        return state

    def _run(
        self,
        X,
        feedback,
        from_state=None,
        stateful=True,
        shift_fb=True,
        return_states=None,
        submodel=None,
    ):
        if submodel is None:
            submodel = self

        states = allocate_returned_states(submodel, X, return_states)

        seq = progress(
            dispatch(X, feedback, shift_fb=shift_fb),
            f"Running {self.name}",
            total=len(X),
        )

        with self.with_state(from_state, stateful=stateful):
            # load proxys after state update to make it have an
            # impact on feedback
            self._load_proxys(keep=True)
            for i, (x, forced_fb, _) in enumerate(seq):

                with self.with_feedback(forced_fb):
                    state = submodel._call(x, return_states=return_states)

                if is_mapping(state):
                    for name, value in state.items():
                        states[name][i, :] = value
                else:
                    states[submodel.output_nodes[0].name][i, :] = state

                self._load_proxys()

        self._clean_proxys()

        return states

    def _unregister_teachers(self):
        """Remove teacher nodes refs from student nodes."""
        for node in self.trainable_nodes:
            node._teacher = None

    def update_graph(
        self,
        new_nodes: Sequence[_Node],
        new_edges: Sequence[Tuple[_Node, _Node]],
    ) -> "Model":
        """Update current Model's with new nodes and edges, inplace (a copy
        is not performed).

        Parameters
        ----------
        new_nodes : list of Node
            New nodes.
        new_edges : list of (Node, Node)
            New edges between nodes.

        Returns
        -------
        Model
            The updated Model.
        """
        nodes = list(set(new_nodes) | set(self.nodes))
        edges = list(set(new_edges) | set(self.edges))

        nodes, edges = self._concat_multi_inputs(nodes, edges)

        self._nodes = nodes
        self._edges = edges

        self._params = {n.name: n.params for n in self._nodes}
        self._hypers = {n.name: n.hypers for n in self._nodes}

        self._inputs, self._outputs = find_entries_and_exits(self._nodes, self._edges)
        self._nodes = topological_sort(self._nodes, self._edges, self._inputs)
        self._node_registry = {n.name: n for n in self.nodes}

        self._dispatcher = DataDispatcher(self)

        self._fitted = all([n.fitted for n in self.nodes])
        self._is_initialized = False

        return self

    def get_node(self, name: str) -> _Node:
        """Get Node in Model, by name.

        Parameters
        ----------
        name : str
            Node name.

        Returns
        -------
        Node
            The requested Node.

        Raises
        ------
        KeyError
            Node not found.
        """
        if self._node_registry.get(name) is not None:
            return self._node_registry[name]
        else:
            raise KeyError(f"No node named '{name}' found in model {self.name}.")

    @property
    def nodes(self) -> List[_Node]:
        """Nodes in the Model, in topological order."""
        return self._nodes

    @property
    def node_names(self):
        """Names of all the Nodes in the Model."""
        return list(self._node_registry.keys())

    @property
    def edges(self):
        """All edges between Nodes, in the form (sender, receiver)."""
        return self._edges

    @property
    def input_dim(self):
        """Input dimension of the Model;
        input dimensions of all input Nodes."""
        if self.is_initialized:
            dims = [n.input_dim for n in self.input_nodes]
            if len(dims) == 0:
                return 0
            elif len(dims) < 2:
                return dims[0]
            return dims
        else:
            return None

    @property
    def output_dim(self):
        """Ouptut dimension of the Model;
        output dimensions of all output Nodes."""
        if self.is_initialized:
            dims = [n.output_dim for n in self.output_nodes]
            if len(dims) == 0:
                return 0
            elif len(dims) < 2:
                return dims[0]
            return dims
        else:
            return None

    @property
    def input_nodes(self):
        """First Nodes in the graph held by the Model."""
        return self._inputs

    @property
    def output_nodes(self):
        """Last Nodes in the graph held by the Model."""
        return self._outputs

    @property
    def trainable_nodes(self):
        """Returns all offline and online
        trainable Nodes in the Model."""
        return [n for n in self.nodes if n.is_trainable]

    @property
    def feedback_nodes(self):
        """Returns all Nodes equiped with a feedback connection
        in the Model."""
        return [n for n in self.nodes if n.has_feedback]

    @property
    def data_dispatcher(self):
        """DataDispatcher object of the Model. Used to
        distribute data to Nodes when
        calling/running/fitting the Model."""
        return self._dispatcher

    @property
    def is_empty(self):
        """Returns True if the Model contains no Nodes."""
        return len(self.nodes) == 0

    @property
    def is_trainable(self) -> bool:
        """Returns True if at least one Node in the Model is trainable."""
        return any([n.is_trainable for n in self.nodes])

    @is_trainable.setter
    def is_trainable(self, value):
        """Freeze or unfreeze trainable Nodes in the Model."""
        trainables = [
            n for n in self.nodes if n.is_trained_offline or n.is_trained_online
        ]
        for node in trainables:
            node.is_trainable = value

    @property
    def is_trained_online(self) -> bool:
        """Returns True if all nodes are online learners."""
        return all([n.is_trained_online or n.fitted for n in self.nodes])

    @property
    def is_trained_offline(self) -> bool:
        """Returns True if all nodes are offline learners."""
        return all([n.is_trained_offline or n.fitted for n in self.nodes])

    @property
    def fitted(self) -> bool:
        """Returns True if all nodes are fitted."""
        return all([n.fitted for n in self.nodes])

    @contextmanager
    def with_state(
        self, state: Dict[str, np.ndarray] = None, stateful=False, reset=False
    ) -> "Model":
        """Modify the state of one or several Nodes in the Model
        using a context manager.
        The modification will have effect only within the context defined,
        before all states return to their previous value.

        Parameters
        ----------
        state : dict of arrays of shape (1, output_dim), optional
            Pairs of keys and values, where keys are Model nodes names and
            value are new ndarray state vectors.
        stateful : bool, default to False
            If set to True, then all modifications made in the context manager
            will remain after leaving the context.
        reset : bool, default to False
            If True, all Nodes will be reset using their :py:meth:`Node.reset`
            method.

        Returns
        -------
        Model
            Modified Model.
        """
        if state is None and not reset:
            current_state = None
            if not stateful:
                current_state = {n.name: n.state() for n in self.nodes}
            yield self
            if not stateful:
                self.reset(to_state=current_state)
        elif isinstance(state, np.ndarray):
            raise TypeError(
                f"Impossible to set state of {self.name} with "
                f"an array. State should be a dict mapping state "
                f"to a Node name within the model."
            )
        else:
            if state is None:
                state = {}

            with ExitStack() as stack:
                for node in self.nodes:
                    value = state.get(node.name)
                    stack.enter_context(
                        node.with_state(value, stateful=stateful, reset=reset)
                    )
                yield self

    @contextmanager
    def with_feedback(
        self, feedback: Dict[str, np.ndarray] = None, stateful=False, reset=False
    ) -> "Model":
        """Modify the feedback received or sent by Nodes in the Model using
        a context manager.
        The modification will have effect only within the context defined,
        before the feedbacks return to their previous states.

        If the Nodes are receiving feedback, then this function will alter the
        states of the Nodes connected to it through feedback connections.

        If the Nodes are sending feedback, then this function will alter the
        states (or state proxies, see :py:meth:`Node.state_proxy`) of the
        Nodes.

        Parameters
        ----------
        feedback : dict of arrays of shape (1, output_dim), optional
            Pairs of keys and values, where keys are Model nodes names and
            value are new ndarray feedback vectors.
        stateful : bool, default to False
            If set to True, then all modifications made in the context manager
            will remain after leaving the context.
        reset : bool, default to False
            If True, all feedbacks  will be reset to zero.

        Returns
        -------
        Model
            Modified Model.
        """

        if feedback is None and not reset:
            yield self
            return

        elif feedback is not None:
            with ExitStack() as stack:
                for node in self.nodes:
                    value = feedback.get(node.name)
                    # maybe node does not send feedback but receives it?
                    if value is None and node.has_feedback:
                        value = feedback.get(node._feedback.name)
                    stack.enter_context(
                        node.with_feedback(value, stateful=stateful, reset=reset)
                    )

                yield self
        else:
            yield self
            return

    def reset(self, to_state: Dict[str, np.ndarray] = None):
        """Reset the last state saved to zero for all Nodes in
        the Model or to other state values.

        Parameters
        ----------
        to_state : dict of arrays of shape (1, output_dim), optional
            Pairs of keys and values, where keys are Model nodes names and
            value are new ndarray state vectors.
        """
        if to_state is None:
            for node in self.nodes:
                node.reset()
        else:
            for node_name, current_state in to_state.items():
                self[node_name].reset(to_state=current_state)
        return self

    def initialize(self, x=None, y=None) -> "Model":
        """Call the Model initializers on some data points.
        Model will be virtually run to infer shapes of all nodes given
        inputs and targets vectors.

        Parameters
        ----------
        x : dict or array-like of shape ([n_inputs], 1, input_dim)
            Input data.
        y : dict or array-like of shape ([n_outputs], 1, output_dim)
            Ground truth data. Used to infer output dimension
            of trainable nodes.

        Returns
        -------
        Model
            Initialized Model.
        """
        self._is_initialized = False
        self._initializer(self, x=x, y=y)
        self.reset()
        self._is_initialized = True
        return self

    def initialize_buffers(self) -> "Model":
        """Call all Nodes buffer initializers. Buffer initializers will create
        buffer arrays on demand to store transient values of the parameters,
        typically during training.

        Returns
        -------
        Model
            Initialized Model.
        """
        for node in self.nodes:
            if node._buffers_initializer is not None:
                node.initialize_buffers()

    def call(
        self,
        x: MappedData,
        forced_feedback: MappedData = None,
        from_state: Dict[str, np.ndarray] = None,
        stateful=True,
        reset=False,
        return_states: Sequence[str] = None,
    ) -> MappedData:
        """Call the Model forward function on a single step of data.
        Model forward function is a composition of all its Nodes forward
        functions.

        Can update the state its Nodes.

        Parameters
        ----------
        x : dict or array-like of shape ([n_inputs], 1, input_dim)
            One single step of input data. If dict, then
            pairs of keys and values, where keys are Model input
            nodes names and values are single steps of input data.
        forced_feedback: dict of arrays of shape ([n_feedbacks], 1, feedback_dim), optional
            Pairs of keys and values, where keys are Model nodes names and
            value are feedback vectors to force into the nodes.
        from_state : dict of arrays of shape ([nodes], 1, nodes.output_dim), optional
            Pairs of keys and values, where keys are Model nodes names and
            value are new ndarray state vectors.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Nodes states will be reset to zero before this operation.
        return_states: list of str, optional
            Names of Nodes from which to return states as output.

        Returns
        -------
        dict or array-like of shape ([n_outputs], 1, output_dim)
            An output vector or pairs of keys and values
            where keys are output nodes names and values
            are corresponding output vectors.
        """

        x, _ = check_xy(
            self,
            x,
            allow_timespans=False,
            allow_n_sequences=False,
        )

        if not self._is_initialized:
            self.initialize(x)

        with self.with_state(from_state, stateful=stateful, reset=reset):
            # load current states in proxys interfaces accessible
            # through feedback. These proxys are not updated during the
            # graph call.
            self._load_proxys(keep=True)

            # maybe load forced feedbacks in proxys ?
            with self.with_feedback(forced_feedback, stateful=stateful, reset=reset):
                state = self._call(x, return_states)

        # wash states proxys
        self._clean_proxys()

        return state

    def run(
        self,
        X: MappedData,
        forced_feedbacks: Dict[str, np.ndarray] = None,
        from_state: Dict[str, np.ndarray] = None,
        stateful=True,
        reset=False,
        shift_fb=True,
        return_states: Sequence[str] = None,
    ) -> MappedData:
        """Run the Model forward function on a sequence of data.
        Model forward function is a composition of all its Nodes forward
        functions.
        Can update the state of the
        Nodes several times.

        Parameters
        ----------
        X : dict or array-like of shape ([n_inputs], timesteps, input_dim)
            A sequence of input data.
            If dict, then pairs of keys and values, where keys are Model input
            nodes names and values are sequence of input data.
        forced_feedbacks: dict of array-like of shape ([n_feedbacks], timesteps, feedabck_dim)
            Pairs of keys and values, where keys are Model nodes names and
            value are sequences of feedback vectors to force into the nodes.
        from_state : dict of arrays of shape ([nodes], 1, nodes.output_dim)
            Pairs of keys and values, where keys are Model nodes names and
            value are new ndarray state vectors.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Nodes states will be reset to zero before this operation.
        shift_fb: bool, defaults to True
            If True, then forced feedbacks are fed to nodes with a
            one timestep delay. If forced feedbacks are a sequence
            of target vectors matching the sequence of input
            vectors, then this parameter should be set to True.
        return_states: list of str, optional
            Names of Nodes from which to return states as output.

        Returns
        -------
        dict or array-like of shape ([n_outputs], timesteps, output_dim)
            A sequence of output vectors or pairs of keys and values
            where keys are output nodes names and values
            are corresponding sequences of output vectors.
        """
        X_, forced_feedbacks_ = to_data_mapping(self, X, forced_feedbacks)

        self._initialize_on_sequence(X_[0], forced_feedbacks_[0])

        states = []
        for X_seq, fb_seq in zip(X_, forced_feedbacks_):
            with self.with_state(reset=reset, stateful=stateful):
                states_seq = self._run(
                    X_seq,
                    fb_seq,
                    from_state=from_state,
                    stateful=stateful,
                    shift_fb=shift_fb,
                    return_states=return_states,
                )

                states.append(states_seq)

        return fold_mapping(self, states, return_states)

    def train(
        self,
        X,
        Y=None,
        force_teachers=True,
        learn_every=1,
        from_state=None,
        stateful=True,
        reset=False,
        return_states=None,
    ) -> MappedData:
        """Train all online Nodes in the Model
        using their online learning rule.

        Parameters
        ----------
        X : dict or array-like of shape ([n_inputs], timesteps, input_dim)
            Input sequence of data. If dict, then pairs
            of keys and values, where keys are Model input
            nodes names and values are sequence of input data.
        Y : dict or array-like of shape ([onlines], timesteps, onlines.output_dim), optional.
            Target sequence of data.
            If dict, then pairs of keys and values, where keys are Model
            online trainable nodes names values are sequences
            of target data. If None, the Nodes will search a feedback
            signal, or train in an unsupervised way, if possible.
        force_teachers : bool, default to True
            If True, this Model will broadcast the available ground truth
            signal
            to all online nodes sending feedabck to other nodes. Otherwise,
            the real state of these nodes will be sent to the feedback
            receivers
            during training.
        learn_every : int, default to 1
            Time interval at which training must occur, when dealing with a
            sequence of input data. By default, the training method is called
            every time the Model receive an input.
        from_state : dict of arrays of shape ([nodes], 1, nodes.output_dim)
            Pairs of keys and values, where keys are Model nodes names and
            value are new ndarray state vectors.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Nodes states will be reset to zero before this operation.
        return_states: list of str, optional
            Names of Nodes from which to return states as output.

        Returns
        -------
        dict or array-like of shape ([n_outputs], timesteps, output_dim)
            All outputs computed during the training
            or pairs of keys and values
            where keys are output nodes names and values
            are corresponding outputs computed.
            If `call` is False,
            outputs will be null vectors.
        """

        self._check_if_only_online()

        X_, Y_ = check_xy(self, X, Y, allow_n_sequences=False)

        self._initialize_on_sequence(X_, Y_)

        states = allocate_returned_states(self, X_, return_states)

        dispatched_data = dispatch(
            X_, Y_, return_targets=True, force_teachers=force_teachers
        )

        with self.with_state(from_state, stateful=stateful, reset=reset):
            # load proxys after state update to make it have an
            # impact on feedback
            self._load_proxys(keep=True)
            for i, (x, forced_feedback, y) in enumerate(dispatched_data):

                if not force_teachers:
                    forced_feedback = None

                with self.with_feedback(forced_feedback):
                    state = self._call(x, return_states=return_states)

                # reload feedbacks from training. Some nodes may need
                # updated feedback signals to train.
                self._load_proxys()

                if i % learn_every == 0 or len(X) == 1:
                    self._train(self, x=x, y=y, force_teachers=force_teachers)

                if is_mapping(state):
                    for name, value in state.items():
                        states[name][i, :] = value
                else:
                    states[self.output_nodes[0].name][i, :] = state

        self._clean_proxys()
        self._unregister_teachers()

        # dicts are only relevant if model has several outputs (len > 1) or
        # if we want to return states from hidden nodes
        if len(states) == 1 and return_states is None:
            return states[self.output_nodes[0].name]

        return states

    def fit(
        self,
        X: MappedData,
        Y: MappedData,
        warmup=0,
        force_teachers=True,
        from_state=None,
        stateful=True,
        reset=False,
    ) -> "Model":
        """Fit all offline Nodes in the Model
        using their offline learning rule.

        Parameters
        ----------
        X : dict or array-like of shape ([n_inputs], [series], timesteps, input_dim)
            Input sequence of data. If dict, then pairs
            of keys and values, where keys are Model input
            nodes names and values are sequence of input data.
        Y : dict or array-like of shape ([offlines], [series], timesteps, offlines.output_dim)
            Target sequence of data. If dict, then pairs
            of keys and values, where keys are Model input
            nodes names and values are sequence of target data.
        warmup : int, default to 0
            Number of timesteps to consider as warmup and
            discard at the begining of each timeseries before training.
        force_teachers : bool, default to True
            If True, this Model will broadcast the available ground truth
            signal
            to all online nodes sending feedback to other nodes. Otherwise,
            the real state of these nodes will be sent to the feedback
            receivers
            during training.
        from_state : dict of arrays of shape ([nodes], 1, nodes.output_dim)
            Pairs of keys and values, where keys are Model nodes names and
            value are new ndarray state vectors.
        stateful : bool, default to True
            If True, Node state will be updated by this operation.
        reset : bool, default to False
            If True, Nodes states will be reset to zero before this operation.

        Returns
        -------
        Model
            Model trained offline.
        """

        if not any([n for n in self.trainable_nodes if n.is_trained_offline]):
            raise TypeError(
                f"Impossible to fit model {self} offline: "
                "no offline nodes found in model."
            )

        X, Y = to_data_mapping(self, X, Y)

        self._initialize_on_sequence(X[0], Y[0])
        self.initialize_buffers()

        subgraphs = get_offline_subgraphs(self.nodes, self.edges)

        trained = set()
        next_X = None

        with self.with_state(from_state, reset=reset, stateful=stateful):
            for i, ((nodes, edges), relations) in enumerate(subgraphs):
                submodel, offlines = build_forward_sumodels(nodes, edges, trained)

                if next_X is not None:
                    for j in range(len(X)):
                        X[j].update(next_X[j])

                return_states = None
                if len(relations) > 0:
                    return_states = list(relations.keys())

                # next inputs for next submodel
                next_X = []
                seq = progress(X, f"Running {self.name}")

                _partial_fit_fn = partial(
                    run_and_partial_fit,
                    force_teachers=force_teachers,
                    complete_model=self,
                    submodel=submodel,
                    warmup=warmup,
                    reset=reset,
                    offlines=offlines,
                    relations=relations,
                    stateful=stateful,
                    return_states=return_states,
                )

                # for seq/batch in dataset
                for x_seq, y_seq in zip(seq, Y):
                    next_X += [_partial_fit_fn(x_seq=x_seq, y_seq=y_seq)]

                for node in offlines:
                    if verbosity():
                        print(f"Fitting node {node.name}...")
                    node.fit()

                trained |= set(offlines)

        return self

    def copy(self, *args, **kwargs):
        raise NotImplementedError()


class FrozenModel(Model):
    """A FrozenModel is a Model that can not be
    linked to other nodes or models.
    """

    def __init__(self, *args, **kwargs):
        super(FrozenModel, self).__init__(*args, **kwargs)

    def update_graph(self, new_nodes, new_edges):
        raise TypeError(
            f"Impossible to update FrozenModel {self}: "
            f"model is frozen and cannot be modified."
        )
