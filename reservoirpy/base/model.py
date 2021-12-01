# Author: Nathan Trouvain at 25/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

# Author: Nathan Trouvain at 01/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from contextlib import ExitStack, contextmanager
from typing import Dict, List, Optional, Tuple
from collections.abc import Iterable
from functools import partial

import numpy as np

from ..utils import to_ragged_seq_set, progress, verbosity
from .graphflow import (DataDispatcher, find_entries_and_exits,
                        get_offline_subgraphs, topological_sort, )
from ._utils import (_allocate_returned_states, _dist_states_to_next_subgraph,
                     _build_forward_sumodels, )
from .types import MappedData, GenericNode
from ..utils.validation import check_vector, is_mapping


def _run_and_partial_fit(model, offlines, relations, x_seq, y_seq,
                         stateful=True, reset=False, return_states=None,
                         force_teachers=True):

    if not model.is_empty:
        x_seq = {n: x for n, x in x_seq.items()
                 if n in model.node_names}

        if force_teachers:
            y_seq = {n: y for n, y in y_seq.items()
                     if n in [o.name for o in offlines]}
        else:
            y_seq = None

        model._is_initialized = True
        states = model.run(x_seq, y_seq,
                           return_states=return_states,
                           stateful=stateful,
                           reset=reset)

        dist_states = _dist_states_to_next_subgraph(states,
                                                    relations)
    else:
        dist_states = x_seq

    for node in offlines:
        node.partial_fit(dist_states[node.name],
                         y_seq[node.name])

        return dist_states


def forward(model: "Model",
            x: MappedData) -> List[np.ndarray]:
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
    x : numpy.ndarray or dict of numpy.ndarray
        A vector of shape `(1, ndim)` corresponding to a timestep of data, or
        a dictionnary mapping node names to vector of shapes
        `(1, ndim of corresponding node)`.

    Returns
    -------
        list of numpy.ndarray
            New states of all terminal nodes of the model, i.e. its output.
    """
    data = model.data_dispatcher.load(x)

    for node in model.nodes:
        node(data[node].x)

    return [out_node.state() for out_node in model.output_nodes]


def train(model: "Model", x=None, y: MappedData = None, force_teachers=True):

    data = model.data_dispatcher.load(X=x, Y=y)

    for node in model.nodes:
        if node.is_trained_online:
            node.train(data[node].x, data[node].y,
                       force_teachers=force_teachers,
                       call=False)


def initializer(model: "Model",
                x: MappedData,
                y: Optional[MappedData] = None):
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


class Model(GenericNode):
    _node_registry: Dict[str, GenericNode]
    _nodes:    List[GenericNode]
    _edges:    List[Tuple[GenericNode, GenericNode]]
    _inputs:   List[GenericNode]
    _outputs:  List[GenericNode]
    _dispatcher: "DataDispatcher"

    def __init__(self, nodes=(), edges=(), name=None):

        self._forward = forward
        self._train = train
        self._initializer = initializer

        self._name = self._get_name(name)

        self._edges = edges

        if len(nodes) > 0:
            self._inputs, self._outputs = find_entries_and_exits(nodes, edges)
            self._nodes = topological_sort(nodes, edges, self._inputs)
        else:
            self._inputs = self._outputs = list()
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

    def __iand__(self, other):
        from reservoirpy.base.ops import merge
        return merge(self, other, inplace=True)

    def _check_input(self, x):
        if is_mapping(x):
            input_names = [n.name for n in self.input_nodes]

            for input_name in input_names:
                if input_name not in x:
                    raise NameError(
                        f"Missing input data for node '{input_name}' "
                        f"of model {self.name}.")

            for name, value in x.items():
                value = check_vector(value)
                x[name] = value

                if self._is_initialized:
                    if value.shape[1] != self[name].input_dim:
                        raise ValueError(
                            f"Impossible to call node {name} in model "
                            f"{self}: node input "
                            f"dimension is (1, {self[name].input_dim}) "
                            f"and input dimension is {value.shape}.")
        return x

    def _check_targets(self, y):
        if is_mapping(y):
            for name, value in y.items():
                value = check_vector(value)
                y[name] = value

                if self._is_initialized:
                    if value.shape[1] != self[name].output_dim:
                        raise ValueError(
                            f"Impossible to fit/train node {name} in model "
                            f"{self}: node output "
                            f"dimension is (1, {self[name].output_dim}) "
                            f"and target dimension is {value.shape}.")
        return y

    def _check_if_only_online(self):
        if any([n.is_trained_offline and not n.fitted for n in self.nodes]):
            raise RuntimeError(f"Impossible to train model {self.name} using "
                               f"online method: model contains untrained "
                               f"offline nodes.")

    def _load_proxys(self, keep=False):
        for node in self._nodes:
            if keep and node._state_proxy is not None:
                continue
            node._state_proxy = node.state()

    def _clean_proxys(self):
        for node in self._nodes:
            node._state_proxy = None

    def _initialize_on_sequence(self, X=None, Y=None):
        if not self._is_initialized:
            x_init = None
            if X is not None:
                if is_mapping(X):
                    x_init = {name: np.atleast_2d(x[0])
                              for name, x in X.items()}
                else:
                    x_init = np.atleast_2d(X[0])

            y_init = None
            if Y is not None:
                if is_mapping(Y):
                    y_init = {name: np.atleast_2d(y[0])
                              for name, y in Y.items()}
                else:
                    y_init = np.atleast_2d(Y[0])

            self.initialize(x_init, y_init)

    def _call(self, x=None, return_states=None, *args, **kwargs):

        self._forward(self, x)

        state = {}
        if return_states == "all":
            for node in self.nodes:
                state[node.name] = node.state()

        elif isinstance(return_states, Iterable):
            for name in return_states:
                state[name] = self[name].state()

        else:
            if len(self.output_nodes) > 1:
                for out_node in self.output_nodes:
                    state[out_node.name] = out_node.state()
            else:
                state = self.output_nodes[0].state()

        return state

    def update_graph(self, new_nodes, new_edges):
        self._nodes = list(set(new_nodes) | set(self.nodes))
        self._edges = list(set(new_edges) | set(self.edges))

        self._params = {n.name: n.params for n in self._nodes}
        self._hypers = {n.name: n.hypers for n in self._nodes}

        self._inputs, self._outputs = find_entries_and_exits(self._nodes,
                                                             self._edges)
        self._nodes = topological_sort(self._nodes, self._edges, self._inputs)
        self._node_registry = {n.name: n for n in self.nodes}

        self._dispatcher = DataDispatcher(self)

        self._fitted = all([n.fitted for n in self.nodes])
        self._is_initialized = False

        return self

    def get_node(self, name):
        if self._node_registry.get(name) is not None:
            return self._node_registry[name]
        else:
            raise KeyError(f"No node named '{name}' found in "
                           f"model {self.name}.")

    @property
    def nodes(self):
        return self._nodes

    @property
    def node_names(self):
        return list(self._node_registry.keys())

    @property
    def edges(self):
        return self._edges

    @property
    def input_dim(self):
        dims = [n.input_dim for n in self.input_nodes]
        if len(dims) == 0:
            return 0
        elif len(dims) < 2:
            return dims[0]
        return dims

    @property
    def output_dim(self):
        dims = [n.output_dim for n in self.output_nodes]
        if len(dims) == 0:
            return 0
        elif len(dims) < 2:
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
        return [n for n in self.nodes if n.is_trainable]

    @property
    def feedback_nodes(self):
        return [n for n in self.nodes if n.has_feedback]

    @property
    def data_dispatcher(self):
        return self._dispatcher

    @property
    def is_empty(self):
        return len(self.nodes) == 0

    @property
    def is_trainable(self) -> bool:
        return any([n.is_trainable for n in self.nodes])

    @is_trainable.setter
    def is_trainable(self, value):
        trainables = [n for n in self.nodes
                      if n.is_trained_offline or n.is_trained_online]
        for node in trainables:
            node.is_trainable = value

    @contextmanager
    def with_state(self, state=None, stateful=False, reset=False):
        if state is None and not reset:
            current_state = None
            if not stateful:
                current_state = {n.name: n.state() for n in self.nodes}
            yield self
            if not stateful:
                self.reset(to_state=current_state)
        else:
            if state is None:
                state = {}

            with ExitStack() as stack:
                for node in self.nodes:
                    value = state.get(node.name)
                    stack.enter_context(node.with_state(value,
                                                        stateful=stateful,
                                                        reset=reset))
                yield self

    @contextmanager
    def with_feedback(self, feedback=None, stateful=False, reset=False):

        if feedback is None and not reset:
            yield self
            return

        with ExitStack() as stack:
            if feedback is not None:
                for node in self.nodes:
                    value = feedback.get(node.name)
                    stack.enter_context(node.with_feedback(value,
                                                           stateful=stateful,
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
                self[node_name].reset(to_state=current_state)
        return self

    def initialize(self, x=None, y=None):
        self._is_initialized = False
        self._initializer(self, x=x, y=y)
        self.reset()
        self._is_initialized = True
        return self

    def initialize_buffers(self):
        for node in self.nodes:
            if node._buffers_initializer is not None:
                node.initialize_buffers()

    def call(self, x=None, forced_feedback=None, from_state=None,
             stateful=True, reset=False, return_states=None):

        if x is not None:
            x = self._check_input(x)

        if not self._is_initialized:
            self.initialize(x)

        # load current states in proxys interfaces accessible
        # through feedback. These proxys are not updated during the graph call.
        self._load_proxys()

        with self.with_state(from_state, stateful=stateful, reset=reset):
            # maybe load forced feedbacks in proxys ?
            with self.with_feedback(forced_feedback,
                                    stateful=stateful,
                                    reset=reset):
                state = self._call(x, return_states)

        # wash states proxys
        self._clean_proxys()

        return state

    def run(self, X=None, forced_feedbacks=None, from_state=None,
            stateful=True, reset=False, shift_fb=True, return_states=None):

        self._initialize_on_sequence(X, forced_feedbacks)

        states = _allocate_returned_states(self, X, return_states)

        seq = progress(
            self._dispatcher.dispatch(X, forced_feedbacks, shift_fb=shift_fb),
            f"Running {self.name}", total=len(X))

        with self.with_state(from_state, stateful=stateful, reset=reset):
            for i, (x, forced_feedback, _) in enumerate(seq):

                self._load_proxys()
                with self.with_feedback(forced_feedback):
                    state = self._call(x, return_states=return_states)

                if is_mapping(state):
                    for name, value in state.items():
                        states[name][i, :] = value
                else:
                    states[self.output_nodes[0].name][i, :] = state

        self._clean_proxys()

        # dicts are only relevant if model has several outputs (len > 1) or
        # if we want to return states from hidden nodes
        if len(states) == 1 and return_states is None:
            return states[self.output_nodes[0].name]

        return states

    def train(self, X, Y=None, force_teachers=True, learn_every=1,
              from_state=None, stateful=True,
              reset=False, return_states=None):

        self._check_if_only_online()

        self._initialize_on_sequence(X, Y)

        states = _allocate_returned_states(self, X, return_states)

        seq_len = X.shape[0]
        dispatch = self._dispatcher.dispatch(X, Y, return_targets=True)
        seq = progress(dispatch, f"Training {self.name}", total=seq_len) \
            if seq_len > 1 else dispatch

        self._load_proxys()
        with self.with_state(from_state, stateful=stateful, reset=reset):
            for i, (x, forced_feedback, y) in enumerate(seq):

                if not force_teachers:
                    forced_feedback = None

                with self.with_feedback(forced_feedback):
                    state = self._call(x, return_states=return_states)

                # reload feedbacks from training. Some nodes may need
                # feedback signals to train.
                self._load_proxys()

                y = self._check_targets(y)

                if i % learn_every == 0 or len(X) == 1:
                    self._train(self, x=x, y=y, force_teachers=force_teachers)

                if is_mapping(state):
                    for name, value in state.items():
                        states[name][i, :] = value
                else:
                    states[self.output_nodes[0].name][i, :] = state

        self._clean_proxys()

        # dicts are only relevant if model has several outputs (len > 1) or
        # if we want to return states from hidden nodes
        if len(states) == 1 and return_states is None:
            return states[self.output_nodes[0].name]

        return states

    def fit(self, X=None, Y=None, force_teachers=True,
            from_state=None, stateful=True, reset=False,
            workers=1):

        if not any([n for n in self.trainable_nodes if n.is_trained_offline]):
            raise TypeError(f"Impossible to fit model {self} offline: "
                            "no offline nodes found in model.")

        X, Y = to_ragged_seq_set(X), to_ragged_seq_set(Y)
        data = list(self._dispatcher.dispatch(X, Y, shift_fb=False))
        X = [datum[0] for datum in data]
        Y = [datum[1] for datum in data]

        self._initialize_on_sequence(X[0], Y[0])
        self.initialize_buffers()

        subgraphs = get_offline_subgraphs(self.nodes,
                                          self.edges)

        trained = set()
        next_X = None

        with self.with_state(from_state, reset=reset, stateful=stateful):
            for i, ((nodes, edges), relations) in enumerate(subgraphs):
                submodel, offlines = _build_forward_sumodels(nodes,
                                                             edges,
                                                             trained)

                if next_X is not None:
                    for j in range(len(X)):
                        X[j].update(next_X[j])

                return_states = None
                if len(relations) > 0:
                    return_states = list(relations.keys())

                # next inputs for next submodel
                next_X = []
                seq = progress(X, f"Running {self.name}")

                _partial_fit_fn = partial(_run_and_partial_fit,
                                          force_teachers=force_teachers,
                                          model=submodel,
                                          reset=reset,
                                          offlines=offlines,
                                          relations=relations,
                                          stateful=stateful,
                                          return_states=return_states)

                # for seq/batch in dataset
                for x_seq, y_seq in zip(seq, Y):
                    next_X += [_partial_fit_fn(x_seq=x_seq, y_seq=y_seq)]

                for node in offlines:
                    if verbosity():
                        print(f"Fitting node {node.name}...")
                    node.fit()

                trained |= set(offlines)

        return self

    def partial_fit(self, *args, **kwargs):
        return self


class FrozenModel(Model):

    def __init__(self, *args, **kwargs):
        super(FrozenModel, self).__init__(*args, **kwargs)

    def update_graph(self, new_nodes, new_edges):
        raise TypeError(f"Impossible to update FrozenModel {self}: "
                        f"model is frozen and cannot be modified.")
