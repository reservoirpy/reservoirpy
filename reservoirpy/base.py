# Author: Nathan Trouvain at 22/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from contextlib import ExitStack, contextmanager
from copy import copy, deepcopy
from uuid import uuid4
from typing import Callable, Dict, List, Optional, Union, Tuple, Any, TypeVar
from collections import Iterable

import numpy as np

from tqdm import tqdm

from .model import Model
from .utils import to_ragged_seq_set
from .utils.types import MappedData, Data, GenericNode, ForwardFn, BackwardFn, PartialBackFn, ForwardInitFn, EmptyInitFn, Shape
from .utils.validation import check_vector, is_mapping
from .utils.parallel import memmap_buffer, clean_tempfile


def _initialize_with_seq_set(node, X, Y=None):
    X = to_ragged_seq_set(X)

    if Y is not None:
        Y = to_ragged_seq_set(Y)

    if not node.is_initialized:
        if Y is not None:
            node.initialize(X[0], Y[0])
        else:
            node.initialize(X[0])

    return X, Y


def _node_fb_init_general(node):
    if node.has_feedback:
        feedback = node.feedback()

        fb_dim = None
        if isinstance(feedback, list):
            fb_dim = tuple([fb.shape[1] for fb in feedback])
        elif isinstance(feedback, np.ndarray):
            fb_dim = feedback.shape[1]

        node.set_feedback_dim(fb_dim)


def _remove_input_for_feedback(model: "Model") -> "Node":
    all_nodes = set(model.nodes)
    input_nodes = set(model.input_nodes)
    filtered_nodes = all_nodes - input_nodes
    filtered_edges = [edge for edge in model.edges
                      if edge[0] not in input_nodes]

    # return a Node if Model - Inputs = 1 Node
    # else return a Model - Inputs
    if len(filtered_nodes) == 1:
        return list(filtered_nodes)[0]
    return Model(filtered_nodes, filtered_edges, name=str(uuid4()))



class Node(GenericNode):
    _name: str

    _state: Optional[np.ndarray]
    _state_proxy:    Optional[np.ndarray]
    _feedback:       Optional[GenericNode]

    _params:  Dict[str, Any]
    _hypers:  Dict[str, Any]
    _buffers: Dict[str, Any]

    _input_dim:    Shape
    _output_dim:   Shape
    _feedback_dim: Shape

    _forward:          ForwardFn
    _backward:         BackwardFn
    _partial_backward: PartialBackFn
    _train:            PartialBackFn

    _initializer:          ForwardInitFn
    _buffers_initializer:  EmptyInitFn
    _feedback_initializer: EmptyInitFn

    _trainable: bool
    _fitted:    bool

    def __init__(self,
                 params: Dict[str, Any] = None,
                 hypers: Dict[str, Any] = None,
                 forward: ForwardFn = None,
                 backward: BackwardFn = None,
                 partial_backward: PartialBackFn = None,
                 train: PartialBackFn = None,
                 initializer: ForwardInitFn = None,
                 fb_initializer: EmptyInitFn = _node_fb_init_general,
                 buffers_initializer: EmptyInitFn = None,
                 input_dim: Shape = None,
                 output_dim: Shape = None,
                 feedback_dim: Shape = None,
                 name: str = None,
                 *args, **kwargs):

        self._params = dict() if params is None else params
        self._hypers = dict() if hypers is None else hypers
        # buffers are all node state components that should not live
        # outside the node training loop, like partial computations for
        # linear regressions. They can also be shared across multiple processes
        # when needed.
        self._buffers = dict()

        self._forward = forward
        self._backward = backward
        self._partial_backward = partial_backward
        self._train = train

        self._initializer = initializer
        self._feedback_initializer = fb_initializer
        self._buffers_initializer = buffers_initializer

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._feedback_dim = feedback_dim

        self._name = self._get_name(name)

        self._is_initialized = False
        self._is_fb_initialized = False
        self._state_proxy = None
        self._feedback = None

        # used to store a reduced version of the feedback if needed
        # when feedback is a Model (inputs of the feedback Model are suppressed
        # in the reduced version, as we do not need then to re-run them
        # because we assume they have already run during the forward call)
        self._reduced_fb = None

        self._trainable = self._backward is not None or \
                          self._partial_backward is not None or \
                          self._train is not None

        self._fitted = False if self.is_trainable and self.is_trained_offline\
            else True

    def __getattr__(self, item):
        return self.get_param(item)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def __rshift__(self, other):
        return self.link(other)

    def __lshift__(self, other):
        return self.link_feedback(other)

    def __ilshift__(self, other):
        return self.link_feedback(other, inplace=True)

    def _check_state(self, s):
        s = check_vector(s)

        if not self._is_initialized:
            raise RuntimeError(
                f"Impossible to set state of node {self.name}: node "
                f"is not initialized yet.")

        if s.shape[1] != self.output_dim:
            raise ValueError(f"Impossible to set state of node {self.name}: "
                             f"dimension mismatch between state vector ("
                             f"{s.shape[1]}) "
                             f"and node output dim ({self.output_dim}).")
        return s

    def _check_input(self, x):
        if isinstance(x, np.ndarray):
            x = check_vector(x, allow_reshape=True)

            if self._is_initialized:
                if x.shape[1] != self.input_dim:
                    raise ValueError(
                        f"Impossible to call node {self.name}: node input "
                        f"dimension is (1, {self.input_dim}) and input "
                        f"dimension "
                        f"is {x.shape}.")

        elif isinstance(x, list):
            for i in range(len(x)):
                x[i] = check_vector(x[i], allow_reshape=True)
        return x

    def _check_output(self, y):
        y = check_vector(y, allow_reshape=True)

        if self._is_initialized:
            if y.shape[1] != self.output_dim:
                raise ValueError(
                    f"Impossible to fit node {self.name}: node expected "
                    f"output dimension is (1, {self.output_dim}) and teacher "
                    f"vector dimension is {y.shape}."
                    )

    def _call(self, x=None, from_state=None, stateful=True, reset=False):
        with self.with_state(from_state, stateful=stateful, reset=reset):
            x = check_vector(x, allow_reshape=True)
            state = self._forward(self, x)
            self._state = state

        return state

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def feedback_dim(self):
        return self._feedback_dim

    @property
    def is_initialized(self):
        return self._is_initialized

    @property
    def has_feedback(self):
        return self._feedback is not None

    @property
    def is_trained_offline(self):
        return self.is_trainable and (self._backward is not None or
                                      self._partial_backward is not None)

    @property
    def is_trained_online(self):
        return self.is_trainable and self._train is not None

    @property
    def is_trainable(self):
        return self._trainable

    @is_trainable.setter
    def is_trainable(self, value: bool):
        if type(value) is bool:
            self._trainable = value
        else:
            raise TypeError("'is_trainable' must be a boolean.")

    @property
    def fitted(self):
        return self._fitted

    @property
    def is_fb_initialized(self):
        return self._is_fb_initialized

    def state(self):
        if not self.is_initialized:
            return None
        return self._state

    def state_proxy(self):
        if self._state_proxy is None:
            return self._state
        return self._state_proxy

    def feedback(self):
        if self.has_feedback:
            if not self._feedback.is_initialized:
                raise RuntimeError(f"Impossible to get feedback "
                                   f"from node or model {self._feedback} "
                                   f"to node {self.name}: {self._feedback.name} "
                                   f"is not initialized.")

            if isinstance(self._feedback, Model):
                input_data = {c.name: p.state_proxy()
                              for p, c in self._feedback.edges
                              if p in self._feedback.input_nodes}
                if isinstance(self._reduced_fb, Model):
                    return self._reduced_fb.call(input_data)
                else:
                    reduced_name = self._reduced_fb.name
                    return self._reduced_fb.call(input_data[reduced_name])
            elif isinstance(self._feedback, Node):
                return self._feedback.state_proxy()
            else:
                return None
        else:
            raise RuntimeError(f"Node {self} is not connected to any feedback "
                               f"Node or Model.")

    def set_state_proxy(self, value=None):
        if value is not None:
            value = self._check_state(value)
        self._state_proxy = value

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

    def set_feedback_dim(self, value):
        if not self.is_fb_initialized:
            self._feedback_dim = value
        else:
            raise TypeError(f"Output dimension of {self.name} is "
                            "immutable after initialization.")

    def get_param(self, name):
        if name in self._params:
            return self._params.get(name)
        elif name in self._hypers:
            return self._hypers.get(name)
        else:
            raise AttributeError(f"No attribute named '{name}' "
                                 f"found in node {self}")

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

    def create_buffer(self, name, shape=None, data=None):
        self._buffers[name] = memmap_buffer(self, data=data,
                                            shape=shape, name=name)

    def set_buffer(self, name, value):
        self._buffers[name][:] = value

    def get_buffer(self, name):
        return self._buffers[name]

    def initialize(self, x=None, y=None):
        if not self._is_initialized:
            if isinstance(x, np.ndarray):
                x = np.atleast_2d(check_vector(x))
            elif isinstance(x, list):
                for i in range(len(x)):
                    x[i] = np.atleast_2d(check_vector(x[i]))

            if y is not None:
                y = np.atleast_2d(check_vector(y))
                self._initializer(self, x=x, y=y)
            else:
                self._initializer(self, x=x)

            self.reset()
            self._is_initialized = True

        return self

    def initialize_feedback(self):
        if not self.is_fb_initialized:
            if isinstance(self._feedback, Model):
                input_data = {c.name: p.state_proxy()
                              for p, c in self._feedback.edges
                              if p in self._feedback.input_nodes}
                self._reduced_fb = _remove_input_for_feedback(self._feedback)

                if isinstance(self._reduced_fb, Model):
                    self._reduced_fb.initialize(x=input_data)
                else:
                    reduced_name = self._reduced_fb.name
                    self._reduced_fb.initialize(x=input_data[reduced_name])

            elif isinstance(self._feedback, Node):
                self._feedback_initializer(self)

            self._is_fb_initialized = True
            self._feedback._is_initialized = True

        return self

    def initialize_buffers(self):
        if self._buffers_initializer is not None:
            if len(self._buffers) == 0:
                self._buffers_initializer(self)

    def clean_buffers(self):
        if len(self._buffers) > 0:
            self._buffers = dict()
            clean_tempfile(self)

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
        return self

    def reset_feedback(self, to_feedback=None):
        if to_feedback is None:
            self._feedback.reset()
        else:
            self._feedback.reset(to_state=to_feedback)

    @contextmanager
    def with_state(self, state=None, stateful=False, reset=False):

        if not self._is_initialized:
            raise RuntimeError(
                f"Impossible to set state of node {self.name}: node"
                f"is not initialized yet.")

        current_state = self._state

        if state is None:
            if reset:
                state = self.zero_state()
            else:
                state = current_state

        self.reset(to_state=state)
        yield self

        if not stateful:
            self._state = current_state

    @contextmanager
    def with_feedback(self, feedback=None, stateful=False, reset=False):

        if self.has_feedback:  # if it is a feedback receiver
            current_fb = self._feedback

            if feedback is None:
                if reset:
                    feedback = self.zero_feedback()
                else:
                    feedback = current_fb

            if isinstance(feedback, Node):
                self._feedback = feedback

                yield self

                self._feedback = current_fb

            elif isinstance(feedback, np.ndarray):
                current_proxy = self._feedback._state_proxy
                self._feedback.set_state_proxy(feedback)

                yield self

                if not stateful:
                    self._feedback._state_proxy = current_proxy
            else:
                raise TypeError(f"Impossible to get feedback from {feedback}: "
                                f"it is neither a Node instance nor an array.")

        else:  # maybe a feedback sender then ?
            current_state_proxy = self._state_proxy

            if feedback is None:
                if reset:
                    feedback = self.zero_state()
                else:
                    feedback = current_state_proxy

            self.set_state_proxy(feedback)

            yield self

            if not stateful:
                self._state_proxy = current_state_proxy

    def zero_state(self):
        """A null state vector."""
        if self.output_dim is not None:
            return np.zeros((1, self.output_dim))

    def zero_feedback(self):
        """A null state vector."""
        if self._feedback is not None:
            return self._feedback.zero_state()
        return None

    def link_feedback(self, node, inplace: bool = False, name: str = None):
        from .ops import link_feedback
        return link_feedback(self, node, inplace=inplace, name=name)

    def call(self, x=None, from_state=None, stateful=True, reset=False):

        if x is not None:
            x = self._check_input(x)

        if not self._is_initialized:
            self.initialize(x)

        state = self._call(x, from_state, stateful, reset)

        return state

    def run(self, X=None, n=1, from_state=None, stateful=True, reset=False):

        if not self._is_initialized:
            if X is not None:
                self.initialize(np.atleast_2d(X[0]))
            else:
                self.initialize()

        seq_len = X.shape[0] if X is not None else n

        with self.with_state(from_state, stateful=stateful, reset=reset):
            states = np.zeros((seq_len, self.output_dim))
            for i in range(seq_len):
                x = None
                if X is not None:
                    x = np.atleast_2d(X[i])
                s = self._call(x)
                states[i, :] = s

        return states

    def train(self, X, Y=None, force_teachers=True, call=True,
              learn_every=1, from_state=None, stateful=True, reset=False):

        if self._train is not None:

            if not self._is_initialized:
                x_init = np.atleast_2d(X[0])
                y_init = np.atleast_2d(Y[0]) if Y is not None else None
                self.initialize(x=x_init, y=y_init)
                self.initialize_buffers()

            seq_len = X.shape[0]

            with self.with_state(from_state, stateful=stateful, reset=reset):
                states = np.zeros((seq_len, self.output_dim))
                for i in range(seq_len):
                    x = X[i, :]

                    y = Y[i, :] if Y is not None else None
                    if y is None and self.has_feedback:
                        y = self.feedback()

                    if call:
                        s = self.call(x)
                    else:
                        s = self.zero_state()

                    if force_teachers:
                        self.set_state_proxy(y)

                    if i % learn_every == 0 or seq_len == 1:
                        self._train(self, x=x, y=y)

                    states[i, :] = s

            return states

    def partial_fit(self, X_batch, Y_batch=None):
        if self._partial_backward is not None:

            X_batch, Y_batch = _initialize_with_seq_set(self, X_batch, Y_batch)

            self.initialize_buffers()

            for X, Y in zip(X_batch, Y_batch):
                self._partial_backward(self, X, Y)

            return self

    def fit(self, X: Data=None, Y: Data=None, **kwargs) -> "Node":

        self._fitted = False

        if self._backward is not None:
            if X is not None:
                X, Y = _initialize_with_seq_set(self, X, Y)

                if self._partial_backward is not None:
                    for X_batch, Y_batch in zip(X, Y):
                        self.partial_fit(X_batch, Y_batch)

            elif not self._is_initialized:
                raise RuntimeError(f"Impossible to fit node {self.name}: node"
                                   f"is not initialized, and fit was called "
                                   f"without input and teacher data.")

            self._backward(self, X, Y)

            self._fitted = True

            self.clean_buffers()

            return self

    def copy(self, name: str = None, copy_feedback: bool = False,
             shallow: bool = False):
        if shallow:
            new_obj = copy(self)
        else:
            if self.has_feedback:
                # store feedback node
                fb = self._feedback
                # temporarily remove it
                self._feedback = None

                # copy and restore feedback
                new_obj = deepcopy(self)
                new_obj._feedback = fb
                self._feedback = fb

            else:
                new_obj = deepcopy(self)

        if copy_feedback:
            if self.has_feedback:
                fb_copy = deepcopy(self._feedback)
                new_obj._feedback = fb_copy

        n = self._get_name(name)
        new_obj._name = n

        return new_obj

