# Author: Nathan Trouvain at 27/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Sequence
from multiprocessing import Manager

import numpy as np
from joblib import Parallel, delayed

from .force import FORCE
from .nvar import NVAR
from .reservoir import Reservoir
from .ridge import Ridge
from ..model import FrozenModel
from ..utils import progress, verbosity, _obj_from_kwargs
from reservoirpy._utils import to_ragged_seq_set
from ..types import GenericNode
from ..utils.validation import is_mapping
from ..utils.parallel import get_joblib_backend

_LEARNING_METHODS = {"ridge": Ridge,
                     "force": FORCE}

_RES_METHODS = {"reservoir": Reservoir,
                "nvar": NVAR}


def _allocate_returned_states(model, inputs=None, return_states=None):
    if inputs is not None:
        if is_mapping(inputs):
            seq_len = inputs[list(inputs.keys())[0]].shape[0]
        else:
            seq_len = inputs.shape[0]
    else:
        raise ValueError("'X' and 'n' parameters can't be None at the "
                         "same time.")

    vulgar_names = {"reservoir": model.reservoir,
                    "readout": model.readout}

    # pre-allocate states
    if return_states == "all":
        states = {name: np.zeros((seq_len, n.output_dim))
                  for name, n in vulgar_names.items()}
    elif isinstance(return_states, Sequence):
        states = {name: np.zeros((seq_len, n.output_dim))
                  for name, n in {name: vulgar_names[name]
                                  for name in return_states}}
    else:
        states = {"readout": np.zeros((seq_len, model.readout.output_dim))}

    return states


def _sort_and_unpack(states, return_states=None):
    # maintain input order (even with parallelization on)
    states = sorted(states, key=lambda s: s[0])
    states = {n: [s[1][n] for s in states] for n in states[0][1].keys()}

    for n, s in states.items():
        if len(s) == 1:
            states[n] = s[0]

    if len(states) == 1 and return_states is None:
        states = states["readout"]

    return states


def forward(model: "ESN", x):
    data = model.data_dispatcher.load(x)

    for node in model.nodes:
        node(data[node].x)

    return [out_node.state() for out_node in model.output_nodes]


def intialize_buffers(model: "ESN"):
    model.readout._buffers_initializer(model.readout)


class ESN(FrozenModel):

    def __init__(self, reservoir_method="reservoir",
                 learning_method="ridge", reservoir: GenericNode = None,
                 readout: GenericNode = None, feedback=False, Win_bias=True,
                 Wout_bias=True, workers=1,
                 backend=None, name=None, **kwargs):

        msg = "'{}' is not a valid method. Available methods for {} are {}."

        if reservoir is None:
            if reservoir_method not in _RES_METHODS:
                raise ValueError(msg.format(reservoir_method, "reservoir",
                                            list(_RES_METHODS.keys())))
            else:
                klas = _RES_METHODS[reservoir_method]
                kwargs["input_bias"] = Win_bias
                reservoir = _obj_from_kwargs(klas, kwargs)

        if readout is None:
            if learning_method not in _LEARNING_METHODS:
                raise ValueError(msg.format(learning_method, "readout",
                                            list(_LEARNING_METHODS.keys())))
            else:
                klas = _LEARNING_METHODS[learning_method]
                kwargs["input_bias"] = Wout_bias
                readout = _obj_from_kwargs(klas, kwargs)

        if feedback:
            reservoir <<= readout

        super(ESN, self).__init__(nodes=[reservoir, readout],
                                  edges=[(reservoir, readout)],
                                  name=name)

        self._hypers.update({"workers": workers,
                             "backend": backend,
                             "reservoir_method": reservoir_method,
                             "learning_method": learning_method,
                             "feedback": feedback})

        self._params.update({"reservoir": reservoir,
                             "readout": readout})

        self._trainable = True
        self._is_fb_initialized = False

        # in case an external Model wants to initialize all buffers at once
        self._buffers_initializer = intialize_buffers

    @property
    def is_trained_offline(self) -> bool:
        return True

    @property
    def is_trained_online(self) -> bool:
        return False

    @property
    def is_fb_initialized(self):
        return self._is_fb_initialized

    @property
    def has_feedback(self):
        """Always returns False, ESNs are not supposed to receive external
        feedback. Feedback between reservoir and readout must be defined
        at ESN creation."""
        return False

    def _call(self, x=None, return_states=None, *args, **kwargs):

        if is_mapping(x):
            data = x[self.reservoir.name]
        else:
            data = x

        state = self.reservoir._call(data)
        self.readout._call(state)

        state = {}
        if return_states == "all":
            for node in ["reservoir", "readout"]:
                state[node] = getattr(self, node).state()
        elif isinstance(return_states, Sequence):
            for name in return_states:
                if name in self.node_names:
                    state[name] = self[name].state()
                elif name in ["reservoir", "readout"]:
                    state[name] = getattr(self, name).state()
        else:
            state = self.readout.state()

        return state

    def state(self, which="external"):
        if which == "external":
            return self.readout.state()
        elif which == "internal":
            return self.reservoir.state()
        else:
            raise ValueError(f"'which' parameter of {self.name} "
                             f"'state' function must be "
                             f"one of 'external' or 'internal'.")

    def initialize_feedback(self) -> "Node":
        """Call the Node feedback initializer. The feedback initializer will
        determine feedback dimension given some feedback signal, and intialize
        all parameters related to the feedback connection.

        Feedback sender Node must be initialized, as the feedback intializer
        will probably call the :py:meth:`Node.feedback` method to get
        a sample of feedback signal.

        Returns
        -------
            Node
                Initialized Node.
        """
        if self.feedback:
            if not self.reservoir.is_fb_initialized:
                empty_feedback = self.reservoir.zero_feedback()
                self.reservoir._feedback_initializer(self.reservoir,
                                                     empty_feedback)
                self._is_fb_initialized = True
        return self

    def run(self, X=None, forced_feedbacks=None, from_state=None,
            stateful=True, reset=False, shift_fb=True, return_states=None):

        X = to_ragged_seq_set(X)
        if forced_feedbacks is not None:
            forced_feedbacks = to_ragged_seq_set(forced_feedbacks)
            init_fb = forced_feedbacks[0]
            fb_gen = iter(forced_feedbacks)
        else:
            init_fb = forced_feedbacks
            fb_gen = (None for _ in range(len(X)))

        self._initialize_on_sequence(X[0], init_fb)

        def run_fn(idx, x, forced_fb):

            states = _allocate_returned_states(self, x, return_states)

            with self.with_state(from_state, stateful=stateful, reset=reset):
                for i, (x, forced_feedback, _) in enumerate(
                        self._dispatcher.dispatch(x, forced_fb,
                                                  shift_fb=shift_fb)):
                    self._load_proxys()
                    with self.with_feedback(forced_feedback):
                        state = self._call(x, return_states=return_states)

                    if is_mapping(state):
                        for name, value in state.items():
                            states[name][i, :] = value
                    else:
                        states["readout"][i, :] = state

            self._clean_proxys()

            return idx, states

        backend = get_joblib_backend(workers=self.workers,
                                     backend=self.backend)

        seq = progress(X, f"Running {self.name}")

        with self.with_state(from_state, reset=reset, stateful=stateful):
            with Parallel(n_jobs=self.workers,
                          backend=backend) as parallel:
                states = parallel(delayed(run_fn)(idx, x, y)
                                  for idx, (x, y) in enumerate(
                    zip(seq, fb_gen)))

        return _sort_and_unpack(states, return_states=return_states)

    def fit(self, X=None, Y=None, from_state=None, stateful=True, reset=False):

        if not self.readout.is_trained_offline:
            raise TypeError(f"Impossible to fit {self} offline: "
                            f"readout {self.readout} is not an offline node.")

        X, Y = to_ragged_seq_set(X), to_ragged_seq_set(Y)
        self._initialize_on_sequence(X[0], Y[0])

        self.initialize_buffers()

        if (self.workers > 1 or self.workers == -1) and \
                self.backend not in ("sequential", "threading"):
            lock = Manager().Lock()
        else:
            lock = None

        def run_partial_fit_fn(x, y):
            states = np.zeros((x.shape[0], self.reservoir.output_dim))

            for i, (x, forced_feedback, _) in enumerate(
                    self._dispatcher.dispatch(x, y, shift_fb=True)):
                self._load_proxys()

                with self.readout.with_feedback(
                        forced_feedback[self.readout.name]):
                    states[i, :] = self.reservoir._call(x[self.reservoir.name])

            self._clean_proxys()

            # Avoid any problem related to multiple
            # writes from multiple processes
            if lock is not None:
                with lock:
                    self.readout.partial_fit(states, y)
            else:
                self.readout.partial_fit(states, y)

        backend = get_joblib_backend(workers=self.workers,
                                     backend=self.backend)

        seq = progress(X, f"Running {self.name}")
        with self.with_state(from_state, reset=reset, stateful=stateful):
            with Parallel(n_jobs=self.workers,
                          backend=backend) as parallel:
                parallel(delayed(run_partial_fit_fn)(x, y)
                         for x, y in zip(seq, Y))

            if verbosity():
                print(f"Fitting node {self.name}...")
            self.readout.fit()

        return self
