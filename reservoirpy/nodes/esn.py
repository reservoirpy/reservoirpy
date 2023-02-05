# Author: Nathan Trouvain at 27/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from multiprocessing import Manager

import numpy as np
from joblib import Parallel, delayed

from .._base import _Node, call
from ..model import FrozenModel
from ..utils import _obj_from_kwargs, progress, verbosity
from ..utils.graphflow import dispatch
from ..utils.model_utils import to_data_mapping
from ..utils.parallel import get_joblib_backend
from ..utils.validation import is_mapping
from .io import Input
from .readouts import Ridge
from .reservoirs import NVAR, Reservoir

_LEARNING_METHODS = {"ridge": Ridge}

_RES_METHODS = {"reservoir": Reservoir, "nvar": NVAR}


def _allocate_returned_states(model, inputs, return_states=None):
    """Create empty placeholders for model outputs."""
    seq_len = inputs[list(inputs.keys())[0]].shape[0]
    vulgar_names = {"reservoir": model.reservoir, "readout": model.readout}

    # pre-allocate states
    if return_states == "all":
        states = {
            name: np.zeros((seq_len, n.output_dim)) for name, n in vulgar_names.items()
        }
    elif isinstance(return_states, (list, tuple)):
        states = {
            name: np.zeros((seq_len, n.output_dim))
            for name, n in {name: vulgar_names[name] for name in return_states}.items()
        }
    else:
        states = {"readout": np.zeros((seq_len, model.readout.output_dim))}

    return states


def _sort_and_unpack(states, return_states=None):
    """Maintain input order (even with parallelization on)"""
    states = sorted(states, key=lambda s: s[0])
    states = {n: [s[1][n] for s in states] for n in states[0][1].keys()}

    for n, s in states.items():
        if len(s) == 1:
            states[n] = s[0]

    if len(states) == 1 and return_states is None:
        states = states["readout"]

    return states


class ESN(FrozenModel):
    """Echo State Networks as a Node, with parallelization of state update.

    This Node is provided as a wrapper for reservoir and readout nodes. Execution
    is distributed over several workers when:

    - the ``workers`` parameters is equal to `n>1` (using `n` workers) or
      `n<=-1` (using `max_available_workers - n` workers)
    - Several independent sequences of inputs are fed to the model at runtime.

    When parallelization is enabled, internal states of the reservoir will be reset
    to 0 at the beginning of every independent sequence of inputs.

    Note
    ----
        This node can not be connected to other nodes. It is only provided as a
        convenience Node to speed up processing of large datasets with "vanilla"
        Echo State Networks.

    :py:attr:`ESN.params` **list**:

    ================== =================================================================
    ``reservoir``      A :py:class:`~reservoirpy.nodes.Reservoir` or a :py:class:`~reservoirpy.nodes.NVAR` instance.
    ``readout``        A :py:class:`~reservoirpy.nodes.Ridge` instance.
    ================== =================================================================

    :py:attr:`ESN.hypers` **list**:

    ==================== ===============================================================
    ``workers``          Number of workers for parallelization (1 by default).
    ``backend``          :py:mod:`joblib` backend to use for parallelization  (``loky`` by default,).
    ``reservoir_method`` Type of reservoir, may be "reservoir" or "nvar" ("reservoir" by default).
    ``learning_method``  Type of readout, by default "ridge".
    ``feedback``         Is readout connected to reservoir through feedback (False by default).
    ==================== ===============================================================

    Parameters
    ----------
    reservoir_method : {"reservoir", "nvar"}, default to "reservoir"
        Type of reservoir, either a :py:class:`~reservoirpy.nodes.Reservoir` or
        a :py:class:`~reservoirpy.nodes.NVAR`.
    learning_method : {"ridge"}, default to "ridge"
        Type of readout. The only method supporting parallelization for now is the
        :py:class:`~reservoirpy.nodes.Ridge` readout.
    reservoir : Node, optional
        A Node instance to use as a reservoir,
        such as a :py:class:`~reservoirpy.nodes.Reservoir` node.
    readout : Node, optional
        A Node instance to use as a readout,
        such as a :py:class:`~reservoirpy.nodes.Ridge` node
        (only this one is supported).
    feedback : bool, default to False
        If True, the reservoir is connected to the readout through
        a feedback connection.
    Win_bias : bool, default to True
        If True, add an input bias to the reservoir.
    Wout_bias : bool, default to True
        If True, add a bias term to the reservoir states entering the readout.
    workers : int, default to 1
        Number of workers used for parallelization. If set to -1, all available workers
        (threads or processes) are used.
    backend : a :py:mod:`joblib` backend, default to "loky"
        A parallelization backend.
    name : str, optional
        Node name.

    See Also
    --------
    Reservoir
    Ridge
    NVAR

    Example
    -------

    """

    def __init__(
        self,
        reservoir_method="reservoir",
        learning_method="ridge",
        reservoir: _Node = None,
        readout: _Node = None,
        feedback=False,
        Win_bias=True,
        Wout_bias=True,
        workers=1,
        backend=None,
        name=None,
        use_raw_inputs=False,
        **kwargs,
    ):

        msg = "'{}' is not a valid method. Available methods for {} are {}."

        if reservoir is None:
            if reservoir_method not in _RES_METHODS:
                raise ValueError(
                    msg.format(reservoir_method, "reservoir", list(_RES_METHODS.keys()))
                )
            else:
                klas = _RES_METHODS[reservoir_method]
                kwargs["input_bias"] = Win_bias
                reservoir = _obj_from_kwargs(klas, kwargs)

        if readout is None:
            if learning_method not in _LEARNING_METHODS:
                raise ValueError(
                    msg.format(
                        learning_method, "readout", list(_LEARNING_METHODS.keys())
                    )
                )
            else:
                klas = _LEARNING_METHODS[learning_method]
                kwargs["input_bias"] = Wout_bias
                readout = _obj_from_kwargs(klas, kwargs)

        if feedback:
            reservoir <<= readout

        if use_raw_inputs:
            source = Input()
            super(ESN, self).__init__(
                nodes=[reservoir, readout, source],
                edges=[(source, reservoir), (reservoir, readout), (source, readout)],
                name=name,
            )
        else:
            super(ESN, self).__init__(
                nodes=[reservoir, readout], edges=[(reservoir, readout)], name=name
            )

        self._hypers.update(
            {
                "workers": workers,
                "backend": backend,
                "reservoir_method": reservoir_method,
                "learning_method": learning_method,
                "feedback": feedback,
            }
        )

        self._params.update({"reservoir": reservoir, "readout": readout})

        self._trainable = True
        self._is_fb_initialized = False

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

        data = x[self.reservoir.name]

        state = call(self.reservoir, data)
        call(self.readout, state)

        state = {}
        if return_states == "all":
            for node in ["reservoir", "readout"]:
                state[node] = getattr(self, node).state()
        elif isinstance(return_states, (list, tuple)):
            for name in return_states:
                state[name] = getattr(self, name).state()
        else:
            state = self.readout.state()

        return state

    def state(self, which="reservoir"):
        if which == "reservoir":
            return self.reservoir.state()
        elif which == "readout":
            return self.readout.state()
        else:
            raise ValueError(
                f"'which' parameter of {self.name} "
                f"'state' function must be "
                f"one of 'reservoir' or 'readout'."
            )

    def run(
        self,
        X=None,
        forced_feedbacks=None,
        from_state=None,
        stateful=True,
        reset=False,
        shift_fb=True,
        return_states=None,
    ):

        X, forced_feedbacks = to_data_mapping(self, X, forced_feedbacks)

        self._initialize_on_sequence(X[0], forced_feedbacks[0])

        def run_fn(idx, x, forced_fb):

            states = _allocate_returned_states(self, x, return_states)

            with self.with_state(from_state, stateful=stateful, reset=reset):
                for i, (x, forced_feedback, _) in enumerate(
                    dispatch(x, forced_fb, shift_fb=shift_fb)
                ):
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

        backend = get_joblib_backend(workers=self.workers, backend=self.backend)

        seq = progress(X, f"Running {self.name}")

        with self.with_state(from_state, reset=reset, stateful=stateful):
            with Parallel(n_jobs=self.workers, backend=backend) as parallel:
                states = parallel(
                    delayed(run_fn)(idx, x, y)
                    for idx, (x, y) in enumerate(zip(seq, forced_feedbacks))
                )

        return _sort_and_unpack(states, return_states=return_states)

    def fit(self, X=None, Y=None, warmup=0, from_state=None, stateful=True, reset=False):

        X, Y = to_data_mapping(self, X, Y)
        self._initialize_on_sequence(X[0], Y[0])

        self.initialize_buffers()

        if (self.workers > 1 or self.workers < 0) and self.backend not in (
            "sequential",
            "threading",
        ):
            lock = Manager().Lock()
        else:
            lock = None

        def run_partial_fit_fn(x, y):
            seq_len = len(x[list(x)[0]])
            states = np.zeros((seq_len, self.reservoir.output_dim))

            for i, (x, forced_feedback, _) in enumerate(dispatch(x, y, shift_fb=True)):
                self._load_proxys()

                with self.readout.with_feedback(forced_feedback[self.readout.name]):
                    states[i, :] = call(self.reservoir, x[self.reservoir.name])

            self._clean_proxys()

            # Avoid any problem related to multiple
            # writes from multiple processes
            # if lock is not None:
            # with lock:  # pragma: no cover
            self.readout.partial_fit(states, y[self.readout.name], warmup=warmup, lock=lock)
            # else:
            #     self.readout.partial_fit(states, y[self.readout.name])

        backend = get_joblib_backend(workers=self.workers, backend=self.backend)

        seq = progress(X, f"Running {self.name}")
        with self.with_state(from_state, reset=reset, stateful=stateful):
            with Parallel(n_jobs=self.workers, backend=backend) as parallel:
                parallel(delayed(run_partial_fit_fn)(x, y) for x, y in zip(seq, Y))

            if verbosity():  # pragma: no cover
                print(f"Fitting node {self.name}...")

            self.readout.fit()

        return self
