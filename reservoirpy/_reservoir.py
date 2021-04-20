from typing import Callable, Union, Tuple, Optional

import numpy as np
from numpy.random import Generator, default_rng

from .activationsfunc import get_function

from ._utils import _check_vector, _add_bias
from ._types import Weights


def _noisify(vector: np.ndarray,
             coef: float,
             random_generator: Generator = None
             ) -> np.ndarray:
    if coef > 0.0:
        if random_generator is None:
            random_generator = default_rng()
        noise = coef * random_generator.uniform(-1, 1, vector.shape)
        return vector + noise
    return vector


def _isquare(array: np.ndarray) -> bool:
    return array.shape[0] == array.shape[1]


def _is2dimensional(array: np.ndarray) -> bool:
    return array.ndim == 2


class Reservoir:
    """Base tool for reservoir computing
    using analog artificial neural networks.

    This can basically be schematized as
    a recurrent cell working on different
    streams of data:

    - external data (the inputs) is projected
    in the reservoir. The weights of the
    connections between the inputs and the
    reservoir units are stored in the `Win`
    matrix.

    - the reservoir represents these inputs
    along with the memory of the previously
    seen inputs in its `state`. The state and
    the new inputs are projected recurrently
    in the reservoir, through the connections
    stored in the `W` matrix.

    - optionally, some external feedback can
    also be projected back to the reservoir
    through the connections stored in the
    `Wfb` matrix. This feedback can come from
    some readout neurons, for instance.

    - finally, the freshly computed
    states can be retrieved and used to feed
    the readout neurons. See the
    :py:class:`Readout` class to know more
    about this part of the process.

    Parameters
    ----------
        Win : numpy.ndarray or scipy.sparse matrix
            Input matrix of the reservoir, storing the
            connections between the inputs and the reservoir.
        W : numpy.ndarray or scipy.sparse matrix
            Internal matrix of the reservoir, storing the
            recurrent connections between the reservoir
            and itself.
        Wfb : numpy.ndarray or scipy.sparse matrix, optional
            An optional feedback matrix storing the connections
            between some output value and the reservoir,
            by default None.
        fb_activation : Union[str, Callable[np.ndarray] ], optional
            An activation function to apply to the feedback
            values, by default "id"
        lr : float, optional
            The leaking rate, a valu controlling the time
            constant of the memory of the reservoir.
            By default 1.0, meaning that no previous states
            values will leak into the current state value
            of the reservoir.
        input_bias : bool, optional
            If ``True``, will add a constant bias
            of 1.0 to the input before projecting them
            to the reservoir, by default True.
        noise_rc : float, optional
            Maximum noise value appliyed to the input,
            used to regularize the activities, by default 0.0.
        noise_in : float, optional
            Maximum noise value appliyed to the activities
            (state) of the reservoir, by default 0.0.
        noise_out : float, optional
            Maximum noise value appliyed to the feedback
            values, by default 0.0.

    Attributes
    ----------
        state: numpy.ndarray
            Last saved internal state.
        last_input: numpy.ndarray
            Last input received.
        shape: tuple of int
            Dimensions of the reservoir, in the form
            (input dimension, reservoir size, feedback dimension).
            If no feedback is allowed, feedback dimension will
            be None.
            This can also be seen as:
            ``(Win.shape[1], W.shape[0], Wfb.shape[1])``.
    """

    _state: np.ndarray
    _last_input: np.ndarray

    W: Weights
    Win: Weights
    Wfb: Weights
    lr: float
    input_bias: bool
    noise_rc: float
    noise_in: float
    noise_out: float
    shape: Tuple[int, int, Optional[int]]

    @property
    def state(self) -> np.ndarray:
        """Last internal state produced by
        the reservoir.

        Can be reset to 0 or to a previous
        value using :py:method:`reset_state`.

        Is only useful during "stateful"
        operations (operations during which
        the state must be kept between
        different sequences of inputs).

        Returns
        -------
        numpy.ndarray
            Last internal state produced.
        """
        return self._state

    @property
    def last_input(self) -> np.ndarray:
        """Last input fed to the reservoir.
        Is only used when the reservoir needs
        feedback from a readout matrix that
        is trained on both the internal states
        of the reservoir and the raw inputs.

        Can be reset to 0 or to a previous
        value using :py:method:`reset_last_input`.

        Is only useful during "stateful"
        operations (operations during which
        the state must be kept between
        different sequences of inputs).

        Returns
        -------
        numpy.ndarray
            Last input received.
        """
        return self._last_input

    def __init__(self,
                 Win: Weights,
                 W: Weights,
                 Wfb: Weights = None,
                 fb_activation: Union[str,
                                      Callable
                                      ] = "id",
                 lr: float = 1.0,
                 input_bias: bool = True,
                 noise_rc: float = 0.0,
                 noise_in: float = 0.0,
                 noise_out: float = 0.0):
        self.lr = lr
        self.Win = _check_vector(Win)
        self.W = _check_vector(W)
        self.Wfb = None
        self.feedback = False

        if Wfb is not None:
            self.Wfb = _check_vector(Wfb)
            self.feedback = True

            if callable(fb_activation):
                self.fb_activation = fb_activation
            else:
                self.fb_activation = get_function(fb_activation)

        self.noise_rc = noise_rc
        self.noise_in = noise_in
        self.noise_out = noise_out
        self.input_bias = input_bias

        self.shape = tuple()
        self._check_matrices()

        self.reset_state()
        self.reset_last_input()

    def __call__(self,
                 inputs: np.ndarray = None,
                 readout: Callable = None,
                 n_steps: int = None,
                 from_state: np.ndarray = None,
                 from_input: np.ndarray = None,
                 stateful: bool = True,
                 generative: bool = False,
                 noise_generator: Generator = None) -> np.ndarray:
        """Make the data flow inside the reservoir, and output
        the corresponding states produced.

        Parameters
        ----------
        inputs : numpy.ndarray, optional
            A sequence of input data to feed the
            reservoir with, by default None.
        readout : Callable, optional
            A :py:class:`Readout` instance, when some
            feedback is needed, by default None.
        n_steps : int, optional
            Number of time steps to compute.

            This parameter is only used when no
            inputs are provided, triggering computation
            in isolated or generative mode for instance.
            In that case, `n_steps` states are computed.
            If an input sequence is provided, one state
            per input is produced and this parameter
            will raise an error if not None.
            By default None.
        from_state : numpy.ndarray, optional
            State vector from which to start
            computations.

            Useful when you need to reset
            the state to a particular value.
            By default None, in which case
            either the `state` saved internally
            (if `stateful` is ``True``) or
            a null vector will be used to initialize
            the internal states.
        from_input : numpy.ndarray, optional
            If a readout is passed as parameter
            and needs both states and inputs to
            make predictions, it might be necessary
            to pass a previous input along with the
            `from_state` parameter, by default None,
            in which case either the `last_input`
            saved internally (if `stateful` is ``True``) or
            a null vector will be used to compute the
            initial readout vector.
        stateful : bool, optional
            If ``True``, the last state saved
            will be used as an initial value
            for this run, along with the last input
            if needed by the readout. At the end of
            the flow of data, the last state freshly
            computed will be saved, so the next run
            will used this state as an initial state.

            Otherwise, the state passed in the
            `from_state` parameter will be used as
            a previous value, or a null vector.
            By default True.
        generative : bool, optional
            If ``True``, if a readout is
            available, and if the readout is trained
            on a regression task (i.e. the readout
            outputs have the same shape than the reservoir
            inputs), will use the outputs of
            the readout at each time step as an
            input for the next step. By default False.
        noise_generator : Generator, optional
            A `numpy` random state generator, to
            ensure reproducibility when using noise as
            a regularization of the states.
            By default None.

        Returns
        -------
        numpy.ndarray
            A series of internal states generated during
            the run.

        Raises
        ------
        ValueError
            Inputs can't be accepted if the generative mode is
            on, or if the `n_steps` parameter is used, because
            it is impossible to decide what to do in that case:
            use the inputs or ignore them ?
        """

        # first connections : from inputs to reservoir
        if inputs is None:
            # no inputs: the data flowing in the reservoir is null
            # (use zeroes instead)
            inputs = self._zero_input_signal(n_steps=n_steps)
            output_states = self._prepare_states_from_inputs(self._zero_input(),
                                                             n_steps=n_steps)
        else:
            # check that we really want to do that: having external inputs
            # and using generative mode is not compatible
            if n_steps is not None or generative:
                raise ValueError("No inputs are allowed in generative mode. "
                                 "Consider using only the n_steps parameter "
                                 "to specify how many steps to generate.")
            inputs = _check_vector(inputs)
            output_states = self._prepare_states_from_inputs(inputs,
                                                             n_steps=n_steps)

            if self.input_bias:
                inputs = _add_bias(inputs, 1.0)

        # second connections: from reservoir to itself
        if from_state is None:
            # resume previous memory loop
            if stateful:
                state = self.state
            # or restart from scratch (zero)
            else:
                state = self._zero_state()
        else:
            # resume with a memory externally saved
            state = _check_vector(from_state)

        # third connections: from inputs to readout
        # (if feedback is needed)
        if from_input is None:
            # resume from last seen data
            if stateful:
                last_input = self.last_input
            # or restart from scratch (no data)
            else:
                last_input = self._zero_input()
        else:
            # resume from an externally saved step
            last_input = _check_vector(from_input)

        # fourth connections: from readout to reservoir
        # (feedback loop)
        # use the provided readout or use a null signal
        # of data (a stream of zeroes) if no feedback
        # is required.
        readout = readout or self._zero_readout_signal

        for i, u in enumerate(inputs):

            # get feedback if needed (or zeroes)
            y = readout(state, last_input)
            last_input = u

            # plug outputs with inputs in a closed loop
            if generative:
                if self.input_bias:
                    y = _add_bias(y, 1.0)
                state = self._next_state(state, y, y, noise_generator)
            # just a forward pass
            else:
                state = self._next_state(state, u, y, noise_generator)

            output_states[i, :] = state

        # update the state internally saved
        if stateful:
            self.reset_state(from_state=state)
            self.reset_last_input(from_input=last_input)

        return output_states

    def _next_state(self,
                    x: np.ndarray,
                    u: np.ndarray,
                    y: np.ndarray,
                    noise_generator: Generator = None
                    ) -> np.ndarray:
        """Given the current internal state `x`, a new
        input `u`, and possibly a feedback `y`, compute
        the next state value.

        Parameters
        ----------
        x : np.ndarray
            Current state.
        u : np.ndarray
            New input.
        y : np.ndarray
            New feedback.
        noise_generator : Generator, optional
            Random state generator, to ensure
            reproducibility when using noise
            to regularize the activities,
            by default None.

        Returns
        -------
        np.ndarray
            Next state.
        """
        # x_ = W . x ...
        x_ = self.W @ x.reshape(-1, 1)
        #    + Win . (u + noise) ...
        x_ += self.Win @ _noisify(u.reshape(-1, 1),
                                  self.noise_in,
                                  noise_generator)

        if self.feedback:
            # + Wfb . (fb_activation(y) + noise)
            y_ = self.fb_activation(y.reshape(-1, 1))
            x_ += self.Wfb @ _noisify(y_,
                                      self.noise_out,
                                      noise_generator)

        # x1 = x * (1 - leak) + leak * (tanh(x_) + noise)
        x1 = (1 - self.lr) * x.reshape(-1, 1)
        x1 += self.lr * _noisify(np.tanh(x_),
                                 self.noise_rc,
                                 noise_generator)

        return x1.T

    def _check_matrices(self):
        """Some checks to be sure the weight matrices have
        a correct format.
        """

        if not _is2dimensional(self.W):
            raise ValueError("Reservoir matrix W should be 2-dimensional "
                             f"but is {self.W.ndim}-dimensional "
                             f"({self.W.shape}).")

        if not _isquare(self.W):
            raise ValueError("Reservoir matrix W should be square "
                             f"but is {self.W.shape}.")

        dim_internal = self.W.shape[0]

        if not _is2dimensional(self.Win):
            raise ValueError("Input matrix Win should be 2-dimensional "
                             f"but is {self.Win.shape}.")

        if not self.Win.shape[0] == self.W.shape[0]:
            raise ValueError("Input matrix Win should be of size "
                             f"({self.W.shape[0]}, input dimension + bias) "
                             f"but is of size {self.Win.shape}.")

        dim_in = self.Win.shape[1]

        if self.Wfb is None:
            self.Wfb = np.zeros((dim_internal, 1))
            dim_out = None
        else:
            if not _is2dimensional(self.Wfb):
                raise ValueError("Feedback matrix Wfb should be 2-dimensional "
                                 f"but is {self.Wfb.shape}.")
            if not self.Wfb.shape[0] == self.W.shape[0]:
                raise ValueError("Feedback matrix Wfb should be of size "
                                 f"({self.W.shape[0]}, output dimension) but "
                                 f"is of size {self.Wfb.shape}.")
            dim_out = self.Wfb.shape[1]

        self.shape = (
            dim_in,
            dim_internal,
            dim_out
        )

    def _zero_input_signal(self,
                           n_steps: int = 1
                           ) -> np.ndarray:
        """Yields a stream of null inputs.
        Usefull in generative mode using feedback
        only.

        Parameters
        ----------
        n_steps : int, optional
            Max number of steps to yield, by default 1

        Yields
        -------
        numpy.ndarray
            A null input vector for the reservoir.
        """
        for t in range(n_steps):
            yield self._zero_input()

    def _zero_readout_signal(self, *args) -> np.ndarray:
        """Return a null feedback, when feedback is not
        needed.

        Returns
        -------
        np.ndarray
            A null feedback vector for the reservoir.
        """
        if self.shape[2] is None:
            return np.zeros((1, 1))
        return np.zeros((1, self.shape[2]))

    def _zero_state(self) -> np.ndarray:
        """A null state vector.
        """
        return np.zeros((1, self.shape[1]))

    def _zero_input(self) -> np.ndarray:
        """A null input vector
        """
        return np.zeros((1, self.shape[0]))

    def _prepare_states_from_inputs(self,
                                    inputs: np.ndarray,
                                    n_steps: int = None
                                    ) -> np.ndarray:
        """Allocate an array of zeroes to fetch the
        computed states.
        """
        if n_steps:
            return np.zeros((n_steps, self.shape[1]))
        return np.zeros((inputs.shape[0], self.shape[1]))

    def reset_state(self, from_state: np.ndarray = None):
        """Reset the last state saved to zero or to
        another state value `from_state`.

        Parameters
        ----------
        from_state : np.ndarray, optional
            New state value for stateful
            computations, by default None.
        """
        if from_state is None:
            self._state = self._zero_state()
        else:
            self._state = _check_vector(from_state)

    def reset_last_input(self, from_input: np.ndarray = None):
        """Reset the last input saved to zero or to
        another input value `from_input`.

        Useful when using a readout working on both the
        internal states of the reservoir and the raw
        inputs.

        Parameters
        ----------
        from_input : np.ndarray, optional
            New input value to use for stateful
            computations, by default None
        """
        if from_input is None:
            self._last_input = self._zero_input()
        else:
            self._last_input = _check_vector(from_input)
