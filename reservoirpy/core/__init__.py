from ._ops import _next_state, _next_state_feedback

from typing import Callable, Union, Tuple, Optional

import numpy as np
from numpy.random import Generator, default_rng

from ..activationsfunc import get_function

from .._utils import _check_vector, _add_bias
from .._types import Weights, RandomSeed, is_iterable
from .. import initializers


class ReservoirCell:
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

    _Win_initializer: Callable = None
    _W_initializer: Callable = None
    _Wfb_initializer: Callable = None

    W: Weights
    Win: Weights
    Wfb: Weights
    lr: float
    input_bias: bool
    noise: Union[Tuple[float, float], Tuple[float, float, float]]
    shape: Union[Tuple[int, int], Tuple[int, int, int]]
    connectivity:  Union[Tuple[float, float], Tuple[float, float, float]]

    def __init__(self,
                 lr: float = 1.0,
                 shape: Union[Tuple[int, int],
                              Tuple[int, int, int]] = None,
                 activation: Callable = np.tanh,
                 noise: Tuple[float, float, float] = (0., 0., 0.),
                 connectivity: Union[Tuple[float, float],
                                     Tuple[float, float, float]] = (0.1, 0.1, 0.1),
                 Win_initializer="bimodal",
                 W_initializer="norm",
                 Wfb_initializer="bimodal",
                 fb_activation: Union[str,
                                      Callable] = "id",
                 input_bias: bool = True,
                 Win: Optional[Weights] = None,
                 W: Optional[Weights] = None,
                 Wfb: Optional[Weights] = None,
                 seed: RandomSeed = None,
                 **kwargs
                 ):

        if not is_iterable(noise):
            raise ValueError("'noise' should be a tuple of floats "
                             f"and not {type(noise)}.")

        if not is_iterable(connectivity):
            raise ValueError("'connectivity should be a tuple "
                             f"of float and not {type(connectivity)}.")

        if shape is not None:
            if len(shape) != len(connectivity):
                raise ValueError(f"Reservoir has {len(shape)} components {shape}"
                                 f"defined in 'shape' but {len(connectivity)} "
                                 f"different connectivity parameters {connectivity}.")

            if len(shape) != len(noise):
                raise ValueError(f"Reservoir has {len(shape)} components {shape}"
                                 f"defined in 'shape' but {len(noise)} "
                                 f"different noise coefficients {noise}.")
        else:
            if W is None and Win is None:
                raise ValueError("Can not build a reservoir without "
                                 "specifying its shape or providing "
                                 "weight matrices.")

            self.shape = shape

        inp_kwargs, res_kwargs, fb_kwargs = initializers._filter_reservoir_kwargs(self, **kwargs)

        if Win is None:
            self._Win_initializer = initializers.get(Win_initializer, connectivity[0], seed=seed, **inp_kwargs)

        if W is None:
            self._W_initializer = initializers.get(W_initializer, connectivity[1], seed=seed, **inp_kwargs)

        if len(shape) == 3 and Wfb is None:
            self._Wfb_initializer = initializers.get(Wfb_initializer, connectivity[2], seed=seed, **inp_kwargs)

        self.lr = lr
        self.shape = shape
        self.activation = activation
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

        self.noise = noise

        self.shape = tuple()
        self._check_matrices()

        self.reset_state()
        self.reset_last_input()

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self,
             inputs: np.ndarray = None,
             states: np.ndarray = None,
             feedback: np.ndarray = None,
             noise_generator: Generator = None) -> np.ndarray:
        """Make the data flow inside the reservoir, and output
        the corresponding states produced.

        Parameters
        ----------
        inputs : numpy.ndarray, optional
            A sequence of input data to feed the
            reservoir with, by default None.
        states : numpy.ndarray, optional

        readout : Callable, optional
            A :py:class:`Readout` instance, when some
            feedback is needed, by default None.

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
            inputs = self._zero_input()

        if self.input_bias is not None:
            inputs = _add_bias(inputs, self.input_bias)

        # second connections: from reservoir to itself
        if states is None:
            states = self._zero_state()

        if feedback is None:
            return _next_state(self.lr, self.Win, self.W, self.activation,
                               self.noise_in, self.noise_rc, states, inputs)
        else:
            return _next_state_feedback(self.lr, self.Win, self.W, self.Wfb, self.activation,
                                        self.noise_in, self.noise_rc, self.noise_out,
                                        states, inputs, feedback)

    def _zero_state(self) -> np.ndarray:
        """A null state vector.
        """
        return np.zeros((1, self.shape[1]))

    def _zero_input(self) -> np.ndarray:
        """A null input vector
        """
        return np.zeros((1, self.shape[0]))



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
