from typing import Callable, Union, Optional
from functools import partial

import numpy as np
from numpy.random import Generator, default_rng

from ..initializers import BimodalScaling, NormalSpectralScaling, Ones
from .._types import Weights
from ..utils.validation import check_matrix, is_array
from ..model import Node
from ..mixins import FeedbackReceiver
from ..activationsfunc import identity, tanh


def forward(reservoir: "Reservoir", u: np.ndarray) -> np.ndarray:

    state = reservoir.state().reshape(1, -1)
    lr = reservoir.lr
    activation = reservoir.activation
    W = reservoir.W
    Win = reservoir.Win
    noise_in = reservoir.noise_in
    noise_rc = reservoir.noise_rc
    rg = reservoir.random_generator

    x_ = state @ W
    x_ += noisify(u.reshape(1, -1), noise_in, rg) @ Win

    if reservoir.get_param("Wfb") is not None:
        Wfb = reservoir.Wfb
        noise_out = reservoir.noise_out
        fb_activation = reservoir.fb_activation
        if reservoir.is_fb_initialized:
            y_ = fb_activation(reservoir.feedback().reshape(1, -1))
            x_ += noisify(y_, noise_out, rg) @ Wfb

    x1 = (1 - lr) * state + lr * noisify(activation(x_), noise_rc, rg)

    return x1


def initialize(reservoir,
               x=None,
               sr=None,
               input_scaling=None,
               input_connectivity=None,
               rc_connectivity=None,
               W_init=None,
               Win_init=None,
               bias_init=None,
               seed=None):

    if x is not None:
        reservoir.set_input_dim(x.shape)

        if is_array(W_init):
            W = check_matrix(W_init)

            if not W.shape[0] == W.shape[1]:
                raise ValueError("Dimension mismatch inside W: "
                                 f"W is {W.shape} but should be "
                                 f"a square matrix.")

            if not W.shape[0] == reservoir.output_dim:
                reservoir.set_output_dim(W.shape[0])

            reservoir.set_param("W", W)
            reservoir.set_param("units", W.shape[0])

        elif callable(W_init):
            shape = (reservoir.output_dim, ) * 2
            if sr is not None and hasattr(W_init, "sr"):
                W_init.sr = sr
            if rc_connectivity is not None and hasattr(W_init, "connectivity"):
                W_init.connectivity = rc_connectivity
            if seed is not None and hasattr(W_init, "reset_seed"):
                W_init.reset_seed(seed)
            W = W_init(shape)
            reservoir.set_param("W", W)

        out_dim = reservoir.output_dim

        if is_array(Win_init):
            Win = check_matrix(Win_init)

            if not Win.shape[0] == x.shape[1]:
                raise ValueError("Dimension mismatch between Win and input "
                                 f"vector: Win is {Win.shape} and input is "
                                 f"{x.shape} ({x.shape[1]} != {Win.shape[0]})")

            if not Win.shape[1] == out_dim:
                raise ValueError("Dimension mismatch between Win and W: "
                                 f"Win is {Win.shape} and W is "
                                 f"{(out_dim, out_dim)}"
                                 f" ({Win.shape[1]} != {out_dim})")
            reservoir.set_param("Win", Win)

        elif callable(Win_init):
            shape = (x.shape[1], out_dim)
            if input_scaling is not None and hasattr(Win_init, "scaling"):
                Win_init.scaling = input_scaling
            if input_connectivity is not None and hasattr(Win_init, "connectivity"):
                Win_init.connectivity = input_connectivity
            if seed is not None and hasattr(Win_init, "reset_seed"):
                Win_init.reset_seed(seed)
            Win = Win_init(shape)
            reservoir.set_param("Win", Win)


def initialize_feedback(reservoir,
                        Wfb_init=None,
                        fb_scaling=None,
                        fb_connectivity=None,
                        fb_dim: int = None,
                        seed=None):

    if Wfb_init is not None:
        if reservoir.feedback is not None:
            feedback = reservoir.feedback()
            fb_dim = feedback.shape[1]
            reservoir.set_feedback_dim(fb_dim)
        elif fb_dim is not None:
            reservoir.set_feedback_dim((1, fb_dim))

        if fb_dim is not None:
            if is_array(Wfb_init):
                Wfb = check_matrix(Wfb_init)
                if not fb_dim == Wfb.shape[0]:
                    raise ValueError("Dimension mismatch between Wfb and feedback "
                                     f"vector: Wfb is {Wfb.shape} and feedback is "
                                     f"{(1, fb_dim)} ({fb_dim} != {Wfb.shape[0]})")

                if not Wfb.shape[1] == reservoir.output_dim:
                    raise ValueError("Dimension mismatch between Wfb and W: "
                                     f"Wfb is {Wfb.shape} and W is "
                                     f"{reservoir.W.shape} ({Wfb.shape[1]} "
                                     f"!= {reservoir.output_dim})")

                reservoir.set_param("Wfb", Wfb)

            elif callable(Wfb_init):
                shape = (fb_dim, reservoir.output_dim)
                if fb_scaling is not None and hasattr(Wfb_init, "scaling"):
                    Wfb_init.scaling = fb_scaling
                if fb_connectivity is not None and hasattr(Wfb_init, "connectivity"):
                    Wfb_init.connectivity = fb_connectivity
                if seed is not None and hasattr(Wfb_init, "reset_seed"):
                    Wfb_init.reset_seed(seed)
                Wfb = Wfb_init(shape)
                reservoir.set_param("Wfb", Wfb)


def noisify(vector: np.ndarray,
            coef: float,
            random_generator: Generator = None
            ) -> np.ndarray:
    if coef > 0.0:
        if random_generator is None:
            random_generator = default_rng()
        noise = coef * random_generator.uniform(-1, 1, vector.shape)
        return vector + noise
    return vector


class Reservoir(FeedbackReceiver, Node):

    def __init__(self,
                 units: int = None,
                 lr: float = 1.0,
                 sr: Optional[float] = None,
                 bias: Union[Weights, Callable] = Ones(),
                 noise_rc: float = 0.0,
                 noise_in: float = 0.0,
                 noise_fb: float = 0.0,
                 input_scaling: Optional[float] = None,
                 fb_scaling: Optional[float] = None,
                 input_connectivity: Optional[float] = None,
                 rc_connectivity: Optional[float] = None,
                 fb_connectivity: Optional[float] = None,
                 Win: Union[Weights, Callable] = BimodalScaling(connectivity=0.1, scaling=1.0),
                 W: Union[Weights, Callable] = NormalSpectralScaling(connectivity=0.1),
                 Wfb: Union[Weights, Callable] = BimodalScaling(connectivity=0.1, scaling=1.0),
                 fb_dim: int = None,
                 fb_activation: Union[str, Callable] = tanh,
                 activation: Union[str, Callable] = identity,
                 name=None,
                 seed=None):

        if units is None and not is_array(W):
            raise ValueError("'units' parameter must not be None if 'W' parameter is not"
                             "a matrix.")

        super(Reservoir, self).__init__(fb_initializer=partial(initialize_feedback,
                                                               Wfb_init=Wfb,
                                                               fb_scaling=fb_scaling,
                                                               fb_connectivity=fb_connectivity,
                                                               fb_dim=fb_dim,
                                                               seed=seed),
                                        params={"W": None, "Win": None, "Wfb": None, "bias": None},
                                        hypers={"lr": lr,
                                                "sr": sr,
                                                "input_scaling": input_scaling,
                                                "fb_scaling": fb_scaling,
                                                "rc_connectivity": rc_connectivity,
                                                "input_connectivity": input_connectivity,
                                                "fb_connectivity": fb_connectivity,
                                                "noise_in": noise_in,
                                                "noise_rc": noise_rc,
                                                "noise_out": noise_fb,
                                                "activation": activation,
                                                "fb_activation": fb_activation,
                                                "units": units},
                                        forward=forward,
                                        initializer=partial(initialize,
                                                            sr=sr,
                                                            input_scaling=input_scaling,
                                                            input_connectivity=input_connectivity,
                                                            rc_connectivity=rc_connectivity,
                                                            W_init=W,
                                                            Win_init=Win,
                                                            bias_init=bias,
                                                            seed=seed),
                                        output_dim=units,
                                        name=name)

        self._seed = seed
        self._rg = np.random.default_rng(seed)

    @property
    def random_generator(self):
        return self._rg

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._rg = np.random.default_rng(value)
        self._seed = value
