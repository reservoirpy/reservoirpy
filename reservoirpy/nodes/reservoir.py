from typing import Callable, Union, Optional
from functools import partial

import numpy as np
from numpy.random import Generator, default_rng

from ..utils.types import Weights
from ..node import Node
from ..mat_gen import generate_internal_weights, generate_input_weights
from ..utils.validation import is_array
from ..activationsfunc import identity, tanh


def forward(reservoir: "Reservoir", u: np.ndarray) -> np.ndarray:

    lr = reservoir.lr
    activation = reservoir.activation
    W = reservoir.W
    Win = reservoir.Win
    noise_in = reservoir.noise_in
    noise_rc = reservoir.noise_rc
    rg = reservoir.random_generator

    _x = W @ reservoir.state().T
    _x += Win @ _noisify(u.reshape(-1, 1), noise_in, rg)

    if reservoir.has_feedback:
        Wfb = reservoir.Wfb
        noise_out = reservoir.noise_out
        fb_activation = reservoir.fb_activation

        _y = fb_activation(reservoir.feedback().reshape(-1, 1))
        _x += Wfb @ _noisify(_y, noise_out, rg)

    x1 = (1 - lr) * reservoir.state().T + lr * _noisify(activation(_x), noise_rc, rg)

    return x1.T


def initialize(reservoir,
               x=None,
               sr=None,
               input_scaling=None,
               input_connectivity=None,
               rc_connectivity=None,
               W_init=None,
               Win_init=None,
               input_bias=None,
               seed=None):

    if x is not None:
        reservoir.set_input_dim(x.shape[1])

        if is_array(W_init):
            W = W_init
            if W.shape[0] != W.shape[1]:
                raise ValueError("Dimension mismatch inside W: "
                                 f"W is {W.shape} but should be "
                                 f"a square matrix.")

            if W.shape[0] != reservoir.output_dim:
                reservoir.set_output_dim(W.shape[0])

        elif callable(W_init):
            W = W_init(N=reservoir.output_dim, sr=sr,
                       proba=rc_connectivity, seed=seed)
        else:
            raise ValueError(f"Data type {type(W_init)} not "
                             f"understood for matrix initializer "
                             f"'W_init' in {reservoir.name}. W "
                             f"should be an array or a callable "
                             f"returning an array.")

        reservoir.set_param("units", W.shape[0])
        reservoir.set_param("W", W)

        out_dim = reservoir.output_dim

        if is_array(Win_init):
            Win = Win_init
            bias_dim = 1 if input_bias else 0
            bias_msg = "+ 1 (bias)" if input_bias else ""
            if Win.shape[0] != x.shape[1] + bias_dim:
                raise ValueError("Dimension mismatch between Win and input "
                                 f"vector in {reservoir.name}: Win is {Win.shape} "
                                 f"and input is {x.shape} ({x.shape[1]} {bias_msg} "
                                 f"!= {Win.shape[0] - bias_dim} {bias_msg})")

            if Win.shape[1] != out_dim:
                raise ValueError(f"Dimension mismatch between Win and W in "
                                 f"{reservoir.name}: "
                                 f"Win is {Win.shape} and W is "
                                 f"{(out_dim, out_dim)}"
                                 f" ({Win.shape[1]} != {out_dim})")

        elif callable(Win_init):
            Win = Win_init(N=reservoir.output_dim, dim_input=x.shape[1],
                           input_bias=input_bias, input_scaling=input_scaling,
                           proba=input_connectivity, seed=seed)
        else:
            raise ValueError(f"Data type {type(Win_init)} not "
                             f"understood for matrix initializer "
                             f"'Win_init' in {reservoir.name}. Win "
                             f"should be an array or a callabl returning "
                             f"an array.")

        reservoir.set_param("Win", Win)


def initialize_feedback(reservoir,
                        Wfb_init=None,
                        fb_scaling=None,
                        fb_connectivity=None,
                        fb_dim: int = None,
                        seed=None):

    if reservoir.has_feedback:
        feedback = reservoir.feedback()
        fb_dim = feedback.shape[1]
        reservoir.set_feedback_dim(fb_dim)
    elif fb_dim is not None:
        reservoir.set_feedback_dim(fb_dim)
    else:
        reservoir.set_feedback_dim(0)

    if fb_dim is not None:
        if is_array(Wfb_init):
            Wfb = Wfb_init
            if not fb_dim == Wfb.shape[0]:
                raise ValueError("Dimension mismatch between Wfb and feedback "
                                 f"vector in {reservoir.name}: Wfb is {Wfb.shape} "
                                 f"and feedback is {(1, fb_dim)} "
                                 f"({fb_dim} != {Wfb.shape[0]})")

            if not Wfb.shape[1] == reservoir.output_dim:
                raise ValueError(f"Dimension mismatch between Wfb and W in "
                                 f"{reservoir.name}: Wfb is {Wfb.shape} and W is "
                                 f"{reservoir.W.shape} ({Wfb.shape[1]} "
                                 f"!= {reservoir.output_dim})")

        elif callable(Wfb_init):
            Wfb = Wfb_init(N=reservoir.output_dim, dim_input=fb_dim,
                           input_bias=False, input_scaling=fb_scaling,
                           proba=fb_connectivity, seed=seed)
        else:
            raise ValueError(f"Data type {type(Wfb_init)} not understood "
                             f"for matrix initializer 'Wfb_init' in "
                             f"{reservoir.name}. Wfb should be an array "
                             f"or a callable returning an array.")

        reservoir.set_param("Wfb", Wfb)


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


class Reservoir(Node):

    def __init__(self,
                 units: int = None,
                 lr: float = 1.0,
                 sr: Optional[float] = None,
                 input_bias: bool = True,
                 noise_rc: float = 0.0,
                 noise_in: float = 0.0,
                 noise_fb: float = 0.0,
                 input_scaling: Optional[float] = 1.0,
                 fb_scaling: Optional[float] = 1.0,
                 input_connectivity: Optional[float] = 0.1,
                 rc_connectivity: Optional[float] = 0.1,
                 fb_connectivity: Optional[float] = 0.1,
                 Win: Union[Weights, Callable] = generate_input_weights,
                 W: Union[Weights, Callable] = generate_internal_weights,
                 Wfb: Union[Weights, Callable] = generate_input_weights,
                 fb_dim: int = None,
                 fb_activation: Union[str, Callable] = identity,
                 activation: Union[str, Callable] = tanh,
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
                                                            input_bias=input_bias,
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
