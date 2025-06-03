from functools import partial
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from ..activationsfunc import get_function, tanh
from ..mat_gen import bernoulli, normal
from ..node import Node
from ..type import NodeInput, Timestep, Weights, is_array
from ..utils import random


class Reservoir(Node):
    initialized: bool
    input_dim: Optional[int]
    output_dim: int

    # params
    units: int
    lr: float
    sr: float
    input_scaling: Union[float, Sequence]
    input_connectivity: float
    rc_connectivity: float
    Win: Union[Weights, Callable]
    W: Union[Weights, Callable]
    bias: Union[Weights, Callable]
    activation: Callable
    rng: np.random.Generator
    # state
    state: Tuple[np.ndarray]

    def __init__(
        self,
        units: Optional[int] = None,
        lr: Union[float, np.ndarray] = 1.0,
        sr: float = 1.0,
        input_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = normal,
        bias: Union[Weights, Callable, float] = 0.0,
        activation: Union[str, Callable] = tanh,
        input_dim: Optional[int] = None,
        dtype: type = np.float64,
        seed=None,
    ):

        self.lr = lr
        self.sr = sr
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.Win = Win
        self.W = W
        self.bias = bias
        self.activation = get_function(activation)
        self.dtype = dtype
        self.rng = random.rand_generator(seed=seed)
        self.initialized = False

        # set units / output_dim
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not a matrix."
            )
        if units is not None and is_array(W) and W.shape[-1] != units:
            raise ValueError(
                f"Both 'units' and 'W' are set but their dimensions doesn't match: "
                f"{units} != {W.shape[-1]}."
            )
        self.units = units if units is not None else W.shape[-1]
        self.output_dim = self.units

        # set input_dim (if possible)
        if input_dim is not None and is_array(Win) and Win.shape[-1] != input_dim:
            raise ValueError(
                f"Both 'input_dim' and 'Win' are set but their dimensions doesn't "
                f"match: {input_dim} != {Win.shape[-1]}."
            )
        self.input_dim = Win.shape[-1] if is_array(Win) else input_dim

    def initialize(self, x: Optional[Union[NodeInput, Timestep]]):

        # set input_dim
        if self.input_dim is None:
            self.input_dim = x.shape[-1] if not isinstance(x, list) else x[0].shape[-1]

        [Win_rng, W_rng, bias_rng] = self.rng.spawn(3)

        if callable(self.Win):
            self.Win = self.Win(
                self.units,
                self.input_dim,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=Win_rng,
            )

        if callable(self.W):
            self.W = self.W(
                self.units,
                self.units,
                sr=self.sr,
                connectivity=self.rc_connectivity,
                dtype=self.dtype,
                seed=W_rng,
            )

        if callable(self.bias):
            self.bias = self.bias(
                self.units,
                connectivity=1.0,
                dtype=self.dtype,
                seed=bias_rng,
            )

        self.state = (np.zeros((self.units,)),)

        self.initialized = True

    def _step(self, state, x):
        W = self.W  # NxN
        Win = self.Win  # NxI
        bias = self.bias  # N or float
        f = self.activation
        lr = self.lr
        (state,) = state

        next_state = f(W @ state + Win @ x + bias)
        next_state = (1 - lr) * state + lr * next_state

        return (next_state,), next_state
