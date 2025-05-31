from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np
from scipy import linalg

from ..node import ParallelNode
from ..type import NodeInput, Timeseries, Timestep, Weights


class Ridge(ParallelNode):
    # TODO: bias
    # TODO: input_bias
    def __init__(
        self,
        ridge: float = 0.0,
        input_bias: bool = True,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        self.ridge = ridge
        self.input_bias = input_bias
        self.Wout = Wout
        self.bias = bias

        # TODO: dimension checks

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initialized = False
        self.state = ()

    def initialize(
        self, x: Optional[NodeInput | Timestep], y: Optional[NodeInput | Timestep]
    ):
        # set input_dim
        if self.input_dim is None:
            self.input_dim = x.shape[-1] if not isinstance(x, list) else x[0].shape[-1]
        # set output_dim
        if self.output_dim is None:
            self.output_dim = y.shape[-1] if not isinstance(y, list) else y[0].shape[-1]

        self.initialized = True

    def _step(self, state: tuple, x: Timestep) -> Tuple[tuple, Timestep]:
        return (), x @ self.Wout

    def _run(self, state: tuple, x: Timeseries) -> Tuple[tuple, Timeseries]:
        return (), x @ self.Wout

    def worker(self, x: Timeseries, y: Timeseries):
        XXT = x.T @ x
        YXT = x.T @ y
        return XXT, YXT

    def master(self, generator: Generator):
        XXT = np.zeros((self.input_dim, self.input_dim))
        YXT = np.zeros((self.input_dim, self.output_dim))
        ridge_In = self.ridge * np.eye(self.input_dim)

        for (xxt, yxt) in generator:
            XXT += xxt
            YXT += yxt

        Wout = Ridge._solve_ridge(XXT=XXT, YXT=YXT, ridge=ridge_In)

        self.Wout = Wout

    def _solve_ridge(XXT, YXT, ridge):
        """Solve Tikhonov regression."""
        return linalg.solve(XXT + ridge, YXT, assume_a="sym")
