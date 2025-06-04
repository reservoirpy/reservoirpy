from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np
from scipy import linalg

from ..node import ParallelNode
from ..type import NodeInput, Timeseries, Timestep, Weights


class Ridge(ParallelNode):
    def __init__(
        self,
        ridge: float = 0.0,
        fit_bias: bool = True,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        self.ridge = ridge
        self.fit_bias = fit_bias
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
        return (), x @ self.Wout + self.bias

    def _run(self, state: tuple, x: Timeseries) -> Tuple[tuple, Timeseries]:
        return (), x @ self.Wout + self.bias

    def worker(self, x: Timeseries, y: Timeseries):
        x_sum = np.sum(x, axis=0)
        y_sum = np.sum(y, axis=0)
        sample_size = x.shape[-1]
        XXT = x.T @ x
        YXT = x.T @ y
        return XXT, YXT, x_sum, y_sum, sample_size

    def master(self, generator: Generator):
        XXT = np.zeros((self.input_dim, self.input_dim))
        YXT = np.zeros((self.input_dim, self.output_dim))
        X_sum = 0.0
        Y_sum = 0.0
        total_samples = 0
        ridge_In = self.ridge * np.eye(self.input_dim)

        for (xxt, yxt, x_sum, y_sum, sample_size) in generator:
            XXT += xxt
            YXT += yxt
            X_sum += x_sum
            Y_sum += y_sum
            total_samples = sample_size

        if self.fit_bias:
            X_means = X_sum / total_samples
            Y_means = Y_sum / total_samples
            XXT -= total_samples * np.outer(X_means, X_means)
            YXT -= total_samples * np.outer(X_means, Y_means)

        Wout = Ridge._solve_ridge(XXT=XXT, YXT=YXT, ridge=ridge_In)

        self.Wout = Wout
        if self.fit_bias:
            self.bias = Y_means - X_means @ Wout
        else:
            self.bias = np.zeros((self.output_dim,))

    def _solve_ridge(XXT, YXT, ridge):
        """Solve Tikhonov regression."""
        return linalg.solve(XXT + ridge, YXT, assume_a="sym")
