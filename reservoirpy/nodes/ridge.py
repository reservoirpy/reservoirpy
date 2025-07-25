from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg

from ..node import ParallelNode
from ..type import NodeInput, State, Timeseries, Timestep, Weights


class Ridge(ParallelNode):
    ridge: float
    fit_bias: bool
    Wout: Weights
    bias: Weights
    name: Optional[str]

    def __init__(
        self,
        ridge: float = 0.0,
        fit_bias: bool = True,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.ridge = ridge
        self.fit_bias = fit_bias
        self.Wout = Wout
        self.bias = bias
        self.name = name

        # TODO: dimension checks

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initialized = False
        self.state = {}

    def initialize(
        self,
        x: Union[NodeInput, Timestep],
        y: Optional[Union[NodeInput, Timestep]] = None,
    ):
        # set input_dim
        if self.input_dim is None:
            if self.Wout is not None:
                self.input_dim = self.Wout.shape[0]
            self.input_dim = (
                x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
            )
        # set output_dim
        if self.output_dim is None:
            if y is not None:
                self.output_dim = (
                    y.shape[-1] if not isinstance(y, Sequence) else y[0].shape[-1]
                )
            elif self.Wout is not None:
                self.output_dim = self.Wout.shape[1]
            elif self.bias is not None:
                self.output_dim = self.bias.shape[0]
            else:
                raise ValueError("Could not infer output_dim at initialization.")

        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        return {"out": x @ self.Wout + self.bias}

    def _run(self, state: State, x: Timeseries) -> Tuple[State, Timeseries]:
        out = x @ self.Wout + self.bias
        return {"out": out[-1]}, out

    def worker(self, x: Timeseries, y: Timeseries):
        x_sum = np.sum(x, axis=0)
        y_sum = np.sum(y, axis=0)
        sample_size = x.shape[0]
        XXT = x.T @ x
        YXT = x.T @ y
        return XXT, YXT, x_sum, y_sum, sample_size

    def master(self, generator: Iterable):
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

        Wout = linalg.solve(XXT + ridge_In, YXT, assume_a="sym")

        self.Wout = Wout
        if self.fit_bias:
            self.bias = Y_means - X_means @ Wout
        else:
            self.bias = np.zeros((self.output_dim,))
