from typing import Callable, Optional, Tuple, Union

import numpy as np

from ..mat_gen import zeros
from ..node import OnlineNode
from ..type import NodeInput, Timeseries, Timestep, Weights


class LMS(OnlineNode):
    def __init__(
        self,
        learning_rate: float = 1e-6,
        Wout: Union[Weights, Callable] = zeros,
        bias: Union[Weights, Callable] = zeros,
        fit_bias: bool = True,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        self.learning_rate = learning_rate
        self.Wout = Wout
        self.bias = bias
        self.fit_bias = fit_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initialized = False

    def _run(self, state: tuple, x: Timeseries) -> Tuple[tuple, Timeseries]:
        return (), x @ self.Wout + self.bias  # (len, in) @ (in, out) + (out,)

    def _step(self, state: tuple, x: Timestep) -> Tuple[tuple, Timestep]:
        return (), x @ self.Wout + self.bias  # (in, ) @ (in, out) + (out,)

    def initialize(
        self,
        x: Optional[NodeInput | Timestep],
        y: Optional[NodeInput | Timestep] = None,
    ):
        # set input_dim
        if self.input_dim is None:
            if isinstance(self.Wout, Weights):
                self.input_dim = self.Wout.shape[0]
            self.input_dim = x.shape[-1] if not isinstance(x, list) else x[0].shape[-1]
        # set output_dim
        if self.output_dim is None:
            if isinstance(self.Wout, Weights):
                self.output_dim = self.Wout.shape[1]
            if isinstance(self.Wout, Weights):
                self.output_dim = self.bias.shape[0]
            if y is not None:
                self.output_dim = (
                    y.shape[-1] if not isinstance(y, list) else y[0].shape[-1]
                )

        # initialize matrices
        if isinstance(self.Wout, Callable):
            self.Wout = self.Wout(self.input_dim, self.output_dim)
        if isinstance(self.bias, Callable):
            self.bias = self.bias(self.output_dim)

        self.initialized = True

    def _learning_step(self, Wout: Weights, bias: Weights, x: Timestep, y: Timestep):
        alpha: float = self.learning_rate

        y_pred_before = x @ Wout + bias  # (out,) = (in,) @ (in, out) + (out,)
        error = y - y_pred_before  # (out,)
        dWout = -alpha * np.outer(x, error)  # (in, out)
        Wout_next = Wout + dWout  # (in, out)
        if self.fit_bias:
            dbias = -alpha * error
            bias_next = bias + dbias
        else:
            bias_next = bias
        y_pred_after = x @ Wout_next

        return (Wout_next, bias_next), y_pred_after

    def partial_fit(self, x: Timeseries, y: Timeseries):
        if not self.initialized:
            self.initialize(x, y)

        Wout = self.Wout
        bias = self.bias
        n_timesteps = x.shape[-2]
        out_dim = y.shape[-1]
        y_pred = np.empty((n_timesteps, out_dim))
        for i, (x_, y_) in enumerate(zip(x, y)):
            (Wout, bias), y_pred_ = self._learning_step(Wout, bias, x_, y_)
            y_pred[i] = y_pred_

        self.Wout = Wout
        self.bias = bias
        return y_pred
