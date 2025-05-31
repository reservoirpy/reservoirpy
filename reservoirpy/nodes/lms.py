from typing import Optional, Tuple

import numpy as np

from ..mat_gen import zeros
from ..node import OnlineNode
from ..type import NodeInput, Timeseries, Timestep


class LMS(OnlineNode):
    def __init__(
        self,
        learning_rate=1e-6,
        Wout=zeros,
        bias=zeros,
        input_bias=True,
        output_dim=None,
    ):
        self.learning_rate = learning_rate
        self.Wout = Wout
        self.bias = bias
        self.input_bias = input_bias
        self.output_dim = output_dim

    def _run(self, state: tuple, x: Timeseries) -> Tuple[tuple, Timeseries]:
        return (), x @ self.Wout  # (len, in) @ (in, out)

    def _step(self, state: tuple, x: Timestep) -> Tuple[tuple, Timestep]:
        return (), x @ self.Wout  # (in, ) @ (in, out)

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

    def learning_step(self, Wout, x: Timestep, y: Timestep):
        alpha: float = self.learning_rate

        y_pred_before: np.ndarray = x @ Wout  # (out, ) = (in, ) @ (in, out)
        error: np.ndarray = y - y_pred_before  # (out, )
        dWout: np.ndarray = -alpha * np.linalg.outer(x, error)  # (in, out)
        Wout_next: np.ndarray = Wout + dWout  # (in, out)
        y_pred_after: "(out, )" = x @ Wout_next

        return (Wout_next,), y_pred_after

    def partial_fit(self, x: Timeseries, y: Timeseries):
        if not self.initialized:
            self.initialize(x, y)

        Wout = self.Wout
        n_timesteps = x.shape[-2]
        out_dim = y.shape[-1]
        y_pred = np.empty((n_timesteps, out_dim))
        for i, (x_, y_) in enumerate(zip(x, y)):
            (Wout,), y_pred_ = self.learning_step(Wout, x_, y_)
            y_pred[i] = y_pred_

        self.Wout = Wout
        return y_pred
