from typing import Callable, Optional, Tuple, Union

import numpy as np

from ..mat_gen import zeros
from ..node import OnlineNode
from ..type import NodeInput, State, Timeseries, Timestep, Weights


class RLS(OnlineNode):
    def __init__(
        self,
        alpha: float = 1e-6,
        Wout: Union[Weights, Callable] = zeros,
        bias: Union[Weights, Callable] = zeros,
        fit_bias: bool = True,
        forgetting: float = 1.0,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.alpha = alpha
        self.Wout = Wout
        self.bias = bias
        self.fit_bias = fit_bias
        self.forgetting = forgetting
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        self.initialized = False
        self.state = {}

    def _run(self, state: State, x: Timeseries) -> Tuple[State, Timeseries]:
        out = x @ self.Wout + self.bias  # (len, in) @ (in, out) + (out,)
        return {"out": out[-1]}, out

    def _step(self, state: State, x: Timestep) -> State:
        return {"out": x @ self.Wout + self.bias}  # (in, ) @ (in, out) + (out,)

    def initialize(
        self,
        x: Optional[Union[NodeInput, Timestep]],
        y: Optional[Union[NodeInput, Timestep]] = None,
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
            if isinstance(self.bias, Weights):
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
        self.P = np.eye(self.input_dim) / self.alpha
        self.S = 0

        self.initialized = True

    def _learning_step(
        self,
        Wout: Weights,
        bias: Weights,
        P: np.ndarray,
        forgetting: float,
        S: float,
        x: Timestep,
        y: Timestep,
    ):
        """
        Wout: np.ndarray (in, out)
        bias: np.ndarray (out,)
        P: np.ndarray (in, in)
        x: np.ndarray (in,)
        y: np.ndarray (out,)
        Returns
        (Wout_next, bias_next, P_next, S_next), y_pred
        """

        Px = P @ x  # (in,)
        dP = -np.outer(Px, Px) / (forgetting + x @ Px)  # (in, in)
        P_next = (P + dP) / forgetting
        S_next = forgetting * S + 1

        prediction = x @ Wout + bias  # (out,) = (in,) @ (in, out) + (out,)
        error = prediction - y  # (out,)
        dWout = -np.outer(P_next @ x, error)  # (in, out)
        Wout_next = Wout + dWout  # (in, out)

        if self.fit_bias:
            bias_next = (S_next * bias - error) / S_next
        else:
            bias_next = bias
        y_pred = x @ Wout_next + bias

        return (Wout_next, bias_next, P_next, S_next), y_pred

    def partial_fit(self, x: Timeseries, y: Timeseries):
        if not self.initialized:
            self.initialize(x, y)

        Wout = self.Wout
        bias = self.bias
        forgetting = self.forgetting
        P = self.P
        n_timesteps = x.shape[-2]
        out_dim = y.shape[-1]
        y_pred = np.empty((n_timesteps, out_dim))
        S = self.S
        for i, (x_, y_) in enumerate(zip(x, y)):
            (Wout, bias, P, S), y_pred_ = self._learning_step(
                Wout, bias, P, forgetting, S, x_, y_
            )
            y_pred[i] = y_pred_

        self.Wout = Wout
        self.bias = bias
        self.state = {"out": y_pred_}
        self.S = S
        return y_pred
