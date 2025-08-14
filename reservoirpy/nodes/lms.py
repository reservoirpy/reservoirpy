from typing import Callable, Optional, Sequence, Union

import numpy as np

from reservoirpy.utils.data_validation import check_node_input

from ..mat_gen import zeros
from ..node import OnlineNode
from ..type import NodeInput, State, Timeseries, Timestep, Weights, is_array


class LMS(OnlineNode):
    """Single layer of neurons learning connections using Least Mean Squares
    algorithm.

    The learning rules is well described in [1]_.

    Parameters
    ----------
    learning_rate : float or Python generator or iterable, default to 1e-6
        Learning rate. If an iterable or a generator is provided, the learning rate can
        be changed at each timestep of training. A new learning rate will be drawn from
        the iterable or generator at each timestep.
    Wout : callable or array-like of shape (units, targets), default to :py:func:`~reservoirpy.mat_gen.zeros`
        Output weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.zeros`
        Bias weights vector or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    fit_bias : bool, default to True
        If True, then a bias parameter will be learned along with output weights.
    input_dim : int, optional
        Number of input dimensions in the readout, can be inferred at first call.
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    name : str, optional
        Node name.

    Examples
    --------
    >>> x = np.random.normal(size=(100, 3))
    >>> noise = np.random.normal(scale=0.01, size=(100, 1))
    >>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.

    >>> from reservoirpy.nodes import LMS
    >>> lms_node = LMS(alpha=1e-1)

    >>> lms_node.train(x[:50], y[:50])
    >>> print(lms_node.Wout.T, lms_node.bias)
    [[ 9.156 -0.967   6.411]] [[11.564]]
    >>> lms_node.train(x[50:], y[50:])
    >>> print(lms_node.Wout.T, lms_node.bias)
    [[ 9.998 -0.202  7.001]] [[12.005]]

    References
    ----------

    .. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
           Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
           https://doi.org/10.1016/j.neuron.2009.07.018
    """

    #: Learned output weights (:math:`\mathbf{W}_{out}`).
    Wout: Weights
    #: Learned bias (:math:`\mathbf{b}`).
    bias: Weights
    #: Learning rate (:math:`\alpha`) (:math:`1\cdot 10^{-6}` by default).
    learning_rate: float
    #: If True, learn a bias term (True by default).
    fit_bias: bool

    def __init__(
        self,
        learning_rate: float = 1e-6,
        Wout: Union[Weights, Callable] = zeros,
        bias: Union[Weights, Callable] = zeros,
        fit_bias: bool = True,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.learning_rate = learning_rate
        self.Wout = Wout
        self.bias = bias
        self.fit_bias = fit_bias
        self.initialized = False
        self.state = {}
        self.name = name

        # set input_dim/output_dim (if possible)
        self.input_dim = input_dim
        self.output_dim = output_dim
        if is_array(Wout):
            if input_dim is not None and Wout.shape[0] != input_dim:
                raise ValueError(
                    f"Both 'input_dim' and 'Wout' are set but their dimensions doesn't "
                    f"match: {input_dim} != {Wout.shape[0]}."
                )
            self.input_dim = Wout.shape[0]
            if output_dim is not None and Wout.shape[1] != output_dim:
                raise ValueError(
                    f"Both 'output_dim' and 'Wout' are set but their dimensions doesn't "
                    f"match: {output_dim} != {Wout.shape[1]}."
                )
            self.output_dim = Wout.shape[1]
        if is_array(bias):
            if output_dim is not None and bias.shape[0] != output_dim:
                raise ValueError(
                    f"Both 'output_dim' and 'Wout' are set but their dimensions doesn't "
                    f"match: {output_dim} != {bias.shape[0]}."
                )
            self.output_dim = bias.shape[0]

    def _run(self, state: tuple, x: Timeseries) -> tuple[State, Timeseries]:
        out = x @ self.Wout + self.bias
        return {"out": out[-1]}, out  # (len, in) @ (in, out) + (out,)

    def _step(self, state: tuple, x: Timestep) -> State:
        return {"out": x @ self.Wout + self.bias}  # (in, ) @ (in, out) + (out,)

    def initialize(
        self,
        x: Union[NodeInput, Timestep],
        y: Optional[Union[NodeInput, Timestep]] = None,
    ):
        self._set_input_dim(x)
        self._set_output_dim(y)

        # initialize matrices
        if isinstance(self.Wout, Callable):
            self.Wout = self.Wout(self.input_dim, self.output_dim)
        if isinstance(self.bias, Callable):
            self.bias = self.bias(self.output_dim)

        self.initialized = True

    def _learning_step(self, x: Timestep, y: Timestep) -> Timestep:
        alpha: float = self.learning_rate
        Wout: Weights = self.Wout
        bias: Weights = self.bias

        prediction = x @ Wout + bias  # (out,) = (in,) @ (in, out) + (out,)
        error = prediction - y  # (out,)
        dWout = -alpha * np.outer(x, error)  # (in, out)
        Wout_next = Wout + dWout  # (in, out)
        if self.fit_bias:
            dbias = -alpha * error
            bias_next = bias + dbias
        else:
            bias_next = bias
        y_pred = x @ Wout_next + bias

        self.Wout = Wout_next
        self.bias = bias_next
        return y_pred

    def partial_fit(self, x: Timeseries, y: Timeseries):
        check_node_input(x, expected_dim=self.input_dim)
        if y is not None:
            check_node_input(y)

        if not self.initialized:
            self.initialize(x, y)

        n_timesteps = x.shape[-2]
        out_dim = y.shape[-1]
        y_pred = np.empty((n_timesteps, out_dim))
        for i, (x_, y_) in enumerate(zip(x, y)):
            y_pred_ = self._learning_step(x_, y_)
            y_pred[i] = y_pred_

        self.state = {"out": y_pred_}
        return y_pred
