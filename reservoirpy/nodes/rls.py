from typing import Callable, Optional, Tuple, Union

import numpy as np

from reservoirpy.utils.data_validation import check_node_input

from ..mat_gen import zeros
from ..node import OnlineNode
from ..type import NodeInput, State, Timeseries, Timestep, Weights, is_array


class RLS(OnlineNode):
    """Single layer of neurons learning connections using Recursive Least Squares
    algorithm.

    The learning rules is well described in [1]_.
    The forgetting factor version of the RLS algorithm used here is described in [2]_.

    Parameters
    ----------
    alpha : float or Python generator or iterable, default to 1e-6
        Diagonal value of matrix P.
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
    forgetting : float, default to 1.0
        The forgetting factor controls the weight given to past observations in the RLS update.
        A value less than 1.0 gives more weight to recent observations.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    name : str, optional
        Node name.

    References
    ----------

    .. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
           Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
           https://doi.org/10.1016/j.neuron.2009.07.018

    .. [2] Waegeman, T., Wyffels, F., & Schrauwen, B. (2012). Feedback Control by Online
           Learning an Inverse Model. IEEE Transactions on Neural Networks and Learning
           Systems, 23(10), 1637–1648. https://doi.org/10.1109/TNNLS.2012.2208655

    Examples
    --------
    >>> x = np.random.normal(size=(100, 3))
    >>> noise = np.random.normal(scale=0.1, size=(100, 1))
    >>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.

    >>> from reservoirpy.nodes import RLS
    >>> rls_node = RLS(alpha=1e-1)

    >>> _ = rls_node.partial_fit(x[:5], y[:5])
    >>> print(rls_node.Wout.T, rls_node.bias)
    [[ 9.90731641 -0.06884784  6.87944632]] [[12.07802068]]
    >>> _ = rls_node.partial_fit(x[5:], y[5:])
    >>> print(rls_node.Wout.T, rls_node.bias)
    [[ 9.99223366 -0.20499636  6.98924066]] [[12.01128622]]
    """

    #: Learned output weights (:math:`\\mathbf{W}_{out}`).
    Wout: Weights
    #: Learned bias (:math:`\\mathbf{b}`).
    bias: Weights
    #: Matrix :math:`\\mathbf{P}` of RLS rule.
    P: Weights
    #: Diagonal value of matrix P (:math:`\\alpha`) (:math:`1\\cdot 10^{-6}` by default).
    alpha: float
    #: If True, learn a bias term (True by default).
    fit_bias: bool
    #: Forgetting factor (:math:`\\lambda`) (:math:`1` by default).
    forgetting: float

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
        self.name = name

        self.initialized = False
        self.state = {}

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
        self._set_input_dim(x)
        self._set_output_dim(y)

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
        Wout: Weights = self.Wout
        bias: Weights = self.bias
        P: np.ndarray = self.P
        forgetting: float = self.forgetting
        S: float = self.S

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

        self.Wout = Wout_next
        self.bias = bias_next
        self.P = P_next
        self.S = S_next
        return y_pred

    def partial_fit(self, x: Timeseries, y: Timeseries):
        check_node_input(x, expected_dim=self.input_dim)
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
