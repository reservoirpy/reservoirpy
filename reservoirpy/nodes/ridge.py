from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg

from ..node import ParallelNode
from ..type import NodeInput, State, Timeseries, Timestep, Weights, is_array


class Ridge(ParallelNode):
    """A single layer of neurons learning with Tikhonov linear regression.

    Output weights of the layer are computed following:

    .. math::

        \\hat{\\mathbf{W}}_{out} = \\mathbf{YX}^\\top ~ (\\mathbf{XX}^\\top +
        \\lambda\\mathbf{Id})^{-1}

    Outputs :math:`\\mathbf{y}` of the node are the result of:

    .. math::

        \\mathbf{y} = \\mathbf{W}_{out}^\\top \\mathbf{x} + \\mathbf{b}

    where:
        - :math:`\\mathbf{X}` is the accumulation of all inputs during training;
        - :math:`\\mathbf{Y}` is the accumulation of all targets during training;
        - :math:`\\mathbf{b}` is the first row of :math:`\\hat{\\mathbf{W}}_{out}`;
        - :math:`\\mathbf{W}_{out}` is the rest of :math:`\\hat{\\mathbf{W}}_{out}`.

    If ``fit_bias`` is True, then :math:`\\mathbf{b}` is non-zero, and a constant
    term is added to :math:`\\mathbf{X}` to compute it.


    Parameters
    ----------
    ridge: float, default to 0.0
        L2 regularization parameter.
    fit_bias : bool, default to True
        If True, then a bias parameter will be learned along with output weights.
    Wout : callable or array-like of shape (input_dim, units), optional
        Output weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    bias : callable or array-like of shape (units,), optional
        Bias weights vector or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    name : str, optional
        Node name.

    Example
    -------

    >>> x = np.random.normal(size=(100, 3))
    >>> noise = np.random.normal(scale=0.1, size=(100, 1))
    >>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.
    >>>
    >>> from reservoirpy.nodes import Ridge
    >>> ridge_regressor = Ridge(ridge=0.001)
    >>>
    >>> ridge_regressor.fit(x, y)
    >>> ridge_regressor.Wout, ridge_regressor.bias
    array([[ 9.992, -0.205,  6.989]]).T, array([[12.011]])
    """

    #: Regularization parameter (:math:`\\lambda`) (0.0 by default).
    ridge: float
    #: If True, learn a bias term (True by default).
    fit_bias: bool
    #: Learned output weights (:math:`\\mathbf{W}_{out}`).
    Wout: Weights
    #: Learned bias (:math:`\\mathbf{b}`).
    bias: Weights

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
