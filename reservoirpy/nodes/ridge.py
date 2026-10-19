# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Callable, Iterable, Literal, Optional, Union

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import lsqr

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

    The ``solver`` sets how the regression is solved:

    - ``"normal"`` (default): the normal equations, with :math:`\\mathbf{XX}^\\top` and
      :math:`\\mathbf{YX}^\\top` accumulated over the timeseries on the raw inputs,
      centered afterwards (if ``fit_bias``) by subtracting the outer products of the
      means, and solved with a symmetric (LDL) decomposition. Fast and memory-light,
      but this centering loses precision when the inputs have a large mean compared to
      their spread: with nearly dependent inputs and a small ``ridge``, the learned
      weights can then be inaccurate.
    - ``"cholesky"``: the normal equations formed on the centered inputs (no loss in
      the centering), solved with a Cholesky decomposition. Forming
      :math:`\\mathbf{XX}^\\top` still squares the condition number of the inputs.
    - ``"qr"``: a QR decomposition of the centered inputs augmented with
      :math:`\\sqrt{\\lambda}\\mathbf{Id}` (the least squares form of the ridge
      regression), then a triangular solve: no normal equations, cheaper than
      ``"svd"``. Needs ``ridge > 0`` or linearly independent inputs.
    - ``"svd"``: the regression solved with a singular value decomposition of the
      centered inputs themselves, as scikit-learn's ``Ridge(solver="svd")``: no normal
      equations, the most stable choice for ill-conditioned inputs.
    - ``"lsqr"``: the iterative least squares solver :py:func:`scipy.sparse.linalg.lsqr`
      (damped by :math:`\\sqrt{\\lambda}`), for each output, as scikit-learn's
      ``Ridge(solver="lsqr")``: no normal equations, its precision is set by its
      stopping tolerance.

    All the solvers except ``"normal"`` keep all the training data in memory during
    ``fit``.


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
    solver : {"normal", "cholesky", "qr", "svd", "lsqr"}, default to "normal"
        Method used to solve the regression (see above).
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
    #: Method used to solve the regression ("normal" by default).
    solver: str

    def __init__(
        self,
        ridge: float = 0.0,
        fit_bias: bool = True,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        solver: Literal["normal", "cholesky", "qr", "svd", "lsqr"] = "normal",
        name: Optional[str] = None,
    ):
        if solver not in ("normal", "cholesky", "qr", "svd", "lsqr"):
            raise ValueError(f"Unknown solver '{solver}': must be 'normal', 'cholesky', 'qr', 'svd' or 'lsqr'.")
        self.ridge = ridge
        self.fit_bias = fit_bias
        self.solver = solver
        self.Wout = Wout
        self.bias = bias
        self.name = name
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
        self.state = {"out": np.zeros((self.output_dim,))}

        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        return {"out": x @ self.Wout + self.bias}

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        out = x @ self.Wout + self.bias
        return {"out": out[-1]}, out

    def worker(self, x: Timeseries, y: Timeseries):
        if self.solver != "normal":
            return x, y  # solved on all the data at once in master
        x_sum = np.sum(x, axis=0)
        y_sum = np.sum(y, axis=0)
        sample_size = x.shape[0]
        XXT = x.T @ x
        YXT = x.T @ y
        return XXT, YXT, x_sum, y_sum, sample_size

    def master(self, generator: Iterable):
        if self.solver == "normal":
            Wout, bias = self._fit_from_moments(generator)
        else:
            Wout, bias = self._fit_from_samples(generator)

        self.Wout = Wout
        self.bias = bias

    def _fit_from_moments(self, generator: Iterable):
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
            total_samples += sample_size

        if self.fit_bias:
            X_means = X_sum / total_samples
            Y_means = Y_sum / total_samples
            XXT -= total_samples * np.outer(X_means, X_means)
            YXT -= total_samples * np.outer(X_means, Y_means)

        Wout = linalg.solve(XXT + ridge_In, YXT, assume_a="sym")

        if self.fit_bias:
            bias = Y_means - X_means @ Wout
        else:
            bias = np.zeros((self.output_dim,))

        return Wout, bias

    def _fit_from_samples(self, generator: Iterable):
        xs, ys = zip(*generator)
        X, Y = np.concatenate(xs), np.concatenate(ys)

        if self.fit_bias:
            X_means, Y_means = X.mean(axis=0), Y.mean(axis=0)
            X, Y = X - X_means, Y - Y_means

        if self.solver == "cholesky":
            Wout = linalg.solve(X.T @ X + self.ridge * np.eye(self.input_dim), X.T @ Y, assume_a="pos")
        elif self.solver == "qr":
            # [X; sqrt(ridge) Id] = QR, and the targets augmented with zeros: Q^T [Y; 0] = Q[:len(X)]^T Y
            Q, R = np.linalg.qr(np.vstack([X, np.sqrt(self.ridge) * np.eye(self.input_dim)]))
            Wout = linalg.solve_triangular(R, Q[: len(X)].T @ Y)
        elif self.solver == "lsqr":
            damp = np.sqrt(self.ridge)
            Wout = np.stack([lsqr(X, y, damp=damp, atol=1e-12, btol=1e-12)[0] for y in Y.T], axis=1)
        else:  # svd
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            # (negligible singular values ignored, tolerance of np.linalg.pinv: no division by 0 with ridge=0)
            d = np.divide(s, s**2 + self.ridge, out=np.zeros_like(s), where=s > s[0] * np.finfo(float).eps * max(X.shape))
            Wout = Vt.T @ (d[:, None] * (U.T @ Y))

        if self.fit_bias:
            bias = Y_means - X_means @ Wout
        else:
            bias = np.zeros((self.output_dim,))

        return Wout, bias
