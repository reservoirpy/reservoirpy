"""
===========================================================
Metrics and observables (:py:mod:`reservoirpy.observables`)
===========================================================

Metrics and observables for Reservoir Computing:

.. autosummary::
   :toctree: generated/

    spectral_radius
    mse
    rmse
    nrmse
    rsquare
    memory_capacity
    effective_spectral_radius
"""

# Author: Nathan Trouvain at 01/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal, Optional, Union
else:
    from typing import Literal, Optional, Union

from copy import deepcopy

import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs

from .model import Model
from .type import Weights
from .utils.random import rand_generator


def _check_arrays(y_true, y_pred):
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if not y_true_array.shape == y_pred_array.shape:
        raise ValueError(
            f"Shape mismatch between y_true and y_pred: "
            f"{y_true_array.shape} != {y_pred_array.shape}"
        )

    return y_true_array, y_pred_array


def spectral_radius(W: Weights, maxiter: Optional[int] = None) -> float:
    """Compute the spectral radius of a matrix `W`.

    Spectral radius is defined as the maximum absolute
    eigenvalue of `W`.

    Parameters
    ----------
    W : array-like (sparse or dense) of shape (N, N)
        Matrix from which the spectral radius will
        be computed.

    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed.
        By default, is equal to `W.shape[0] * 20`.
        See `Scipy documentation <https://docs.scipy.org/
        doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_
        for more informations.

    Returns
    -------
    float
        Spectral radius of `W`.

    Raises
    ------
    ArpackNoConvergence
        When computing spectral radius on large
        sparse matrices, it is possible that the
        Fortran ARPACK algorithm used to compute
        eigenvalues don't converge towards precise
        values. To avoid this problem, set the `maxiter`
        parameter to an higher value. Be warned that
        this may drastically increase the computation
        time.

    Examples
    --------
    >>> from reservoirpy.observables import spectral_radius
    >>> from reservoirpy.mat_gen import normal
    >>> W = normal(1000, 1000, degree=8)
    >>> print(spectral_radius(W))
    2.8758915077733564
    """
    if issparse(W):
        if maxiter is None:
            maxiter = W.shape[0] * 20

        return max(
            abs(
                eigs(
                    W,
                    k=1,
                    which="LM",
                    maxiter=maxiter,
                    return_eigenvectors=False,
                    v0=np.ones(W.shape[0], W.dtype),
                )
            )
        )

    return max(abs(linalg.eig(W)[0]))


def mse(y_true: np.ndarray, y_pred: np.ndarray, dimensionwise: bool = False) -> float:
    """Mean squared error metric:

    .. math::

        \\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    dimensionwise: boolean, optional
        If True, return a mean squared error for each dimension of the timeseries

    Returns
    -------
    float
        Mean squared error.
    If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy.nodes import ESN
    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = ESN(units=100, sr=1).fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import mse
    >>> print(mse(y_true=y_test, y_pred=y_pred))
    0.03962918253990291
    """
    y_true_array, y_pred_array = _check_arrays(y_true, y_pred)

    if dimensionwise:
        if len(y_true_array.shape) == 3:
            axis = (0, 1)
        else:
            axis = 0
    else:
        axis = None
    return np.mean((y_true_array - y_pred_array) ** 2, axis=axis)


def rmse(y_true: np.ndarray, y_pred: np.ndarray, dimensionwise: bool = False) -> float:
    """Root mean squared error metric:

    .. math::

        \\sqrt{\\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}}

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    dimensionwise: boolean, optional
        If True, return a mean squared error for each dimension of the timeseries

    Returns
    -------
    float
        Root mean squared error.
    If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = model.fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import rmse
    >>> print(rmse(y_true=y_test, y_pred=y_pred))
    0.00034475744480521534
    """
    return np.sqrt(mse(y_true, y_pred, dimensionwise=dimensionwise))


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    norm: Literal["minmax", "var", "mean", "q1q3"] = "minmax",
    norm_value: Optional[float] = None,
    dimensionwise: bool = False,
) -> float:
    """Normalized root mean squared error metric:

    .. math::

       \\frac{1}{\\lambda} * \\sqrt{\\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}}

    where :math:`\\lambda` may be:
        - :math:`\\max y - \\min y` (Peak-to-peak amplitude) if ``norm="minmax"``;
        - :math:`\\mathrm{Var}(y)` (variance over time) if ``norm="var"``;
        - :math:`\\mathbb{E}[y]` (mean over time) if ``norm="mean"``;
        - :math:`Q_{3}(y) - Q_{1}(y)` (quartiles) if ``norm="q1q3"``;
        - or any value passed to ``norm_value``.

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    norm : {"minmax", "var", "mean", "q1q3"}, default to "minmax"
        Normalization method.
    norm_value : float, optional
        A normalization factor. If set, will override the ``norm`` parameter.
    dimensionwise: boolean, optional
        If True, return a mean squared error for each dimension of the timeseries

    Returns
    -------
    float
        Normalized root mean squared error.
    If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = model.fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import nrmse
    >>> print(nrmse(y_true=y_test, y_pred=y_pred, norm="var"))
    0.007854318015438394
    """
    error = rmse(y_true, y_pred, dimensionwise=dimensionwise)
    if norm_value is not None:
        return error / norm_value

    else:
        y_true_array, y_pred_array = _check_arrays(y_true, y_pred)

        if dimensionwise:
            if len(y_true_array.shape) == 3:
                axis = (0, 1)
            else:
                axis = 0
        else:
            axis = None

        norms = {
            "minmax": lambda y: np.ptp(y, axis=axis),
            "var": lambda y: y.var(axis=axis),
            "mean": lambda y: y.mean(axis=axis),
            "q1q3": lambda y: np.quantile(y, 0.75, axis=axis)
            - np.quantile(y, 0.25, axis=axis),
        }

        if norms.get(norm) is None:
            raise ValueError(
                f"Unknown normalization method. "
                f"Available methods are {list(norms.keys())}."
            )
        else:
            return error / norms[norm](y_true_array)


def rsquare(
    y_true: np.ndarray, y_pred: np.ndarray, dimensionwise: bool = False
) -> float:
    """Coefficient of determination :math:`R^2`:

    .. math::

        1 - \\frac{\\sum^{N-1}_{i=0} (y - \\hat{y})^2}
        {\\sum^{N-1}_{i=0} (y - \\bar{y})^2}

    where :math:`\\bar{y}` is the mean value of ground truth.

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    dimensionwise: boolean, optional
        If True, return a mean squared error for each dimension of the timeseries

    Returns
    -------
    float
        Coefficient of determination.
    If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = model.fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import rsquare
    >>> print(rsquare(y_true=y_test, y_pred=y_pred))
    0.9999972921653904
    """
    y_true_array, y_pred_array = _check_arrays(y_true, y_pred)

    if dimensionwise:
        if len(y_true_array.shape) == 3:
            axis = (0, 1)
        else:
            axis = 0
    else:
        axis = None

    d = (y_true_array - y_pred_array) ** 2
    D = (y_true_array - y_true_array.mean(axis=axis)) ** 2
    return 1 - np.sum(d, axis=axis) / np.sum(D, axis=axis)


def memory_capacity(
    model: Model,
    k_max: int,
    as_list: bool = False,
    series: Optional[np.ndarray] = None,
    test_size: Union[int, float] = 0.2,
    seed: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
):
    """Memory Capacity of a model

    The Memory Capacity [1]_ (MC) measure is defined as:

    .. math::
        MC = \\sum_{k=1}^{k_{max}} MC_k

    where:

    .. math::
        MC_k = \\rho^2(u(t-k), y_k(t)) = {cov^2[u(t-k), y_k(t)] \\over var(u(t-k)) \\cdot var(y_k(t))}

    This measure is commonly used in reservoir computing to evaluate the ability of a network to
    recall the previous timesteps. By default, the timeseries :math:`u` is an i.i.d. uniform signal in [-0.8, 0.8].

    Parameters
    ----------
    model : :class:`reservoirpy.Model`
        A ReservoirPy model on which the memory capacity is tested.
        Must have only one input and output node.
    k_max : int
        Maximum time lag between input and output.
        A common rule of thumb is to choose `k_max = 2*reservoir.units`.
    as_list: bool, optional, defaults to False
        If True, returns an array in which `out[k]` :math:`= MC_{k+1}`
    series: array of shape (timesteps, 1), optional
        If specified, is used as the timeseries :math:`u`.
    test_size : int or float
        Number of timesteps for the training phase. Can also be specified
        as a float ratio.
    seed : int or :py:class:`numpy.random.Generator`, optional
        Random state seed for reproducibility.

    Returns
    -------
    float, between 0 and `k_max`.
    If `as_list` is set to True, returns an array of shape `(k_max, )`.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> from reservoirpy.observables import memory_capacity
    >>> model = Reservoir(100, sr=1, seed=1) >> Ridge(ridge=1e-4)
    >>> mcs = memory_capacity(model, k_max=50, as_list=True, seed=1)
    >>> print(f"Memory capacity of {model.name}: {np.sum(mcs):.4}")
    Memory capacity of Model-0: 12.77

    .. plot::

        from reservoirpy.nodes import Reservoir, Ridge
        from reservoirpy.observables import memory_capacity
        model = Reservoir(100, sr=1, seed=1) >> Ridge(ridge=1e-4)
        mcs = memory_capacity(model, k_max=50, as_list=True, seed=1)
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, 51), mcs)
        plt.grid(); plt.xlabel("Time lag (timestep)"); plt.ylabel("$MC_k$")
        plt.show()

    References
    ----------
    .. [1] Jaeger, H. (2001). Short term memory in echo state networks.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # Task definition
    if series is None:
        rng = rand_generator(seed)
        series = rng.uniform(low=-0.8, high=0.8, size=(10 * k_max, 1))

    if isinstance(test_size, float) and test_size < 1 and test_size >= 0:
        test_len = round(series.shape[0] * test_size)
    elif isinstance(test_size, int):
        test_len = test_size
    else:
        raise ValueError(
            "invalid test_size argument: "
            "test_size can be an integer or a float "
            f"in [0, 1[, but is {test_size}."
        )

    # sliding_window_view creates a matrix of the same
    # timeseries with an incremental shift on each column
    dataset = sliding_window_view(series[:, 0], k_max + 1)[:, ::-1]
    X_train = dataset[:-test_len, :1]
    X_test = dataset[-test_len:, :1]
    Y_train = dataset[:-test_len, 1:]
    Y_test = dataset[-test_len:, 1:]
    # Model
    model_clone = deepcopy(model)
    model_clone.fit(X_train, Y_train, warmup=k_max)
    Y_pred = model_clone.run(X_test)

    # u[t-k] - z_k[t] square correlation
    capacities = np.square(
        [
            np.corrcoef(y_pred, y_test, rowvar=False)[1, 0]
            for y_pred, y_test in zip(Y_pred.T, Y_test.T)
        ]
    )

    if as_list:
        return capacities
    else:
        return np.sum(capacities)


def effective_spectral_radius(
    W: np.ndarray, lr: float = 1.0, maxiter: Optional[int] = None
):
    """Effective spectral radius

    The effective spectral radius is defined as the maximal singular value of the matrix
    :math:`lr \\cdot W + (1-lr) \\cdot I_{n}`.

    This concept was first introduced by Jaeger & al. [1]_, with an important result on leaky echo
    state networks:

    Supposing:

    #. The activation function is `tanh`
    #. There is no added noise inside the reservoir (`noise_rc = noise_in = 0.0`)
    #. There is no feedback
    #. There is no bias inside the reservoir (`bias = 0`)

    Then, if the effective spectral radius exceeds 1, the ESN does not have the echo state property.

    Parameters
    ----------
    W : array of shape `(units, units)`
        Adjacency matrix of a reservoir
    lr : float
        Leak rate
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed.
        By default, is equal to `W.shape[0] * 20`.
        See `Scipy documentation <https://docs.scipy.org/
        doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_
        for more informations.

    Returns
    -------
    float
        Spectral radius of :math:`lr \\cdot W + (1-lr) \\cdot I_{n}`.

    Raises
    ------
    ArpackNoConvergence
        When computing spectral radius on large
        sparse matrices, it is possible that the
        Fortran ARPACK algorithm used to compute
        eigenvalues don't converge towards precise
        values. To avoid this problem, set the `maxiter`
        parameter to an higher value. Be warned that
        this may drastically increase the computation
        time.

    Examples
    --------

    >>> from reservoirpy.observables import spectral_radius, effective_spectral_radius
    >>> from reservoirpy.mat_gen import uniform
    >>> W = uniform(100, 100, sr=0.5, seed=0)
    >>> lr = 0.5
    >>> print(f"{spectral_radius(W)=:.3}")
    >>> print(f"{effective_spectral_radius(W, lr=lr)=:.3}")
    spectral_radius(W)=0.5
    effective_spectral_radius(W, lr=lr)=0.701

    References
    ----------
    .. [1] Jaeger, H., Lukoševičius, M., Popovici, D., & Siewert, U. (2007).
        Optimization and applications of echo state networks with leaky-integrator
        neurons. Neural networks, 20(3), 335-352.
    """
    units = W.shape[0]
    return spectral_radius(W=lr * W + (1 - lr) * np.eye(units), maxiter=maxiter)
