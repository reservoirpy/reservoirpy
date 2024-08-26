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
"""

# Author: Nathan Trouvain at 01/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs

from .type import Weights


def _check_arrays(y_true, y_pred):
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if not y_true_array.shape == y_pred_array.shape:
        raise ValueError(
            f"Shape mismatch between y_true and y_pred: "
            "{y_true_array.shape} != {y_pred_array.shape}"
        )

    return y_true_array, y_pred_array


def spectral_radius(W: Weights, maxiter: int = None) -> float:
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


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error metric:

    .. math::

        \\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.

    Returns
    -------
    float
        Mean squared error.

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
    return float(np.mean((y_true_array - y_pred_array) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error metric:

    .. math::

        \\sqrt{\\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}}

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.

    Returns
    -------
    float
        Root mean squared error.

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
    return np.sqrt(mse(y_true, y_pred))


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    norm: Literal["minmax", "var", "mean", "q1q3"] = "minmax",
    norm_value: float = None,
) -> float:
    """Normalized mean squared error metric:

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

    Returns
    -------
    float
        Normalized mean squared error.

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
    error = rmse(y_true, y_pred)
    if norm_value is not None:
        return error / norm_value

    else:
        norms = {
            "minmax": lambda y: y.ptp(),
            "var": lambda y: y.var(),
            "mean": lambda y: y.mean(),
            "q1q3": lambda y: np.quantile(y, 0.75) - np.quantile(y, 0.25),
        }

        if norms.get(norm) is None:
            raise ValueError(
                f"Unknown normalization method. "
                f"Available methods are {list(norms.keys())}."
            )
        else:
            return error / norms[norm](np.asarray(y_true))


def rsquare(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    Returns
    -------
    float
        Coefficient of determination.

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

    d = (y_true_array - y_pred_array) ** 2
    D = (y_true_array - y_true_array.mean()) ** 2
    return 1 - np.sum(d) / np.sum(D)
