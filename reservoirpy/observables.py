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
    lyapunov
    ky_dim
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import sys
import warnings
from copy import deepcopy
from typing import Literal, Optional, Union

import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs
from tqdm import tqdm

from .type import Weights
from .utils.random import rand_generator


def _check_arrays(y_true, y_pred):
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if not y_true_array.shape == y_pred_array.shape:
        raise ValueError(f"Shape mismatch between y_true and y_pred: " f"{y_true_array.shape} != {y_pred_array.shape}")

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
            "q1q3": lambda y: np.quantile(y, 0.75, axis=axis) - np.quantile(y, 0.25, axis=axis),
        }

        if norms.get(norm) is None:
            raise ValueError(f"Unknown normalization method. " f"Available methods are {list(norms.keys())}.")
        else:
            return error / norms[norm](y_true_array)


def rsquare(y_true: np.ndarray, y_pred: np.ndarray, dimensionwise: bool = False) -> float:
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
    model: "Model",
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
    >>> print(f"Memory capacity of the model: {np.sum(mcs):.4}")
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
            "invalid test_size argument: " "test_size can be an integer or a float " f"in [0, 1[, but is {test_size}."
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
        [np.corrcoef(y_pred, y_test, rowvar=False)[1, 0] for y_pred, y_test in zip(Y_pred.T, Y_test.T)]
    )

    if as_list:
        return capacities
    else:
        return np.sum(capacities)


def effective_spectral_radius(W: np.ndarray, lr: float = 1.0, maxiter: Optional[int] = None):
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


def _estimate_lambda_1(model, epsilon, n_probe=1000, sub_cycle=5, probe_sub_cycle_cap=None, progress_bar=False):
    """Estimates the first Lyapunov exponent, using a simplified version of
    the Bennetin algorithm with a single perturbation. This is used by
    `_lyapunov()` to determine the characteristic timescale 1/λ1 of the
    reservoir computing system, which that function uses to set
    reorthonormalization cycle length"""

    t_step = getattr(model, "t_step", 1.0)
    stdev = getattr(model, "stdev", 1.0)
    pert_scale = epsilon * stdev

    raw = model.state
    if isinstance(raw, list):
        model.state = np.asarray(raw[0], dtype=float).copy()

    def _run_pass(sub_cyc, min_windows=4, label="probe"):
        fid = np.asarray(model.state, dtype=float).copy()
        pert = fid.copy()
        pert[0] += pert_scale
        n_windows = max(min_windows, n_probe // sub_cyc)
        log_growths = np.empty(n_windows)
        discard = n_windows // 2

        bar = tqdm(
            total=n_windows, desc="  " + label.ljust(24), unit="renorm", disable=not progress_bar, file=sys.stdout
        )

        for w in range(n_windows):
            model.state = fid
            model.run(sub_cyc)
            fid = np.asarray(model.state, dtype=float)
            model.state = pert
            model.run(sub_cyc)
            pert = np.asarray(model.state, dtype=float)
            delta = pert - fid
            d = np.linalg.norm(delta)
            if not np.isfinite(d) or d == 0.0:
                bar.close()
                return None, fid
            log_growths[w] = np.log(d / pert_scale)
            pert = fid + delta * (pert_scale / d)
            bar.update(1)

        bar.close()
        lam = float(log_growths[discard:].mean()) / sub_cyc
        return (lam / t_step if lam > 0 else None), fid

    dim = np.asarray(model.state, dtype=float).size
    slow_sub = max(50, dim)
    if probe_sub_cycle_cap is not None:
        slow_sub = min(slow_sub, probe_sub_cycle_cap)
    state_before = np.asarray(model.state, dtype=float).copy()
    result_slow, fid = _run_pass(slow_sub, min_windows=200, label="lyapunov time probe")
    model.state = fid
    if result_slow is not None:
        return result_slow

    model.state = state_before
    result_fast, fid = _run_pass(sub_cycle, label="lyapunov time probe")
    model.state = fid
    return result_fast


def _ordering_violated(log_growths, prob_threshold=0.8):
    """Return True if adjacent spectrum entries are probably out of order.

    The Benettin algorithm for Lyapunov exponents should produce an ordered
    spectrum if it is properly broken in. This helper function takes the growth
    factors used to determine that spectrum, and checks if the probability of
    mis-ordering is greated than `prob_threshold`.
    """
    from scipy.special import ndtr

    n, k = log_growths.shape
    if k < 2 or n < 2:
        return False
    mean = log_growths.mean(axis=0)
    se = log_growths.std(axis=0, ddof=1) / np.sqrt(n)
    diff = mean[1:] - mean[:-1]
    se_diff = np.sqrt(se[:-1] ** 2 + se[1:] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = ndtr(diff / se_diff)
    # Zero-variance pairs: determined purely by the sign of the mean gap.
    prob = np.where(se_diff == 0, (diff > 0).astype(float), prob)
    prob = prob[np.isfinite(prob)]  # drop collapsed (NaN std) directions
    if prob.size == 0:
        return False
    return bool(prob.max() > prob_threshold)


def _is_converged(spectrum, ci_half_rate, ky_dim, ky_dim_ci, rtol, mode, atol=0.0):
    """Check Lyapunov convergence under the selected criterion.

    Converged when EITHER the absolute CI half-width is below ``atol`` OR the
    relative CI half-width is below ``rtol`` times the reference magnitude. For
    covergence type "all", this either/or is evaluated for each element.
    """
    if rtol <= 0 and atol <= 0:
        return False
    effective = mode
    if mode == "ky_dim" and (ky_dim is None or ky_dim_ci is None):
        effective = "all"
    if effective == "ky_dim":
        if ky_dim == 0 or not np.isfinite(ky_dim_ci):
            return False
        ci, scale = ky_dim_ci, abs(ky_dim)
    elif effective == "lambda_1":
        lam = spectrum[0]
        if not (np.isfinite(lam) and lam != 0 and np.isfinite(ci_half_rate[0])):
            return False
        ci, scale = ci_half_rate[0], abs(lam)
    elif effective == "all":
        finite = np.isfinite(ci_half_rate)
        lam1 = spectrum[0]
        if not finite.any() or not np.isfinite(lam1) or lam1 == 0:
            return False
        ci, scale = float(np.max(ci_half_rate[finite])), abs(lam1)
    else:
        raise ValueError(f"unknown convergence mode: {mode!r}")
    return bool((atol > 0 and ci < atol) or (rtol > 0 and ci < rtol * scale))


def _warn_in_order(message: str, stacklevel: int = 3) -> None:
    sys.stdout.flush()
    orig_showwarning = warnings.showwarning

    def _to_stdout(msg, cat, fn, ln, file=None, line=None):
        orig_showwarning(msg, cat, fn, ln, file=sys.stdout, line=line)

    warnings.showwarning = _to_stdout
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)
    finally:
        warnings.showwarning = orig_showwarning
    sys.stdout.flush()


def _lyapunov(
    model,
    k: int,
    cycle_length: Optional[int] = None,
    breakin_cycles: int = 500,
    breakin_groups: int = 3,
    ordering_threshold: float = 0.8,
    min_cycles: int = 500,
    max_cycles: int = 2000,
    lyap_time_per_cycle: float = 0.5,
    epsilon: float = 1e-5,
    convergence: str = "ky_dim",
    rtol: float = 0.0,
    atol: float = 0.0,
    progress_bar: bool = False,
    probe_sub_cycle_cap: int = 50,
) -> dict:
    """Calculate the Lyapunov spectrum for a dynamical system, using the Benettin
    algorithm [1]_.

    The algorithm works by creating a set of small orthonormal perturbations of
    the state of a dynamical system. It then evolves each of these perturbed trajectories
    along with a fiducial trajectory. After some interval of time, the growth factor of
    these perturbations is measured, and the perturbations are reorthonormalized
    using the QR factorization. For a detailed and clear exposition of the
    Benettin algorith, see Wolf et al. (1985) [2]_. Note that [1]_ and [2]_
    reorthonormalize the perturbed states by Gram-Schmidt process instead of QR
    factorization used here.

    Solver works in three steps:
    - Initial Probe: Gives a rough estimate of the maximal
    Lyapunov exponent, which is used to determine the characteristic timescale
    of the system and set the reorthonormalization cycle length
    - Break-in: Initializes perturbed states, then goes through cycles of
    evolution and reorthonormalization to allow all transients to decay
    - Main iterative solving: Performs reorthormalization cycles and records growth
    factors, continues until estimated spectrum converges or maximum number of
    cycles is reached.


    Parameters
    ----------
    model : class, with the following attributes:
        - `state`: array-like
            The dynamical system's internal state as a 1-D array.
        - `run(n_steps)`: callable, with integer inputs
            Evolves the dynamical system `n_steps` forward. Assumes the system
            is discrete.
        - `t_step`: float, optional, defaults to 1.0
            Time per discrete system step. Used for converting spectrum units
            in final output.
        - `stdev`: float, optional, defaults to 1.0
            Used to set scale of perturbations
    k : int
        Number of lyapunov exponents to find. Maximum is the length of
        `model.state`
    cycle_length: int, optional, defaults to None
        Length of reorthonormalization cycle. If not provided, found by probe
    breakin_cycles: int, defaults to 500
        Orthonormalization cycles per break-in group.
    breakin_groups: int, defaults to 3
        Break-in uses a maximum of `breakin_cycles` x 'breakin_groups` cycles.
        After each group of `breakin_cycles` cycles, a fast diagnostic checks
        if the estimated spectrum is ordered. A lack of ordering indicated
        undecayed transients; in this case another `breakin_cycles` cycles are
        run. If the estimated spectrum is properly ordered, break-in stops early.
    ordering_threshold: float, defaults to 0.8
        Minimum probability of the spectrum being out-of-order that will lead
        to additional break-in cycles.
    min_cycles: int, defaults to 500
        Number of cycles in main iterative phase before convergence check.
        Algorithm will iterate forward `min_cycles` and check convergence until
        convergence or `max_cycles` is reached.
    max_cycles: int, defaults to 2000
        Maximum number of cycles in main iterative phase
    lyap_time_per_cycle: float, defaults to 0.5
        Reorthonormalization cycle length, in units of lyapunov time (1/λ_1).
        Used for break-in and iterative solving steps, with λ_1 approximated
        by initial probe step.
    epsilon: float, defaults to 1e-5
        Perturbation scale: perturbation size after reorthonormalization is
        `epsilon` x `model.stdev`
    convergence: str, defaults to "ky_dim"
        Convergence criteria for early termination of iterative solving
        - "ky_dim" (default): 95% confidence interval for Kaplan-Yorke
        dimension (D_KY) is less than `atol` or estimated D_KY x 'rtol'
        - "lambda_1": 95% confidence interval of maximal lyapunpov exponent (λ1)
        is less than `atol` or λ1 x `rtol`
        - "all": 95% confidence interval of each spectral element λ_i is less
        than `atol` or λ1 x `rtol`
    rtol: float, defaults to 0.0
        Relative tolerance for convergence
    atol: float, defaults to 0.0
        Absolute tolerance for convergence
    progress_bar: bool, defaults to False
        Displays a tqdm progress bar for each solver stage
    probe_sub_cycle_cap: int, optional, defaults to 50
        Number of cycles used in probe phase for rough estimate of λ1

    Returns
    -------
    dict of results:

    spectrum : ndarray of shape (k,)
        Lyapunov exponents in descending order, in base e
    spectrum_ci : ndarray of shape (k,)
        Half-width of the 95 % confidence interval for each exponent
    ky_dim : float
        Kaplan-Yorke dimension estimated from ``spectrum``.
    ky_dim_ci : float
        Half-width of the 95 % confidence interval for ``ky_dim``
    n_cycles : int
        Number of reorthonormalization cycles completed in the main solver.
    converged : bool
        ``True`` if the convergence criterion was satisfied before
        ``max_cycles`` was reached; ``False`` otherwise.
    cycle_length : int
        Reorthonormalization cycle length used (in discrete steps), either as
        provided or as determined by the initial probe.
    breakin_cycles : int
        Total number of break-in cycles actually run across all break-in groups.
    lambda_1_probe : float or None
        Rough estimate of the maximal Lyapunov exponent from the initial probe
        phase, used to set ``cycle_length``. ``None`` if ``cycle_length`` was
        provided by the caller.

    References
    ----------
    .. [1] Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J.-M. (1980).
       Lyapunov characteristic exponents for smooth dynamical systems and for
       Hamiltonian systems; a method for computing all of them. Part 1: Theory.
       *Meccanica*, 15(1), 9–20. https://doi.org/10.1007/BF02128236

    .. [2] Wolf, A., Swift, J. B., Swinney, H. L., & Vastano, J. A. (1985).
       Determining Lyapunov exponents from a time series.
       *Physica D: Nonlinear Phenomena*, 16(3), 285–317.
       https://doi.org/10.1016/0167-2789(85)90011-9
    """
    state0 = np.asarray(model.state, dtype=float)
    t_step = getattr(model, "t_step", 1.0)
    stdev = getattr(model, "stdev", 1.0)
    pert_scale = epsilon * stdev

    if cycle_length is None:
        lambda_1_probe = _estimate_lambda_1(
            model, epsilon, probe_sub_cycle_cap=probe_sub_cycle_cap, progress_bar=progress_bar
        )
        if lambda_1_probe is None or lambda_1_probe <= 0:
            cycle_length = 100
        else:
            cycle_length = int(np.ceil(lyap_time_per_cycle / (lambda_1_probe * t_step)))
    else:
        lambda_1_probe = None

    dim = state0.size
    perturbed_states = np.empty((k + 1, dim))
    perturbed_states[0] = state0
    for j in range(k):
        perturbed_states[j + 1] = state0.copy()
        perturbed_states[j + 1, j % dim] += pert_scale

    def _do_qr_cycles(n, record=False, pbar=None):
        """Run n QR-Benettin cycles; optionally record log-growths."""
        buf = np.empty((n, k)) if record else None
        c = 0
        for _ in range(n):
            for j in range(k + 1):
                model.state = perturbed_states[j]
                model.run(cycle_length)
                perturbed_states[j] = np.asarray(model.state, dtype=float)
            fiducial = perturbed_states[0]
            error = perturbed_states[1:] - fiducial
            Q, R = np.linalg.qr(error.T)
            diag_R = np.abs(np.diag(R))
            diag_R = np.where(diag_R > 0, diag_R, np.finfo(float).tiny)
            if record:
                buf[c] = np.log(diag_R / pert_scale)
                c += 1
            signs = np.sign(np.diag(R))
            signs[signs == 0] = 1.0
            Q = Q * signs
            perturbed_states[1:] = fiducial + (Q[:, :k] * pert_scale).T
            if pbar is not None:
                pbar.update(1)
        return buf[:c] if record else None

    breakin_used = 0
    _DIAG_CYCLES = 50
    for group in range(1, breakin_groups + 1):
        _bi_label = "perturbation breakin" if group == 1 else "perturbation breakin ext"
        pbar_bi = tqdm(
            total=breakin_cycles,
            desc="  " + _bi_label.ljust(24),
            unit="cyc",
            disable=not progress_bar,
            file=sys.stdout,
        )
        _do_qr_cycles(breakin_cycles, pbar=pbar_bi)
        breakin_used += breakin_cycles
        pbar_bi.close()
        if group < breakin_groups:
            sample = _do_qr_cycles(_DIAG_CYCLES, record=True)
            if not _ordering_violated(sample, prob_threshold=ordering_threshold):
                break

    _SAT_LOG_THRESH = np.log(1e-12)
    log_growths = np.zeros((max_cycles, k))
    sat_count = np.zeros(k, dtype=int)
    n_cycles = 0
    converged = False
    collapsed = np.zeros(k, dtype=bool)

    cycles_per_batch = max(1, min_cycles)

    pbar = tqdm(
        total=max_cycles, desc="  " + "lyapunov spectrum finder", unit="cyc", disable=not progress_bar, file=sys.stdout
    )

    while n_cycles < max_cycles and not converged:
        batch = min(cycles_per_batch, max_cycles - n_cycles)
        for _ in range(batch):
            if n_cycles >= max_cycles:
                break
            for j in range(k + 1):
                model.state = perturbed_states[j]
                model.run(cycle_length)
                perturbed_states[j] = np.asarray(model.state, dtype=float)

            fiducial = perturbed_states[0]
            if not np.isfinite(fiducial).all():
                break

            error = perturbed_states[1:] - fiducial
            Q, R = np.linalg.qr(error.T)
            diag_R = np.abs(np.diag(R))
            diag_R = np.where(diag_R > 0, diag_R, np.finfo(float).tiny)
            lg = np.log(diag_R / pert_scale)
            log_growths[n_cycles] = lg
            sat_count += lg < _SAT_LOG_THRESH
            n_cycles += 1
            pbar.update(1)

            signs = np.sign(np.diag(R))
            signs[signs == 0] = 1.0
            Q = Q * signs
            perturbed_states[1:] = fiducial + (Q[:, :k] * pert_scale).T

        if n_cycles == 0:
            break

        collapsed = sat_count > 0.5 * n_cycles

        recorded = log_growths[:n_cycles].copy()
        recorded[:, collapsed] = np.nan

        with np.errstate(invalid="ignore"):
            spectrum = np.nanmean(recorded, axis=0) / (cycle_length * t_step)
        std = np.nanstd(recorded, axis=0, ddof=1) if n_cycles > 1 else np.full(k, np.inf)
        ci_half = 1.96 * std / np.sqrt(n_cycles)
        ci_half_rate = ci_half / (cycle_length * t_step)

        ky, ky_ci = ky_dim(spectrum, ci_half_rate, warn=False)

        if rtol > 0 or atol > 0:
            converged = _is_converged(spectrum, ci_half_rate, ky, ky_ci, rtol, convergence, atol)

    pbar.close()

    recorded = log_growths[:n_cycles].copy()
    recorded[:, collapsed] = np.nan

    with np.errstate(invalid="ignore"):
        spectrum = np.nanmean(recorded, axis=0) / (cycle_length * t_step)
    std_final = np.nanstd(recorded, axis=0, ddof=1) if n_cycles > 1 else np.full(k, np.inf)
    ci_half_rate = 1.96 * std_final / np.sqrt(max(n_cycles, 1)) / (cycle_length * t_step)

    ky, ky_ci = ky_dim(spectrum, ci_half_rate)

    if _ordering_violated(recorded, prob_threshold=ordering_threshold):
        _warn_in_order(
            "Final Lyapunov spectrum has statistically significant out-of-order "
            "adjacent exponents — consider increasing max_cycles or breakin_cycles.",
        )

    finite_spec = spectrum[np.isfinite(spectrum)]
    if len(finite_spec) >= 2 and np.isfinite(spectrum[0]) and spectrum[0] != 0:
        n_near_zero = int(np.sum(np.abs(finite_spec) < 1e-2 * np.abs(spectrum[0])))
        if n_near_zero >= 2:
            _warn_in_order(
                f"{n_near_zero} Lyapunov exponents are within 0.01·|λ₁| of zero; "
                "some may be spurious and the Kaplan-Yorke dimension may be off by one.",
            )

    return {
        "spectrum": spectrum,
        "spectrum_ci": ci_half_rate,
        "ky_dim": ky,
        "ky_dim_ci": ky_ci,
        "n_cycles": n_cycles,
        "converged": converged,
        "cycle_length": cycle_length,
        "breakin_cycles": breakin_used,
        "lambda_1_probe": lambda_1_probe,
    }


def ky_dim(spectrum, ci_half=None, warn=True):
    """Compute the Kaplan-Yorke dimension [1]_ from a Lyapunov spectrum.

    .. math::

        D_{KY} = j + \\frac{\\sum_{i=1}^{j} \\lambda_i}{|\\lambda_{j+1}|}

    where :math:`j` is the largest index such that:

    .. math::

        `\\sum_{i=1}^{j} \\lambda_i \\geq 0`.

    Parameters
    ----------
    spectrum : array-like
        Lyapunov exponents in descending order.
    ci_half : array-like, optional
        Half-width 95 % confidence intervals per exponent (same length as
        ``spectrum``).  When given, the CI on :math:`D_{KY}` is propagated
        via first-order error propagation
    warn : bool
        When ``True`` (default), emit a :class:`RuntimeWarning` if all
        cumulative sums are positive (dimension exceeds the spectrum length).
        Set to ``False`` inside convergence loops to suppress repeated
        warnings.

    Returns
    -------
    float or None, or tuple of (float or None, float or None)
        Normal case: returns :math:`D_{KY}` as a float.

        Degenerate cases:

        * All cumulative sums :math:`\\leq 0` (purely contracting spectrum):
          returns ``0.0``.
        * Total sum :math:`> 0` True dimension is greater than or equal to
        spectrum length: returns ``float(len(spectrum))`` and, if ``warn`` is
         ``True``, emits a ``RuntimeWarning``.
        * :math:`\\lambda_{j+1}` non-finite: returns ``None``.

        When ``ci_half`` is provided each case returns the corresponding pair
        ``(dim, dim_ci)``; the CI is ``None`` for all three degenerate cases.

    References
    ----------
    .. [1] Kaplan, J. L., & Yorke, J. A. (1979). Chaotic behavior of
       multidimensional difference equations. In H.-O. Peitgen & H.-O. Walther
       (Eds.), *Functional Differential Equations and Approximation of Fixed
       Points*, Lecture Notes in Mathematics, vol. 730, pp. 204–227. Springer.
       https://doi.org/10.1007/BFb0064319
    """
    s = np.asarray(spectrum, dtype=float)
    n = len(s)
    cum = np.cumsum(np.where(np.isfinite(s), s, 0.0))
    pos = np.where(cum > 0)[0]

    def _ret(dim, dim_ci=None):
        return (dim, dim_ci) if ci_half is not None else dim

    if len(pos) == 0:
        return _ret(0.0)

    j = pos[-1]

    if j + 1 >= n:
        if warn:
            warnings.warn(
                f"Dimension not found, greater than or equal to {n}",
                RuntimeWarning,
                stacklevel=2,
            )
        return _ret(float(n))

    lam_j1 = s[j + 1]
    if not np.isfinite(lam_j1) or lam_j1 == 0:
        return _ret(None)

    dim = float(j + 1) + cum[j] / np.abs(lam_j1)

    if ci_half is None:
        return dim

    ci = np.asarray(ci_half, dtype=float)
    var = float(np.nansum((ci[: j + 1] / np.abs(lam_j1)) ** 2))
    var += float((ci[j + 1] * cum[j] / lam_j1**2) ** 2)
    return dim, float(np.sqrt(var))


# ---------------------------------------------------------------------------
# RC Lyapunov spectrum — public wrapper
# ---------------------------------------------------------------------------


class _ReservoirStepper:
    """Wrapper for :class:`reservoirpy.Model` that exposes the full model state
    as a single flat vector for use by :func:`_lyapunov`.

    **State representation**

    The dynamical state tracked by the Benettin algorithm is the concatenation
    of every node's ``state["out"]`` array, in the order returned by
    ``model.nodes``.  :meth:`_get_model_state` concatenates them into a 1-D
    array; :meth:`_set_model_state` writes a flat vector back by splitting it at
    breakpoints cached in ``self._splits`` (computed once in ``__init__`` via
    ``np.cumsum`` of per-node sizes) and reshaping each segment to the
    corresponding ``self._shapes`` entry.  For a single-reservoir ESN the flat
    state is just the reservoir activation vector.  For a model with multiple
    stateful nodes (e.g. two reservoirs in series) the state is the
    concatenation of all their activation vectors.

    **Closed-loop stepping**

    ``run(n_steps)`` unflatten the current ``self.state`` into the model,
    advances ``n_steps`` closed-loop steps, then re-flattens.  Each step feeds
    the current output node(s) back as the next input.  If the model has
    feedback buffers (``model.feedback_buffers`` is non-empty), the model's own
    feedback mechanism is used (``model.run(iters=1)``); otherwise the output is
    assembled manually from ``model.outputs`` and fed in as a ``(1, d)`` array.
    """

    t_step: float = 1.0

    def __init__(self, model, init_states: list, stdev: float = 1.0, t_step: float = 1.0):
        self._model = model
        self.state = init_states[0] if len(init_states) == 1 else init_states
        self.stdev = stdev
        self.t_step = t_step
        sizes = [n.state["out"].size for n in model.nodes]
        self._splits = np.cumsum(sizes)[:-1]
        self._shapes = [n.state["out"].shape for n in model.nodes]

    def _get_model_state(self) -> np.ndarray:
        """Flat 1-D concatenation of every node's ``state["out"]``."""
        return np.concatenate([np.asarray(n.state["out"], float).ravel() for n in self._model.nodes])

    def _set_model_state(self, flat: np.ndarray) -> None:
        """Write a flat state vector back into each node using cached segment boundaries."""
        for n, seg, shape in zip(self._model.nodes, np.split(flat, self._splits), self._shapes):
            n.state["out"] = seg.reshape(shape)

    def run(self, n_steps: int) -> None:
        """Advance ``n_steps`` closed-loop steps; update ``self.state``."""
        self._set_model_state(np.asarray(self.state, float))
        for _ in range(n_steps):
            self._closed_loop_step()
        self.state = self._get_model_state()

    def _closed_loop_step(self) -> None:
        """Advance the underlying model one step in closed-loop mode."""
        fb = getattr(self._model, "feedback_buffers", None)
        if fb:
            self._model.run(iters=1)
        else:
            pred = np.concatenate([np.asarray(n.state["out"], float).ravel() for n in self._model.outputs])
            self._model.run(pred.reshape(1, -1))

    def probe_stdev(self, n_probe: int = 200) -> float:
        """Estimate attractor stdev from a short closed-loop run.

        Starts from the first init state (restores it after probing so the
        saved state is not consumed).  Returns at least a small positive floor
        so perturbation scaling is never zero.
        """
        saved = np.asarray(self.state if not isinstance(self.state, list) else self.state[0], float).copy()
        self._set_model_state(saved)
        records = []
        for _ in range(n_probe):
            self._closed_loop_step()
            records.append(self._get_model_state())
        self._set_model_state(saved)
        stdev = float(np.std(np.stack(records)))
        return stdev if stdev > 0 else 1.0


def lyapunov(
    model,
    init_traj=None,
    init_state=None,
    k: int = 10,
    **lyap_kwargs,
) -> dict:
    """Compute the Lyapunov spectrum of a trained reservoir computer, using
    a version of the Benettin algorithm [1]_.

    The Lyapunov spectrum describes the way in which the distance between
    nearby trajectories of a dynamical system grow or shrink as the dynamical
    system evolves. The most positive exponent :math:`\\lambda_1` measures the
    rate at which almost all pairs of initially nearby trajectories diverge:

    For two trajectories :math:`x_1(t), x_2(t); x_2(t_0) = x_1(t_0) + \\delta_0`:

    .. math::
        \\lambda_1 = \\lim_{t\\rightarrow\\infty}\\lim_{\\delta_0 \\rightarrow 0}
        \\frac{1}{t}\\ln\\frac{\\|\\delta(t)\\|}{\\|\\delta_0\\|}

    where :math:`\\delta(t) = x_2(t) - x_1(t),\\; t > t_0`, or more simply:

    .. math::
        \\|x_1(t) - x_2(t)\\| \\approx \\delta_0 e^{\\lambda_1 t}

    Additional elements of the Lyapunov spectrum :math:`\\lambda_2, \\lambda_3, \\ldots`
    characterize the rate at which perturbations to a trajectory grow or decay
    in spaces orthogonal to the direction of growth measured by :math:`\\lambda_1`.
    For a thorough explanation see the paper by Wolf et al (1985) [2]_.

    The algorithm for finding the Lyapunov spectrum assumes that the dynamical
    system is ergodic; that is that almost every trajectory eventually passes
    arbitrarily close to any point on the attractor. If a system has multiple
    basins of attraction, these basins may have different lyapunov spectra.

    Wraps :func:`_lyapunov` for use with a trained reservoirpy :class:`Model`
    via a :class:`_ReservoirStepper`. For details of the actual solver used,
    see documentation for :func:`_lyapunov`.

    Parameters
    ----------
    model : :class:`reservoirpy.Model`
        A trained reservoirpy model.  Both simple echo state networks (ESNs)
        and more complex models with multiple components with internal states
        are supported (e.g. a model with multiple reservoirs)
    init_traj : array-like of shape (T, input_dim) or (T,), optional
        Timeseries used to initalize the reservoir state. Must be provided if
        `init_state` is not provided.
    init_state : array-like of shape (D,), optional
        a flat state vector of length equal to the dimension
        of the flattened state of ``model``. For a ``model`` with multiple
        state-preserving components, see :class:`_ReservoirStepper` for
        parsing details. If both `init_traj` and `init_state` are provided,
        uses `init_state` by default.
    k : int, default is 10
        Number of Lyapunov exponents to compute.
    **lyap_kwargs
        Additional keyword arguments forwarded to :func:`_lyapunov`
        (e.g. ``breakin_cycles``, ``breakin_groups``, ``ordering_threshold``,
        ``min_cycles``, ``max_cycles``, ``rtol``, ``atol``,
        ``epsilon``, ``lyap_time_per_cycle``, ``progress_bar``).

    Returns
    -------
    dict
        Same output as :func:`_lyapunov`: ``"spectrum"``, ``"spectrum_ci"``,
        ``"ky_dim"``, ``"ky_dim_ci"``, ``"n_cycles"``, ``"converged"``,
        ``"cycle_length"``, ``"breakin_cycles"``, ``"lambda_1_probe"``.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> from reservoirpy.datasets import lorenz
    >>> from reservoirpy.observables import lyapunov
    >>> data = lorenz(12000)
    >>> x_train, y_train = data[:10000], data[1:10001]
    >>> esn = Reservoir(300, sr=0.95) >> Ridge(ridge=1e-6)
    >>> esn = esn.fit(x_train, y_train)
    >>> result = lyapunov(esn, init_traj=x_train[:200], k=3,
    ...                   min_cycles=1000, max_cycles=2000)
    >>> print(result["spectrum"])
    """
    if init_state is not None:
        init_states = [np.asarray(init_state, dtype=float).ravel()]
        probe = False
    elif init_traj is not None:
        init_arr = np.asarray(init_traj, dtype=float)
        if init_arr.ndim == 1:
            init_arr = init_arr[:, None]
        model.reset()
        model.run(init_arr)
        init_states = [np.concatenate([np.asarray(n.state["out"], float).ravel() for n in model.nodes])]
        probe = True
    else:
        raise ValueError("lyapunov() requires either init_traj or init_state.")

    stepper = _ReservoirStepper(model, init_states)
    if probe:
        stepper.stdev = stepper.probe_stdev()

    return _lyapunov(stepper, k=k, **lyap_kwargs)
