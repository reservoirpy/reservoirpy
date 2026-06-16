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
    """Estimate the largest Lyapunov exponent via a Benettin probe.

    Primary pass (``"lyapunov time probe"``):
    ``sub_cycle = max(50, state_dim)`` — each window spans at least one full
    ring-buffer traversal, so the DDE delayed-feedback channel has time to act
    before renormalization resets the perturbation direction.  The first half
    of windows is discarded as an alignment transient.
    The primary pass uses ``min_windows=200`` (about 100 recording windows
    after the half-discard), enough to average out Lorenz-scale variance while
    spanning the full ring buffer.
    Fallback pass (``"lyapunov time probe"``): ``sub_cycle=5`` — runs only if
    the primary pass returns ``None`` (divergence or degenerate perturbation
    norm).  Returns ``None`` only if both passes fail.
    Leaves ``model.state`` restored to its pre-probe value.

    ``probe_sub_cycle_cap`` caps the primary-pass sub_cycle (useful for
    large-state models like ESNs where the state dimension is not a ring-buffer
    length).

    When ``model.state`` is a list (multiple initial conditions), the probe
    runs from the first element only; ``_lyapunov`` saves the init states
    before calling this, so the list is unchanged for the actual computation.
    """
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


def _ordering_violated(log_growths, cycle_length, t_step):
    """Return True if adjacent spectrum entries are significantly out of order.

    Uses a max-adjacent-violation criterion: a pair is out of order when the
    later exponent exceeds the earlier one by more than their combined 95 % CI.
    """
    n, k = log_growths.shape
    if k < 2 or n < 2:
        return False
    rate_scale = 1.0 / (cycle_length * t_step)
    spec = log_growths.mean(axis=0) * rate_scale
    std = log_growths.std(axis=0, ddof=1)
    ci = 1.96 * std / np.sqrt(n) * rate_scale
    for i in range(k - 1):
        diff = spec[i + 1] - spec[i]
        combined_ci = np.sqrt(ci[i] ** 2 + ci[i + 1] ** 2)
        if diff > combined_ci:
            return True
    return False


def _is_converged(spectrum, ci_half_rate, ky_dim, ky_dim_ci, rtol, mode):
    """Check Lyapunov convergence under the selected criterion."""
    if rtol <= 0:
        return False
    effective = mode
    if mode == "ky_dim" and (ky_dim is None or ky_dim_ci is None):
        effective = "all"
    if effective == "ky_dim":
        return ky_dim != 0 and ky_dim_ci < rtol * abs(ky_dim)
    if effective == "lambda_1":
        lam = spectrum[0]
        return bool(np.isfinite(lam) and lam != 0 and ci_half_rate[0] < rtol * abs(lam))
    if effective == "all":
        finite = np.isfinite(ci_half_rate)
        lam1 = spectrum[0]
        if not finite.any() or not np.isfinite(lam1) or lam1 == 0:
            return False
        return bool(np.max(ci_half_rate[finite]) < rtol * abs(lam1))
    raise ValueError(f"unknown convergence mode: {mode!r}")


def _warn_in_order(message: str, stacklevel: int = 3) -> None:
    """Emit a RuntimeWarning to stdout so it stays in order with progress bars.

    Progress bars and status prints go to stdout; the default ``warnings.warn``
    writes to stderr.  Stdout and stderr are independent streams — flushing
    both does not guarantee display order at the terminal level.  This helper
    temporarily redirects warning output to stdout (the same stream as tqdm),
    which guarantees the warning appears immediately after the bar that
    triggered it.

    ``stacklevel=3`` (default) makes the reported call-site point to
    *_lyapunov*'s caller rather than to the helper itself — equivalent to
    ``stacklevel=2`` in a direct ``warnings.warn`` call inside ``_lyapunov``.
    """
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
    breakin_cycles: Optional[int] = None,
    min_cycles: int = 500,
    max_cycles: int = 2000,
    rtol: float = 0.0,
    epsilon: float = 1e-5,
    lyap_time_per_cycle: float = 0.5,
    convergence: str = "ky_dim",
    progress_bar: bool = False,
    probe_sub_cycle_cap: Optional[int] = None,
) -> dict:
    """Compute the Lyapunov spectrum of a dynamical system via QR-Benettin.

    Uses a modified Benettin algorithm with QR reorthonormalization. Growth
    factors are read from ``|diag(R)|`` of the QR step, which is equivalent
    to computing parallelotope volumes but simpler and numerically stabler.

    Parameters
    ----------
    model : :class:`ReservoirStepper` or equivalent
        A stepper object implementing the following contract (reservoirs via
        :class:`ReservoirStepper`; baseline ODE systems via a compatible
        wrapper):

        - ``state`` : array of shape ``(dim,)``, or list of ``m`` such arrays
          for multiple input states.  Readable and writable.
        - ``run(steps: int)`` : advance the model ``steps`` iterations,
          updating ``model.state`` in place.
        - ``t_step`` : float, default 1 — real time per iteration.
        - ``stdev`` : float, default 1 — attractor data standard deviation,
          used to scale perturbations.

    k : int
        Number of Lyapunov exponents to compute.
    cycle_length : int or None, optional
        Number of model steps between QR reorthonormalizations.  If ``None``
        (default), a 100-step probe estimates λ₁ and sets ``cycle_length``
        to approximately half the Lyapunov time.
    breakin_cycles : int or None, optional
        Cycles run before accumulation begins (discarded).  If ``None``
        (default), an adaptive scheme runs a 500-cycle break-in, checks
        spectrum ordering, and reruns up to two more times (1500 cycles
        total) if ordering is still violated.
    min_cycles : int, optional
        Minimum post-breakin cycles between convergence checks.  Default 500.
    max_cycles : int, optional
        Hard cap on total post-breakin cycles.  Default 5000.
    rtol : float, optional
        Relative convergence tolerance.  Default 0 runs to ``max_cycles``.
    epsilon : float, optional
        Relative perturbation magnitude; actual size is ``epsilon * stdev``.
    lyap_time_per_cycle : float, optional
        Target number of Lyapunov times spanned per QR cycle when
        ``cycle_length`` is auto-estimated from the λ₁ probe.  Ignored if
        ``cycle_length`` is supplied explicitly.  Default 0.5.
    convergence : str, optional
        Which quantity the ``rtol`` criterion is applied to.  One of
        ``"ky_dim"`` (default — falls back to ``"all"`` when KY is undefined),
        ``"lambda_1"`` (CI of λ₁ only), or ``"all"`` (max CI vs |λ₁|).
    progress_bar : bool, optional
        If True, show tqdm progress bars for the probe, break-in, and
        accumulation phases.  Default False.

    Returns
    -------
    dict with keys:

    - ``"spectrum"`` : array of shape ``(k,)`` — Lyapunov exponents, descending.
      ``nan`` for collapsed directions.
    - ``"spectrum_ci"`` : array of shape ``(k,)`` — 95 % CI half-widths.
    - ``"ky_dim"`` : float or None — Kaplan-Yorke dimension; ``None`` if the
      spectrum does not support the KY formula.
    - ``"ky_dim_ci"`` : float or None — 95 % CI half-width of KY dimension.
    - ``"n_cycles"`` : int — post-breakin cycles used.
    - ``"converged"`` : bool — True if ``rtol`` criterion was met.
    - ``"log_growths"`` : array of shape ``(n_cycles, k)`` — per-cycle log
      expansion/contraction factors; ``nan`` for collapsed directions.
    - ``"collapsed_directions"`` : bool array of shape ``(k,)`` — True for
      directions whose R_ii was saturated in >50% of cycles.
    - ``"cycle_length"`` : int — cycle_length used (auto or supplied).
    - ``"breakin_cycles"`` : int — breakin cycles actually run.
    - ``"lambda_1_probe"`` : float or None — λ₁ estimate from the probe;
      ``None`` if ``cycle_length`` was supplied explicitly.

    Notes
    -----
    For multiple input states (``m > 1``), each state collection is broken in
    independently, then cycled ``min_cycles // m`` times per batch before a
    shared convergence check.  ``max_cycles`` is the total across all
    collections.

    References
    ----------
    .. [1] Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J.-M. (1980).
           Lyapunov characteristic exponents for smooth dynamical systems and
           for Hamiltonian systems; a method for computing all of them.
           Meccanica, 15(1), 9-20.
    """
    raw_state = model.state
    if isinstance(raw_state, list):
        init_states = [np.asarray(s, dtype=float) for s in raw_state]
    else:
        init_states = [np.asarray(raw_state, dtype=float)]
    m = len(init_states)
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

    perturbed_states = []
    for s in init_states:
        dim = s.size
        collection = np.empty((k + 1, dim))
        collection[0] = s
        for j in range(k):
            collection[j + 1] = s.copy()
            collection[j + 1, j % dim] += pert_scale
        perturbed_states.append(collection)

    def _do_qr_cycles(n, record=False, pbar=None):
        """Run n QR-Benettin cycles over all m collections; optionally record log-growths."""
        buf = np.empty((n * m, k)) if record else None
        c = 0
        for _ in range(n):
            for i in range(m):
                for j in range(k + 1):
                    model.state = perturbed_states[i][j]
                    model.run(cycle_length)
                    perturbed_states[i][j] = np.asarray(model.state, dtype=float)
                fiducial = perturbed_states[i][0]
                error = perturbed_states[i][1:] - fiducial
                Q, R = np.linalg.qr(error.T)
                diag_R = np.abs(np.diag(R))
                diag_R = np.where(diag_R > 0, diag_R, np.finfo(float).tiny)
                if record:
                    buf[c] = np.log(diag_R / pert_scale)
                    c += 1
                signs = np.sign(np.diag(R))
                signs[signs == 0] = 1.0
                Q = Q * signs
                perturbed_states[i][1:] = fiducial + (Q[:, :k] * pert_scale).T
            if pbar is not None:
                pbar.update(1)
        return buf[:c] if record else None

    breakin_used = 0
    if breakin_cycles is not None:
        pbar_bi = tqdm(
            total=breakin_cycles,
            desc="  " + "perturbation breakin".ljust(24),
            unit="cyc",
            disable=not progress_bar,
            file=sys.stdout,
        )
        _do_qr_cycles(breakin_cycles, pbar=pbar_bi)
        breakin_used = breakin_cycles
        pbar_bi.close()
    else:
        _BREAKIN_CHUNK = 500
        _DIAG_CYCLES = 50
        _MAX_ATTEMPTS = 3
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            _bi_label = "perturbation breakin" if attempt == 1 else "perturbation breakin ext"
            pbar_bi = tqdm(
                total=_BREAKIN_CHUNK,
                desc="  " + _bi_label.ljust(24),
                unit="cyc",
                disable=not progress_bar,
                file=sys.stdout,
            )
            _do_qr_cycles(_BREAKIN_CHUNK, pbar=pbar_bi)
            breakin_used += _BREAKIN_CHUNK
            pbar_bi.close()
            sample = _do_qr_cycles(_DIAG_CYCLES, record=True)
            if not _ordering_violated(sample, cycle_length, t_step):
                break

    _SAT_LOG_THRESH = np.log(1e-12)
    log_growths = np.zeros((max_cycles, k))
    sat_count = np.zeros(k, dtype=int)
    n_cycles = 0
    converged = False
    collapsed = np.zeros(k, dtype=bool)

    cycles_per_batch = max(1, min_cycles // m)

    pbar = tqdm(
        total=max_cycles, desc="  " + "lyapunov spectrum finder", unit="cyc", disable=not progress_bar, file=sys.stdout
    )

    while n_cycles < max_cycles and not converged:
        batch = min(cycles_per_batch, max_cycles - n_cycles)
        for _ in range(batch):
            for i in range(m):
                if n_cycles >= max_cycles:
                    break
                for j in range(k + 1):
                    model.state = perturbed_states[i][j]
                    model.run(cycle_length)
                    perturbed_states[i][j] = np.asarray(model.state, dtype=float)

                fiducial = perturbed_states[i][0]
                if not np.isfinite(fiducial).all():
                    break

                error = perturbed_states[i][1:] - fiducial
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
                perturbed_states[i][1:] = fiducial + (Q[:, :k] * pert_scale).T

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

        ky_dim = _kaplan_yorke(spectrum)
        ky_dim_ci = _kaplan_yorke_ci(spectrum, ci_half_rate)

        if rtol > 0:
            converged = _is_converged(spectrum, ci_half_rate, ky_dim, ky_dim_ci, rtol, convergence)

    pbar.close()

    recorded = log_growths[:n_cycles].copy()
    recorded[:, collapsed] = np.nan

    with np.errstate(invalid="ignore"):
        spectrum = np.nanmean(recorded, axis=0) / (cycle_length * t_step)
    std_final = np.nanstd(recorded, axis=0, ddof=1) if n_cycles > 1 else np.full(k, np.inf)
    ci_half_rate = 1.96 * std_final / np.sqrt(max(n_cycles, 1)) / (cycle_length * t_step)

    ky_dim = _kaplan_yorke(spectrum)
    ky_dim_ci = _kaplan_yorke_ci(spectrum, ci_half_rate)

    if _ordering_violated(recorded, cycle_length, t_step):
        _warn_in_order(
            "Final Lyapunov spectrum has statistically significant out-of-order "
            "adjacent exponents — consider increasing max_cycles or breakin_cycles.",
        )

    finite_spec = spectrum[np.isfinite(spectrum)]
    if len(finite_spec) >= 2 and np.isfinite(spectrum[0]) and spectrum[0] != 0:
        n_near_zero = int(np.sum(np.abs(finite_spec) < 1e-3 * np.abs(spectrum[0])))
        if n_near_zero >= 2:
            _warn_in_order(
                f"{n_near_zero} Lyapunov exponents are within 1e-3·|λ₁| of zero; "
                "some may be spurious and the Kaplan-Yorke dimension may be off by one.",
            )

    return {
        "spectrum": spectrum,
        "spectrum_ci": ci_half_rate,
        "ky_dim": ky_dim,
        "ky_dim_ci": ky_dim_ci,
        "n_cycles": n_cycles,
        "converged": converged,
        "log_growths": recorded,
        "collapsed_directions": collapsed,
        "cycle_length": cycle_length,
        "breakin_cycles": breakin_used,
        "lambda_1_probe": lambda_1_probe,
    }


def _kaplan_yorke(spectrum: np.ndarray) -> Optional[float]:
    """Compute Kaplan-Yorke dimension from a Lyapunov spectrum.

    Returns ``None`` if the KY formula is not applicable: no positive
    cumulative sum, all cumulative sums positive, or the j+1 entry is
    nan/zero.  NaN entries (collapsed directions) contribute 0 to the
    cumulative sum so trailing collapsed directions don't prevent computation.
    """
    s = np.asarray(spectrum, dtype=float)
    cum = np.cumsum(np.where(np.isfinite(s), s, 0.0))
    pos = np.where(cum > 0)[0]
    if len(pos) == 0:
        return None
    j = pos[-1]
    if j + 1 >= len(s):
        return None
    if not np.isfinite(s[j + 1]) or s[j + 1] == 0:
        return None
    return float(j + 1) + cum[j] / np.abs(s[j + 1])


def _kaplan_yorke_ci(spectrum: np.ndarray, ci_half: np.ndarray) -> Optional[float]:
    """Propagate CI half-widths through the Kaplan-Yorke formula.

    Returns ``None`` in the same cases as ``_kaplan_yorke``.

    Notes
    -----
    With :math:`D_{KY} = j + 1 + \\sum_{i \\le j} \\lambda_i / |\\lambda_{j+1}|`,
    the propagated variance combines :math:`\\partial D / \\partial \\lambda_i =
    1 / |\\lambda_{j+1}|` for :math:`i \\le j` and
    :math:`\\partial D / \\partial \\lambda_{j+1} = -\\sum_{i \\le j} \\lambda_i
    / \\lambda_{j+1}^2`.
    """
    s = np.asarray(spectrum, dtype=float)
    cum = np.cumsum(np.where(np.isfinite(s), s, 0.0))
    pos = np.where(cum > 0)[0]
    if len(pos) == 0:
        return None
    j = pos[-1]
    if j + 1 >= len(s):
        return None
    lam_j1 = s[j + 1]
    if not np.isfinite(lam_j1) or lam_j1 == 0:
        return None
    ci = np.asarray(ci_half, dtype=float)
    var = float(np.nansum((ci[: j + 1] / np.abs(lam_j1)) ** 2))
    var += float((ci[j + 1] * cum[j] / lam_j1**2) ** 2)
    return float(np.sqrt(var))


# ---------------------------------------------------------------------------
# RC Lyapunov spectrum — public wrapper
# ---------------------------------------------------------------------------


def _flatten_model_state(model) -> np.ndarray:
    """Return a 1-D concatenation of every node's ``state["out"]`` vector."""
    return np.concatenate([np.asarray(n.state["out"], float).ravel() for n in model.nodes])


def _unflatten_model_state(model, flat: np.ndarray) -> None:
    """Write a flat state vector back into each node's ``state["out"]``."""
    idx = 0
    for n in model.nodes:
        d = n.state["out"].size
        end = idx + d
        n.state["out"] = flat[idx:end].reshape(n.state["out"].shape)
        idx = end


def _parse_init(init, kind: str):
    """Return a list of arrays ready for spin-up.

    For ``kind="trajectory"`` each element is shape ``(T, input_dim)``.
    For ``kind="state"`` each element is a 1-D flat array.
    """
    if isinstance(init, (list, tuple)):
        items = [np.asarray(x, float) for x in init]
    else:
        a = np.asarray(init, float)
        if kind == "trajectory":
            items = [a] if a.ndim <= 2 else list(a)
        else:
            items = [a] if a.ndim == 1 else list(a)
    if kind == "trajectory":
        items = [x[:, None] if x.ndim == 1 else x for x in items]
    return items


class ReservoirStepper:
    """Stepper that drives a trained reservoirpy :class:`Model` for :func:`_lyapunov`.

    This is the **primary** object fed to the QR-Benettin engine.  It wraps a
    trained model and exposes the stepper contract:

    - ``state`` — flat 1-D array (or list of arrays for multi-IC runs),
      readable and writable.
    - ``run(n_steps)`` — advance the model ``n_steps`` closed-loop steps,
      updating ``state`` in place.
    - ``t_step`` — real time per step (default 1.0; ``lyapunov()`` sets this
      from the model's training ``time_step`` HP if available).
    - ``stdev`` — attractor scale used to size perturbations; set by
      :meth:`probe_stdev` or supplied directly.

    Closed-loop stepping: if the model has delayed-feedback edges
    (``feedback_buffers`` is non-empty), each step calls
    ``model.run(iters=1)``, which routes output through the buffer
    automatically.  Otherwise the current output-node state is read and fed
    back as the next-step input — the common case of a plain
    ``Reservoir >> Ridge`` model without explicit feedback wiring.
    """

    t_step: float = 1.0

    def __init__(self, model, init_states: list, stdev: float = 1.0, t_step: float = 1.0):
        self._model = model
        self.state = init_states[0] if len(init_states) == 1 else init_states
        self.stdev = stdev
        self.t_step = t_step

    def run(self, n_steps: int) -> None:
        """Advance ``n_steps`` closed-loop steps; update ``self.state``."""
        _unflatten_model_state(self._model, np.asarray(self.state, float))
        for _ in range(n_steps):
            self._closed_loop_step()
        self.state = _flatten_model_state(self._model)

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
        _unflatten_model_state(self._model, saved)
        records = []
        for _ in range(n_probe):
            self._closed_loop_step()
            records.append(_flatten_model_state(self._model))
        _unflatten_model_state(self._model, saved)
        stdev = float(np.std(np.stack(records)))
        return stdev if stdev > 0 else 1.0


def lyapunov(
    model,
    init,
    kind: Literal["trajectory", "state"] = "trajectory",
    spinup: int = 200,
    n_spinup: Optional[int] = 1,
    k: Optional[int] = None,
    **lyap_kwargs,
) -> dict:
    """Compute the Lyapunov spectrum of a trained reservoir computer.

    Wraps :func:`_lyapunov` for use with a trained reservoirpy :class:`Model`
    via a :class:`ReservoirStepper`.  The model is driven in closed-loop
    (autoregressive) mode: at each step the output-node prediction is fed back
    as the next-step input.  Models with explicit feedback wiring (``<<``) and
    plain ``Reservoir >> Ridge`` models are both supported.

    Parameters
    ----------
    model : :class:`reservoirpy.Model`
        A trained reservoirpy model.  The standard construction
        ``Reservoir(...) >> Ridge(...)`` works directly; explicit feedback
        wiring (``<<``) is also supported.
    init : array-like or list of array-like
        Initialising data.  Interpretation depends on ``kind``:

        - ``"trajectory"`` (default): one or more input timeseries of shape
          ``(T, input_dim)`` or ``(T,)`` for scalar input.  Each is split into
          ``spinup``-length chunks; the model is run on each chunk to produce
          one initialised reservoir state.  Chunks shorter than ``spinup`` are
          discarded.
        - ``"state"``: one or more pre-computed flat state vectors of length
          equal to the total concatenated ``node.state["out"]`` dimension.
          Loaded directly without spin-up.
    kind : {"trajectory", "state"}, default "trajectory"
        How to interpret ``init``.  Use ``"trajectory"`` when you have
        training/validation data; ``"state"`` when you already have a
        post-spinup reservoir state.
    spinup : int, default 200
        Length of each spin-up chunk in timesteps.  Ignored when
        ``kind="state"``.
    n_spinup : int or None, default 1
        Number of spin-up trajectories (initial reservoir states) to use.
        ``1`` (default) uses a single state — the fastest option and sufficient
        for most purposes.  ``None`` uses every chunk that fits in ``init``,
        which can improve spectral averaging but multiplies cost by the number
        of chunks.  Ignored when ``kind="state"`` (all provided states are used
        unless capped here).
    k : int, optional
        Number of Lyapunov exponents to compute.  Defaults to the full flat
        state dimension (``Σ node.units``), which is the maximum meaningful
        value.
    **lyap_kwargs
        Additional keyword arguments forwarded to :func:`_lyapunov`
        (e.g. ``breakin_cycles``, ``min_cycles``, ``max_cycles``, ``rtol``,
        ``epsilon``, ``lyap_time_per_cycle``, ``progress_bar``).

    Returns
    -------
    dict
        Same output as :func:`_lyapunov`: ``"spectrum"``, ``"spectrum_ci"``,
        ``"ky_dim"``, ``"ky_dim_ci"``, ``"n_cycles"``, ``"converged"``,
        ``"log_growths"``, ``"collapsed_directions"``, ``"cycle_length"``,
        ``"breakin_cycles"``, ``"lambda_1_probe"``.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> from reservoirpy.datasets import lorenz
    >>> from reservoirpy.observables import lyapunov
    >>> data = lorenz(12000)
    >>> x_train, y_train = data[:10000], data[1:10001]
    >>> esn = Reservoir(300, sr=0.95) >> Ridge(ridge=1e-6)
    >>> esn = esn.fit(x_train, y_train)
    >>> result = lyapunov(esn, init=x_train, spinup=200, k=3,
    ...                   min_cycles=1000, max_cycles=2000)
    >>> print(result["spectrum"])
    """
    items = _parse_init(init, kind)
    init_states = []

    for item in items:
        if kind == "state":
            init_states.append(np.asarray(item, float).ravel())
        else:
            for i in range(0, len(item) - spinup + 1, spinup):
                end = i + spinup
                chunk = item[i:end]
                model.reset()
                model.run(chunk)
                init_states.append(_flatten_model_state(model))
                if n_spinup is not None and len(init_states) >= n_spinup:
                    break
        if n_spinup is not None and len(init_states) >= n_spinup:
            break

    if not init_states:
        raise ValueError("init produced zero spin-up states — " "are all trajectories shorter than spinup?")

    stepper = ReservoirStepper(model, init_states)
    if kind == "trajectory":
        stepper.stdev = stepper.probe_stdev()

    if k is None:
        k = init_states[0].size

    return _lyapunov(stepper, k=k, **lyap_kwargs)
