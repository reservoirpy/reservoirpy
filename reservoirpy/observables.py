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
    valid_time
    valid_time_multitest
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import sys
import time
import warnings
from copy import deepcopy
from typing import Literal, Optional, Union

import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs

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


def _estimate_lambda_1(model, epsilon, n_probe=1000, sub_cycle=5,
                       probe_sub_cycle_cap=None,
                       progress_bar=False, progress_verbose=False):
    """Estimate the largest Lyapunov exponent via a two-pass Benettin probe.

    Pass 1 (fast): ``sub_cycle=5`` — good for high-λ ODE systems.
    Pass 2 (slow): ``sub_cycle = max(50, state_dim)`` — each window spans at
    least one full ring-buffer traversal, so the DDE delayed-feedback channel
    has time to act before renormalization resets the perturbation direction.
    The first half of windows is discarded as alignment transient in both passes.
    Returns ``None`` only if both passes give a non-positive or diverging result.
    Leaves ``model.state`` restored to its pre-probe value.

    ``probe_sub_cycle_cap`` caps the slow-pass sub_cycle (useful for large-state
    models like ESNs where the state dimension is not a ring-buffer length).
    """
    t_step = getattr(model, "t_step", 1.0)
    stdev = getattr(model, "stdev", 1.0)
    pert_scale = epsilon * stdev

    # If model.state is a list (multiple initial conditions), probe from the
    # first element only.  _lyapunov saves init_states before calling this so
    # the list is unchanged for the actual computation.
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

        if progress_bar:
            total_steps = n_windows * sub_cyc * 2   # fid + pert each window
            print(f"    {label}:  {n_windows} renorms × {sub_cyc} steps "
                  f"({discard} transient / {n_windows - discard} kept)"
                  f"  = {total_steps} steps total",
                  flush=True)
        bar = (_LyapProgress(n_windows, label, cycles_per_mark=1,
                             verbose=progress_verbose)
               if progress_bar else None)

        for w in range(n_windows):
            model.state = fid
            model.evolve(sub_cyc)
            fid = np.asarray(model.state, dtype=float)
            model.state = pert
            model.evolve(sub_cyc)
            pert = np.asarray(model.state, dtype=float)
            delta = pert - fid
            d = np.linalg.norm(delta)
            if not np.isfinite(d) or d == 0.0:
                if bar:
                    bar.close()
                return None, fid
            log_growths[w] = np.log(d / pert_scale)
            pert = fid + delta * (pert_scale / d)
            if bar:
                inst = log_growths[w] / sub_cyc / t_step
                phase = "transient" if w < discard else "kept    "
                bar.set_metric(
                    f"log(d/ε)={log_growths[w]:+.3f}  "
                    f"λ₁(inst)={inst:+.5f}/step  [{phase}]"
                )
                bar.update(1)

        if bar:
            bar.close()
        # discard first half as alignment transient; mean over second half
        lam = float(log_growths[discard:].mean()) / sub_cyc
        return (lam / t_step if lam > 0 else None), fid

    result_fast, fid = _run_pass(sub_cycle, label="probe-fast")
    # always run slow pass — dim-aware window size handles DDE/PDE ring-buffer states
    model.state = fid
    dim = fid.size
    # min_windows=200 → 100 recording windows after half-discard, enough to average
    # out Lorenz-scale variance while still spanning the full ring buffer for DDE/PDE
    slow_sub = max(50, dim)
    if probe_sub_cycle_cap is not None:
        slow_sub = min(slow_sub, probe_sub_cycle_cap)
    result_slow, fid = _run_pass(slow_sub, min_windows=200, label="probe-slow")
    model.state = fid
    # prefer the slow-pass result; fall back to fast; return None if both fail
    results = [r for r in [result_slow, result_fast] if r is not None]
    return results[0] if results else None


def _ordering_violated(log_growths, cycle_length, t_step):
    """Return True if the spectrum from log_growths has statistically significant
    out-of-order adjacent pairs (max adjacent violation criterion).
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


_PROGRESS_INTERVAL = 50


def _progress_bar(done: int, total: int, done_flag: bool, phase: str = "accum") -> None:
    width = 30
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    tag = " converged" if done_flag else ""
    sys.stderr.write(f"\r  {phase} [{bar}] {done}/{total}{tag}")
    sys.stderr.flush()


class _LyapProgress:
    """Plain-stdout progress bar: a ruler line of '_' then '|' marks, one mark
    per ``cycles_per_mark`` cycles.  Times itself for s/cycle + ETA reporting.

    Non-verbose output example (total=2000, cpm=100 → 20 marks)::

        Breakin: 2000 cycles, 1 '|' = 100 cycles
        ____________________
        ||||||||||||||||||||

    Verbose output: one status line printed per mark (count, elapsed,
    ms/cyc, optional metric string).
    """

    def __init__(self, total: int, label: str,
                 cycles_per_mark: int = 100, verbose: bool = False) -> None:
        self.total = max(1, int(total))
        self.cpm = max(1, int(cycles_per_mark))
        self.n_marks = max(1, -(-self.total // self.cpm))  # ceiling division
        self.count = 0
        self.marks = 0
        self.verbose = verbose
        self.metric = ""
        self.label = label
        self.t0 = time.perf_counter()
        print(f"  {label}: {self.total} cycles, 1 '|' = {self.cpm} cycles",
              flush=True)
        print("  " + "_" * self.n_marks, flush=True)
        if not verbose:
            sys.stdout.write("  ")
            sys.stdout.flush()

    def set_metric(self, text: str) -> None:
        self.metric = text

    def update(self, n: int = 1) -> None:
        self.count += n
        tgt = min(self.n_marks, self.count // self.cpm)
        while self.marks < tgt:
            self.marks += 1
            if self.verbose:
                elapsed = self.elapsed
                spc = self.per_cycle
                print(
                    f"  [{self.label}] cyc {self.count}/{self.total}  "
                    f"mark {self.marks}/{self.n_marks}  "
                    f"elapsed {elapsed:5.1f}s  {spc * 1e3:6.2f} ms/cyc"
                    + (f"  {self.metric}" if self.metric else ""),
                    flush=True,
                )
            else:
                sys.stdout.write("|")
                sys.stdout.flush()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @property
    def per_cycle(self) -> float:
        return self.elapsed / self.count if self.count else float("nan")

    def close(self) -> None:
        if not self.verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _lyapunov(
    model,
    k: int,
    cycle_length: Optional[int] = None,
    breakin_cycles: Optional[int] = None,
    min_cycles: int = 500,
    max_cycles: int = 2000,
    rtol: float = 0.01,
    epsilon: float = 1e-5,
    lyap_time_per_cycle: float = 0.5,
    convergence: str = "ky_dim",
    display: bool = False,
    progress_bar: bool = False,
    progress_verbose: bool = False,
    probe_sub_cycle_cap: Optional[int] = None,
) -> dict:
    """Compute the Lyapunov spectrum of a dynamical system via QR-Benettin.

    Uses a modified Benettin algorithm with QR reorthonormalization. Growth
    factors are read from ``|diag(R)|`` of the QR step, which is equivalent
    to computing parallelotope volumes but simpler and numerically stabler.

    Parameters
    ----------
    model : object
        Duck-typed model with the following attributes:

        - ``state`` : array of shape ``(dim,)``, or list of ``m`` such arrays
          for multiple input states.  Readable and writable.
        - ``evolve(steps: int)`` : advance the model ``steps`` iterations,
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
        (default), an adaptive scheme runs 200-cycle chunks until spectrum
        ordering is satisfied, up to a hard cap of 1000 cycles.
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
    display : bool, optional
        If True, show a growth-factor time series and a convergence plot.

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
    # --- parse multi/single-state input ---
    raw_state = model.state
    if isinstance(raw_state, list):
        init_states = [np.asarray(s, dtype=float) for s in raw_state]
    else:
        init_states = [np.asarray(raw_state, dtype=float)]
    m = len(init_states)
    t_step = getattr(model, "t_step", 1.0)
    stdev = getattr(model, "stdev", 1.0)
    pert_scale = epsilon * stdev

    # --- probe to auto-set cycle_length ---
    if cycle_length is None:
        if progress_bar:
            _cap_str = (f", cap={probe_sub_cycle_cap}" if probe_sub_cycle_cap
                        else "")
            print(f"  [probe] estimating λ₁  (fast sub_cycle=5 + slow{_cap_str})",
                  flush=True)
        lambda_1_probe = _estimate_lambda_1(
            model, epsilon, probe_sub_cycle_cap=probe_sub_cycle_cap,
            progress_bar=progress_bar, progress_verbose=progress_verbose)
        if lambda_1_probe is None or lambda_1_probe <= 0:
            cycle_length = 100
        else:
            cycle_length = int(np.ceil(lyap_time_per_cycle / (lambda_1_probe * t_step)))
        if display or progress_bar:
            probe_str = (f"{lambda_1_probe:.5f}" if lambda_1_probe is not None
                         else "None (fallback)")
            steps_per_qr = cycle_length * (k + 1)
            print(f"  [probe] λ₁ ≈ {probe_str}/step  →  cycle_length={cycle_length}"
                  f"  ({steps_per_qr} steps/QR-cycle, k+1={k+1} states)",
                  flush=True)
    else:
        lambda_1_probe = None

    # --- init perturbed states ---
    # perturbed_states[i]: shape (k+1, dim), index 0 = fiducial
    perturbed_states = []
    for s in init_states:
        dim = s.size
        collection = np.empty((k + 1, dim))
        collection[0] = s
        for j in range(k):
            collection[j + 1] = s.copy()
            collection[j + 1, j % dim] += pert_scale
        perturbed_states.append(collection)

    def _do_qr_cycles(n, record=False):
        """Run n QR-Benettin cycles over all m collections; optionally record log-growths."""
        buf = np.empty((n * m, k)) if record else None
        c = 0
        for _ in range(n):
            for i in range(m):
                for j in range(k + 1):
                    model.state = perturbed_states[i][j]
                    model.evolve(cycle_length)
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
        return buf[:c] if record else None

    # --- plain-stdout progress bar factory (replaces tqdm) ---
    _make_bar = (
        lambda total, label: _LyapProgress(  # noqa: E731
            total, label, cycles_per_mark=100, verbose=progress_verbose
        )
    ) if progress_bar else None

    # --- adaptive or fixed breakin ---
    breakin_used = 0
    breakin_secs: Optional[float] = None
    if breakin_cycles is not None:
        b = _make_bar(breakin_cycles, "Breakin") if _make_bar else None
        remaining = breakin_cycles
        while remaining > 0:
            chunk = min(remaining, _PROGRESS_INTERVAL)
            _do_qr_cycles(chunk)
            breakin_used += chunk
            remaining -= chunk
            if display:
                _progress_bar(breakin_used, breakin_cycles, False, "breakin")
            if b is not None:
                b.update(chunk)
        if display:
            sys.stderr.write("\n")
            sys.stderr.flush()
        if b is not None:
            breakin_secs = b.elapsed
            b.close()
    else:
        _BREAKIN_CHUNK = 200
        _DIAG_CYCLES = 50
        _MAX_BREAKIN = 1000
        b = _make_bar(_MAX_BREAKIN, "Breakin") if _make_bar else None
        while breakin_used < _MAX_BREAKIN:
            _do_qr_cycles(_BREAKIN_CHUNK)
            breakin_used += _BREAKIN_CHUNK
            sample = _do_qr_cycles(_DIAG_CYCLES, record=True)
            violated = _ordering_violated(sample, cycle_length, t_step)
            if display:
                ok_str = "ok" if not violated else "violated"
                _progress_bar(breakin_used, _MAX_BREAKIN, not violated,
                               f"breakin  order={ok_str}")
            if b is not None:
                b.update(_BREAKIN_CHUNK)
            if not violated:
                break
        if display:
            sys.stderr.write("\n")
            sys.stderr.flush()
        if b is not None:
            breakin_secs = b.elapsed
            b.close()

    # --- ETA estimate before accumulation ---
    if progress_bar and breakin_used > 0 and breakin_secs is not None:
        spc = breakin_secs / breakin_used
        print(
            f"  Breakin: {breakin_used} cycles in {breakin_secs:.1f}s "
            f"({spc * 1e3:.2f} ms/cyc, cycle_length={cycle_length})"
            f"  →  est. accumulation ≤ {spc * max_cycles:.1f}s "
            f"for {max_cycles} cycles",
            flush=True,
        )

    # --- accumulation phase ---
    _SAT_LOG_THRESH = np.log(1e-12)
    log_growths = np.zeros((max_cycles, k))
    sat_count = np.zeros(k, dtype=int)
    n_cycles = 0
    converged = False
    collapsed = np.zeros(k, dtype=bool)

    cycles_per_batch = max(1, min_cycles // m)

    acc = _make_bar(max_cycles, "Lyapunov") if _make_bar else None

    while n_cycles < max_cycles and not converged:
        batch = min(cycles_per_batch, max_cycles - n_cycles)
        for _ in range(batch):
            for i in range(m):
                if n_cycles >= max_cycles:
                    break
                for j in range(k + 1):
                    model.state = perturbed_states[i][j]
                    model.evolve(cycle_length)
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
                if acc is not None:
                    acc.update(1)
                if display and n_cycles % _PROGRESS_INTERVAL == 0:
                    _progress_bar(n_cycles, max_cycles, False)

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
        std = (np.nanstd(recorded, axis=0, ddof=1) if n_cycles > 1
               else np.full(k, np.inf))
        ci_half = 1.96 * std / np.sqrt(n_cycles)
        ci_half_rate = ci_half / (cycle_length * t_step)

        ky_dim = _kaplan_yorke(spectrum)
        ky_dim_ci = _kaplan_yorke_ci(spectrum, ci_half_rate)

        if acc is not None and np.isfinite(spectrum[0]):
            acc.set_metric(
                f"λ₁={spectrum[0] * t_step:+.4f}/step  conv={converged}"
            )

        if rtol > 0:
            converged = _is_converged(
                spectrum, ci_half_rate, ky_dim, ky_dim_ci, rtol, convergence
            )

    if acc is not None:
        acc.close()
    if display:
        _progress_bar(n_cycles, max_cycles, converged)
        sys.stderr.write("\n")
        sys.stderr.flush()

    # --- finalize ---
    recorded = log_growths[:n_cycles].copy()
    recorded[:, collapsed] = np.nan

    with np.errstate(invalid="ignore"):
        spectrum = np.nanmean(recorded, axis=0) / (cycle_length * t_step)
    std_final = (np.nanstd(recorded, axis=0, ddof=1) if n_cycles > 1
                 else np.full(k, np.inf))
    ci_half_rate = (1.96 * std_final / np.sqrt(max(n_cycles, 1))
                    / (cycle_length * t_step))

    ky_dim = _kaplan_yorke(spectrum)
    ky_dim_ci = _kaplan_yorke_ci(spectrum, ci_half_rate)

    if _ordering_violated(recorded, cycle_length, t_step):
        warnings.warn(
            "Final Lyapunov spectrum has statistically significant out-of-order "
            "adjacent exponents — consider increasing max_cycles or breakin_cycles.",
            RuntimeWarning,
            stacklevel=2,
        )

    finite_spec = spectrum[np.isfinite(spectrum)]
    if len(finite_spec) >= 2 and np.isfinite(spectrum[0]) and spectrum[0] != 0:
        n_near_zero = int(np.sum(np.abs(finite_spec) < 1e-3 * np.abs(spectrum[0])))
        if n_near_zero >= 2:
            warnings.warn(
                f"{n_near_zero} Lyapunov exponents are within 1e-3·|λ₁| of zero; "
                "some may be spurious and the Kaplan-Yorke dimension may be off by one.",
                RuntimeWarning,
                stacklevel=2,
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
    # partial derivative of D_KY = j+1 + sum(lam_0..j)/|lam_{j+1}|
    # d/d(lam_i for i<=j) = 1/|lam_{j+1}|, d/d(lam_{j+1}) = -cum[j]/lam_{j+1}^2
    ci = np.asarray(ci_half, dtype=float)
    var = float(np.nansum((ci[: j + 1] / np.abs(lam_j1)) ** 2))
    var += float((ci[j + 1] * cum[j] / lam_j1**2) ** 2)
    return float(np.sqrt(var))


# ---------------------------------------------------------------------------
# Valid prediction time
# ---------------------------------------------------------------------------


def _data_std(data: np.ndarray) -> float:
    """RMS Euclidean deviation from the mean across a trajectory.

    For univariate data this equals the standard deviation.  For multivariate
    data it is :math:`\\sqrt{\\mathbb{E}[\\|x - \\bar{x}\\|_2^2]}`, matching
    the normalisation used in :func:`valid_time_multitest`.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    mean = arr.mean(axis=0)
    return float(np.sqrt(np.mean(np.linalg.norm(arr - mean, axis=-1) ** 2)))


def _valid_time(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    norm_std: float,
    eps: Union[float, list] = None,
    ks: Optional[list] = None,
) -> dict:
    """Measure valid prediction time and normalised RMSE against ground truth.

    Parameters
    ----------
    y_pred : array-like of shape (T, d)
        Predicted trajectory.
    y_true : array-like of shape (T, d)
        Ground-truth trajectory.
    norm_std : float
        RMS Euclidean scale of the data (e.g. from :func:`_data_std`).
        Thresholds are expressed as multiples of this value.
    eps : float or list of float, optional
        Threshold multiples.  Valid prediction time is the first step at
        which the per-step error exceeds ``eps * norm_std``.
        Defaults to ``[0.2, 0.4]``.
    ks : list of int, optional
        Horizons for normalised RMSE computation.  For each ``k``, the
        RMSE over the first ``k`` steps is divided by ``norm_std``.
        If ``k`` exceeds the trajectory length the available steps are
        used.  If None (default), no RMSE is computed.

    Returns
    -------
    dict
        Keys:

        * ``"valid_time_<eps>"`` – first step index at which the error
          exceeds ``eps * norm_std``; equals ``len(y_pred)`` if the
          threshold is never crossed (censored).
        * ``"valid_time_<eps>_reached"`` – ``True`` if the threshold was
          actually crossed.
        * ``"rmse_<k>"`` – normalised RMSE over the first ``k`` steps
          (only present if ``ks`` is not None).
        * ``"all_thresholds_reached"`` – ``True`` if every threshold in
          ``eps`` was crossed.
    """
    y_pred_arr, y_true_arr = _check_arrays(y_pred, y_true)
    if eps is None:
        eps = [0.2, 0.4]
    eps_list = [eps] if isinstance(eps, (int, float)) else list(eps)

    # Per-step Euclidean distance
    if y_pred_arr.ndim == 1:
        err = np.abs(y_pred_arr - y_true_arr)
    else:
        err = np.linalg.norm(y_pred_arr - y_true_arr, axis=-1)

    result = {}
    n_reached = 0
    for e in eps_list:
        threshold = float(e) * norm_std
        exceeded = err > threshold
        if exceeded.any():
            result[f"valid_time_{e}"] = int(np.argmax(exceeded))
            result[f"valid_time_{e}_reached"] = True
            n_reached += 1
        else:
            result[f"valid_time_{e}"] = len(err)  # censored — threshold never crossed
            result[f"valid_time_{e}_reached"] = False
    result["all_thresholds_reached"] = n_reached == len(eps_list)

    if ks is not None:
        for k in ks:
            k_eff = min(int(k), len(err))
            rms_k = float(np.sqrt(np.mean(err[:k_eff] ** 2)))
            result[f"rmse_{k}"] = rms_k / norm_std if norm_std > 0 else rms_k

    return result


def _rc_run_generative(
    model, n_steps: int, prepend_current: bool = False
) -> np.ndarray:
    """Run a trained RC model in closed-loop for ``n_steps`` timesteps.

    Supports two common topologies:

    * **Explicit feedback** (``model.feedback_buffers`` non-empty, e.g.
      ``ESN(feedback=True)`` trained with a zero-dimensional external input):
      calls ``model.run(iters=1)`` each step, letting the framework route the
      delayed readout output back to the reservoir automatically.
    * **Manual feed-back** (plain ``Reservoir >> Ridge`` or any model whose
      reservoir gets its external input, not a feedback buffer): reads the
      readout's current output state and feeds it back as the next-step input
      via ``model.run(out.reshape(1, -1))``.  This is the standard generative
      pattern for 1-step-ahead prediction models.

    Parameters
    ----------
    model : reservoirpy Model
        A trained model whose state has already been advanced (e.g. via a
        warmup ``model.run(x_spinup)`` call).
    n_steps : int
        Number of prediction steps to return.
    prepend_current : bool, optional
        When ``True``, the readout's *current* output state (``ŷ[S]``,
        already computed during warmup) is recorded as the first element of
        the returned array before any new ``model.run`` calls are made, and
        only ``n_steps - 1`` additional steps are generated.  Use this on the
        first call after warmup so that ``preds[0]`` corresponds to the same
        timestep as ``val_traj[spinup]``.  Default is ``False`` (legacy
        behaviour: generate ``n_steps`` brand-new steps starting from
        ``ŷ[S+1]``).

    Returns
    -------
    np.ndarray of shape ``(n_steps, output_dim)``
        Predicted trajectory.
    """
    fb = getattr(model, "feedback_buffers", None)
    preds = []

    if prepend_current:
        # Record the post-warmup readout output ŷ[S] as the seed prediction.
        seed_out = np.concatenate(
            [np.asarray(n.state["out"], float).ravel() for n in model.outputs]
        )
        preds.append(seed_out)
        n_iters = n_steps - 1
    else:
        n_iters = n_steps

    for _ in range(n_iters):
        if fb:
            out = np.asarray(model.run(iters=1), dtype=float)
        else:
            prev_out = np.concatenate(
                [np.asarray(n.state["out"], float).ravel() for n in model.outputs]
            )
            out = np.asarray(model.run(prev_out.reshape(1, -1)), dtype=float)
        preds.append(out.reshape(-1))
    return np.stack(preds, axis=0)


def valid_time(
    model: "Model",
    val_traj: np.ndarray,
    spinup: int = 200,
    eps: Union[float, list] = None,
    ks: Optional[list] = None,
    norm_std: Optional[float] = None,
    block: int = 300,
    t_step: float = 1.0,
) -> dict:
    """Measure valid prediction time for a trained reservoir-computing model.

    The model is warmed up on the first ``spinup`` steps of ``val_traj`` by
    running it open-loop.  It then predicts autoregressively (closed-loop)
    and the predicted trajectory is compared to ``val_traj[spinup:]`` using
    :func:`_valid_time`.  Prediction extends in blocks of ``block`` steps
    until every threshold in ``eps`` has been crossed or the end of
    ``val_traj`` is reached.

    The model must support closed-loop (generative) prediction via
    ``model.run(iters=n)`` after a warmup ``model.run(x_spinup)``.  An
    :class:`~reservoirpy.ESN` built with ``feedback=True`` satisfies this.

    Parameters
    ----------
    model : :class:`reservoirpy.Model`
        A trained reservoirpy model.  Not modified; a deep copy is used.
    val_traj : array-like of shape (T, d)
        Validation trajectory.  The first ``spinup`` steps warm up the
        model; the remainder provide ground truth.
    spinup : int, optional
        Number of open-loop warmup steps.  Default is 200.
    eps : float or list of float, optional
        Threshold multiples for valid prediction time.
        Defaults to ``[0.2, 0.4]``.
    ks : list of int, optional
        Horizons for normalised RMSE.  See :func:`_valid_time`.
    norm_std : float, optional
        Pre-computed data scale.  If None, computed from ``val_traj`` via
        :func:`_data_std`.
    block : int, optional
        Number of prediction steps per extension block.  Default is 300.
    t_step : float, optional
        Physical time per step.  Valid times in the returned dict are
        multiplied by ``t_step``.  Default is 1.0 (steps).

    Returns
    -------
    dict
        Result from :func:`_valid_time` with valid-time values scaled by
        ``t_step``.

    Examples
    --------
    >>> from reservoirpy import ESN
    >>> from reservoirpy.datasets import lorenz, to_forecasting
    >>> data = lorenz(10000)
    >>> x_train, _, y_train, _ = to_forecasting(data[:5000], test_size=0.0)
    >>> model = ESN(units=500, sr=0.9, feedback=True, ridge=1e-6)
    >>> model.fit(x_train, y_train)
    >>> from reservoirpy.observables import valid_time
    >>> vt = valid_time(model, data[5000:], spinup=200, eps=0.2)
    """
    val_arr = np.asarray(val_traj, dtype=float)
    if norm_std is None:
        norm_std = _data_std(val_arr)
    if eps is None:
        eps = [0.2, 0.4]
    eps_list = [eps] if isinstance(eps, (int, float)) else list(eps)

    truth = val_arr[spinup:]
    n_truth = len(truth)

    model_copy = deepcopy(model)
    if spinup > 0:
        model_copy.run(val_arr[:spinup])  # warm up; reservoir state persists

    pred_chunks = []
    offset = 0
    result = None
    first_block = True

    while offset < n_truth:
        steps = min(block, n_truth - offset)
        chunk = _rc_run_generative(model_copy, steps, prepend_current=first_block)
        first_block = False
        pred_chunks.append(chunk)
        offset += steps
        y_pred = np.concatenate(pred_chunks, axis=0)
        result = _valid_time(y_pred, truth[:offset], norm_std=norm_std, eps=eps, ks=ks)
        if result["all_thresholds_reached"]:
            break

    if result is None:
        # val_traj too short for any prediction steps
        result = {f"valid_time_{e}": 0 for e in eps_list}
        result.update({f"valid_time_{e}_reached": False for e in eps_list})
        result["all_thresholds_reached"] = False

    if t_step != 1.0:
        for e in eps_list:
            result[f"valid_time_{e}"] = result[f"valid_time_{e}"] * t_step

    return result


def valid_time_multitest(
    model: "Model",
    val_data: Union[np.ndarray, list],
    n_segments: int,
    spinup: int = 200,
    eps: Union[float, list] = None,
    ks: Optional[list] = None,
    block: int = 300,
    t_step: float = 1.0,
) -> dict:
    """Estimate valid prediction time over multiple validation segments.

    The full ``val_data`` is used to compute a single normalisation scale,
    ensuring consistent thresholds across segments.  The data are then split
    into ``n_segments`` contiguous pieces and :func:`valid_time` is called on
    each.  Summary statistics are aggregated across segments.

    This mirrors the multi-trial aggregation in ``loss_funcs.PredictionLoss``
    from the ``reservoir`` framework: geometric-mean RMSE, mean and median
    valid times, and a ``"valid_time_fitness"`` headline equal to
    ``max(medians)`` across thresholds.

    Parameters
    ----------
    model : :class:`reservoirpy.Model`
        A trained reservoirpy model.  Not modified; deep copies are used.
    val_data : array-like of shape (T, d) or list of arrays
        Validation data.  If a single array it is split into ``n_segments``
        contiguous pieces of equal length.  If a list, the first
        ``n_segments`` elements are used as-is.
    n_segments : int
        Number of validation segments.
    spinup : int, optional
        Open-loop warmup steps per segment.  Default is 200.
    eps : float or list of float, optional
        Threshold multiples.  Defaults to ``[0.2, 0.4]``.
    ks : list of int, optional
        Horizons for normalised RMSE.  See :func:`_valid_time`.
    block : int, optional
        Prediction extension block size.  Default is 300.
    t_step : float, optional
        Physical time per step for valid-time scaling.  Default is 1.0.

    Returns
    -------
    dict
        Keys:

        * ``"valid_time_<eps>_mean"`` / ``"valid_time_<eps>_median"`` –
          mean and median valid times across segments (in units of
          ``t_step``).
        * ``"valid_time_<eps>_fraction_reached"`` – fraction of segments
          where the threshold was actually crossed.
        * ``"rmse_<k>_gmean"`` – geometric mean of normalised RMSE across
          segments (only if ``ks`` is not None).
        * ``"valid_time_fitness"`` – ``max`` of the per-threshold medians.
          Equivalent to ``prediction valid time fitness`` in
          ``loss_funcs.PredictionLoss``.

    Examples
    --------
    >>> from reservoirpy import ESN
    >>> from reservoirpy.datasets import lorenz
    >>> from reservoirpy.datasets import to_forecasting
    >>> from reservoirpy.observables import valid_time_multitest
    >>> data = lorenz(30000)
    >>> x_tr, _, y_tr, _ = to_forecasting(data[:20000], test_size=0.0)
    >>> model = ESN(units=500, sr=0.9, feedback=True, ridge=1e-6)
    >>> model.fit(x_tr, y_tr)
    >>> results = valid_time_multitest(model, data[20000:], n_segments=10)
    >>> print(f"Valid time fitness: {results['valid_time_fitness']:.2f}")
    """
    if isinstance(val_data, np.ndarray):
        all_arr = val_data.astype(float)
    else:
        all_arr = np.concatenate([np.asarray(s, dtype=float) for s in val_data], axis=0)

    norm_std = _data_std(all_arr)

    if eps is None:
        eps = [0.2, 0.4]
    eps_list = [eps] if isinstance(eps, (int, float)) else list(eps)

    # Build segment list
    if isinstance(val_data, (list, tuple)):
        segments = [np.asarray(s, dtype=float) for s in val_data[:n_segments]]
    else:
        seg_len = len(all_arr) // n_segments
        segments = [all_arr[i * seg_len: (i + 1) * seg_len] for i in range(n_segments)]

    # Run valid_time on each segment with the global norm_std
    seg_results = [
        valid_time(
            model, seg, spinup=spinup, eps=eps, ks=ks,
            norm_std=norm_std, block=block, t_step=t_step,
        )
        for seg in segments
    ]

    output = {}

    # Aggregate valid times (mean, median, fraction reached)
    medians = []
    for e in eps_list:
        key = f"valid_time_{e}"
        times = np.array([r[key] for r in seg_results], dtype=float)
        output[f"{key}_mean"] = float(np.mean(times))
        med = float(np.median(times))
        output[f"{key}_median"] = med
        medians.append(med)
        reached = np.array([r[f"{key}_reached"] for r in seg_results])
        output[f"{key}_fraction_reached"] = float(np.mean(reached))

    output["valid_time_fitness"] = float(max(medians)) if medians else float("nan")

    # Aggregate RMSE via geometric mean across segments
    if ks is not None:
        for k in ks:
            key = f"rmse_{k}"
            vals = np.array([r.get(key, np.nan) for r in seg_results], dtype=float)
            valid_vals = vals[np.isfinite(vals) & (vals > 0)]
            output[f"{key}_gmean"] = (
                float(np.exp(np.mean(np.log(valid_vals))))
                if len(valid_vals) > 0
                else float("nan")
            )

    return output


# ---------------------------------------------------------------------------
# RC Lyapunov spectrum — public wrapper
# ---------------------------------------------------------------------------


def _flatten_model_state(model) -> np.ndarray:
    """Return a 1-D concatenation of every node's ``state["out"]`` vector."""
    return np.concatenate(
        [np.asarray(n.state["out"], float).ravel() for n in model.nodes]
    )


def _unflatten_model_state(model, flat: np.ndarray) -> None:
    """Write a flat state vector back into each node's ``state["out"]``."""
    idx = 0
    for n in model.nodes:
        d = n.state["out"].size
        n.state["out"] = flat[idx: idx + d].reshape(n.state["out"].shape)
        idx += d


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


def _rc_closed_loop_step(model) -> None:
    """Advance a trained RC one step in closed-loop (autoregressive) mode.

    If the model has delayed-feedback edges (feedback_buffers is non-empty),
    use ``model.run(iters=1)`` which routes output through the buffer
    automatically.  Otherwise read the current output-node state and feed it
    back manually as the next-step input — this covers the common case of a
    plain ``Reservoir >> Ridge`` model without explicit feedback wiring.
    """
    fb = getattr(model, "feedback_buffers", None)
    if fb:
        model.run(iters=1)
    else:
        pred = np.concatenate(
            [np.asarray(n.state["out"], float).ravel() for n in model.outputs]
        )
        model.run(pred.reshape(1, -1))


def _probe_stdev(model, n_probe: int = 200) -> float:
    """Estimate attractor stdev from a short closed-loop run."""
    records = []
    for _ in range(n_probe):
        _rc_closed_loop_step(model)
        records.append(_flatten_model_state(model))
    stdev = float(np.std(np.stack(records)))
    return stdev if stdev > 0 else 1.0


class _RCAdapter:
    """Duck-typed wrapper that exposes a trained reservoirpy Model to ``_lyapunov``.

    ``_lyapunov`` sets ``adapter.state``, calls ``adapter.evolve(steps)``,
    then reads ``adapter.state`` back.  This class translates those operations
    to ``_unflatten_model_state`` / closed-loop stepping /
    ``_flatten_model_state``.
    """

    t_step: float = 1.0

    def __init__(self, model, init_states: list, stdev: float):
        self._model = model
        self.state = init_states if len(init_states) > 1 else init_states[0]
        self.stdev = stdev

    def evolve(self, steps: int) -> None:
        _unflatten_model_state(self._model, np.asarray(self.state, float))
        for _ in range(steps):
            _rc_closed_loop_step(self._model)
        self.state = _flatten_model_state(self._model)


def lyapunov(
    model,
    init,
    kind: Literal["trajectory", "state"] = "trajectory",
    spinup: int = 200,
    n_spinup: Optional[int] = 1,
    k: Optional[int] = None,
    **lyap_kwargs,
) -> dict:
    """Lyapunov spectrum of a trained reservoir computer.

    Wraps :func:`_lyapunov` for use with a trained reservoirpy :class:`Model`.
    The model is driven in closed-loop (autoregressive) mode: at each step the
    output-node prediction is fed back as the next-step input.  Models with
    explicit feedback wiring (``<<``) and plain ``Reservoir >> Ridge`` models
    are both supported.

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
        ``epsilon``, ``lyap_time_per_cycle``, ``display``).

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
                chunk = item[i: i + spinup]
                model.reset()
                model.run(chunk)
                init_states.append(_flatten_model_state(model))
                if n_spinup is not None and len(init_states) >= n_spinup:
                    break
        if n_spinup is not None and len(init_states) >= n_spinup:
            break

    if not init_states:
        raise ValueError(
            "init produced zero spin-up states — "
            "are all trajectories shorter than spinup?"
        )

    if kind == "trajectory":
        stdev = _probe_stdev(model)
    else:
        stdev = 1.0

    if k is None:
        k = init_states[0].size

    adapter = _RCAdapter(model, init_states, stdev)
    return _lyapunov(adapter, k=k, **lyap_kwargs)
