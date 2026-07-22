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
    mae
    memory_capacity
    effective_spectral_radius
    lyapunov_exponent
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

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
    2.886926160973429
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

    return max(abs(linalg.eigvals(W)))


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
    >>> from reservoirpy import ESN
    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = ESN(units=100, sr=1).fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import mse
    >>> print(mse(y_true=y_test, y_pred=y_pred))
    1.0103865448822693e-07
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
    0.0004845326475756944
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
    0.007249877565336089
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
    0.9999970566302174
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


def mae(y_true: np.ndarray, y_pred: np.ndarray, dimensionwise: bool = False) -> float:
    """Mean absolute error metric:

    .. math::

        \\frac{1}{N} \sum_{i=0}^{N-1} |y_i - \hat{y}_i|


    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    dimensionwise: boolean, optional
        If True, return a mean absolute error for each dimension of the timeseries.

    Returns
    -------
    float
        Mean absolute error. If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = model.fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import mae
    >>> print(mae(y_true=y_test, y_pred=y_pred))
    0.00025325136638810585
    """
    y_true_array, y_pred_array = _check_arrays(y_true, y_pred)

    if dimensionwise:
        if len(y_true_array.shape) == 3:
            axis = (0, 1)
        else:
            axis = 0
    else:
        axis = None

    error = np.abs(y_true_array - y_pred_array)
    return np.mean(error, axis=axis)


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
    Memory capacity of the model: 18.78

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
    spectral_radius(W)=0.5
    >>> print(f"{effective_spectral_radius(W, lr=lr)=:.3}")
    effective_spectral_radius(W, lr=lr)=0.731

    References
    ----------
    .. [1] Jaeger, H., Lukoševičius, M., Popovici, D., & Siewert, U. (2007).
        Optimization and applications of echo state networks with leaky-integrator
        neurons. Neural networks, 20(3), 335-352.
    """
    units = W.shape[0]
    return spectral_radius(W=lr * W + (1 - lr) * np.eye(units), maxiter=maxiter)


def lyapunov_exponent(
    reservoir: "Node",
    x: np.ndarray,
    warmup: int = 0,
    initial_distance: float = 1e-8,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> float:
    """Largest Lyapunov exponent of a reservoir driven by an input timeseries.

    The largest Lyapunov exponent :math:`\\lambda_{max}` measures the average
    exponential rate at which two infinitesimally close reservoir states diverge
    (or converge) when the reservoir is driven by the same input. It is a standard
    way to locate the *edge of chaos* of an Echo State Network [1]_:

    - :math:`\\lambda_{max} < 0`: nearby trajectories contract. The reservoir has
      the echo state property (its state asymptotically depends only on the input).
    - :math:`\\lambda_{max} > 0`: nearby trajectories diverge. The reservoir is in
      a chaotic regime and does not have the echo state property.

    It is estimated with the two-trajectory algorithm of Benettin et al. [2]_: a
    reference trajectory and a copy perturbed by a small distance ``initial_distance``
    are evolved under the same input; at each timestep the log-ratio of the new to
    the initial separation is accumulated and the perturbed state is rescaled back to
    ``initial_distance``. The exponent (per timestep) is the mean of these log-ratios.

    Parameters
    ----------
    reservoir : :class:`reservoirpy.Node`
        A reservoir (or any recurrent node) whose internal state is used to measure
        the divergence of nearby trajectories.
    x : array of shape (timesteps, input_dim)
        Input timeseries used to drive the reservoir. The exponent is conditioned
        on this input.
    warmup : int, defaults to 0
        Number of initial timesteps used to let the reservoir reach its attractor.
        These timesteps are discarded from the exponent estimate. Must be strictly
        smaller than the number of timesteps.
    initial_distance : float, defaults to 1e-8
        Distance between the reference and the perturbed state, kept constant by
        rescaling at each timestep. Should be small enough to stay in the linear
        regime of the dynamics.
    seed : int or :py:class:`numpy.random.Generator`, optional
        Random state seed for the initial perturbation direction, for reproducibility.

    Returns
    -------
    float
        Estimate of the largest Lyapunov exponent (per timestep).

    Examples
    --------
    >>> import numpy as np
    >>> from reservoirpy.nodes import Reservoir
    >>> from reservoirpy.observables import lyapunov_exponent
    >>> rng = np.random.default_rng(seed=2504)
    >>> x = rng.uniform(-0.8, 0.8, size=(2000, 1))
    >>> reservoir = Reservoir(100, sr=0.9, lr=1.0, seed=1)
    >>> print(f"{lyapunov_exponent(reservoir, x, warmup=200, seed=1):.3f}")
    -0.147

    References
    ----------
    .. [1] Boedecker, J., Obst, O., Lizier, J. T., Mayer, N. M., & Asada, M. (2012).
        Information processing in echo state networks at the edge of chaos.
        Theory in Biosciences, 131(3), 205-213.
    .. [2] Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J. M. (1980).
        Lyapunov characteristic exponents for smooth dynamical systems and for
        Hamiltonian systems; a method for computing all of them. Meccanica, 15, 9-20.
    """
    x = np.asarray(x)
    if x.ndim < 2:
        raise ValueError(f"x must be a timeseries of shape (timesteps, input_dim), but has shape {x.shape}.")
    n_timesteps = x.shape[-2]
    if warmup >= n_timesteps:
        raise ValueError(f"warmup ({warmup}) must be strictly smaller than the number of timesteps ({n_timesteps}).")

    reservoir = deepcopy(reservoir)
    if not reservoir.initialized:
        reservoir.initialize(x)
    reservoir.reset()
    if warmup > 0:
        reservoir.run(x[:warmup])
    ref_state = reservoir.state

    # The recurrent state may hold several arrays: flatten them into a single
    # vector to measure the distance between two trajectories.
    keys = list(ref_state.keys())

    def flatten(state):
        return np.concatenate([np.ravel(state[k]) for k in keys])

    def unflatten(vec, template):
        state, start = {}, 0
        for k in keys:
            stop = start + np.size(template[k])
            state[k] = vec[start:stop].reshape(np.shape(template[k]))
            start = stop
        return state

    rng = rand_generator(seed)
    ref_vec = flatten(ref_state)
    direction = rng.normal(size=ref_vec.shape)
    direction /= np.linalg.norm(direction)
    pert_vec = ref_vec + initial_distance * direction

    log_divergences = np.empty((n_timesteps - warmup,))
    for i, t in enumerate(range(warmup, n_timesteps)):
        ref_state = reservoir._step(ref_state, x[t])
        pert_state = reservoir._step(unflatten(pert_vec, ref_state), x[t])
        ref_vec = flatten(ref_state)
        pert_vec = flatten(pert_state)
        difference = pert_vec - ref_vec
        distance = np.linalg.norm(difference)
        log_divergences[i] = np.log(distance / initial_distance)
        # Rescale the perturbed state back to `initial_distance` from the reference.
        pert_vec = ref_vec + (initial_distance / distance) * difference

    return float(np.mean(log_divergences))
