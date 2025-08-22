"""
===================================================
Weights initialization (:mod:`reservoirpy.mat_gen`)
===================================================

Quick tools for weight matrices initialization.

This module provides simple tools for reservoir internal weights
and input/feedback weights initialization. Spectral radius of the
internal weights, input scaling and sparsity are fully parametrizable.

Because most of the architectures developed in Reservoir Computing
involve sparsely-connected neuronal units, the preferred format for all
generated matrices is a :py:mod:`scipy.sparse` format (in most cases *csr*).
Sparse arrays allow fast computations and compact representations of
weights matrices, and remains easily readable. They can be parsed back to
simple Numpy arrays just by calling their ``toarray()`` method.

All functions can take as parameter a :py:class:`numpy.random.Generator`
instance, or a seed number, to ensure reproducibility. Both distribution
of weights and distribution of non-zero connections are controlled with the
seed.

.. autosummary::
   :toctree: generated/

    random_sparse
    uniform
    normal
    bernoulli
    ring
    line
    orthogonal
    cluster
    small_world
    zeros
    ones
    fast_spectral_initialization
    Initializer

Example
=======

Random sparse matrix initializer from uniform distribution,
with spectral radius to 0.9 and connectivity of 0.1.

Matrix creation can be delayed...

.. ipython:: python

    from reservoirpy.mat_gen import random_sparse

    initializer = random_sparse(dist="uniform", sr=0.9, connectivity=0.1)
    matrix = initializer(100, 100)
    print(type(matrix), "\\n", matrix[:5, :5])

...or can be performed right away.

.. ipython:: python

    matrix = random_sparse(100, 100, dist="uniform", sr=0.9, connectivity=0.1)
    print(type(matrix), "\\n", matrix[:5, :5])

Random sparse matrix from Gaussian distribution,
with mean of 0 and variance of 0.5 and an out-degree of 2:

.. ipython:: python

    from reservoirpy.mat_gen import normal

    matrix = normal(7, 10, degree=2, direction="out", loc=0, scale=0.5)
    print(type(matrix), "\\n", matrix)

Dense matrix from Gaussian distribution,
with mean of 0 and variance of 0.5:

.. ipython:: python

    from reservoirpy.mat_gen import normal

    matrix = normal(50, 100, loc=0, scale=0.5)
    print(type(matrix), "\\n", matrix[:5, :5])

Sparse matrix from uniform distribution in [-0.5, 0.5],
with connectivity of 0.9 and input_scaling of 0.3:

.. ipython:: python

    from reservoirpy.mat_gen import uniform

    matrix = uniform(200, 60, low=0.5, high=0.5, connectivity=0.9, input_scaling=0.3)
    print(type(matrix), "\\n", matrix[:5, :5])

Sparse matrix from a Bernoulli random variable
giving 1 with probability p and -1 with probability 1-p,
with p=0.5 (by default) with connectivity of 0.2
and fixed seed, in Numpy format:

.. ipython:: python

    from reservoirpy.mat_gen import bernoulli

    matrix = bernoulli(10, 60, connectivity=0.2, sparsity_type="dense")
    print(type(matrix), "\\n", matrix[:5, :5])


References
==========

    .. [1] C. Gallicchio, A. Micheli, and L. Pedrelli,
           ‘Fast Spectral Radius Initialization for Recurrent
           Neural Networks’, in Recent Advances in Big Data and
           Deep Learning, Cham, 2020, pp. 380–390,
           doi: 10.1007/978-3-030-16841-4_39.
"""

import copy
from functools import partial
from inspect import signature
from typing import Callable, Iterable, Literal, Optional, Union

import numpy as np
from numpy.random import Generator
from scipy import sparse, stats
from scipy.sparse.linalg import ArpackNoConvergence

from .observables import spectral_radius
from .type import global_dtype
from .utils.random import rand_generator

__all__ = [
    "fast_spectral_initialization",
    "random_sparse",
    "uniform",
    "normal",
    "bernoulli",
    "zeros",
    "ones",
]

_epsilon = 1e-8  # used to avoid division by zero when rescaling spectral radius


class Initializer:
    """Base class for initializer functions. Allow updating initializer function
    parameters several times before calling. May perform spectral radius rescaling
    or input scaling as a post-processing to initializer function results.

    Parameters
    ----------
    func : callable
        Initializer function. Should have a `shape` argument and return a Numpy array
        or Scipy sparse array.
    allow_sr : bool, default to True
        Authorize spectral radius rescaling for this initializer.
    allow_input_scaling : bool, default to True
        Authorize input_scaling for this initializer.
    allow_rescaling : bool, default to True
        Authorize any kind of rescaling (spectral radius or input scaling) for this
        initializer.

    Example
    -------
    >>> from reservoirpy.mat_gen import random_sparse
    >>> init_func = random_sparse(dist="uniform")
    >>> init_func = init_func(connectivity=0.1)
    >>> matrix = init_func(5, 5)  # actually creates the matrix
    >>> matrix = random_sparse(5, 5, dist="uniform", connectivity=0.1)  # also creates the matrix
    """

    def __init__(
        self,
        func,
        allow_sr=True,
        allow_input_scaling=True,
        allow_rescaling=True,
    ):
        self._func = func
        self._kwargs = dict()
        self._allow_sr = allow_sr
        self._allow_input_scaling = allow_input_scaling
        self._allow_rescaling = allow_rescaling

        self.__doc__ = func.__doc__
        self.__annotations__ = func.__annotations__
        if self._allow_sr:
            self.__annotations__.update({"sr": float})
        if self._allow_input_scaling:
            self.__annotations__.update({"input_scaling": Union[float, Iterable[float]]})

    def __repr__(self):
        return f"{self._func.__name__}({str(self._kwargs)[1:-1]})"

    def __call__(self, *shape, **kwargs):
        if "sr" in kwargs and not self._allow_sr:
            raise ValueError("Spectral radius rescaling is not supported by this initializer.")

        if "input_scaling" in kwargs and not self._allow_input_scaling:
            raise ValueError("Input scaling is not supported by this initializer.")

        init = copy.deepcopy(self)
        init._kwargs = kwargs | init._kwargs

        if len(shape) > 0:
            if init._allow_rescaling:
                return init._func_post_process(*shape, **init._kwargs)
            else:
                return init._func(*shape, **init._kwargs)
        else:
            if len(kwargs) > 0:
                return init
            else:
                return init._func(**init._kwargs)  # should raise, shape is None

    def _func_post_process(self, *shape, sr=None, input_scaling=None, **kwargs):
        """Post process initializer with spectral radius or input scaling factors."""
        if sr is not None and input_scaling is not None:
            raise ValueError("'sr' and 'input_scaling' parameters are mutually exclusive for a " "given matrix.")

        if sr is not None:
            return _scale_spectral_radius(self._func, shape, sr, **kwargs)
        elif input_scaling is not None:
            return _scale_inputs(self._func, shape, input_scaling, **kwargs)
        else:
            return self._func(*shape, **kwargs)


def _get_rvs(dist: str, random_state: Generator, **kwargs) -> Callable:
    """Get a scipy.stats random variable generator.

    Parameters
    ----------
    dist : str
        A scipy.stats distribution.
    random_state : Generator
        A Numpy random generator.

    Returns
    -------
    scipy.stats.rv_continuous or scipy.stats.rv_discrete
        A scipy.stats random variable generator.
    """
    if dist == "custom_bernoulli":
        return _bernoulli_discrete_rvs(**kwargs, random_state=random_state)
    elif dist in dir(stats):
        distribution = getattr(stats, dist)
        return partial(distribution(**kwargs).rvs, random_state=random_state)
    else:
        raise ValueError(
            f"'{dist}' is not a valid distribution name. " "See 'scipy.stats' for all available distributions."
        )


def _bernoulli_discrete_rvs(p=0.5, value: float = 1.0, random_state: Union[Generator, int] = None) -> Callable:
    """Generator of Bernoulli random variables, equal to +value or -value.

    Parameters
    ----------
    p : float, default to 0.5
        Probability of single success (+value). Single failure (-value) probability
        is (1-p).
    value : float, default to 1.0
        Success value. Failure value is equal to -value.

    Returns
    -------
    callable
        A random variable generator.
    """
    rg = rand_generator(random_state)

    def rvs(size: int = 1):
        return rg.choice([value, -value], p=[p, 1 - p], replace=True, size=size)

    return rvs


def _scale_spectral_radius(w_init, shape, sr, **kwargs):
    """Change the spectral radius of a matrix created with an
    initializer.

    Parameters
    ----------
    w_init : Initializer
        An initializer.
    shape : tuple of int
        Shape of the matrix.
    sr : float
        New spectral radius.
    seed: int or Generator
        A random generator or an integer seed.

    Returns
    -------
    Numpy array or Scipy sparse array
        Rescaled matrix.
    """
    convergence = False

    if "seed" in kwargs:
        seed = kwargs.pop("seed")
    else:
        seed = None
    rng = rand_generator(seed)

    w = w_init(*shape, seed=rng, **kwargs)

    while not convergence:
        # make sure the eigenvalues are reachable.
        # (maybe find a better way to do this on day)
        try:
            current_sr = spectral_radius(w)
            if -_epsilon < current_sr < _epsilon:
                current_sr = _epsilon  # avoid div by zero exceptions.
            w *= sr / current_sr
            convergence = True
        except ArpackNoConvergence:  # pragma: no cover
            # as rng has been used before, a new matrix should be generated
            w = w_init(*shape, seed=rng, **kwargs)

    return w


def _scale_inputs(w_init, shape, input_scaling, **kwargs):
    """Rescale a matrix created with an initializer.

    Parameters
    ----------
    w_init : Initializer
        An initializer.
    shape : tuple of int
        Shape of the matrix.
    input_scaling : float
        Scaling parameter.

    Returns
    -------
    Numpy array or Scipy sparse array
        Rescaled matrix.
    """
    w = w_init(*shape, **kwargs)
    if sparse.issparse(w):
        return w.multiply(input_scaling)
    else:
        return np.multiply(w, input_scaling)


def _random_degree(
    m: int,
    n: int,
    degree: int = 10,
    direction: Literal["in", "out"] = "out",
    format: str = "coo",
    dtype: np.dtype = None,
    random_state: Union[None, int, np.random.Generator, np.random.RandomState] = None,
    data_rvs=None,
):
    """Generate a sparse array of the given shape with randomly distributed values.
    - If `direction=out`, each column has `degree` non-zero values.
    - If `direction=in`, each line has `degree` non-zero values.

    Parameters
    ----------
    m, n : int
        shape of the array
    degree : int, optional
        in-degree or out-degree of each node of the corresponding graph of the
        generated matrix:
    direction : {"in", "out"}, defaults to "out"
        Specify the direction of the `degree` value. Allowed values:
        - "in": `degree` corresponds to in-degrees
        - "out": `degree` corresponds to out-degrees
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        This random state will be used
        for sampling the sparsity structure, but not necessarily for sampling
        the values of the structurally nonzero entries of the matrix.
    data_rvs : callable, optional
        Samples a requested number of random values.
        This function should take a single argument specifying the length
        of the ndarray that it will return. The structurally nonzero entries
        of the sparse random array will be taken from the array sampled
        by this function. By default, uniform [0, 1) random values will be
        sampled using the same random state as is used for sampling
        the sparsity structure.

    Returns
    -------
    res : sparse array

    Notes
    -----
    Only float types are supported for now.

    """
    dtype = np.dtype(dtype)

    if data_rvs is None:  # pragma: no cover
        if np.issubdtype(dtype, np.complexfloating):

            def data_rvs(n):
                return random_state.uniform(size=n) + random_state.uniform(size=n) * 1j

        else:
            data_rvs = partial(random_state.uniform, 0.0, 1.0)
    mn = m * n

    tp = np.intc
    if mn > np.iinfo(tp).max:  # pragma: no cover
        tp = np.int64

    if mn > np.iinfo(tp).max:  # pragma: no cover
        msg = """\
Trying to generate a random sparse array such as the product of dimensions is
greater than %d - this is not supported on this machine
"""
        raise ValueError(msg % np.iinfo(tp).max)

    # each column has `degree` non-zero values
    if direction == "out":
        if not 0 <= degree <= m:
            raise ValueError(f"'degree'={degree} must be between 0 and m={m}.")

        i = np.zeros((n * degree), dtype=tp)
        j = np.zeros((n * degree), dtype=tp)
        for column in range(n):
            ind = random_state.choice(m, size=degree, replace=False)
            i[column * degree : (column + 1) * degree] = ind
            j[column * degree : (column + 1) * degree] = column

    # each line has `degree` non-zero values
    elif direction == "in":
        if not 0 <= degree <= n:
            raise ValueError(f"'degree'={degree} must be between 0 and n={n}.")

        i = np.zeros((m * degree), dtype=tp)
        j = np.zeros((m * degree), dtype=tp)
        for line in range(m):
            ind = random_state.choice(n, size=degree, replace=False)
            i[line * degree : (line + 1) * degree] = line
            j[line * degree : (line + 1) * degree] = ind

    else:
        raise ValueError(f'\'direction\'={direction} must either be "out" or "in".')

    vals = data_rvs(len(i)).astype(dtype, copy=False)
    return sparse.coo_array((vals, (i, j)), shape=(m, n)).asformat(format, copy=False)


def _random_sparse(
    *shape: int,
    dist: str,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed: Union[int, np.random.Generator] = None,
    degree: Union[int, None] = None,
    direction: Literal["in", "out"] = "out",
    **kwargs,
):
    """Create a random array.

    Parameters
    ----------
    *shape : int, int, ..., optional
        Shape (row, columns, ...) of the matrix.
    dist: str
        A distribution name from :py:mod:`scipy.stats` module, such as "norm" or
        "uniform". Parameters like `loc` and `scale` can be passed to the distribution
        functions as keyword arguments to this function. Usual distributions for
        internal weights are :py:class:`scipy.stats.norm` with parameters `loc` and
        `scale` to obtain weights following the standard normal distribution,
        or :py:class:`scipy.stats.uniform` with parameters `loc=-1` and `scale=2`
        to obtain weights uniformly distributed between -1 and 1.
        Can also have the value "custom_bernoulli". In that case, weights will be drawn
        from a Bernoulli discrete random variable alternating between -1 and 1 and
        drawing 1 with a probability `p` (default `p` parameter to 0.5).
    connectivity: float, default to 1.0
        Also called density of the sparse array. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse array format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    degree: int, default to None
        If not None, override the `connectivity` argument and corresponds to the number
        of non-zero values along the axis specified by `direction`
    direction: {"in", "out"}, default to "out"
        If `degree` is not None, specifies the axis along which the `degree` non-zero
        values are distributed.
        - If `direction` is "in", each line will have `degree` non-zero values. In other
        words, each node of the corresponding graph will have `degree` in-degrees
        - If `direction` is "out", each column will have `degree` non-zero values. In
        other words, each node of the corresponding graph will have `degree` out-degrees
    **kwargs : optional
        Arguments for the scipy.stats distribution.

    Returns
    -------
    scipy.sparse array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.
    """

    rng = rand_generator(seed)
    rvs = _get_rvs(dist, **kwargs, random_state=rng)

    if degree is not None:
        if len(shape) != 2:
            raise ValueError(f"Matrix shape must have 2 dimensions, got {len(shape)}: {shape}")
        m, n = shape
        matrix = _random_degree(
            m=m,
            n=n,
            degree=degree,
            direction=direction,
            format=sparsity_type,
            dtype=dtype,
            random_state=rng,
            data_rvs=rvs,
        )
    else:
        if 0 < connectivity > 1.0:
            raise ValueError("'connectivity' must be >0 and <1.")

        if connectivity >= 1.0 or len(shape) != 2:
            matrix = np.zeros(shape, dtype=dtype)
            non_zeros_count = int(connectivity * np.prod(shape))
            non_zero_weights = rvs(size=non_zeros_count).astype(dtype)
            raveled_indices = rng.choice(matrix.size, non_zeros_count, replace=False)
            indices = np.unravel_index(raveled_indices, matrix.shape)
            matrix[indices] = non_zero_weights
        else:
            if "rng" in signature(sparse.random_array).parameters:
                rng_arg = {"rng": rng}
            else:
                # compatibility with scipy<1.15 (& thus Python<3.10). TODO: remove when updating those
                rng_arg = {"random_state": rng.integers(4294967295)}
            matrix = sparse.random_array(
                shape,
                density=connectivity,
                format=sparsity_type,
                **rng_arg,
                data_sampler=rvs,
                dtype=dtype,
            )

    # sparse.random may return np.matrix if format="dense".
    # Only ndarray are supported though, hence the explicit cast.
    if type(matrix) is np.matrix:
        matrix = np.asarray(matrix)

    return matrix


random_sparse = Initializer(_random_sparse)


def _uniform(
    *shape: int,
    low: float = -1.0,
    high: float = 1.0,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed: Union[int, np.random.Generator] = None,
    degree: Union[int, None] = None,
    direction: Literal["in", "out"] = "out",
):
    """Create an array with uniformly distributed values.

    Parameters
    ----------
    *shape : int, int, ..., optional
        Shape (row, columns, ...) of the array.
    low, high : float, float, default to -1, 1
        Boundaries of the uniform distribution.
    connectivity: float, default to 1.0
        Also called density of the sparse array. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse array format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    degree: int, default to None
        If not None, override the `connectivity` argument and corresponds to the number
        of non-zero values along the axis specified by `direction`
    direction: {"in", "out"}, default to "out"
        If `degree` is not None, specifies the axis along which the `degree` non-zero
        values are distributed.
        - If `direction` is "in", each line will have `degree` non-zero values. In other
        words, each node of the corresponding graph will have `degree` in-degrees
        - If `direction` is "out", each column will have `degree` non-zero values. In
        other words, each node of the corresponding graph will have `degree` out-degrees

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.
    """
    if high < low:
        raise ValueError("'high' boundary must be > to 'low' boundary.")
    return _random_sparse(
        *shape,
        dist="uniform",
        loc=low,
        scale=high - low,
        connectivity=connectivity,
        degree=degree,
        direction=direction,
        dtype=dtype,
        sparsity_type=sparsity_type,
        seed=seed,
    )


uniform = Initializer(_uniform)


def _normal(
    *shape: int,
    loc: float = 0.0,
    scale: float = 1.0,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed: Union[int, np.random.Generator] = None,
    degree: Union[int, None] = None,
    direction: Literal["in", "out"] = "out",
):
    """Create an array with values distributed following a Gaussian distribution.

    Parameters
    ----------
    *shape : int, int, ..., optional
        Shape (row, columns, ...) of the array.
    loc, scale : float, float, default to 0, 1
        Mean and scale of the Gaussian distribution.
    connectivity: float, default to 1.0
        Also called density of the sparse array. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse array format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    degree: int, default to None
        If not None, override the `connectivity` argument and corresponds to the number
        of non-zero values along the axis specified by `direction`
    direction: {"in", "out"}, default to "out"
        If `degree` is not None, specifies the axis along which the `degree` non-zero
        values are distributed.
        - If `direction` is "in", each line will have `degree` non-zero values. In other
        words, each node of the corresponding graph will have `degree` in-degrees
        - If `direction` is "out", each column will have `degree` non-zero values. In
        other words, each node of the corresponding graph will have `degree` out-degrees

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.
    """
    return _random_sparse(
        *shape,
        dist="norm",
        loc=loc,
        scale=scale,
        connectivity=connectivity,
        degree=degree,
        direction=direction,
        dtype=dtype,
        sparsity_type=sparsity_type,
        seed=seed,
    )


normal = Initializer(_normal)


def _bernoulli(
    *shape: int,
    p: float = 0.5,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed: Union[int, np.random.Generator] = None,
    degree: Union[int, None] = None,
    direction: Literal["in", "out"] = "out",
):
    """Create an array with values equal to either 1 or -1. Probability of success
    (to obtain 1) is equal to p.

    Parameters
    ----------
    *shape : int, int, ..., optional
        Shape (row, columns, ...) of the array.
    p : float, default to 0.5
        Probability of success (to obtain 1).
    connectivity: float, default to 1.0
        Also called density of the sparse array. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse array format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    degree: int, default to None
        If not None, override the `connectivity` argument and corresponds to the number
        of non-zero values along the axis specified by `direction`
    direction: {"in", "out"}, default to "out"
        If `degree` is not None, specifies the axis along which the `degree` non-zero
        values are distributed.
        - If `direction` is "in", each line will have `degree` non-zero values. In other
        words, each node of the corresponding graph will have `degree` in-degrees
        - If `direction` is "out", each column will have `degree` non-zero values. In
        other words, each node of the corresponding graph will have `degree` out-degrees

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.
    """
    if 1 < p or p < 0:
        raise ValueError("'p' must be <= 1 and >= 0.")
    return _random_sparse(
        *shape,
        p=p,
        dist="custom_bernoulli",
        connectivity=connectivity,
        dtype=dtype,
        sparsity_type=sparsity_type,
        seed=seed,
        degree=degree,
        direction=direction,
    )


bernoulli = Initializer(_bernoulli)


def _ones(*shape: int, dtype: np.dtype = global_dtype, **kwargs):
    """Create an array filled with 1.

    Parameters
    ----------
    *shape : int, int, ..., optional
        Shape (row, columns, ...) of the array.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.
    """
    return np.ones(shape, dtype=dtype)


ones = Initializer(_ones)


def _zeros(*shape: int, dtype: np.dtype = global_dtype, **kwargs):
    """Create an array filled with 0.

    Parameters
    ----------
    *shape : int, int, ..., optional
        Shape (row, columns, ...) of the array.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.

    Note
    ----

    `sr` parameter is not available for this initializer. The spectral radius of a null
    matrix can not be rescaled.
    """
    return np.zeros(shape, dtype=dtype)


zeros = Initializer(_zeros, allow_sr=False)


def _fast_spectral_initialization(
    N: int,
    *args,
    sr: float = None,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed: Union[int, np.random.Generator] = None,
    degree: Union[int, None] = None,
    direction: Literal["in", "out"] = "out",
):
    """Fast spectral radius (FSI) approach for weights
    initialization [1]_ of square matrices.

    This method is well suited for computation and rescaling of
    very large weights matrices, with a number of neurons typically
    above 500-1000.

    Parameters
    ----------
    N : int, optional
        Shape :math:`N \\times N` of the array.
        This function only builds square matrices.
    connectivity: float, default to 1.0
        Also called density of the sparse array. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse array format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    degree: int, default to None
        If not None, override the `connectivity` argument and corresponds to the number
        of non-zero values along the axis specified by `direction`
    direction: {"in", "out"}, default to "out"
        If `degree` is not None, specifies the axis along which the `degree` non-zero
        values are distributed.
        - If `direction` is "in", each line will have `degree` non-zero values. In other
        words, each node of the corresponding graph will have `degree` in-degrees
        - If `direction` is "out", each column will have `degree` non-zero values. In
        other words, each node of the corresponding graph will have `degree` out-degrees

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.

    Note
    ----

    This function was designed for initialization of a reservoir's internal weights.
    In consequence, it can only produce square matrices. If more than one positional
    argument of shape are provided, only the first will be used.

    References
    -----------

    .. [1] C. Gallicchio, A. Micheli, and L. Pedrelli,
           ‘Fast Spectral Radius Initialization for Recurrent
           Neural Networks’, in Recent Advances in Big Data and
           Deep Learning, Cham, 2020, pp. 380–390,
           doi: 10.1007/978-3-030-16841-4_39.
    """
    if connectivity < 0.0 or connectivity > 1.0:
        raise ValueError("'connectivity' must be >0 and <1.")

    if sr is None or connectivity <= 0.0:
        a = 1
    else:
        a = -(6 * sr) / (np.sqrt(12) * np.sqrt((connectivity * N)))

    return _uniform(
        N,
        N,
        low=np.min((a, -a)),
        high=np.max((a, -a)),
        connectivity=connectivity,
        dtype=dtype,
        sparsity_type=sparsity_type,
        seed=seed,
        degree=degree,
        direction=direction,
    )


fast_spectral_initialization = Initializer(
    _fast_spectral_initialization,
    allow_input_scaling=False,
    allow_rescaling=False,
)


def _ring(
    *shape: int,
    weights: Optional[np.ndarray] = None,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    **kwargs,
):
    """Create a lower cyclic shift matrix.

    This is used for ring reservoirs, which have a circular topology
    (each node `n` is connected to the node `n+1 % units`).

    Note that the `connectivity` and `seed` parameters have no effect.

    Parameters
    ----------
    *shape : (int, int), optional
        Shape (row, columns) of the array. Must be a square matrix, i.e. row == columns.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    weights : array of shape (units, ), optional
        If defined, corresponds to the weights of each connection.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        Format of the output matrix. "csr" and "csc" corresponds to the Scipy sparse
        matrix formats, and "dense" corresponds to a regular Numpy array.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    **kwargs : optional
        This argument is kept for compatibility reasons. This is not used.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.
    """

    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Shape of the ring matrix must be (units, units), got {shape}.")
    units = shape[0]

    if weights is None:
        weights = np.ones((units,), dtype=dtype)
    row = np.roll(np.arange(units, dtype=np.int32), shift=-1)
    col = np.arange(units, dtype=np.int32)

    matrix = sparse.coo_array((weights, (row, col)), shape=(units, units)).asformat(sparsity_type, copy=False)

    if type(matrix) is np.matrix:
        matrix = np.asarray(matrix)

    return matrix


ring = Initializer(_ring)


def _line(
    *shape: int,
    weights: Optional[np.ndarray] = None,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    **kwargs,
):
    """Create a lower shift matrix.

    This is used for line reservoirs, which have a linear topology
    (each node `n` is connected to the node `n+1`, the last node has no successor).

    Note that the `connectivity` and `seed` parameters have no effect.

    Parameters
    ----------
    *shape : (int, int), optional
        Shape (row, columns) of the array. Must be a square matrix, i.e. row == columns.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    weights : array of shape (units-1, ), optional
        If defined, corresponds to the weights of each connection.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        Format of the output matrix. "csr" and "csc" corresponds to the Scipy sparse
        matrix formats, and "dense" corresponds to a regular Numpy array.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    **kwargs : optional
        This argument is kept for compatibility reasons. This is not used.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.
    """

    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Shape of the ring matrix must be (units, units), got {shape}.")
    units = shape[0]

    if weights is None:
        weights = np.ones((units - 1,), dtype=dtype)
    row = np.arange(1, units, dtype=np.int32)
    col = np.arange(units - 1, dtype=np.int32)

    matrix = sparse.coo_array((weights, (row, col)), shape=(units, units)).asformat(sparsity_type, copy=False)

    if type(matrix) is np.matrix:
        matrix = np.asarray(matrix)

    return matrix


line = Initializer(_line)


def _orthogonal(
    *shape: int,
    dtype: np.dtype = global_dtype,
    seed: Union[int, np.random.Generator] = None,
    **kwargs,
):
    """Create a random orthogonal matrix, drawn from the O(N) Haar distribution (the only uniform distribution on O(N)).

    Note that the `connectivity` parameter in the Reservoir node has no effect with an orthogonal matrix initializer.

    Parameters
    ----------
    *shape : (int, int), optional
        Shape (row, columns) of the array. Must be a square matrix, i.e. row == columns.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    **kwargs : optional
        This argument is kept for compatibility reasons. This is not used.

    Returns
    -------
    numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partially initialized with the given keyword
        parameters, which can be called with a shape and returns a matrix.
    """

    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Shape of the ring matrix must be (units, units), got {shape}.")
    units = shape[0]
    rg = rand_generator(seed)

    matrix = stats.ortho_group.rvs(dim=units, random_state=rg)

    return matrix


orthogonal = Initializer(_orthogonal)


def _cluster(
    *shape: int,
    dtype: np.dtype = global_dtype,
    seed: Union[int, np.random.Generator] = None,
    p_in=0.1,
    p_out=0.01,
    cluster=3,
    distribution="normal",
    sparsity_type="csr",
    **kwargs,
):
    """
    Create a cluster matrix with given distribution and p_in/p_out parameters. Based on the Erdős-Rényi clustered matrix
    algorithm.

    Parameters
    ----------
    *shape : (int, int), optional
        Shape (row, columns) of the array. Must be a square matrix, i.e. row == columns.
    cluster : int, optional
        Number of clusters to create within the matrix. Default is 3.
    p_in : float, optional
        Defines the connectivity within a cluster. Default is 0.1.
    p_out : float, optional
        Defines the connectivity between clusters. Default is 0.01.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        Format of the output matrix. "csr" and "csc" corresponds to the Scipy sparse
        matrix formats, and "dense" corresponds to a regular Numpy array.
    distribution: {"normal", "uniform", "random", "bernoulli"}, default to "normal"
        A distribution name from :py:mod:`scipy.stats` module, such as "norm" or
        "uniform". Parameters like `loc` and `scale` can be passed to the distribution
        functions as keyword arguments to this function. Usual distributions for
        internal weights are :py:class:`scipy.stats.norm` with parameters `loc` and
        `scale` to obtain weights following the standard normal distribution,
        or :py:class:`scipy.stats.uniform` with parameters `loc=-1` and `scale=2`
        to obtain weights uniformly distributed between -1 and 1.
        Can also have the value "custom_bernoulli". In that case, weights will be drawn
        from a Bernoulli discrete random variable alternating between -1 and 1 and
        drawing 1 with a probability `p` (default `p` parameter to 0.5).
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    **kwargs : optional
        This argument is kept for compatibility reasons. This is not used.

    Returns
    -------
    numpy array or callable

    """

    # Check that the shape is divisible by the amount of cluster
    if shape[0] % cluster != 0:
        raise ValueError("Units must be a multiple of the amount of cluster.")

    rng = rand_generator(seed)
    # Define dictionary for corresponding weight distribution matrices
    matrix_dict = dict(normal=normal, uniform=uniform, random=random_sparse, bernoulli=bernoulli)

    # Check for valid distribution
    if distribution not in matrix_dict:
        raise ValueError(
            f"Distribution {distribution} is not supported. Must be 'normal', 'uniform', 'random', 'bernoulli'."
        )

    # Define the global matrix
    matrix = matrix_dict[distribution](
        shape[0],
        shape[1],
        connectivity=p_out,
        dtype=dtype,
        sparsity_type=sparsity_type,
        seed=rng,
    )

    # Define the number of neurons inside each cluster
    n_c = shape[0] // cluster

    # Define the cluster matrix
    c_matrices = matrix_dict[distribution](cluster, n_c, n_c, connectivity=p_in, dtype=dtype, seed=rng)

    # Create the cluster matrix
    for i in range(cluster):
        matrix[i * n_c : i * n_c + n_c, i * n_c : i * n_c + n_c] = c_matrices[i]

    return matrix


cluster = Initializer(_cluster)


def _small_world(
    *shape: int,
    dtype: np.dtype = global_dtype,
    seed: Union[int, np.random.Generator] = None,
    nb_close_neighbours: int = 2,
    proba_rewire: float = 0.1,
    distribution: Literal["normal", "uniform", "random_sparse", "bernoulli"] = "normal",
    sparsity_type: str = "csr",
    **kwargs,
):
    """
    Create a small-world network using the Watts-Strogatz model.
    This function generates a small-world network adjacency matrix with a specified
    number of close neighbours and a probability of rewiring edges.
    The small-world network is characterized by a high clustering coefficient and
    a short average path length, making it suitable for modeling complex networks.
    Small-world networks are often used in various fields such as sociology,
    neuroscience, and computer science to represent systems with local connections.
    See : Watts, D. J.; Strogatz, S. H. (1998). "Collective dynamics of 'small-world' networks"

    Parameters
    ----------
    *shape : (int, int)
        Shape of the matrix. Must be square.
    nb_close_neighbours : int
        Number of close neighbours per node (must be even).
    proba_rewire : float
        Probability of rewiring edges (between 0 and 1).
    dtype : numpy.dtype
        Output matrix dtype.
    seed : int or np.random.Generator
        Random seed or generator.
    distribution : Literal["normal", "uniform", "random_sparse", "bernoulli"], default to "normal"
        Distribution to use for the weights of the connections.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        Format of the output matrix. "csr" and "csc" corresponds to the Scipy sparse
    **kwargs : unused
        For compatibility.

    Returns
    -------
    numpy.ndarray
        Adjacency matrix of the small-world network.
    """

    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Shape of the small-world matrix must be (units, units), got {shape}.")
    if nb_close_neighbours % 2 != 0:
        raise ValueError("nb_close_neighbours must be even.")
    if not (0 <= proba_rewire <= 1):
        raise ValueError("proba_rewire must be between 0 and 1.")

    units = shape[0]
    rng = rand_generator(seed)
    half_neighbours = nb_close_neighbours // 2

    matrix = sparse.lil_array((units, units), dtype=dtype)

    weight_distribution = dict(normal=normal, uniform=uniform, random_sparse=random_sparse, bernoulli=bernoulli)
    if distribution not in weight_distribution:
        raise ValueError(
            f"Distribution {distribution} is not supported. Must be 'normal', 'uniform', 'random_sparse', 'bernoulli'."
        )
    lower_diag_weights = weight_distribution[distribution](
        units * half_neighbours,
        dtype=dtype,
        seed=seed,
        **kwargs,
    )
    upper_diag_weights = weight_distribution[distribution](
        units * half_neighbours,
        dtype=dtype,
        seed=seed,
        **kwargs,
    )

    # Branching nb_close_neighbours nodes to each node (creating the ring lattice)
    indices = np.arange(units)  # (units,)
    offsets = np.arange(1, half_neighbours + 1).reshape(-1, 1)  # (half_neigh, 1)
    i = (indices + offsets) % units  # (units, half_neighbours)
    j = np.tile(indices, (1, half_neighbours))
    i, j = i.flatten(), j.flatten()
    matrix[i, j] = lower_diag_weights
    matrix[j, i] = upper_diag_weights

    # Rewiring edges with probability proba_rewire
    # We only consider the upper triangle of the matrix to avoid double connections
    edges = np.vstack((i, j)).T

    rewire_mask = rng.random(units * half_neighbours) < proba_rewire
    rewired_edges = edges[rewire_mask]

    for i, j in rewired_edges:
        possible_nodes = np.where(np.logical_and(matrix[[i]].toarray().ravel() == 0, np.arange(units) != i))[0]
        if len(possible_nodes) == 0:
            continue
        new_j = rng.choice(possible_nodes)
        matrix[i, new_j] = matrix[i, j]
        matrix[new_j, i] = matrix[j, i]
        matrix[i, j] = 0
        matrix[j, i] = 0
    return matrix.asformat(sparsity_type, copy=False)


small_world = Initializer(_small_world)
