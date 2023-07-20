"""
===================================================
Weights initialization (:mod:`reservoirpy.mat_gen`)
===================================================

Quick tools for weight matrices initialization.

This module provides simples tools for reservoir internal weights
and input/feedback weights initialization. Spectral radius of the
internal weights, input scaling and sparsity are fully parametrizable.

Because most of the architectures developped in Reservoir Computing
involve sparsely-connected neuronal units, the prefered format for all
generated matrices is a :py:mod:`scipy.sparse` format (in most cases *csr*).
Sparse arrays allow fast computations and compact representations of
weights matrices, and remains easily readable. They can be parsed back to
simple Numpy arrays just by calling their ``toarray()`` method.

All functions can take as paramater a :py:class:`numpy.random.Generator`
instance, or a seed number, to ensure reproducibility. Both distribution
of weights and distribution of non-zero connections are controled with the
seed.

.. autosummary::
   :toctree: generated/

    random_sparse
    uniform
    normal
    bernoulli
    zeros
    ones
    generate_internal_weights
    generate_input_weights
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
import warnings
from functools import partial
from typing import Callable, Iterable, Union

import numpy as np
from numpy.random import Generator
from scipy import sparse, stats
from scipy.sparse.linalg import ArpackNoConvergence

from .observables import spectral_radius
from .type import global_dtype
from .utils.random import rand_generator

__all__ = [
    "fast_spectral_initialization",
    "generate_internal_weights",
    "generate_input_weights",
    "random_sparse",
    "uniform",
    "normal",
    "bernoulli",
    "zeros",
    "ones",
]

_epsilon = 1e-8  # used to avoid division by zero when rescaling spectral radius


def _filter_deprecated_kwargs(kwargs):

    deprecated = {
        "proba": "connectivity",
        "typefloat": "dtype",
        "N": None,
        "dim_input": None,
    }

    new_kwargs = {}
    args = [None, None]
    args_order = ["N", "dim_input"]
    for depr, repl in deprecated.items():
        if depr in kwargs:
            depr_argument = kwargs.pop(depr)

            msg = f"'{depr}' parameter is deprecated since v0.3.1."
            if repl is not None:
                msg += f" Consider using '{repl}' instead."
                new_kwargs[repl] = depr_argument
            else:
                args[args_order.index(depr)] = depr_argument

            warnings.warn(msg, DeprecationWarning)

    args = [a for a in args if a is not None]
    kwargs.update(new_kwargs)

    return args, kwargs


class Initializer:
    """Base class for initializer functions. Allow updating initializer function
    parameters several times before calling. May perform spectral radius rescaling
    or input scaling as a post-processing to initializer function results.

    Parameters
    ----------
    func : callable
        Initializer function. Should have a `shape` argument and return a Numpy array
        or Scipy sparse matrix.
    autorize_sr : bool, default to True
        Autorize spectral radius rescaling for this initializer.
    autorize_input_scaling : bool, default to True
        Autorize input_scaling for this initializer.
    autorize_rescaling : bool, default to True
        Autorize any kind of rescaling (spectral radius or input scaling) for this
        initializer.

    Example
    -------
    >>> from reservoirpy.mat_gen import random_sparse
    >>> init_func = random_sparse(dist="uniform")
    >>> init_func = init_func(connectivity=0.1)
    >>> init_func(5, 5)  # actually creates the matrix
    >>> random_sparse(5, 5, dist="uniform", connectivity=0.1)  # also creates the matrix
    """

    def __init__(
        self,
        func,
        autorize_sr=True,
        autorize_input_scaling=True,
        autorize_rescaling=True,
    ):
        self._func = func
        self._kwargs = dict()
        self._autorize_sr = autorize_sr
        self._autorize_input_scaling = autorize_input_scaling
        self._autorize_rescaling = autorize_rescaling

        self.__doc__ = func.__doc__
        self.__annotations__ = func.__annotations__
        if self._autorize_sr:
            self.__annotations__.update({"sr": float})
        if self._autorize_input_scaling:
            self.__annotations__.update(
                {"input_scaling": Union[float, Iterable[float]]}
            )

    def __repr__(self):
        split = super().__repr__().split(" ")
        return split[0] + f" ({self._func.__name__}) " + " ".join(split[1:])

    def __call__(self, *shape, **kwargs):
        if "sr" in kwargs and not self._autorize_sr:
            raise ValueError(
                "Spectral radius rescaling is not supported by this initializer."
            )

        if "input_scaling" in kwargs and not self._autorize_input_scaling:
            raise ValueError("Input scaling is not supported by this initializer.")

        new_shape, kwargs = _filter_deprecated_kwargs(kwargs)

        if len(new_shape) > 1:
            shape = new_shape
        elif len(new_shape) > 0:
            shape = (new_shape[0], new_shape[0])

        init = copy.deepcopy(self)
        init._kwargs.update(kwargs)

        if len(shape) > 0:
            if init._autorize_rescaling:
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
            raise ValueError(
                "'sr' and 'input_scaling' parameters are mutually exclusive for a "
                "given matrix."
            )

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
            f"'{dist}' is not a valid distribution name. "
            "See 'scipy.stats' for all available distributions."
        )


def _bernoulli_discrete_rvs(
    p=0.5, value: float = 1.0, random_state: Union[Generator, int] = None
) -> Callable:
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
    Numpy array or Scipy sparse matrix
        Rescaled matrix.
    """
    convergence = False

    if "seed" in kwargs:
        seed = kwargs.pop("seed")
    else:
        seed = None
    rg = rand_generator(seed)

    w = w_init(*shape, seed=seed, **kwargs)

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
            if seed is None:
                seed = rg.integers(1, 9999)
            else:
                seed = rg.integers(1, seed + 1)  # never stuck at 1
            w = w_init(*shape, seed=seed, **kwargs)

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
    Numpy array or Scipy sparse matrix
        Rescaled matrix.
    """
    w = w_init(*shape, **kwargs)
    if sparse.issparse(w):
        return w.multiply(input_scaling)
    else:
        return np.multiply(w, input_scaling)


def _random_sparse(
    *shape: int,
    dist: str,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed: Union[int, np.random.Generator] = None,
    **kwargs,
):
    """Create a random matrix.

    Parameters
    ----------
    *shape : int, int, ..., optional
        Shape (row, columns, ...) the matrix.
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
        Also called density of the sparse matrix. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    **kwargs : optional
        Arguments for the scipy.stats distribution.

    Returns
    -------
    scipy.sparse array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partialy initialized with the given keyword parameters,
        which can be called with a shape and returns a matrix.
    """
    if 0 < connectivity > 1.0:
        raise ValueError("'connectivity' must be >0 and <1.")

    rg = rand_generator(seed)
    rvs = _get_rvs(dist, **kwargs, random_state=rg)

    if connectivity >= 1.0 or len(shape) != 2:
        matrix = rvs(size=shape).astype(dtype)
        if connectivity < 1.0:
            matrix[rg.random(shape) > connectivity] = 0.0
    else:
        matrix = sparse.random(
            shape[0],
            shape[1],
            density=connectivity,
            format=sparsity_type,
            random_state=rg,
            data_rvs=rvs,
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
):
    """Create an array with uniformly distributed values.

    Parameters
    ----------
    *shape : int, int, ..., optional
        Shape (row, columns, ...) of the array.
    low, high : float, float, default to -1, 1
        Boundaries of the uniform distribution.
    connectivity: float, default to 1.0
        Also called density of the sparse matrix. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partialy initialized with the given keyword parameters,
        which can be called with a shape and returns a matrix.
    """
    if high < low:
        raise ValueError("'high' boundary must be > to 'low' boundary.")
    return _random_sparse(
        *shape,
        dist="uniform",
        loc=low,
        scale=high - low,
        connectivity=connectivity,
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
):
    """Create an array with values distributed following a Gaussian distribution.

    Parameters
    ----------
    *shape : int, int, ..., optional
        Shape (row, columns, ...) of the array.
    loc, scale : float, float, default to 0, 1
        Mean and scale of the Gaussian distribution.
    connectivity: float, default to 1.0
        Also called density of the sparse matrix. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partialy initialized with the given keyword parameters,
        which can be called with a shape and returns a matrix.
    """
    return _random_sparse(
        *shape,
        dist="norm",
        loc=loc,
        scale=scale,
        connectivity=connectivity,
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
        Also called density of the sparse matrix. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partialy initialized with the given keyword parameters,
        which can be called with a shape and returns a matrix.
    """
    if 1 < p < 0:
        raise ValueError("'p' must be <= 1 and >= 0.")
    return _random_sparse(
        *shape,
        p=p,
        dist="custom_bernoulli",
        connectivity=connectivity,
        dtype=dtype,
        sparsity_type=sparsity_type,
        seed=seed,
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
        Else, returns a function partialy initialized with the given keyword parameters,
        which can be called with a shape and returns a matrix.
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
        Else, returns a function partialy initialized with the given keyword parameters,
        which can be called with a shape and returns a matrix.

    Note
    ----

    `sr` parameter is not available for this initializer. The spectral radius of a null
    matrix can not be rescaled.
    """
    return np.zeros(shape, dtype=dtype)


zeros = Initializer(_zeros, autorize_sr=False)


def _fast_spectral_initialization(
    N: int,
    *args,
    sr: float = None,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed: Union[int, np.random.Generator] = None,
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
        Also called density of the sparse matrix. By default, creates dense arrays.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partialy initialized with the given keyword parameters,
        which can be called with a shape and returns a matrix.

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
    if 0 > connectivity < 1.0:
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
    )


fast_spectral_initialization = Initializer(
    _fast_spectral_initialization,
    autorize_input_scaling=False,
    autorize_rescaling=False,
)


def _generate_internal_weights(
    N: int,
    *args,
    dist="norm",
    connectivity=0.1,
    dtype=global_dtype,
    sparsity_type="csr",
    seed=None,
    **kwargs,
):
    """Generate the weight matrix that will be used for the internal connections of a
     reservoir.

    Warning
    -------

    This function is deprecated since version v0.3.1 and will be removed in future
    versions. Please consider using :py:func:`normal`, :py:func:`uniform` or
    :py:func:`random_sparse` instead.

    Parameters
    ----------
    N : int, optional
        Shape :math:`N \\times N` of the array.
        This function only builds square matrices.
    dist: str, default to "norm"
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
    connectivity: float, default to 0.1
        Also called density of the sparse matrix.
    sr : float, optional
        If defined, then will rescale the spectral radius of the matrix to this value.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    **kwargs : optional
        Arguments for the scipy.stats distribution.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partialy initialized with the given keyword parameters,
        which can be called with a shape and returns a matrix.
    """

    warnings.warn(
        "'generate_internal_weights' is deprecated since v0.3.1 and will be removed in "
        "future versions. Consider using 'bernoulli' or 'random_sparse'.",
        DeprecationWarning,
    )

    return _random_sparse(
        N,
        N,
        connectivity=connectivity,
        dtype=dtype,
        dist=dist,
        sparsity_type=sparsity_type,
        seed=seed,
        **kwargs,
    )


generate_internal_weights = Initializer(
    _generate_internal_weights, autorize_input_scaling=False
)


def _generate_input_weights(
    N,
    dim_input,
    dist="custom_bernoulli",
    connectivity=1.0,
    dtype=global_dtype,
    sparsity_type="csr",
    seed=None,
    input_bias=False,
    **kwargs,
):
    """Generate input or feedback weights for a reservoir.

    Weights are drawn by default from a discrete Bernoulli random variable,
    i.e. are always equal to 1 or -1. Then, they can be rescaled to a specific constant
    using the `input_scaling` parameter.

    Warning
    -------

    This function is deprecated since version v0.3.1 and will be removed in future
    versions. Please consider using :py:func:`bernoulli` or :py:func:`random_sparse`
    instead.

    Parameters
    ----------
    N: int
        Number of units in the connected reservoir.
    dim_input: int
        Dimension of the inputs connected to the reservoir.
    dist: str, default to "norm"
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
    connectivity: float, default to 0.1
        Also called density of the sparse matrix.
    input_scaling: float or array, optional
        If defined, then will rescale the matrix using this coefficient or array
        of coefficients.
    input_bias: bool, optional
        'input_bias' parameter is deprecated. Bias should be initialized
        separately from the input matrix.
        If True, will add a row to the matrix to take into
        account a constant bias added to the input.
    dtype : numpy.dtype, default to numpy.float64
        A Numpy numerical type.
    sparsity_type : {"csr", "csc", "dense"}, default to "csr"
        If connectivity is inferior to 1 and shape is only 2-dimensional, then the
        function will try to use one of the Scipy sparse matrix format ("csr" or "csc").
        Else, a Numpy array ("dense") will be used.
    seed : optional
        Random generator seed. Default to the global value set with
        :py:func:`reservoirpy.set_seed`.
    **kwargs : optional
        Arguments for the scipy.stats distribution.

    Returns
    -------
    Numpy array or callable
        If a shape is given to the initializer, then returns a matrix.
        Else, returns a function partialy initialized with the given keyword parameters,
        which can be called with a shape and returns a matrix.
    """
    warnings.warn(
        "'generate_input_weights' is deprecated since v0.3.1 and will be removed in "
        "future versions. Consider using 'normal', 'uniform' or 'random_sparse'.",
        DeprecationWarning,
    )

    if input_bias:
        warnings.warn(
            "'input_bias' parameter is deprecated. Bias should be initialized "
            "separately from the input matrix.",
            DeprecationWarning,
        )

        dim_input += 1

    return _random_sparse(
        N,
        dim_input,
        connectivity=connectivity,
        dtype=dtype,
        dist=dist,
        sparsity_type=sparsity_type,
        seed=seed,
        **kwargs,
    )


generate_input_weights = Initializer(_generate_input_weights, autorize_sr=False)
