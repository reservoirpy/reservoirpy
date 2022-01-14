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

    generate_internal_weights
    generate_input_weights
    fast_spectral_initialization

Example
=======

Here, we generate a 1000 units reservoir `W` with a spectral radius of 0.5,
connected to 5 inputs by the `W_in` matrix, with an input scaling of 0.9.

.. ipython:: python

    from reservoirpy.mat_gen import fast_spectral_initialization
    from reservoirpy.mat_gen import generate_input_weights
    W = fast_spectral_initialization(1000, sr=0.5)
    Win = generate_input_weights(1000, 5, input_scaling=0.9)
    print(W.shape, Win.shape)

References
==========

    .. [1] C. Gallicchio, A. Micheli, and L. Pedrelli,
           ‘Fast Spectral Radius Initialization for Recurrent
           Neural Networks’, in Recent Advances in Big Data and
           Deep Learning, Cham, 2020, pp. 380–390,
           doi: 10.1007/978-3-030-16841-4_39.

"""
import warnings

from functools import partial
from typing import Union, Callable

from scipy import stats
import numpy as np

from numpy.random import Generator, default_rng
from scipy import sparse
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

from .observables import spectral_radius
from .utils.random import rand_generator


__all__ = [
    "fast_spectral_initialization",
    "generate_internal_weights",
    "generate_input_weights"
]


def _is_probability(proba: float) -> bool:
    return 1. - proba >= 0. and proba >= 0.


def _get_rvs(dist: str,
             random_state: Generator,
             **kwargs) -> Callable:
    # override scipy.stats uniform rvs
    # to allow user to set the distribution with
    # common low/high values and not loc/scale
    if dist == "uniform":
        return _uniform_rvs(**kwargs,
                            random_state=random_state)
    elif dist == "bimodal":
        return _bimodal_discrete_rvs(**kwargs,
                                     random_state=random_state)
    elif dist in dir(stats):
        distribution = getattr(stats, dist)
        return partial(distribution(**kwargs).rvs,
                       random_state=random_state)
    else:
        raise ValueError(f"'{dist}' is not a valid distribution name. "
                         "See 'scipy.stats' for all available distributions.")


def _bimodal_discrete_rvs(value: float = 1.,
                          random_state: Union[Generator, int] = None) -> Callable:

    if isinstance(random_state, Generator):
        rg = random_state
    else:
        rg = default_rng(random_state)

    def rvs(size: int = 1):
        return rg.choice([value, -value], replace=True, size=size)

    return rvs


def _uniform_rvs(low: float = -1.0,
                 high: float = 1.0,
                 random_state: Union[Generator, int] = None) -> Callable:

    if isinstance(random_state, Generator):
        rg = random_state
    else:
        rg = default_rng(random_state)

    distribution = getattr(stats, "uniform")
    return partial(distribution(loc=low, scale=high-low).rvs,
                   random_state=rg)


def fast_spectral_initialization(N: int,
                                 sr: float = None,
                                 proba: float = 0.1,
                                 seed: Union[int, Generator] = None,
                                 verbose: bool = False,
                                 sparsity_type: str = 'csr',
                                 typefloat=np.float64,
                                 **kwargs,):
    """Fast spectral radius (FSI) approach for weights
    initialization [1]_.

    This method is well suited for computation and rescaling of
    very large weights matrices, with a number of neurons typically
    above 500-1000.

    Parameters
    ----------
    N : int
        Number of reservoir units, i.e. dimension of
        the square weights matrix.
    sr : float, optional
        Spectral radius, i.e. maximum desired eigenvalue of the
        reservoir weights matrix, by default None.
    proba : float, optional
        Probability of non-zero connection,
        density of the weight matrix, by default 0.1
    seed : int or RandomState, optional
        Random state generator seed, for reproducibility,
        by default None
    verbose : bool, optional
    sparsity_type : {"csr", "csc", "coo", "dense"} optional
        Scipy sparse matrix format. "csr" by default. If "dense"
        is chosen, the matrix will be a Numpy array and not a
        Scipy sparse matrix.
    typefloat : np.dtype, optional

    Returns
    -------
    np.ndarray or scipy.sparse matrix
        A reservoir weights matrix.

    Raises
    ------
    ValueError
        Invalid non-zero connection probability.

    References
    -----------

        .. [1] C. Gallicchio, A. Micheli, and L. Pedrelli,
               ‘Fast Spectral Radius Initialization for Recurrent
               Neural Networks’, in Recent Advances in Big Data and
               Deep Learning, Cham, 2020, pp. 380–390,
               doi: 10.1007/978-3-030-16841-4_39.

    Example
    -------

        >>> from reservoirpy.mat_gen import fast_spectral_initialization
        >>> W = fast_spectral_initialization(5, proba=0.5, seed=42)
        >>> W.toarray()
        array([[ 0.13610996, -0.57035192,  0.        ,  0.        ,  0.        ],
               [ 0.26727631,  0.        , -0.22370579, -0.04951286,  0.        ],
               [ 0.60065244,  0.        , -0.02846888,  0.        ,  0.08786003],
               [ 0.        ,  0.39151551,  0.4157107 ,  0.        ,  0.41754172],
               [ 0.        ,  0.72101228,  0.        ,  0.        ,  0.        ]])
    """
    if kwargs.get("spectral_radius") is not None:
        warnings.warn("Deprecation warning: spectral_radius parameter "
                      "is deprecated since 0.2.2 and will be removed. "
                      "Please use sr instead.")
        sr = kwargs.get("spectral_radius")

    if not _is_probability(proba):
        raise ValueError(f"proba = {proba} not in [0; 1].")

    rg = rand_generator(seed)

    if sr is None or proba <= 0.:
        a = 1
    else:
        a = -(6 * sr) / (np.sqrt(12) * np.sqrt((proba * N)))

    rvs = _get_rvs("uniform",
                   low=min(a, -a),
                   high=max(a, -a),
                   random_state=rg)

    if proba < 1 and sparsity_type != "dense":
        return sparse.random(N, N, density=proba,
                             random_state=rg, format=sparsity_type,
                             data_rvs=rvs)
    else:
        return rvs(size=(N, N))


def generate_internal_weights(N: int,
                              sr: float = None,
                              proba: float = 0.1,
                              dist: str = "norm",
                              sparsity_type: str = 'csr',
                              seed: Union[int, Generator] = None,
                              typefloat=np.float64,
                              Wstd: float = None,
                              **kwargs):
    """Generate the weights matrix that will be used
    for the internal connections of the reservoir.

    Weights will follow a normal distribution  by default,
    of mean 0 and scale `Wstd` (by default 1), and can then be rescaled to
    reach a specific spectral radius.

    Parameters
    ----------
    N : int
        Number of reservoir units, i.e. dimension of
        the square weights matrix.
    sr : float, optional
        Spectral_radius, i.e. maximum desired eigenvalue of the
        reservoir weights matrix, by default None
    proba : float, optional
        Probability of non-zero connection,
        density of the weight matrix, by default 0.1
    Wstd : float, optional
        Standard deviation of internal weights, by default 1.0
    dist: str, optional
        A distribution name from :py:mod:`scipy.stats`
        module. Parameters like ``loc`` and ``scale``
        can be passed to the distribution functions
        as keyword arguments to this function.

        Usual distributions for internal weights
        are ``'norm'`` with parameters ``loc=0`` and
        ``scale=Wstd`` to obtain weights following
        the standard normal distribution, or ``'uniform'``
        with parameters ``low=-1.`` and ``high=1.`` to
        obtain weights uniformly distributed between -1 and 1.
        Default set to ``'norm'``.
    sparsity_type : {"csr", "csc", "coo", "dense"} optional
        Scipy sparse matrix format. "csr" by default. If "dense"
        is chosen, the matrix will be a Numpy array and not a
        Scipy sparse matrix.
    seed : int or RandomState, optional
        Random state generator seed, for reproducibility,
        by default None
    typefloat : numpy.dtype, optional

    Returns
    -------
    np.ndarray or scipy.sparse matrix
        A reservoir weights matrix.

    Raises
    ------
    ValueError
        Invalid non-zero connection probability.

    Example
    -------

        >>> from reservoirpy.mat_gen import fast_spectral_initialization
        >>> W = generate_internal_weights(5, proba=0.5, seed=42)
        >>> W.toarray()
        array([[-1.72491783, -0.2257763 ,  0.        ,  0.        ,  0.        ],
               [-1.4123037 ,  0.        , -1.01283112, -1.91328024,  0.        ],
               [ 0.0675282 ,  0.        , -1.42474819,  0.        ,  1.46564877],
               [ 0.        ,  0.24196227, -0.90802408,  0.        , -0.56228753],
               [ 0.        ,  0.31424733,  0.        ,  0.        ,  0.        ]])
    """
    if 'spectral_radius' in kwargs:
        warnings.warn("Deprecation warning: 'spectral_radius' parameter "
                      "is deprecated since 0.2.2 and will be removed. "
                      "Please use 'sr' instead.")
        sr = kwargs['spectral_radius']
        kwargs.pop('spectral_radius')

    if Wstd is not None:
        warnings.warn("Deprecation warning: 'Wstd' parameter "
                      "is deprecated since 0.2.2 and will be removed. "
                      "Please use 'scale' instead.")
        kwargs["scale"] = Wstd

    if not _is_probability(proba):
        raise ValueError(f"proba = {proba} not in [0; 1].")

    w = None
    done = False
    while not done:
        rg = rand_generator(seed)

        rvs = _get_rvs(dist,
                       random_state=rg,
                       **kwargs)

        # sparse format (default)
        if sparsity_type != "dense":
            w = sparse.random(N, N, density=proba,
                              format=sparsity_type,
                              random_state=rg,
                              data_rvs=rvs)
        # dense format
        else:
            w = rvs(size=(N, N)).astype(typefloat)
            w[rg.uniform(0.0, 1.0, (N, N)) > proba] = 0.0

        if sr is not None:
            # make sure the eigenvalues are reachable.
            # TODO: maybe find a better way to do this
            try:
                current_sr = spectral_radius(w)
                w *= sr / current_sr
                done = True
            except ArpackNoConvergence:
                seed = rg.integers(1, 999)
        else:
            done = True

    return w


def generate_input_weights(N: int,
                           dim_input: int,
                           input_scaling: float = 1.,
                           proba: float = 0.1,
                           input_bias: bool = False,
                           dist: str = "bimodal",
                           seed: Union[int, Generator] = None,
                           typefloat=np.float64,
                           **kwargs):
    """
    Generate input or feedback weights for the reservoir.

    Weights are drawn by default from a discrete ``'bimodal'``
    distribution, i.e. are always equal to 1 or -1.
    Then, they can be rescaled to a specific constant using
    the `input_scaling` parameter.

    Parameters
    ----------
        N: int
            Number of units in the connected reservoir.
        dim_input: int
            Dimension of the inputs connected to the reservoir.
        input_scaling: float, optional
            Constant value used to rescale the weights.
        input_bias: bool, optional
            If True, will add a row to the matrix to take into
            account a constant bias added to the input.
            Mandatory when using a :py:class:`reservoirpy.ESN` with
            `input_bias` set.
        proba: float, optional
            Probability of non-zero connections, density of
            the matrix, by default 0.1.
        dist: str, optional
            A distribution name from :py:mod:`scipy.stats`
            module. Parameters like ``loc`` and ``scale``
            can be passed to the distribution functions
            as keyword arguments to this function.

            Usual distributions for input and feedback weights
            are ``'uniform'`` with parameters ``loc=-1`` and
            ``scale=2`` to obtain weights uniformly distributed
            between -1 and 1, or ``'bimodal'`` to obtain weights
            with value randomly set to 1 or -1 only. Default set
            to ``'bimodal'``.
        seed : int or Generator, optional
            Random state generator seed, for reproducibility,
            by default None
        typefloat : numpy.dtype, optional
    Returns
    -------
    np.ndarray or scipy.sparse matrix
        A reservoir input or feedback weights matrix.

    Raises
    ------
    ValueError
        Invalid non zero connection probability.

    Example
    -------

        >>> from reservoirpy.mat_gen import generate_input_weights
        >>> Win = generate_input_weights(10, 2, input_scaling=2., proba=0.5, seed=42)
        >>> Win
        array([[-2., -0.],
               [ 0.,  0.],
               [ 2.,  2.],
               [ 2., -0.],
               [ 0.,  0.],
               [-2.,  0.],
               [-0.,  2.],
               [-2.,  2.],
               [ 2., -0.],
               [-2., -2.]])

    """
    if not _is_probability(proba):
        raise ValueError(f"proba = {proba} not in [0; 1].")

    if input_bias:
        dim_input += 1

    rg = rand_generator(seed)

    rvs = _get_rvs(dist,
                   **kwargs,
                   random_state=rg)

    w = rvs(size=(N, dim_input))
    w[rg.random((N, dim_input)) > proba] = 0.0
    w *= input_scaling

    return w
