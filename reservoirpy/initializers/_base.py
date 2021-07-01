""":mod: `reservoirpy.intializers._base` provides base
utility for initializer definition.
"""

# Author: Nathan Trouvain at 16/04/2021 <nathan.trouvain@inria.fr>
# Licence: MIT Licence
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Tuple, Callable, Optional
from abc import ABC
from functools import partial

import numpy as np
from scipy import sparse
from scipy import stats
from numpy.random import Generator

from .._types import Weights, RandomSeed
from .._utils import _random_generator


def _get_rvs(dist: str,
             rg: Generator,
             **kwargs) -> Callable:

    # override scipy.stats uniform rvs
    # to allow user to set the distribution with
    # common low/high values and not loc/scale
    if dist == "uniform":
        return _uniform_rvs(rg, **kwargs)

    elif dist == "bimodal":
        return _bimodal_discrete_rvs(rg, **kwargs)

    elif dist in dir(stats):
        distribution = getattr(stats, dist)
        return partial(distribution(**kwargs).rvs,
                       random_state=rg)
    else:
        raise ValueError(f"'{dist}' is not a valid distribution name. "
                         "See 'scipy.stats' for all available distributions.")


def _bimodal_discrete_rvs(rg: Generator,
                          value: float = 1.) -> Callable:
    """ Only value or -value, randomly chosen.
    """

    def rvs(size: int = 1):
        return rg.choice([value, -value], replace=True, size=size)

    return rvs


def _uniform_rvs(rg: Generator,
                 low: float = -1.0,
                 high: float = 1.0,
                 ) -> Callable:

    distribution = getattr(stats, "uniform")
    return partial(distribution(loc=low, scale=high-low).rvs,
                   random_state=rg)


def random_sparse_matrix(shape: Tuple[int, int],
                         connectivity: float,
                         distribution: str,
                         seed: Optional[RandomSeed] = None,
                         sparsity_type: str = "csr",
                         **kwargs
                         ) -> Weights:
    """Generates a random matrix in a dense or sparse format.

    Parameters
    ----------
    shape : (dim1, dim2)
        Shape of the returned matrix.
    connectivity : float
        Probability of connection between units. Density of
        the sparse matrix.
    distribution : str
        A `scipy.stats` distribution name, like "norm" or
        "uniform".
    seed : int or Generator or RandomState, optional
        A random state seed or generator for reproducibility.
    sparsity_type : {'csr', 'csc', 'coo', 'dense'}
        A `scipy.sparse` matrix format. If set to `'dense'`,
        will return a `numpy.ndarray` instead of sparse matrix.
    kwargs :
        Keywords arguments to specify distribution parameters,
        like `high` and `low` for uniform distribution, or
        `loc` and `scale` for normal and log-normal distribution.

    Returns
    -------
    numpy.ndarray or scipy.sparse matrix:
        Initialized weights matrix.
    """
    rg = _random_generator(seed)

    rvs = _get_rvs(distribution,
                   rg,
                   **kwargs)

    if connectivity < 1:
        if sparsity_type == "dense":
            default_sparsity = "csr"
        else:
            default_sparsity = sparsity_type

        matrix = sparse.random(*shape, density=connectivity,
                               random_state=rg,
                               format=default_sparsity,
                               data_rvs=rvs)

        if sparsity_type == "dense":
            matrix = matrix.toarray()
    else:
        matrix = rvs(size=shape)

    return matrix


class Initializer(ABC):
    """Base class for weights initializers. All initializers should
    inherit from this class.

    All initializers should implement their own ``__call__`` method::

        def __call__(self, shape):
            # returns a matrix with the specifiyed shape
            # this matrix should be either of type numpy.ndarray
            # or scipy.sparse

    Parameters:
    -----------
    seed: int or Generator instance, optional
        Random state seed or generator.

    Attributes:
    -----------
    seed: int
        Random state seed.
    random_state: Generator
        Numpy Generator instance created from seed.
        Used for reproducibility.

    Example
    -------
    Here is an example of an initializer building a sparse matrix
    with discrete values between 0 and 1::

        class BinaryInitializer(Initializer):

            def __init__(self, density, seed):
                super(BinaryInitializer, self).__init__(seed)

                # a random generator seed is required as
                # argument by the base Initializer class.
                # the random state is then
                # available in self._rs

            def __call__(self, shape):
                distribution = lambda s: self.random_state.integers(low=0, high=2, size=s)
                return scipy.sparse.random(*shape,
                                            density=self.density,
                                            data_rvs=distribution)
    """
    _rs: Generator = None
    _seed: RandomSeed = None

    @property
    def random_state(self):
        return self._rs

    @property
    def seed(self):
        return self._seed

    def __init__(self, seed: Optional[RandomSeed] = None):
        self.reset_seed(seed)

    def __call__(self, shape: Tuple[int, int]
                 ) -> Weights:
        raise NotImplementedError

    def reset_seed(self, seed: Optional[RandomSeed] = None):
        """Produce a new numpy.random.RandomState object based
        on the given seed. This RandomState generator will then
        be used to compute the random matrices.

        Parameters:
        -----------
        seed: int or RandomState instance, optional
            If set, will be used to randomly generate the matrices.
        """
        if seed is None:
            self._rs = _random_generator(self._seed)
        else:
            self._rs = _random_generator(seed)
            self._seed = seed


class RandomSparse(Initializer):
    """Random sparse matrices generator class.

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    distribution: str, defaults to "norm"
        A `scipy.stats` distribution function name.
        Usual distributions are "norm", "uniform",
        or "bimodal" to randomly draw only -1 or 1.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    **kwargs: optional
        Keywords arguments to pass to the `scipy.stats`
        distribution function.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    distribution: str, defaults to "normal"
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
    seed: int
    random_state: Generator

    Other attributes set in ``kwargs`` parameter and used to describe
    the chosen distribution, like the ``loc`` and ``scale`` parameters of
    the 'normal' distribution. These parameters are readonly.

    Example
    -------
        >>> sparse_initializer = RandomSparse(connectivity=0.2,
        ...                                   distribution="normal",
        ...                                   loc=0, scale=1)
        >>> sparse_initializer((5, 5))  # generate a (5, 5) matrix
    """
    @property
    def distribution(self):
        return self._distribution

    def __init__(self,
                 connectivity: float = 0.1,
                 distribution: str = "norm",
                 sparsity_type: str = "csr",
                 seed: Optional[RandomSeed] = None,
                 **kwargs):
        super(RandomSparse, self).__init__(seed=seed)

        self.connectivity = connectivity
        self.sparsity_type = sparsity_type
        self._distribution = distribution

        # partial function to draw random samples
        # initialized with kwargs
        self._rvs_kwargs = kwargs
        self._upload_properties(kwargs)

    def _upload_properties(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, "_"+key, value)
            getter = self._make_property(key)
            setattr(RandomSparse, key, property(getter))

    def _make_property(self, key):
        def getter(self):
            return getattr(self, "_"+key)
        return getter

    def __call__(self,
                 shape: Tuple[int, int]
                 ) -> Weights:
        """Produce a random sparse matrix of specified shape.

        Parameters
        ----------
        shape : tuple (dim1, dim2)
            Shape of the matrix to build.

        Returns
        -------
        np.ndarray dense array or scipy.sparse matrix
            Generated matrix.
            Can be either in a sparse or a dense format,
            depending on the connectivity parameter set in
            the initializer.
        """
        return random_sparse_matrix(shape,
                                    self.connectivity,
                                    self.distribution,
                                    self.random_state,
                                    self.sparsity_type,
                                    **self._rvs_kwargs)


class Ones(Initializer):

    def __init__(self):
        super(Ones, self).__init__(seed=None)

    def __call__(self, shape):
        return np.ones(shape)


class Zeros(Initializer):

    def __init__(self):
        super(Zeros, self).__init__(seed=None)

    def __call__(self, shape):
        return np.zeros(shape)
