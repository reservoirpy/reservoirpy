""":mod: `reservoirpy.initilializers._input_feedback`
Provides base tools for input and feedback weights initialization.
"""

# Author: Nathan Trouvain at 16/04/2021 <nathan.trouvain@inria.fr>
# Licence: MIT Licence
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Optional

import numpy as np

from ._base import RandomSparse
from .._types import RandomSeed


class NormalScaling(RandomSparse):
    """Class for input and feedback
    weights initialization following a normal
    distribution. A scaling coefficient can be
    appliyed over the weights.

    The scaling coefficient change the standard deviation
    of the normal distribution from which the weights are
    sampled. The mean of this distribution remains 0.

    For input weights initialization, shape of the returned matrix
    should always be (reservoir dimension, input dimension).
    Similarly, for feedback weights initialization,
    shape of the returned matrix
    should always be (reservoir dimension, ouput dimension)

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    scaling: float, defaults to 1
        Scaling coefficient to apply on the weights.
    loc: float, defaults to 0
        Mean of the normal distribution.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator, optional
        Random state generator seed.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    scaling: float, default to 1
    loc: float, defaults to 0
    distribution: {"norm"}
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
    seed: int
    random_state: Generator

    Example
    -------
        >>> norm_scaling = NormalScaling(connectivity=0.2,
        ...                              scaling=0.5)
        >>> norm_scaling((10, 3))
    """

    def __init__(self,
                 connectivity: float = 0.1,
                 scaling: float = 1,
                 loc: float = 0,
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr",
                 ):
        super(NormalScaling, self).__init__(connectivity,
                                            distribution="norm",
                                            sparsity_type=sparsity_type,
                                            seed=seed,
                                            loc=0 if loc is None else loc,
                                            scale=scaling)
        self.scaling = scaling


class UniformScaling(RandomSparse):
    """Class for input and feedback
    weights initialization following an uniform
    distribution. A scaling coefficient can be
    appliyed over the weights.

    The scaling coefficient change the boundaries of the
    uniform distribution from which the weights are
    sampled.

    For input weights initialization, shape of the returned matrix
    should always be (reservoir dimension, input dimension).
    Similarly, for feedback weights initialization,
    shape of the returned matrix
    should always be (reservoir dimension, ouput dimension)

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    scaling: float, optional
        Scaling coefficient to apply on the weights.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator, optional
        Random state generator seed.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    scaling: float, defaults to 1
    high, low: {scaling, -scaling}
    distribution: {"uniform"}
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
    seed: int
    random_state: Generator

    Example
    -------
        >>> uni_scaling = UniformScaling(connectivity=0.2,
        ...                              scaling=0.5)
        >>> uni_scaling((10, 3))

    """

    def __init__(self,
                 connectivity: float = 0.1,
                 scaling: float = 1,
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr"
                 ):
        super(UniformScaling, self).__init__(connectivity,
                                             distribution="uniform",
                                             sparsity_type=sparsity_type,
                                             seed=seed,
                                             low=-scaling,
                                             high=scaling)
        self.scaling = scaling


class BimodalScaling(RandomSparse):
    """Class for input and feedback
    weights initialization with only two
    values, -1 and 1. A scaling coefficient can be
    appliyed over the weights.

    The scaling coefficient is multiplied with
    the dicrete values chosen for the weights.

    For input weights initialization, shape of the returned matrix
    should always be (reservoir dimension, input dimension).
    Similarly, for feedback weights initialization,
    shape of the returned matrix
    should always be (reservoir dimension, ouput dimension)

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    scaling: float, optional
        Scaling coefficient to apply on the weights.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator
        Random state generator seed.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    scaling: float, defaults to 1
    value: {scaling}
    distribution: {"bimodal"}
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
    seed: int
    random_state: Generator

    Example
    -------
        >>> bin_scaling = BimodalScaling(connectivity=0.2,
        ...                               scaling=0.5)
        >>> bin_scaling((10, 3))
    """

    def __init__(self,
                 connectivity: float = 0.1,
                 scaling: float = 1,
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr",
                 ):
        super(BimodalScaling, self).__init__(connectivity,
                                             distribution="bimodal",
                                             sparsity_type=sparsity_type,
                                             seed=seed,
                                             value=scaling)
        self.scaling = scaling


class LogNormalScaling(RandomSparse):
    """"Class for input and feedback
    weights initialization following a log-normal
    distribution. A scaling coefficient can be
    appliyed over the weights.

    The scaling coefficient change the scale of the
    log-normal distribution from which the weights are
    sampled.

    For input weights initialization, shape of the returned matrix
    should always be (reservoir dimension, input dimension).
    Similarly, for feedback weights initialization,
    shape of the returned matrix
    should always be (reservoir dimension, ouput dimension)

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    scaling: float, defaults to 1.
        Scaling coefficient to apply on the weights.
    loc: float, defaults to 0.
        Mean of the log-normal distribution.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator, optional
        Random state generator seed.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    scaling: float, defaults to 1
    loc: float, defaults to 0
    distribution: {"lognorm"}
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
    seed: int
    random_state: Generator


    Example
    -------
        >>> log_scaling = LogNormalScaling(connectivity=0.2,
        ...                                scaling=0.5)
        >>> log_scaling((10, 3))
    """

    def __init__(self,
                 connectivity: float = 0.1,
                 scaling: float = 1.,
                 loc: float = 0.,
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr"
                 ):
        super(LogNormalScaling, self).__init__(connectivity,
                                               distribution="lognorm",
                                               sparsity_type=sparsity_type,
                                               seed=seed,
                                               scale=np.exp(loc),
                                               s=scaling)
        self.scaling = scaling
