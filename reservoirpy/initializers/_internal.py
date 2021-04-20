""":mod: `reservoirpy.initilializers._internal`
Provides base tools for internal reservoir weights initialization.
"""

# Author: Nathan Trouvain at 16/04/2021 <nathan.trouvain@inria.fr>
# Licence: MIT Licence
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Optional

import numpy as np

from ._base import RandomSparse
from .._types import Weights, RandomSeed
from ..observables import spectral_radius


def _fsi_uniform_bounds(units: int,
                        connectivity: float,
                        sr: Optional[float]) -> float:
    """Compute FSI coefficient ``a``.
    """
    if sr is None or connectivity == 0.:
        return 1.
    else:
        return -6 * sr \
               / (np.sqrt(12) * np.sqrt(connectivity * units))


def _rescale_sr(matrix: Weights, sr: Optional[float] = None) -> Weights:
    """Rescale a matrix to a specific spectral radius.
    """
    if sr is not None:
        rho = spectral_radius(matrix)
        matrix *= sr / rho
    return matrix


class SpectralScaling(RandomSparse):
    """Internal weights initialization with spectral radius scaling.

    The weigths follows any specifyed distribution, and are then
    rescaled:

    .. math::
        W := W \\frac{\\mathrm{spectral~radius}}{\\rho(W)}

    where:

    .. math::
        \\rho(W) = \\max |\\mathrm{eigenvalues}(W)|

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    sr: float, optional
        Maximum eigenvalue of the initialized matrix.
    distribution: str, defaults to "normal"
        A numpy.random.RandomState distribution function name.
        Usual distributions are "normal", "uniform", "standard_normal",
        or "choice" with ``a=[-1, 1]``, to randomly draw -1 or 1.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator
        Random state generator seed.
    **kwargs: optional
        Keywords arguments to pass to the numpy.random.RandomState
        distribution function.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    sr: float
    distribution: str, defaults to "normal"
    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
    seed: int
    random_state: Generator

    Example
    -------
        >>> sr_scaling = SpectralScaling(distribution="norm",
        ...                              loc=0, scale=1,
        ...                              sr=0.9)
        >>> sr_scaling(5)  # generate a (5, 5) weight matrix
    """
    def __init__(self,
                 connectivity: float = 0.1,
                 sr: float = None,
                 distribution: str = "norm",
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr",
                 **kwargs):
        super(SpectralScaling, self).__init__(connectivity, distribution,
                                              sparsity_type, seed, **kwargs)
        self.sr = sr

    def __call__(self,
                 units: int,
                 ) -> Weights:
        """Produce a random sparse matrix representing the
        weights of a specifyed number of neuronal units.

        Parameters
        ----------
        units : int
            Number of units.

        Returns
        -------
        np.ndarray dense array or scipy.sparse matrix
            Generated weights.
        """
        matrix = super(SpectralScaling, self).__call__((units, units))
        return _rescale_sr(matrix, self.sr)


class FastSpectralScaling(SpectralScaling):
    """Fast Spectral Initialization (FSI) technique for
    reservoir internal weights.

    Quickly performs spectral radius scaling.

    The weigths $W_{i,j}$ are defined by:

    .. math::

        W_{i,j} \\sim \\mathcal{U}(-a, a)

    where:

    .. math::

        a = -6 \\frac{\\mathrm{spectral~radius}}{\\sqrt{12}\\sqrt{\\mathrm{connectivity}
        \\times \\mathrm{Nb_{neurons}}}}

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    sr: float, optional
        Maximum eigenvalue of the initialized matrix.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator
        Random state generator seed.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    sr: float
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
    distribution: {"uniform"}
        FSI requires an uniform distribution of weights value.
    seed: int
    random_state: Generator

    Example
    -------
        >>> fsi = FastSpectralScaling(spectral_radius=0.9)
        >>> fsi((5, 5)  # generate a 5x5 weight matrix
    """

    def __init__(self,
                 connectivity: float = 0.1,
                 sr: Optional[float] = None,
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr",
                 ):
        # uniform distribution between -1 and 1 by default. this will
        # change at each call.
        super(FastSpectralScaling, self).__init__(connectivity=connectivity,
                                                  sr=sr,
                                                  distribution="uniform",
                                                  seed=seed,
                                                  sparsity_type=sparsity_type,
                                                  low=-1,
                                                  high=1)

    def __call__(self,
                 units: int,
                 ) -> Weights:
        """Produce a random sparse matrix storing the
        weights of a specifyed number of neuronal units.

        Parameters
        ----------
        units : int
            Number of units.

        Returns
        -------
        np.ndarray dense array or scipy.sparse matrix
            Generated weights.
        """
        # adapt FSI coef to the current reservoir shape
        a = _fsi_uniform_bounds(units, self.connectivity, self.sr)

        self._rvs_kwargs = {"high": max(a, -a), "low": min(a, -a)}
        self._upload_properties(self._rvs_kwargs)

        return super(SpectralScaling, self).__call__((units, units))


class NormalSpectralScaling(SpectralScaling):
    """Convenience class for weight initialization
    with spectral radius scaling and normal distribution
    of weights value.

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    sr: float, optional
        Maximum eigenvalue of the initialized matrix.
    loc: float, defaults to 0
        Mean of the distribution
    scale: float, defaults to 1
        Standard deviation of the distribution
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator
        Random state generator seed.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    sr: float
    loc: float, defaults to 0
    scale: float, defaults to 1
    distribution: {"norm"}
    sparsity_type: {"csr", "csc", "coo"}, defaults to "csr"
    seed: int
    random_state: Generator
    """

    def __init__(self,
                 connectivity: float = 0.1,
                 sr: float = None,
                 loc: float = 0.,
                 scale: float = 1.,
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr",
                 ):
        super(NormalSpectralScaling, self).__init__(connectivity,
                                                    sr,
                                                    distribution="norm",
                                                    sparsity_type=sparsity_type,
                                                    seed=seed,
                                                    loc=loc,
                                                    scale=scale)


class UniformSpectralScaling(SpectralScaling):
    """Convenience class for weight initialization
    with spectral radius scaling and uniform distribution
    of weights value.

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    sr: float, optional
        Maximum eigenvalue of the initialized matrix.
    high, low: float, defaults to (-1, 1)
        Boundaries of the uniform distribution of weights.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator
        Random state generator seed.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    sr: float
    high: float, defaults to 1
    low: float, defaults to -1
    distribution: {"uniform"}
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
    seed: int
    random_state: Generator
    """

    def __init__(self,
                 connectivity: float = 0.1,
                 sr: float = None,
                 low: float = -1.,
                 high: float = 1.,
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr",
                 ):
        super(UniformSpectralScaling, self).__init__(connectivity,
                                                     sr,
                                                     distribution="uniform",
                                                     sparsity_type=sparsity_type,
                                                     seed=seed,
                                                     high=high,
                                                     low=low)


class BimodalSpectralScaling(SpectralScaling):
    """Convenience class for weight initialization
    with spectral radius scaling and random discrete
    distribution of weights over two values evenly
    placed around zero.

    For instance, with a value of 1, creates a matrix
    where non zero weights are only 1 or -1.

    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    sr: float, optional
        Maximum eigenvalue of the initialized matrix.
    value: float, defaults to 1
        Authorized positive value for the weights.
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator
        Random state generator seed.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    sr: float
    value: float, defaults to 1
    distribution: {"bimodal"}
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
    seed: int
    random_state: RandomState or Generator
    """

    def __init__(self,
                 connectivity: float = 0.1,
                 sr: float = None,
                 value: float = 1.,
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr",
                 ):
        super(BimodalSpectralScaling, self).__init__(connectivity,
                                                     sr,
                                                     distribution="bimodal",
                                                     sparsity_type=sparsity_type,
                                                     seed=seed,
                                                     value=value)


class LogNormalSpectralScaling(SpectralScaling):
    """Convenience class for weight initialization
    with spectral radius scaling and log-normal distribution
    of weights value.


    Parameters
    ----------
    connectivity: float, defaults to 0.1
        Probability of connection between units. Density of
        the sparse matrix.
    sr: float, optional
        Maximum eigenvalue of the initialized matrix.
    mean: float, defaults to 0
         Mean of the distribution
    sigma: float, default to 1
        Standard deviation of the distribution
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
        scipy.sparse format to use. If "dense", a numpy.ndarray dense
        matrix is used to store the weights.
    seed: int or RandomState or Generator
        Random state generator seed.

    Attributes
    ----------
    connectivity : float, defaults to 0.1
    spectral_radius: float
    mean: float, default to 0
    sigma: float, default to 1
    distribution: {"lognorm"}
    sparsity_type: {"csr", "csc", "coo", "dense"}, defaults to "csr"
    seed: int
    random_state: Generator
    """

    def __init__(self,
                 connectivity: float = 0.1,
                 sr: float = None,
                 loc: float = 0.,
                 scale: float = 1.,
                 seed: Optional[RandomSeed] = None,
                 sparsity_type: str = "csr"
                 ):
        super(LogNormalSpectralScaling, self).__init__(connectivity,
                                                       sr,
                                                       distribution="lognorm",
                                                       sparsity_type=sparsity_type,
                                                       seed=seed,
                                                       scale=np.exp(loc),
                                                       s=scale)
