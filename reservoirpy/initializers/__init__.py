"""
===================
Weight initializers
===================
"""

# Author: Nathan Trouvain at 16/04/2021 <nathan.trouvain@inria.fr>
# Licence: MIT Licence
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import warnings

from typing import Optional

from .._types import RandomSeed
from ._base import Initializer
from ._base import RandomSparse
from ._internal import FastSpectralScaling
from ._internal import SpectralScaling
from ._internal import NormalSpectralScaling
from ._internal import UniformSpectralScaling
from ._internal import BimodalSpectralScaling
from ._internal import LogNormalSpectralScaling
from ._input_feedback import NormalScaling
from ._input_feedback import UniformScaling
from ._input_feedback import BimodalScaling


__all__ = [
    "get", "Initializer", "RandomSparse",
    "FastSpectralScaling", "SpectralScaling",
    "NormalSpectralScaling", "UniformSpectralScaling",
    "BimodalSpectralScaling", "LogNormalSpectralScaling",
    "NormalScaling", "UniformScaling",
    "BimodalScaling"
]


_registry = {
        "normal": {
            "spectral": NormalSpectralScaling,
            "scaling": NormalScaling
        },
        "uniform": {
            "spectral": UniformSpectralScaling,
            "scaling": UniformScaling
        },
        "bimodal": {
            "spectral": BimodalSpectralScaling,
            "scaling": BimodalScaling
        },
        "fsi": FastSpectralScaling,
        "lognormal": LogNormalSpectralScaling
    }


def get(method: str,
        connectivity: float = None,
        scaling: int = None,
        sr: int = None,
        seed: Optional[RandomSeed] = None,
        **kwargs
        ) -> Initializer:
    """Returns an intializer given the
    parameters.

    Parameters
    ----------
    method : {"normal", "uniform", "bivalued", "fsi"}
        Method used for randomly sample the weights.
        "fsi" can only be used with spectral scaling.
    connectivity : float, optional
        Probability of connection between units. Density of
        the sparse matrix.
    scaling : int, optional
        Scaling coefficient to apply on the weights. Can not be used
        with spectral scaling.
    sr : int, optional
        Maximum eigenvalue of the initialized matrix. Can not be used
        with regular scaling.
    seed : int or RandomState or Generator, optional
        Random state generator seed or RandomState or Generator instance.

    Returns
    -------
    Initializer
        An :py:class:`Initializer` object.

    Raises
    ------
    ValueError
        Can't perfom both spectral scaling and regular scaling.
    ValueError
        Method is not recognized.
    """
    if scaling is not None and sr is not None:
        raise ValueError("Parameters 'scaling' and 'sr' are mutually exclusive.")

    selection = _registry.get(method)

    if selection is None:
        raise ValueError(f"'{method}' is not a valid method. "
                         "Must be 'fsi', 'normal', 'uniform' or 'bimodal'.")

    if method == "fsi":
        return selection(sr=sr,
                         connectivity=connectivity,
                         seed=seed,
                         **kwargs)

    if method == "lognormal":
        return selection(sr=sr,
                         connectivity=connectivity,
                         seed=seed,
                         **kwargs)

    if scaling is None:
        if sr is not None:
            selected_initializer = selection["spectral"]
            return selected_initializer(sr=sr,
                                        connectivity=connectivity,
                                        seed=seed,
                                        **kwargs)
        else:
            warnings.warn("Neither 'spectral_radius' nor 'scaling' are "
                          "set. Default initializer returned will then be "
                          "a constant scaling initializer.", UserWarning)

    selected_initializer = selection["scaling"]
    return selected_initializer(scaling=scaling,
                                connectivity=connectivity,
                                seed=seed,
                                **kwargs)
