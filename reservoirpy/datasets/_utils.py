# Author: Nathan Trouvain at 07/05/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from pathlib import Path
from typing import List

import numpy as np

DATA_FOLDER = Path.home() / Path("reservoirpy-data")


def _get_data_folder(folder_path=None):
    if folder_path is None:
        folder_path = DATA_FOLDER
    else:
        folder_path = Path(folder_path)

    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    return folder_path


def from_aeon_classification(
    X: np.ndarray | List[np.ndarray],
):
    """Converts a dataset in the `Aeon <https://aeon-toolkit.org/>`_ classification format into a ReservoirPy-compatible format.

    The Aeon library provides many classical classification datasets, notably all the benchmark datasets from the
    `<https://timeseriesclassification.com>`_ website. You can also use aeon to load datasets in various fileformats.

    Parameters
    ----------
    X : array-like of shape (n_timeseries, n_dimensions, n_timesteps) or list of arrays of shape (n_dimensions, n_timesteps)
        Input data in the aeon dataset format for classification

    Returns
    -------
    X : array of shape (n_timeseries, n_timesteps, n_dimensions) or list of arrays of shape (n_timesteps, n_dimensions)
        Input data in the ReservoirPy dataset format


    Examples
    --------

    >>> from aeon.datasets import load_classification
    >>> X, y = load_classification("FordA")
    >>> print(X.shape)
    (4921, 1, 500)
    >>> from reservoirpy.datasets import from_aeon_classification
    >>> X_ = from_aeon_classification(X)
    >>> print(X_.shape)
    (4921, 500, 1)
    """
    X_out: np.ndarray | List[np.ndarray]

    if isinstance(X, list):
        X_out = [np.swapaxes(np.array(series), 0, 1) for series in X]
        return X_out

    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X)
        except TypeError:
            raise TypeError(f"X must be numpy array or a list, got {type(X)}")

    if not len(X.shape) == 3:
        raise ValueError(
            f"Expected a 3-dimensional array, got {len(X.shape)} dimensions."
        )

    X_out = np.swapaxes(X, 1, 2)
    return X_out
