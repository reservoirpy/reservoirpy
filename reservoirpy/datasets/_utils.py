# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import sys
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from reservoirpy.type import is_array

DATA_FOLDER = Path.home() / Path("reservoirpy-data")


def _get_data_folder(folder_path=None):
    if folder_path is None:
        folder_path = DATA_FOLDER
    else:
        folder_path = Path(folder_path)

    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    return folder_path


def one_hot_encode(y: Union[np.ndarray, Sequence]):
    """Encode categorical features as a one-hot numeric array.

    This functions creates a trailing column for each class from the dataset. This function also supports inputs as
    lists of numpy arrays to stay compatible with the ReservoirPy format.

    Accepted inputs and corresponding outputs:

    - array of shape (n, ) or (n, 1) or list of length n -> array of shape (n, n_classes)
    - array of shape (n, m) or (n, m, 1) -> array of shape (n, m, n_classes)
    - list of arrays with shape (m, ) or (m, 1) -> list of arrays with shape (n, n_classes)

    Parameters
    ----------
    X: array or list of categorical values, or list of array of categorical values
        The data to determine the categories of each features.

    Returns
    -------
    array or list. See above for details.
        One-hot encoded dataset

    Example
    -------
    >>> from reservoirpy.datasets import one_hot_encode
    >>> X = np.random.normal(size=(10, 100, 1))  # 10 series, 100 timesteps
    >>> y = np.mean(X, axis=(1,2)) > 0. # a boolean for each series
    >>> print(y)
    [ True False False False  True False  True  True  True False]
    >>> y_encoded, classes = one_hot_encode(y)
    >>> y_encoded
    array([ [0., 1.],
            [1., 0.],
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [1., 0.]])
    >>> classes
    array([False,  True])

    """
    if isinstance(y, list) and is_array(y[0]):  # multi-sequence
        # treat it as one long timeseries before re-separating them
        series_lengths = [series.shape[0] for series in y]
        series_split_indices = np.cumsum(series_lengths)[:-1]
        concatenated_series = np.concatenate(y)
        concatenated_encoded, classes = one_hot_encode(concatenated_series)
        encoded = np.split(concatenated_encoded, series_split_indices)
        return encoded, classes

    y = np.array(y)

    if y.shape[-1] == 1:
        y = y.reshape(y.shape[:-1])

    classes, y_class_indices = np.unique(y, return_inverse=True)
    y_class_indices = y_class_indices.reshape(y.shape)
    nb_classes = len(classes)
    encoder = np.eye(nb_classes)
    y_encoded = encoder[y_class_indices]
    return y_encoded, classes


def from_aeon_classification(
    X: Union[np.ndarray, Sequence[np.ndarray]],
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
    X_out: Union[np.ndarray, list[np.ndarray]]

    if isinstance(X, Sequence):
        X_out = [np.swapaxes(np.array(series), 0, 1) for series in X]
        return X_out

    if not is_array(X):
        if np.array(X).shape == ():
            raise TypeError(f"X must be numpy array-like or a list, got {type(X)}")
        X = np.array(X)

    if not len(X.shape) == 3:
        raise ValueError(f"Expected a 3-dimensional array, got {len(X.shape)} dimensions.")

    X_out = np.swapaxes(X, 1, 2)
    return X_out
