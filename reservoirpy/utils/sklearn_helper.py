from functools import wraps
from typing import Callable

import numpy as np
try:
    from sklearn import linear_model
except ImportError:
    sklearn = None


def get_linear(name) -> Callable:
    """
    Returns a scikit-learn linear model class given its name as a string.

    Parameters
    ----------
    name : str
        The name of the scikit-learn linear model.

    Returns
    -------
    Callable
        The scikit-learn linear model class.
    """
    return getattr(linear_model, name)


def check_sklearn_dim(X, y, readout):
    """
    Checks input dimensions and ensures that the input and output dimensions are
    consistent with the ScikitNode format (shape: [num of data points, number of time steps, num of features]).
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data.
    y : numpy.ndarray
        Output data.
    readout : ScikitNode
        ScikitNode instance.

    Returns
    -------
    tuple
        Tuple containing the reshaped input and output data (X, y).

    Raises
    ------
    ValueError
        If the input dimensions are incorrect.
    NotImplementedError
        If the input dimensions are not supported by the function.
    """
    if X.ndim == y.ndim:
        return X, y
    else:
        y = y.squeeze()
        if y.ndim == 1:  # classification task
            if X.shape[0] == y.shape[0]:
                if readout.name in ["Ridge", "ElasticNet", "Lasso", "LinearRegression"]:
                    mask = np.array([val.is_integer() for val in y], dtype=np.int)
                    if np.sum(mask) == len(y):
                        y_numpy = np.zeros((len(y), len(np.unique(y))))
                        y_numpy[np.arange(len(y)), y] = 1
                        y = y_numpy
                        if X.ndim != y.ndim:
                            raise ValueError("Current scipy regressors do not support per time step regression task. Use scipy classifiers")
                        return X, y
                if X.ndim == 2 and y.ndim == 1:
                    return X[:, None, :], y[:, None, None]
                elif X.ndim == 3 and y.ndim == 2:
                    return X, y[:, None, :]
            else:
                raise ValueError("Incorrect dimensions. Ensure NxTxD")
        elif (y.sum(axis=1) - np.ones(y.shape[0])).sum() == 0:
            if X.ndim == 3:
                return X, y[:, None, :]
            elif X.ndim == 2:
                return X[:, None, :], y[:, None, :]
        else:
            raise NotImplementedError