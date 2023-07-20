from functools import wraps
from typing import Callable

import numpy as np
try: #pragma: no cover
    from sklearn import linear_model
except ImportError:
    sklearn = None


def get_linear(method) -> Callable:
    """
    Returns a scikit-learn linear model class given its method as a string.

    Parameters
    ----------
    method : str
        The method of the scikit-learn linear model.

    Returns
    -------
    Callable
        The scikit-learn linear model class.
    """
    return getattr(linear_model, method)


class TransformInputSklearn(object):
    def _for_regression(self, X, y):
        # Squeeze X and y to remove any unnecessary dimensions
        X, y = np.squeeze(X), np.squeeze(y)
        # Ensure X and y have the same number of dimensions
        # import pdb;pdb.set_trace()
        if X.ndim != y.ndim:
            if X.ndim == 2:
                X = X[:, :, None]
            if y.ndim == 1:
                y = y[:, None, None]
            elif y.ndim == 2:
                y = y[:, :, None]
            else:
                raise ValueError("X and y must have the same number of dimensions")
        else:
            if X.ndim == y.ndim == 1:
                return X[:, None, None], y[:, None, None]
        return X, y
    
    def _for_classification(self, X, y):
        # Squeeze X and y to remove any unnecessary dimensions
        X, y = np.squeeze(X), np.squeeze(y)

        # Ensure X and y have the same number of dimensions
        if X.ndim != y.ndim:
            if X.ndim == 2:
                N, T = X.shape
                X = X[:, :, None]
            elif y.ndim == 2:
                N, T = y.shape
                y = np.argmax(y, axis=1)
            else:
                raise ValueError("X and y must have the same number of dimensions")
        
        # One-hot encode y if it is not already one-hot encoded
        if y.ndim == 1:
            y_onehot = np.zeros((N, T))
            for i in range(N):
                y_onehot[i, T-1] = y[i]
            y = y_onehot[:, :, None]

        return X, y



    def __call__(self, X, y, task=None):
        """
        Checks input dimensions and ensures that the input and output dimensions are
        consistent with the ScikitNode format (shape: [num of data points, number of time steps, num of features]).
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Output data.
        task : string
            either regression or classification.

        Returns
        -------
        tuple
            Tuple containing the reshaped input and output data (X, y).

        Raises
        ------
        ValueError
            If the input dimensions are incorrect.
        """
        if task == "classification":
            return self._for_classification(X, y)
        elif task == "regression":
            return self._for_regression(X, y)

class TransformOutputSklearn(object):
    def __call__(self, y_pred, y_true):
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if y_pred.shape == y_true.shape:
            return y_pred,y_true 
        # y_true = [y_true[i][-1][0] for i in range(len(y_true))]
        # y_pred = [y_pred[i][-1][0] for i in range(len(y_pred))]
        y_pred = y_pred.reshape(y_true.shape)
        return y_pred, y_true

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
    elif X.shape[:2] == y.shape:
        return X, y[:, :, None]
    else:
        y = y.squeeze()
        if y.ndim == 1:  # classification task
            if X.shape[0] == y.shape[0]:
                if readout.method_name in ["Ridge", "ElasticNet", "Lasso", "LinearRegression"]:
                    mask = np.array([val.is_integer() for val in y], dtype=np.int)
                    if np.sum(mask) == len(y):
                        y_numpy = np.zeros((len(y), len(np.unique(y))))
                        y_numpy[np.arange(len(y)), y] = 1
                        y = y_numpy
                        if X.ndim != y.ndim:
                            raise ValueError("Current sklearn regressors do not support per time step regression task. Use scipy classifiers")
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