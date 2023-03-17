from functools import wraps
from typing import Callable

import numpy as np
from sklearn import linear_model

def get_linear(name: str) -> Callable:
    """Return an activation function from name.

    Parameters
    ----------
    name : str
        Name of the activation function.
        Can be one of {'softmax', 'softplus',
        'sigmoid', 'tanh', 'identity', 'relu'} or
        their respective short names {'smax', 'sp',
        'sig', 'id', 're'}.

    Returns
    -------
    callable
        An activation function.
    """
    index = {
        "linear_regression": linear_regression,
        "ridge_regression": ridge_regression,
        "elastic_net":elastic_net,
        "lasso":lasso
    }

    if index.get(name) is None:
        raise ValueError(f"Function name must be one of {[k for k in index.keys()]}")
    else:
        return index[name]

def linear_regression(**kwargs):
    return linear_model.LinearRegression(fit_intercept=kwargs["fit_intercept"])

def ridge_regression(**kwargs):
    return linear_model.Ridge(alpha=kwargs["alpha"],
        max_iter=kwargs["max_iter"],
        tol=kwargs["tol"],
        fit_intercept=kwargs["fit_intercept"])

def elastic_net(**kwargs):
    return linear_model.ElasticNet(
        alpha=kwargs["alpha"],
        max_iter=kwargs["max_iter"],
        tol=kwargs["tol"],
        fit_intercept=kwargs["fit_intercept"],
        warm_start=kwargs['warm_start'],
        l1_ratio=kwargs['l1_ratio']
        )

def lasso(**kwargs):
    return linear_model.Lasso(
        alpha=kwargs["alpha"],
        max_iter=kwargs["max_iter"],
        tol=kwargs["tol"],
        fit_intercept=kwargs["fit_intercept"],
        warm_start=kwargs['warm_start'],
        )
