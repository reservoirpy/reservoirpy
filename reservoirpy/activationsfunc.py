"""Activations functions for reservoir, feedback and output.
"""
from typing import Callable
from functools import wraps

import numpy as np


def _elementwise(func):
    """Vectorize a function to apply it
    on arrays.
    """
    vect = np.vectorize(func)

    @wraps(func)
    def vect_wrapper(*args, **kwargs):
        u = np.asanyarray(args)
        v = vect(u)
        return v[0]

    return vect_wrapper


def get_function(name: str) -> Callable:  # pragma: no cover
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
    Callable
        An activation function.
    """
    index = {
        "softmax": softmax,
        "softplus": softplus,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "identity": identity,
        "relu": relu,
        "smax": softmax,
        "sp": softplus,
        "sig": sigmoid,
        "id": identity,
        "re": relu,
    }

    if index.get(name) is None:
        raise ValueError(f"Function name must be one of "
                         f"{[k for k in index.keys()]}")
    else:
        return index[name]


def softmax(x: np.ndarray, beta=1) -> np.ndarray:
    """Softmax activation function:

    .. math::

        y_k = \\frac{e^{x_k}}{\\sum_{i=1}^{n} e^{x_i}}

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    return np.exp(beta*x) / np.exp(beta*x).sum()


@_elementwise
def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus acctivation function:

    .. math::

        f(x) = \\mathrm{ln}(1 + e^{x})

    Can be used as a smooth version of ReLU.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    return np.log(1 + np.exp(x))


@_elementwise
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function:

    .. math::

        f(x) = \\frac{1}{1 + e^{-x}}


    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    if x < 0:
        u = np.exp(x)
        return u / (u + 1)
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation function:

    .. math::

        f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    return np.tanh(x)


@_elementwise
def identity(x: np.ndarray) -> np.ndarray:
    """Identity function :

    .. math::

        f(x) = x

    Provided for convenience.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    return x


@_elementwise
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function:

    .. math::
        f(x) = x ~~ \\mathrm{if} ~~ x > 0 ~~ \\mathrm{else} ~~ 0

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    if x < 0:
        return 0
    return x
