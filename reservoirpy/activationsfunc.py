"""
=============================================================
Activations functions (:py:mod:`reservoirpy.activationsfunc`)
=============================================================

Activation functions for reservoir, feedback and output.

.. autosummary::
   :toctree: generated/

    get_function
    identity
    sigmoid
    tanh
    relu
    softmax
    softplus

"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Callable, Union

import numpy as np


def get_function(name: Union[Callable, str]) -> Callable:
    """Return an activation function from name.

    Parameters
    ----------
    name : str, Callable
        Name of the activation function.
        Can be one of {'softmax', 'softplus',
        'sigmoid', 'tanh', 'identity', 'relu'} or
        their respective short names {'smax', 'sp',
        'sig', 'id', 're'}. If `name` is a Callable,
        simply returns `name`.

    Returns
    -------
    callable
        An activation function.
    """
    if isinstance(name, Callable):
        return name

    index = {
        "softmax": softmax,
        "smax": softmax,
        "softplus": softplus,
        "sp": softplus,
        "sigmoid": sigmoid,
        "sig": sigmoid,
        "tanh": tanh,
        "identity": identity,
        "id": identity,
        "relu": relu,
        "re": relu,
    }

    if index.get(name) is None:
        raise ValueError(f"Function name must be one of {[k for k in index.keys()]}")
    else:
        return index[name]


def softmax(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Softmax activation function.

    .. math::

        y_k = \\frac{e^{\\beta x_k}}{\\sum_{i=1}^{n} e^{\\beta x_i}}

    Parameters
    ----------
    x : array
        Input array.
    beta: float, default to 1.0
        Beta parameter of softmax.
    Returns
    -------
    array
        Activated vector.
    """
    _x = np.asarray(x)
    return np.exp(beta * _x) / np.exp(beta * _x).sum(axis=-1, keepdims=True)


def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus activation function.

    .. math::

        f(x) = \\mathrm{ln}(1 + e^{x})

    Can be used as a smooth version of ReLU.

    Parameters
    ----------
    x : array
        Input array.
    Returns
    -------
    array
        Activated vector.
    """
    return np.log(1.0 + np.exp(x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function.

    .. math::

        f(x) = \\frac{1}{1 + e^{-x}}


    Parameters
    ----------
    x : array
        Input array.
    Returns
    -------
    array
        Activated vector.
    """
    # we adapt the formula used to avoid saturation in case of high values in exp (0.0 instead of nans)
    return np.where(x < 0, np.exp(x) / (np.exp(x) + 1.0), 1.0 / (1.0 + np.exp(-x)))


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation function.

    .. math::

        f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}

    Parameters
    ----------
    x : array
        Input array.
    Returns
    -------
    array
        Activated vector.
    """
    return np.tanh(x)


def identity(x: np.ndarray) -> np.ndarray:
    """Identity function.

    .. math::

        f(x) = x

    Provided for convenience.

    Parameters
    ----------
    x : array
        Input array.
    Returns
    -------
    array
        Activated vector.
    """
    return x


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function.

    .. math::

        f(x) = x ~~ \\mathrm{if} ~~ x > 0 ~~ \\mathrm{else} ~~ 0

    Parameters
    ----------
    x : array
        Input array.
    Returns
    -------
    array
        Activated vector.
    """
    return np.maximum(x, 0.0)
