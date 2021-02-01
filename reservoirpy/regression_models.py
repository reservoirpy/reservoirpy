"""Simple regression models for readout matrix learning.

This module provides simple linear models that can be used
to compute the readout matrix coefficients with simple
linear regression algorithms, like ridge regularized regression
or any linear model from scikit-learn API.

These models are already packed in the :py:class:`reservoirpy.ESN`
class, and can instanciated by passing them as arguments to the `ESN`
object.

This module defines all models in the form of parametrizable functions.
This functions then return another function that can be used to fit the
model to the data, as shown in example below.

Example
-------

.. code-block:: python

    from reservoirpy.regression_models import ridge_linear_model
    model = ridge_linear_model(ridge=1.e-6)
    ...
    # get some internal states from a reservoir
    # and ground truth vectors (teachers)
    ...
    # compute the readout
    Wout = model(states, teachers)

Using scikit-learn API (for instance, the `Lasso` regression model):

.. code-block:: python

    from sklearn.linear_model import Lasso
    from reservoirpy.regression_models import sklearn_linear_model
    # parametrize the model
    regressor = Lasso(alpha=0.1)
    # create a model function...
    model = sklearn_linear_model(regressor)
    ...
    # or create an ESN and inject the model
    ...
    esn = ESN(..., reg_model=regressor)

In most cases, you won't need to call this module directly. Simply
pass the models to the `ESN` object as parameters.
See the :py:class:`reservoirpy.ESN` documentation for more information.
"""
from typing import Callable

import numpy as np

from scipy import linalg


def sklearn_linear_model(model: Callable):
    """Create a solver from a scikit-learn linear model.

    Parameters
    ----------
    model : sklearn.linear_model instance
        A scikit-learn linear model.
    """
    def linear_model_solving(X, Y):
        # Learning of the model (first row of X, which contains only ones, is removed)
        model.fit(X[1:, :].T, Y.T)

        # linear_model provides Matrix A and Vector b
        # such as A * X[1:, :] + b ~= Ytarget
        A = np.asmatrix(model.coef_)
        b = np.asmatrix(model.intercept_).T

        # Then the matrix W = "[b | A]" statisfies "W * X ~= Ytarget"
        return np.asarray(np.hstack([b, A]))

    return linear_model_solving


def ridge_linear_model(ridge=0., typefloat=np.float32):
    """Create a solver able to perform a linear regression with
    a L2 regularization, also known as Tikhonov or Ridge regression.

    The ridge regression is performed following this equation:

    .. math::

        W_{out} = YX^{T} \cdot (XX^{T} + \mathrm{ridge} \\times \mathrm{Id}_{N})

    where :math:`W_out` is the readout matrix learnt through this regression,
    :math:`X` are the internal states, :math:`Y` are the ground truth vectors,
    and :math:`N` is the internal state dimension (number of units in the
    reservoir).


    Parameters
    ----------
    ridge : float, optional
        Regularization parameter. By default, equal to 0.

    typefloat : numpy.dtype, optional
    """
    def ridge_model_solving(X, Y):
        ridgeid = (ridge*np.eye(X.shape[0])).astype(typefloat)

        return np.dot(np.dot(Y, X.T), linalg.inv(np.dot(X, X.T) + ridgeid))

    return ridge_model_solving


def pseudo_inverse_linear_model():
    """Create a solver able to perform a linear regression
    using an analytical method without regularization.

    The regression is performed following this equation:

    .. math::

        W_{out} = YX^{T}

    where :math:`W_out` is the readout matrix learnt through this regression,
    :math:`X` are the internal states and :math:`Y` are the ground truth vectors.
    """
    def pseudo_inverse_model_solving(X, Y):
        return np.dot(Y, linalg.pinv(X))

    return pseudo_inverse_model_solving
