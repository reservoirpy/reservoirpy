"""Simple regression models for readout matrix learning.

This module provides simple linear models that can be used
to compute the readout matrix coefficients with simple
linear regression algorithms, like _ridge regularized regression
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
    model = ridge_linear_model(_ridge=1.e-6)
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
from abc import ABCMeta

import numpy as np

from scipy import linalg
from joblib import Parallel, delayed

from .utils.validation import check_vector, add_bias
from .utils.parallel import lock as global_lock
from .utils.parallel import manager, get_joblib_backend, as_memmap, clean_tempfile
from .utils.types import Weights, Data


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

    The _ridge regression is performed following this equation:

    .. math::

        W_{out} = YX^{T} \cdot (XX^{T} + \mathrm{_ridge} \\times \mathrm{Id}_{_dim_in})

    where :math:`W_out` is the readout matrix learnt through this regression,
    :math:`X` are the internal states, :math:`Y` are the ground truth vectors,
    and :math:`_dim_in` is the internal state dimension (number of units in the
    reservoir).


    Parameters
    ----------
    ridge : float, optional
        Regularization parameter. By default, equal to 0.

    typefloat : numpy._dtype, optional
    """
    def ridge_model_solving(X, Y):

        ridgeid = (ridge*np.eye(X.shape[0])).astype(typefloat)
        return linalg.solve(np.dot(X, X.T) + ridgeid, np.dot(Y, X.T).T, assume_a="sym").T

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


def _solve_ridge(XXT, YXT, ridge):
    return linalg.solve(XXT + ridge, YXT.T, assume_a="sym").T


def _prepare_inputs(X, Y, bias=True, allow_reshape=False):
    if bias:
        X = add_bias(X)
    if not isinstance(X, np.ndarray):
        X = np.vstack(X)
    if not isinstance(Y, np.ndarray):
        Y = np.vstack(Y)

    X = check_vector(X, allow_reshape=allow_reshape)
    Y = check_vector(Y, allow_reshape=allow_reshape)

    return X, Y


def _check_tikhnonv_terms(XXT, YXT, x, y):
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Impossible to perform _ridge regression: dimension mismatch "
                         f"between target sequence of shape {y.shape} and state sequence "
                         f"of shape {x.shape} ({x.shape[0]} != {y.shape[0]}).")

    if y.shape[1] != YXT.shape[0]:
        raise ValueError(f"Impossible to perform _ridge regression: dimension mismatch "
                         f"between target sequence of shape {y.shape} and expected ouptut "
                         f"dimension ({YXT.shape[0]}) ({y.shape[1]} != {YXT.shape[0]})")


class Model(metaclass=ABCMeta):

    Wout: Weights
    _dim_in: int
    _dim_out: int
    _initialized = False

    @property
    def initialized(self):
        return self._initialized

    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        return self._dim_out

    def fit(self, X: Data = None, Y: Data = None) -> Weights:
        raise NotImplementedError


class OfflineModel(Model, metaclass=ABCMeta):

    def partial_fit(self, X: Data, Y: Data):
        raise NotImplementedError


class OnlineModel(Model, metaclass=ABCMeta):

    def step_fit(self, X: Data, Y: Data) -> Weights:
        raise NotImplementedError


class RidgeRegression(OfflineModel):

    def __init__(self, ridge=0.0, workers=-1, dtype=np.float64):
        self.workers = workers

        self._dtype = dtype
        self._ridge = ridge
        self._ridgeid = None
        self._XXT = None
        self._YXT = None

    @property
    def ridge(self):
        return self._ridge

    @ridge.setter
    def ridge(self, value):
        self._ridge = value
        if self._initialized:
            self._reset_ridge_matrix()

    def _reset_ridge_matrix(self):
        self._ridgeid = (self._ridge * np.eye(self._dim_in + 1, dtype=self._dtype))

    def initialize(self, dim_in=None, dim_out=None):

        if dim_in is not None:
            self._dim_in = dim_in
        if dim_out is not None:
            self._dim_out = dim_out

        if getattr(self, "Wout", None) is None:
            self.Wout = np.zeros((self._dim_in + 1, self._dim_out), dtype=self._dtype)
        if getattr(self, "_XXT", None) is None:
            self._XXT = as_memmap(np.zeros((self._dim_in + 1, self._dim_in + 1), dtype=self._dtype), caller=self)
        if getattr(self, "_YXT", None) is None:
            self._YXT = as_memmap(np.zeros((self._dim_out, self._dim_in + 1), dtype=self._dtype), caller=self)
        if getattr(self, "_ridgeid", None) is None:
            self._reset_ridge_matrix()

        self._initialized = True

    def clean(self):
        del self._XXT
        del self._YXT
        if self._initialized:
            self.initialize()

    def partial_fit(self, X, Y):

        X, Y = _prepare_inputs(X, Y, allow_reshape=True)

        if not self._initialized:
            raise RuntimeError(f"RidgeRegression model was never initialized. Call "
                               f"'initialize() first.")

        _check_tikhnonv_terms(self._XXT, self._YXT, X, Y)

        xxt = X.T.dot(X)
        yxt = Y.T.dot(X)

        # Lock the memory map to avoid increment from
        # different processes at the same time (Numpy doesn't like that).
        with global_lock:
            self._XXT += xxt
            self._YXT += yxt


    def fit(self, X=None, Y=None):

        if X is not None and Y is not None:
            if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
                self.partial_fit(X, Y)

            elif isinstance(X, list) and isinstance(Y, list):
                # if all states and all teachers are given at once,
                # perform partial fit anyway to avoid memory overload.
                workers = min(self.workers, len(X))
                backend = get_joblib_backend(workers)
                with Parallel(n_jobs=workers, backend=backend) as parallel:
                    parallel(delayed(self.partial_fit)(x, y)
                             for x, y in zip(X, Y))

        self.Wout = _solve_ridge(self._XXT, self._YXT, self._ridgeid)

        clean_tempfile(self)
        self.clean()

        return self.Wout


class SklearnLinearModel(OfflineModel):

    def __init__(self, model, workers=-1, dtype=np.float64):
        self.model = model
        self.workers = workers
        self.dtype = dtype

        self._X = None
        self._Y = None

    def initialize(self, dim_in=None, dim_out=None):

        if dim_in is not None:
            self._dim_in = dim_in
        if dim_out is not None:
            self._dim_out = dim_out

        if getattr(self, "Wout", None) is None:
            self.Wout = np.zeros((self._dim_in + 1, self._dim_out), dtype=self.dtype)
        if getattr(self, "_X", None) is None:
            self._X = manager.list()
        if getattr(self, "_Y", None) is None:
            self._Y = manager.list()

        self._initialized = True

    def clean(self):
        del self._X
        del self._Y
        if self._initialized:
            self.initialize()

    def partial_fit(self, X, Y):

        X, Y = _prepare_inputs(X, Y, bias=False)

        if not self._initialized:
            raise RuntimeError(f"SklearnLinearModel model was never initialized. Call "
                               f"'initialize() first.")

        with global_lock:
            self._X.append(X)
            self._Y.append(Y)

    def fit(self, X=None, Y=None):

        if X is not None and Y is not None:
            X, Y = _prepare_inputs(X, Y, bias=False)
        else:
            X, Y = _prepare_inputs(self._X, self._Y, bias=False)

        self.model.fit(X, Y)

        # linear_model provides Matrix A and Vector b
        # such as A * X[1:, :] + b ~= Ytarget
        A = np.asmatrix(self.model.coef_)
        b = np.asmatrix(self.model.intercept_).T

        # Then the matrix W = "[b | A]" statisfies "W * X ~= Ytarget"
        self.Wout = np.asarray(np.hstack([b, A]), dtype=self.dtype)

        self.clean()

        return self.Wout
