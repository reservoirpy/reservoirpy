"""Simple regression models for readout matrix learning.

This module provides simple linear models that can be used
to compute the readout matrix coefficients with simple
linear regression algorithms, like ridge regularized regression
or any linear model from scikit-learn API.

These models are already packed in the :py:class:`compat.ESN`
class, and can instanciated by passing them as arguments to the `ESN`
object.

In most cases, you won't need to call this module directly. Simply
pass the models to the `ESN` object as parameters.
See the :py:class:`compat.ESN` documentation for more information.
"""
from abc import ABCMeta

import numpy as np
from joblib import Parallel, delayed
from scipy import linalg

from ..utils.parallel import get_joblib_backend, as_memmap, clean_tempfile
from ..types import Weights, Data
from ..utils.validation import check_vector, add_bias


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


class _Model(metaclass=ABCMeta):
    """
    Base template for model learning classes.
    """

    Wout: Weights
    _dim_in: int
    _dim_out: int
    _initialized = False

    @property
    def initialized(self):
        """A boolean indicating wether the internal parameters of the
        model are initialized or not."""
        return self._initialized

    @property
    def dim_in(self):
        """Input dimension of the model (i.e. internal
        states dimension)."""
        return self._dim_in

    @property
    def dim_out(self):
        """Output dimension of the model."""
        return self._dim_out

    def fit(self, X: Data = None, Y: Data = None) -> Weights:
        """Fit states X to targets values Y following the model
        learning rule.

        Parameters
        ----------
        X : numpy.ndarray or list of numpy.ndarray
            Internal states of the reservoir.
        Y : numpy.ndarray or list of numpy.ndarray
            Targets values for each states.
        Returns
        -------
            numpy.ndarray
                A readout matrix of shape (targets dimension,
                states dimension + bias (=1)).
        """
        raise NotImplementedError


class _OfflineModel(_Model, metaclass=ABCMeta):

    def partial_fit(self, X: Data, Y: Data):
        """Partially fit the states X to the targets values
        Y. This method can be used to pre-comppute some
        steps of the final fitting method.

        Parameters
        ----------
        X : numpy.ndarray or list of numpy.ndarray
            Internal states of the reservoir.
        Y : numpy.ndarray or list of numpy.ndarray
            Targets values for each states.
        """
        raise NotImplementedError


class RidgeRegression(_OfflineModel):
    """Ridge regression model for reservoir output weights
    learning.

    .. math::

        W_{out} = YX^{T} \\cdot (XX^{T} + \\mathrm{_ridge} \\times \\mathrm{Id}_{_dim_in})

    where :math:`W_out` is the readout matrix learnt through this regression,
    :math:`X` are the internal states, :math:`Y` are the targets vectors,
    and :math:`_dim_in` is the internal state dimension (number of units in the
    reservoir).

    By default, ridge coefficient is set to :math:`0`, which is equivalent to a simple
    analytic resolution using pseudo-inverse.

    Partial fit method allows to concurrently pre-compute :math:`XX^{T]`
    and :math:`YX^{T}` when several independent state sequences are
    provided, for performance, as suggested by [1]_.

    .. [1] Lukosevicius, M. (2012). A Practical Guide to Applying Echo State
           Networks. Neural Networks: Tricks of the Trade.
    """

    def __init__(self, ridge=0.0, workers=-1, dtype=np.float64):
        self.workers = workers

        self._dtype = dtype
        self._ridge = ridge
        self._ridgeid = None
        self._XXT = None
        self._YXT = None

    @property
    def ridge(self):
        """Regularization coefficient of the model."""
        return self._ridge

    @ridge.setter
    def ridge(self, value):
        self._ridge = value
        if self._initialized:
            self._reset_ridge_matrix()

    def _reset_ridge_matrix(self):
        self._ridgeid = (self._ridge * np.eye(self._dim_in + 1, dtype=self._dtype))

    def initialize(self, dim_in: int = None, dim_out: int = None):
        """
        Initialize the model internal parameters.

        Parameters
        ----------
        dim_in : int
            States dimension.
        dim_out : int
            Targets dimension.
        """
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
        """Clean all internal parameters of the model."""
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
