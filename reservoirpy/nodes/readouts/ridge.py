# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial
from typing import Any, Dict, List

import numpy as np
from scipy import linalg

from ...mat_gen import zeros
from ...node import Node
from ...type import Shape, global_dtype
from ...utils.parallel import clean_tempfile, memmap_buffer
from .base import _initialize_readout, _prepare_inputs_for_learning, readout_forward


def _solve_ridge(XXT, YXT, ridge):
    """Solve Tikhonov regression."""
    return linalg.solve(XXT + ridge, YXT.T, assume_a="sym")


def _accumulate(readout, xxt, yxt):
    """Aggregate Xi.Xi^T and Yi.Xi^T matrices from a state sequence i."""
    XXT = readout.get_buffer("XXT")
    YXT = readout.get_buffer("YXT")
    XXT += xxt
    YXT += yxt


def partial_backward(readout: Node, X_batch, Y_batch=None, lock=None):
    """Pre-compute XXt and YXt before final fit."""
    X, Y = _prepare_inputs_for_learning(
        X_batch,
        Y_batch,
        bias=readout.input_bias,
        allow_reshape=True,
    )

    xxt = X.T.dot(X)
    yxt = Y.T.dot(X)

    if lock is not None:
        # This is not thread-safe using Numpy memmap as buffers
        # ok for parallelization then with a lock (see ESN object)
        with lock:
            _accumulate(readout, xxt, yxt)
    else:
        _accumulate(readout, xxt, yxt)


def backward(readout: Node, *args, **kwargs):
    ridge = readout.ridge
    XXT = readout.get_buffer("XXT")
    YXT = readout.get_buffer("YXT")

    input_dim = readout.input_dim
    if readout.input_bias:
        input_dim += 1

    ridgeid = ridge * np.eye(input_dim, dtype=global_dtype)

    Wout_raw = _solve_ridge(XXT, YXT, ridgeid)

    if readout.input_bias:
        Wout, bias = Wout_raw[1:, :], Wout_raw[0, :][np.newaxis, :]
        readout.set_param("Wout", Wout)
        readout.set_param("bias", bias)
    else:
        readout.set_param("Wout", Wout_raw)

    readout.clean_buffers()


def initialize(readout: Node, x=None, y=None, bias_init=None, Wout_init=None):
    _initialize_readout(
        readout, x, y, bias=readout.input_bias, init_func=Wout_init, bias_init=bias_init
    )
    readout.initialize_buffers()


class Ridge(Node):
    _buffers: Dict[str, Any]

    _X: List  # For partial_fit default behavior (store first, then fit)
    _Y: List

    """A single layer of neurons learning with Tikhonov linear regression.

    Output weights of the layer are computed following:

    .. math::

        \\hat{\\mathbf{W}}_{out} = \\mathbf{YX}^\\top ~ (\\mathbf{XX}^\\top +
        \\lambda\\mathbf{Id})^{-1}

    Outputs :math:`\\mathbf{y}` of the node are the result of:

    .. math::

        \\mathbf{y} = \\mathbf{W}_{out}^\\top \\mathbf{x} + \\mathbf{b}

    where:
        - :math:`\\mathbf{X}` is the accumulation of all inputs during training;
        - :math:`\\mathbf{Y}` is the accumulation of all targets during training;
        - :math:`\\mathbf{b}` is the first row of :math:`\\hat{\\mathbf{W}}_{out}`;
        - :math:`\\mathbf{W}_{out}` is the rest of :math:`\\hat{\\mathbf{W}}_{out}`.

    If ``input_bias`` is True, then :math:`\\mathbf{b}` is non-zero, and a constant
    term is added to :math:`\\mathbf{X}` to compute it.

    :py:attr:`Ridge.params` **list**

    ================== =================================================================
    ``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
    ``bias``           Learned bias (:math:`\\mathbf{b}`).
    ================== =================================================================

    :py:attr:`Ridge.hypers` **list**

    ================== =================================================================
    ``ridge``          Regularization parameter (:math:`\\lambda`) (0.0 by default).
    ``input_bias``     If True, learn a bias term (True by default).
    ================== =================================================================


    Parameters
    ----------
    ridge: float, default to 0.0
        L2 regularization parameter.
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    Wout : callable or array-like of shape (units, targets), default to :py:func:`~reservoirpy.mat_gen.zeros`
        Output weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.zeros`
        Bias weights vector or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    input_bias : bool, default to True
        If True, then a bias parameter will be learned along with output weights.
    name : str, optional
        Node name.

    Example
    -------

    >>> x = np.random.normal(size=(100, 3))
    >>> noise = np.random.normal(scale=0.1, size=(100, 1))
    >>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.
    >>>
    >>> from reservoirpy.nodes import Ridge
    >>> ridge_regressor = Ridge(ridge=0.001)
    >>>
    >>> ridge_regressor.fit(x, y)
    >>> ridge_regressor.Wout, ridge_regressor.bias
    array([[ 9.992, -0.205,  6.989]]).T, array([[12.011]])
    """

    def __init__(
        self,
        ridge=0.0,
        output_dim=None,
        Wout=zeros,
        bias=zeros,
        input_bias=True,
        name=None,
    ):
        # buffers are all node state components that should not live
        # outside the node training loop, like partial computations for
        # linear regressions. They can also be shared across multiple processes
        # when needed.
        self._buffers = dict()

        super(Ridge, self).__init__(
            params={"Wout": None, "bias": None},
            hypers={"ridge": ridge, "input_bias": input_bias},
            forward=readout_forward,
            partial_backward=partial_backward,
            backward=backward,
            output_dim=output_dim,
            initializer=partial(initialize, Wout_init=Wout, bias_init=bias),
            name=name,
        )

    def initialize_buffers(self):
        """create memmaped buffers for matrices X.X^T and Y.X^T pre-computed
        in parallel for ridge regression
        ! only memmap can be used ! Impossible to share Numpy arrays with
        different processes in r/w mode otherwise (with proper locking)
        """
        if len(self._buffers) == 0:
            input_dim = self.input_dim
            output_dim = self.output_dim

            if self.input_bias:
                input_dim += 1

            self.create_buffer("XXT", (input_dim, input_dim))
            self.create_buffer("YXT", (output_dim, input_dim))

        return self

    def create_buffer(
        self, name: str, shape: Shape = None, data: np.ndarray = None, as_memmap=True
    ):
        """Create a buffer array on disk, using numpy.memmap. This can be
        used to store transient variables on disk. Typically, called inside
        a `buffers_initializer` function.

        Parameters
        ----------
        name : str
            Name of the buffer array.
        shape : tuple of int, optional
            Shape of the buffer array.
        data : array-like
            Data to store in the buffer array.
        """
        if as_memmap:
            self._buffers[name] = memmap_buffer(self, data=data, shape=shape, name=name)
        else:
            if data is not None:
                self._buffers[name] = data
            else:
                self._buffers[name] = np.empty(shape)

    def set_buffer(self, name: str, value: np.ndarray):
        """Dump data in the buffer array.

        Parameters
        ----------
        name : str
            Name of the buffer array.
        value : array-like
            Data to store in the buffer array.
        """
        self._buffers[name][:] = value.astype(self.dtype)

    def get_buffer(self, name) -> np.memmap:
        """Get data from a buffer array.

        Parameters
        ----------
        name : str
            Name of the buffer array.

        Returns
        -------
            numpy.memmap
                Data as Numpy memory map.
        """
        if self._buffers.get(name) is None:
            raise AttributeError(f"No buffer named '{name}' in {self}.")
        return self._buffers[name]

    def clean_buffers(self):
        """Clean Node's buffer arrays."""
        if len(self._buffers) > 0:
            self._buffers = dict()
            clean_tempfile(self)

        # Empty possibly stored inputs and targets in default buffer.
        self._X = self._Y = []
