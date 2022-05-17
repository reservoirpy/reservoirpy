# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial

import numpy as np
from scipy import linalg

from ...mat_gen import zeros
from ...node import Node
from ...type import global_dtype
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


def initialize(readout: Node, x=None, y=None, bias_init=None, Wout_init=None):

    _initialize_readout(
        readout, x, y, bias=readout.input_bias, init_func=Wout_init, bias_init=bias_init
    )


def initialize_buffers(readout):
    """create memmaped buffers for matrices X.X^T and Y.X^T pre-computed
    in parallel for ridge regression
    ! only memmap can be used ! Impossible to share Numpy arrays with
    different processes in r/w mode otherwise (with proper locking)
    """
    input_dim = readout.input_dim
    output_dim = readout.output_dim

    if readout.input_bias:
        input_dim += 1

    readout.create_buffer("XXT", (input_dim, input_dim))
    readout.create_buffer("YXT", (output_dim, input_dim))


class Ridge(Node):
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
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    ridge: float, default to 0.0
        L2 regularization parameter.
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
    """

    def __init__(
        self,
        output_dim=None,
        ridge=0.0,
        Wout=zeros,
        bias=zeros,
        input_bias=True,
        name=None,
    ):
        super(Ridge, self).__init__(
            params={"Wout": None, "bias": None},
            hypers={"ridge": ridge, "input_bias": input_bias},
            forward=readout_forward,
            partial_backward=partial_backward,
            backward=backward,
            output_dim=output_dim,
            initializer=partial(initialize, Wout_init=Wout, bias_init=bias),
            buffers_initializer=initialize_buffers,
            name=name,
        )
