# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from scipy import linalg

from ..node import Node
from ..utils.parallel import get_lock
from ..utils.types import global_dtype
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


def forward(readout: Node, x):
    return (x @ readout.Wout.T + readout.bias.T).T


def partial_backward(readout: Node, X_batch, Y_batch=None):
    X, Y = _prepare_inputs(X_batch, Y_batch, allow_reshape=True)

    xxt = X.T.dot(X)
    yxt = Y.T.dot(X)

    # Lock the memory map to avoid increment from
    # different processes at the same time (Numpy doesn't like that).
    with get_lock():
        readout.set_buffer("XXT", readout.get_buffer("XXT") + xxt)
        readout.set_buffer("YXT", readout.get_buffer("YXT") + yxt)


def backward(readout: Node, X=None, Y=None):
    ridge = readout.ridge
    XXT = readout.get_buffer("XXT")
    YXT = readout.get_buffer("YXT")

    ridgeid = (ridge * np.eye(readout.input_dim + 1, dtype=global_dtype))

    Wout_raw = _solve_ridge(XXT, YXT, ridgeid)

    Wout, bias = Wout_raw[:, 1:], Wout_raw[:, 1]

    readout.set_param("Wout", Wout)
    readout.set_param("bias", bias)


def initialize(readout: Node,
               x=None,
               y=None):

    if x is not None:

        in_dim = x.shape[1]

        if y is not None:
            out_dim = y.shape[1]
        else:
            out_dim = readout.output_dim

        readout.set_input_dim(in_dim)
        readout.set_output_dim(out_dim)

        Wout = np.zeros((out_dim, in_dim))
        bias = np.zeros((out_dim, 1))

        readout.set_param("Wout", Wout)
        readout.set_param("bias", bias)

        # create memmaped buffers for matrices X.X^T and Y.X^T pre-computed
        # in parallel for ridge regression
        # ! only memmap can be used ! Impossible to share Numpy arrays with
        # different processes in r/w mode otherwise (with proper locking)
        readout.create_buffer("XXT", (readout.input_dim + 1,
                                      readout.input_dim + 1))
        readout.create_buffer("YXT", (readout.output_dim,
                                      readout.input_dim + 1))


class Ridge(Node):

    def __init__(self, output_dim=None, ridge=0.0, transient=0, name=None):
        super(Ridge, self).__init__(params={"Wout": None, "bias": None},
                                    hypers={"ridge": ridge,
                                            "transient": transient},
                                    forward=forward,
                                    partial_backward=partial_backward,
                                    backward=backward,
                                    output_dim=output_dim,
                                    initializer=initialize,
                                    name=name)
