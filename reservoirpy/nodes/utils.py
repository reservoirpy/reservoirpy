# Author: Nathan Trouvain at 27/09/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from ..utils.validation import check_vector, add_bias
from ..types import global_dtype
from reservoirpy.node import Node


def _initialize_readout(readout, x=None, y=None, init_func=None, bias=True):

    if x is not None:

        in_dim = x.shape[1]

        if readout.output_dim is not None:
            out_dim = readout.output_dim
        elif y is not None:
            out_dim = y.shape[1]
        else:
            raise RuntimeError(f"Impossible to initialize {readout.name}: "
                               f"output dimension was not specified at "
                               f"creation, and no teacher vector was given.")

        readout.set_input_dim(in_dim)
        readout.set_output_dim(out_dim)

        if init_func is None:
            init_func = np.zeros

        if callable(init_func):
            if bias:
                in_dim += 1

            W = init_func((in_dim, out_dim), dtype=global_dtype)

        elif isinstance(init_func, np.ndarray):
            W = init_func
            W = W.reshape(readout.input_dim + int(bias), readout.output_dim)
        else:
            raise ValueError(f"Data type {type(init_func)} not "
                             f"understood for matrix initializer "
                             f"'Wout_init'. It should be an array or "
                             f"a callable returning an array.")

        if bias:
            Wout = W[1:, :]
            bias = W[:1, :].reshape((1, out_dim))
        else:
            Wout = W
            bias = np.zeros((1, out_dim)) # TODO: bias from callable

        readout.set_param("Wout", Wout)
        readout.set_param("bias", bias)


def _prepare_inputs_for_learning(X=None, Y=None, bias=True, transient=0,
                                 allow_reshape=False):

    seq_len = None
    if X is not None:

        seq_len = len(X)

        if bias:
            X = add_bias(X)
        if not isinstance(X, np.ndarray):
            X = np.vstack(X)

        X = check_vector(X, allow_reshape=allow_reshape)

        if seq_len > transient:
            X = X[transient:]

    if Y is not None:

        if not isinstance(Y, np.ndarray):
            Y = np.vstack(Y)

        Y = check_vector(Y, allow_reshape=allow_reshape)

        if seq_len is None:
            seq_len = len(Y)

        if seq_len > transient:
            Y = Y[transient:]
        # TODO: else raise ?

    return X, Y


def readout_forward(node: Node, x):
    return (node.Wout.T @ x.T + node.bias.T).T


def _assemble_wout(Wout,  bias, has_bias=True):
    wo = Wout
    if has_bias:
        wo = np.r_[bias, wo]
    return wo


def _split_and_save_wout(node, wo):
    if node.input_bias:
        Wout, bias = wo[1:, :], wo[0, :][np.newaxis, :]
        node.set_param("Wout", Wout)
        node.set_param("bias", bias)
    else:
        node.set_param("Wout", wo)
