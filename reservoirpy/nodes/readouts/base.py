# Author: Nathan Trouvain at 27/09/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from ...node import Node
from ...utils.validation import add_bias, check_vector


def _initialize_readout(
    readout, x=None, y=None, init_func=None, bias_init=None, bias=True
):

    if x is not None:

        in_dim = x.shape[1]

        if readout.output_dim is not None:
            out_dim = readout.output_dim
        elif y is not None:
            out_dim = y.shape[1]
        else:
            raise RuntimeError(
                f"Impossible to initialize {readout.name}: "
                f"output dimension was not specified at "
                f"creation, and no teacher vector was given."
            )

        readout.set_input_dim(in_dim)
        readout.set_output_dim(out_dim)

        if callable(init_func):
            W = init_func(in_dim, out_dim, dtype=readout.dtype)
        elif isinstance(init_func, np.ndarray):
            W = (
                check_vector(init_func, caller=readout)
                .reshape(readout.input_dim, readout.output_dim)
                .astype(readout.dtype)
            )
        else:
            raise ValueError(
                f"Data type {type(init_func)} not "
                f"understood for matrix initializer "
                f"'Wout'. It should be an array or "
                f"a callable returning an array."
            )

        if bias:
            if callable(bias_init):
                bias = bias_init(1, out_dim, dtype=readout.dtype)
            elif isinstance(bias_init, np.ndarray):
                bias = (
                    check_vector(bias_init)
                    .reshape(1, readout.output_dim)
                    .astype(readout.dtype)
                )
            else:
                raise ValueError(
                    f"Data type {type(bias_init)} not "
                    f"understood for matrix initializer "
                    f"'bias'. It should be an array or "
                    f"a callable returning an array."
                )
        else:
            bias = np.zeros((1, out_dim), dtype=readout.dtype)

        readout.set_param("Wout", W)
        readout.set_param("bias", bias)


def _prepare_inputs_for_learning(X=None, Y=None, bias=True, allow_reshape=False):
    if X is not None:

        if bias:
            X = add_bias(X)
        if not isinstance(X, np.ndarray):
            X = np.vstack(X)

        X = check_vector(X, allow_reshape=allow_reshape)

    if Y is not None:

        if not isinstance(Y, np.ndarray):
            Y = np.vstack(Y)

        Y = check_vector(Y, allow_reshape=allow_reshape)

    return X, Y


def readout_forward(node: Node, x):
    return (node.Wout.T @ x.reshape(-1, 1) + node.bias.T).T


def _assemble_wout(Wout, bias, has_bias=True):
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


def _compute_error(node, x, y=None):
    """Error between target and prediction."""
    prediction = node.state()
    error = prediction - y
    return error, x.T
