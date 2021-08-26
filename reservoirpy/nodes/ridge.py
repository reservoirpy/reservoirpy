# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from ..node import Node


def forward(readout: Node, x):

    return (x @ readout.Wout.T + readout.bias.T).T


def partial_backward(readout: None, x, y):
    ...


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


class Ridge(Node):

    def __init__(self, output_dim=None, ridge=0.0, name=None):
        # we add _XXT and _YXT as protected parameters to be used by parallel
        # manager in a thread safe way.
        super(Ridge, self).__init__(params={"Wout": None, "bias": None,
                                            "_XXT": None, "_YXT": None},
                                    hypers={"ridge": ridge},
                                    forward=forward,
                                    output_dim=output_dim,
                                    initializer=initialize,
                                    name=name)
