# Author: Nathan Trouvain at 24/09/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial
import numpy as np

from .utils import (readout_forward, _initialize_readout,
                    _prepare_inputs_for_learning)

from ..node import Node


def train(node: "FORCE", x=None, y=None):
    x, y = _prepare_inputs_for_learning(x, y, bias=node.input_bias,
                                        allow_reshape=True)

    P = node.P
    r = x.T

    k = np.dot(P, r)
    rPr = np.dot(r.T, k)
    c = float(1.0 / (1.0 + rPr))
    P = P - c * np.outer(k, k)

    wo = node.Wout
    if node.input_bias:
        bias = node.bias
        wo = np.r_[bias, wo]

    if y is None and node.has_feedback:
        y = node.feedback()

    e = node.state() - y
    dw = -c * np.outer(e, k)

    wo = wo + dw.T

    if node.input_bias:
        Wout, bias = wo[1:, :], wo[0, :][np.newaxis, :]
        node.set_param("Wout", Wout)
        node.set_param("bias", bias)
    else:
        node.set_param("Wout", wo)

    node.set_param("P", P)


def initialize(readout: "FORCE",
               x=None,
               y=None,
               init_func=None,
               bias=None):

    _initialize_readout(readout, x, y, init_func, bias)

    if x is not None:
        input_dim, alpha = readout.input_dim, readout.alpha

        if readout.input_bias:
            input_dim += 1

        P = np.asmatrix(np.eye(input_dim)) / alpha

        readout.set_param("P", P)
        readout.set_param("step", 0)


class FORCE(Node):

    # A special thanks to Lionel Eyraud-Dubois and
    # Olivier Beaumont for their improvement of this method.

    def __init__(self, output_dim=None, alpha=1e-6,
                 Wout_init=np.zeros, input_bias=True, name=None):
        super(FORCE, self).__init__(params={"Wout": None,
                                            "bias": None,
                                            "P": None,
                                            "step": None},
                                    hypers={"alpha": alpha,
                                            "input_bias": input_bias},
                                    forward=readout_forward,
                                    train=train,
                                    initializer=partial(initialize,
                                                        init_func=Wout_init,
                                                        bias=input_bias),
                                    output_dim=output_dim,
                                    name=name)
