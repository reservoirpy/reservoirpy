# Author: Nathan Trouvain at 06/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
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

from reservoirpy.node import Node


def _reset_buffers(step, rTPs, factors):
    step[:] = np.zeros_like(step)
    rTPs[:] = np.zeros_like(rTPs)
    factors[:] = np.zeros_like(factors)


def train(readout: "BatchFORCE", x=None, y=None):

    if x is not None:

        x, y = _prepare_inputs_for_learning(x, y, bias=readout.has_bias,
                                            allow_reshape=True)

        W = readout.Wout
        if readout.has_bias:
            bias = readout.bias
            W = np.c_[bias, W]

        P = readout.P
        r = x.T
        output = readout.state()

        factors = readout.get_buffer("factors")
        rTPs = readout.get_buffer("rTPs")
        steps = readout.get_buffer("step")
        step = int(steps[0])

        error = output.T - y.T

        rt = r.T
        rTP = (rt @ P) - (rt @ (factors * rTPs)) @ rTPs.T
        factor = float(1.0 / (1.0 + rTP @ r))

        factors[step] = factor
        rTPs[:, step] = rTP

        new_rTP = rTP * (1 - factor * (rTP @ r).item())

        W -= error @ new_rTP

        if readout.has_bias:
            readout.set_param("Wout", W[:, 1:])
            readout.set_param("bias", W[:, :1])
        else:
            readout.set_param("Wout", W)

        step += 1

        if step == readout.batch_size:
            P -= (factors * rTPs) @ rTPs.T
            _reset_buffers(steps, rTPs, factors)


def initialize(readout: "BatchFORCE",
               x=None,
               y=None,
               init_func=None,
               bias=None):

    _initialize_readout(readout, x, y, init_func, bias)

    if x is not None:
        input_dim, alpha = readout.input_dim, readout.alpha

        if readout.has_bias:
            input_dim += 1

        P = np.asmatrix(np.eye(input_dim)) / alpha

        readout.set_param("P", P)


def initialize_buffers(readout: "BatchFORCE"):
    bias_dim = 0
    if readout.has_bias:
        bias_dim = 1

    readout.create_buffer("rTPs", (readout.input_dim + bias_dim,
                                   readout.batch_size))
    readout.create_buffer("factors", (readout.batch_size, ))
    readout.create_buffer("step", (1, ))


class BatchFORCE(Node):

    # A special thanks to Lionel Eyraud-Dubois and
    # Olivier Beaumont for their improvement of this method.

    def __init__(self, output_dim=None, alpha=1e-6, batch_size=1,
                 Wout_init=np.zeros, bias=True, name=None):
        super(BatchFORCE, self).__init__(params={"Wout": None,
                                            "bias": None,
                                            "P": None},
                                    hypers={"alpha": alpha,
                                            "batch_size": batch_size,
                                            "has_bias": bias},
                                    forward=readout_forward,
                                    train=train,
                                    initializer=partial(initialize,
                                                        init_func=Wout_init,
                                                        bias=bias),
                                    buffers_initializer=initialize_buffers,
                                    output_dim=output_dim,
                                    name=name)
