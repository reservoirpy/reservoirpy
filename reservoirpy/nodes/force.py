# Author: Nathan Trouvain at 24/09/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from numbers import Number
from typing import Iterable
from functools import partial

import numpy as np

from .utils import (readout_forward, _initialize_readout,
                    _prepare_inputs_for_learning, _assemble_wout,
                    _split_and_save_wout)

from reservoirpy.node import Node


def _rls_like_rule(P, r, e):
    k = np.dot(P, r)
    rPr = np.dot(r.T, k)
    c = float(1.0 / (1.0 + rPr))
    P = P - c * np.outer(k, k)

    dw = -c * np.outer(e, k)

    return dw, P


def _lms_like_rule(alpha, r, e):
    # learning rate is a generator to allow scheduling
    dw = -next(alpha) * np.outer(e, r)
    return dw


def _compute_error(node, x, y=None):
    prediction = node.state()

    if y is None and node.has_feedback:
        y = node.feedback()

    error = prediction - y

    return error, x.T


def rls_like_train(node: "FORCE", x, y=None):
    x, y = _prepare_inputs_for_learning(x, y, bias=node.input_bias,
                                        allow_reshape=True)

    error, r = _compute_error(node, x, y)

    P = node.P
    dw, P = _rls_like_rule(P, r, error)
    wo = _assemble_wout(node.Wout, node.bias, node.input_bias)
    wo = wo + dw.T

    _split_and_save_wout(node, wo)

    node.set_param("P", P)


def lms_like_train(node: "FORCE", x, y=None):
    x, y = _prepare_inputs_for_learning(x, y, bias=node.input_bias,
                                        allow_reshape=True)

    error, r = _compute_error(node, x, y)

    alpha = node._alpha_gen
    dw = _lms_like_rule(alpha, r, error)
    wo = _assemble_wout(node.Wout, node.bias, node.input_bias)
    wo = wo + dw.T

    _split_and_save_wout(node, wo)


def initialize_rls(readout: "FORCE",
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


def initialize_lms(readout: "FORCE",
                   x=None,
                   y=None,
                   init_func=None,
                   bias=None):

    _initialize_readout(readout, x, y, init_func, bias)


class FORCE(Node):

    _rules = ("lms", "rls")

    def __init__(self, output_dim=None, alpha=1e-6, rule="rls",
                 Wout_init=np.zeros, input_bias=True, name=None):

        params = {"Wout": None,
                  "bias": None}

        if rule not in self._rules:
            raise ValueError(f"Unknown rule for FORCE learning. "
                             f"Available rules are {self._rules}.")
        else:
            if rule == "lms":
                train = lms_like_train
                initialize = initialize_lms
            else:
                train = rls_like_train
                initialize = initialize_rls
                params["P"] = None

        if isinstance(alpha, Number):
            def _alpha_gen():
                while True:
                    yield alpha
            alpha_gen = _alpha_gen()
        elif isinstance(alpha, Iterable):
            alpha_gen = alpha
        else:
            raise TypeError("'alpha' parameter should be a float or an "
                            "iterable yielding floats.")

        super(FORCE, self).__init__(params=params,
                                    hypers={"alpha": alpha,
                                            "_alpha_gen": alpha_gen,
                                            "input_bias": input_bias,
                                            "rule": rule},
                                    forward=readout_forward,
                                    train=train,
                                    initializer=partial(initialize,
                                                        init_func=Wout_init,
                                                        bias=input_bias),
                                    output_dim=output_dim,
                                    name=name)
