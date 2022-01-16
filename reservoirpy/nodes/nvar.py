# Author: Nathan Trouvain at 14/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import sys

import itertools as it

if sys.version_info < (3, 8):
    from scipy.special import comb
else:
    from math import comb

import numpy as np

from reservoirpy.node import Node


def forward(node, x):
    store = node.store
    strides = node.strides
    idxs = node._monomial_idx

    # store the current input
    new_store = np.roll(store, 1, axis=0)
    new_store[0] = x
    node.set_param("store", new_store)

    output = np.zeros((node.output_dim, 1))

    # select all previous inputs, including the current, with strides
    linear_feats = np.ravel(new_store[::strides, :]).reshape(-1, 1)
    linear_len = linear_feats.shape[0]

    output[:linear_len, :] = linear_feats

    # select monomial terms and compute them
    output[linear_len:, :] = np.prod(linear_feats[idxs], axis=1)

    return output.reshape(1, -1)


def initialize(node, x=None, *args, **kwargs):
    if x is not None:
        input_dim = x.shape[1]

        order = node.order
        delay = node.delay
        strides = node.strides

        linear_dim = delay * input_dim
        # number of non linear components is (d + n - 1)! / (d - 1)! n!
        # i.e. number of all unique monomials of order n made from the
        # linear components.
        nonlinear_dim = comb(linear_dim + order - 1, order)

        output_dim = output_dim = int(linear_dim + nonlinear_dim)

        node.set_output_dim(output_dim)
        node.set_input_dim(input_dim)

        # for each monomial created in the non linear part, indices
        # of the n components involved, n being the order of the
        # monomials. Precompute them to improve efficiency.
        idx = np.array(
            list(it.combinations_with_replacement(np.arange(linear_dim),
                                                  order)))

        node.set_param("_monomial_idx", idx)

        # to store the k*s last inputs, k being the delay and s the strides
        node.set_param("store", np.zeros((delay * strides, node.input_dim)))


class NVAR(Node):

    def __init__(self, delay, order, strides=1, name=None):
        super(NVAR, self).__init__(params={"store": None,
                                           "_monomial_idx": None},
                                   hypers={"delay": delay,
                                           "order": order,
                                           "strides": strides},
                                   forward=forward,
                                   initializer=initialize,
                                   name=name)
