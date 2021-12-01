# Author: Nathan Trouvain at 18/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import sys

if sys.version_info < (3, 8):
    from scipy.special import comb
else:
    from math import comb

import numpy as np

from reservoirpy.nodes import NVAR


def _get_output_dim(input_dim, delay, order):
    linear_dim = delay * input_dim
    nonlinear_dim = comb(linear_dim + order - 1, order)
    return int(linear_dim + nonlinear_dim)


def test_nvar_init():

    node = NVAR(3, 2)

    data = np.ones((1, 10))
    res = node(data)

    assert node.store is not None
    assert node.strides == 1
    assert node.delay == 3
    assert node.order == 2

    data = np.ones((10000, 10))
    res = node.run(data)

    assert res.shape == (10000, _get_output_dim(10, 3, 2))


def test_nvar_chain():

    node1 = NVAR(3, 1)
    node2 = NVAR(3, 2, strides=2)

    data = np.ones((1, 10))
    res = (node1 >> node2)(data)

    assert res.shape == (1, _get_output_dim(_get_output_dim(10, 3, 1), 3, 2))
