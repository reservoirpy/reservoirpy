# Author: Nathan Trouvain at 18/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from math import comb

import numpy as np

from reservoirpy.nodes import NVAR


def _get_output_dim(input_dim, delay, order):
    linear_dim = delay * input_dim
    nonlinear_dim = comb(linear_dim + order - 1, order)
    return int(linear_dim + nonlinear_dim)


def test_nvar():
    node = NVAR(3, 2)

    data = np.ones((10,))
    res = node(data)

    assert node.store is not None
    assert node.strides == 1
    assert node.delay == 3
    assert node.order == 2
    assert node.input_dim == 10
    assert node.output_dim == _get_output_dim(10, 3, 2)
    assert res.shape == (_get_output_dim(10, 3, 2),)

    data = np.ones((1000, 10))
    res = node.run(data)

    assert res.shape == (1000, _get_output_dim(10, 3, 2))
