# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from math import comb

import jax.numpy as jnp

from reservoirpy.jax.nodes import NVAR


def _get_output_dim(input_dim, delay, order):
    linear_dim = delay * input_dim
    nonlinear_dim = comb(linear_dim + order - 1, order)
    return int(linear_dim + nonlinear_dim)


def test_nvar():
    node = NVAR(3, 2)

    data = jnp.ones((10,))
    res = node(data)

    assert "store" in node.state
    assert node.strides == 1
    assert node.delay == 3
    assert node.order == 2
    assert node.input_dim == 10
    assert node.output_dim == _get_output_dim(10, 3, 2)
    assert res.shape == (_get_output_dim(10, 3, 2),)

    data = jnp.ones((1000, 10))
    res = node.run(data)

    assert res.shape == (1000, _get_output_dim(10, 3, 2))
