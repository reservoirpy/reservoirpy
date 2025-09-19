# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal

from reservoirpy.jax.nodes import RLS


def test_rls_init():
    node = RLS(10, output_dim=3)

    data = jnp.ones((100,))
    # y = jnp.ones((1, 3))
    res = node(data)

    assert node.Wout.shape == (100, 3)
    assert node.bias.shape == (3,)
    assert node.alpha == 10

    data = jnp.ones((10000, 100))
    res = node.run(data)
    assert res.shape == (10000, 3)

    # with initialized Wout and bias
    Wout = jnp.ones((100, 3))
    bias = jnp.ones((3,))
    data = jnp.ones((100,))
    # unspecified dimensions
    node = RLS(10, Wout=Wout, bias=bias)
    res = node(data)
    # correct specified dimensions
    node = RLS(10, Wout=Wout, bias=bias, input_dim=100, output_dim=3)
    res = node(data)
    # incorrect specified dimensions
    with pytest.raises(ValueError):
        node = RLS(10, Wout=Wout, bias=bias, input_dim=101, output_dim=1)
    with pytest.raises(ValueError):
        bias = jnp.ones((10,))
        node = RLS(bias=bias, output_dim=1)


def test_rls_train_one_step():
    node = RLS(10)

    x = jnp.ones((10, 5, 2))
    y = jnp.ones((10, 5, 10))

    for x, y in zip(x, y):
        res = node.partial_fit(x, y)

    assert node.Wout.shape == (2, 10)
    assert node.bias.shape == (10,)
    assert node.alpha == 10

    data = jnp.ones((17, 2))
    res = node.run(data)

    assert res.shape == (17, 10)


def test_rls_train():
    jax.config.update("jax_enable_x64", True)
    node = RLS(alpha=1e-6)

    X, Y = jnp.ones((38, 100)), jnp.ones((38, 10))

    res = node.partial_fit(X, Y)

    assert res.shape == (38, 10)
    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, jnp.ones((100, 10)) * 0.005, decimal=4)
    assert node.bias.shape == (10,)
    assert_array_almost_equal(node.bias, jnp.ones((10,)) * 0.5, decimal=4)

    node = RLS(1e-6, fit_bias=False)

    X, Y = jnp.ones((38, 100)), jnp.ones((38, 10))

    res = node.partial_fit(X, Y)

    assert res.shape == (38, 10)
    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, jnp.ones((100, 10)) * 0.01, decimal=4)
    assert node.bias.shape == (10,)
    assert_array_almost_equal(node.bias, jnp.ones((10,)) * 0.0, decimal=4)

    node = RLS(1e-6, fit_bias=True)

    X, Y = jnp.ones((5, 24, 100)), jnp.ones((5, 24, 10))

    for x, y in zip(X, Y):
        res = node.partial_fit(x, y)

    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, jnp.ones((100, 10)) * 0.005, decimal=4)
    assert node.bias.shape == (10,)
    assert_array_almost_equal(node.bias, jnp.ones((10,)) * 0.5, decimal=4)

    data = jnp.ones((120, 100))
    res = node.run(data)

    assert res.shape == (120, 10)
