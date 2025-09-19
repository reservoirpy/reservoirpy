# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from reservoirpy.jax.activationsfunc import relu
from reservoirpy.jax.mat_gen import bernoulli, zeros
from reservoirpy.jax.nodes import ES2N


def test_es2n_init():
    node = ES2N(100, proximity=0.8, bias=zeros)

    data = jnp.ones((10,))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert node.Win.shape == (100, 10)
    assert node.proximity == 0.8
    assert node.units == 100

    data = jnp.ones((120, 10))
    res = node.run(data)

    assert res.shape == (120, 100)

    with pytest.raises(ValueError):
        ES2N()

    res = ES2N(100, activation="relu")
    assert id(res.activation) == id(jax.nn.relu)


def test_es2n_init_from_proximity_is_arrays():
    proximity = jnp.ones((100,)) * 0.5
    input_scaling = jnp.ones((10,)) * 0.8
    node = ES2N(100, proximity=proximity, input_scaling=input_scaling)

    data = jnp.ones((2, 10))
    res = node.run(data)

    assert node.W.shape == (100, 100)
    assert node.Win.shape == (100, 10)
    assert_array_equal(node.proximity, jnp.ones(100) * 0.5)
    assert_array_equal(node.input_scaling, jnp.ones(10) * 0.8)


def test_es2n_init_from_matrices():
    Win = jnp.ones((100, 10))

    node = ES2N(100, proximity=0.8, Win=Win, bias=bernoulli)

    data = jnp.ones((10,))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert_array_equal(node.Win, Win)
    assert node.proximity == 0.8
    assert node.units == 100

    data = jnp.ones((10000, 10))
    res = node.run(data)

    assert res.shape == (10000, 100)

    Win = jnp.ones((100, 10))
    bias = jnp.ones((100,))

    node = ES2N(100, proximity=0.8, Win=Win, bias=bias)

    data = jnp.ones((10,))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert node.proximity == 0.8
    assert node.units == 100
    assert node.output_dim == 100

    data = jnp.ones((10000, 10))
    res = node.run(data)

    assert res.shape == (10000, 100)

    with pytest.raises(ValueError):  # Shape override (matrix.shape > units parameter)
        data = jnp.ones((10))
        W = jnp.ones((10, 10))
        res = ES2N(100, W=W)
        _ = res(data)

    with pytest.raises(TypeError):  # Bad matrix shape
        W = jnp.ones((10, 11))
        res = ES2N(W=W)
        res(data)

    with pytest.raises(ValueError):  # Bad matrix format
        res = ES2N(100, W=1.0)
        res(data)

    with pytest.raises((ValueError, TypeError)):  # Bad Win shape
        res = ES2N(100, Win=jnp.ones((100, 20)))
        res(data)

    with pytest.raises((ValueError, TypeError)):  # Bad Win shape
        res = ES2N(100, Win=jnp.ones((101, 10)))
        res(data)

    with pytest.raises(ValueError):  # Bad matrix format
        res = ES2N(100, Win=1.0)
        res(data)


def test_es2n_bias():
    node = ES2N(100, proximity=0.8, bias=zeros)
    data = jnp.ones((10,))
    res = node(data)
    assert node.W.shape == (100, 100)
    assert node.Win.shape == (100, 10)
    assert node.bias.shape == (100,)
    assert_array_equal(node.bias, jnp.zeros((100,)))
    assert node.proximity == 0.8
    assert node.units == 100

    node = ES2N(100, proximity=0.8, bias=bernoulli)
    data = jnp.ones((10,))
    res = node(data)
    assert node.bias.shape == (100,)

    bias = jnp.ones((100,))
    node = ES2N(100, bias=bias)
    res = node(data)
    assert_array_equal(node.bias, bias)

    bias = jnp.ones((100,))
    node = ES2N(100, bias=bias)
    res = node(data)
    assert_array_equal(node.bias, bias)

    data = jnp.zeros((100,))
    node = ES2N(100, bias=1.0)
    res = node(data)
    assert_array_equal(res, jnp.tanh(jnp.ones((100,))))

    with pytest.raises(AssertionError):
        bias = jnp.ones((100, 1))
        node = ES2N(100, bias=bias)
        res = node(data)
        assert res.shape == (100,)

    with pytest.raises(ValueError):
        bias = jnp.ones((100, 2))
        node = ES2N(100, bias=bias)
        node(data)


def test_es2n_run():
    x = jnp.ones((10, 5))

    res = ES2N(100)
    out = res.run(x)
    assert out.shape == (10, 100)


def test_es2n_chain():
    node1 = ES2N(100, proximity=0.8)
    node2 = ES2N(50, proximity=1.0)

    data = jnp.ones((10,))
    res = (node1 >> node2)(data)

    assert node1.W.shape == (100, 100)
    assert node1.Win.shape == (100, 10)
    assert node2.W.shape == (50, 50)
    assert node2.Win.shape == (50, 100)

    assert res.shape == (50,)


def test_es2n_seed():
    node1 = ES2N(100, seed=123)
    node2 = ES2N(100, seed=123)

    data = jnp.ones((10, 10))

    assert_array_almost_equal(node1.run(data), node2.run(data))
