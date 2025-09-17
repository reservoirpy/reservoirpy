# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax.numpy as jnp
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal

from reservoirpy.jax.nodes import LMS, Reservoir


def test_lms_init():
    X = jnp.ones((2, 100))
    Y = jnp.ones((2, 10))

    # with callable Wout & bias
    node = LMS(0.1)
    assert node.input_dim is None
    assert node.output_dim is None
    node.initialize(X, Y)
    assert_array_equal(node.Wout, jnp.zeros((100, 10)))
    assert_array_equal(node.bias, jnp.zeros((10,)))
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.learning_rate == 0.1

    # with initialized weights
    node = LMS(1e-6, Wout=jnp.ones((100, 10)), bias=jnp.ones((10,)))
    node.initialize(X, Y)
    assert_array_equal(node.Wout, jnp.ones((100, 10)))
    assert_array_equal(node.bias, jnp.ones((10,)))
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.learning_rate == 1e-6


def test_lms_fit():
    node = LMS(1e-3)

    x = jnp.ones((7, 5, 2))
    y = jnp.ones((7, 5, 10))

    res = node.fit(x, y)

    assert node.Wout.shape == (2, 10)
    assert not jnp.all(node.Wout == jnp.zeros((2, 10)))
    assert node.bias.shape == (10,)
    assert not jnp.all(node.bias == jnp.zeros((10,)))
    assert node.learning_rate == 1e-3

    data = jnp.ones((1000, 2))
    res = node.run(data)

    assert res.shape == (1000, 10)


def test_lms_partial_fit():
    rng = numpy.random.default_rng(seed=2368)
    X = rng.normal(size=(10, 5, 2))
    Y = rng.normal(size=(10, 5, 10))
    data = rng.normal(size=(1000, 2))

    node1 = LMS(1e-4)
    for x, y in zip(X, Y):
        node1.partial_fit(x, y)

    assert node1.Wout.shape == (2, 10)
    assert node1.bias.shape == (10,)

    res1 = node1.run(data)
    assert res1.shape == (1000, 10)

    # compare with .fit()
    node2 = LMS(1e-4)
    node2.fit(X.reshape(10 * 5, 2), Y.reshape(10 * 5, 10))
    res2 = node2.run(data)
    assert_array_almost_equal(res1, res2)
    assert not jnp.all(res1 == jnp.zeros((1000, 10)))
    assert_array_almost_equal(node1.Wout, node2.Wout)
    assert not jnp.all(node1.Wout == jnp.zeros((2, 10)))
    assert_array_almost_equal(node1.bias, node2.bias)
    assert not jnp.all(node1.bias == jnp.zeros((10,)))


def test_lms_call():
    x = jnp.array([0.1, 0.2, 0.3])

    node = LMS(output_dim=3)
    y1 = node.step(x)
    assert_array_equal(y1, jnp.zeros((3,)))
    node = LMS(output_dim=2)
    y2 = node(x)
    assert_array_equal(y2, jnp.zeros((2,)))
