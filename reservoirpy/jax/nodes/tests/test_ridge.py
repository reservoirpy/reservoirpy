# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax.numpy as jnp
import numpy
import pytest
from joblib import Parallel, delayed
from numpy.testing import assert_array_almost_equal, assert_array_equal

from reservoirpy.jax.nodes import Ridge


def test_ridge_init():
    x: jnp.ndarray = jnp.ones((100,))
    X: jnp.ndarray = jnp.ones((120, 100))
    Y: jnp.ndarray = jnp.ones((120, 10))

    node = Ridge(ridge=1e-8)
    node.initialize(X, Y)
    assert node.output_dim == 10
    assert node.input_dim == 100
    assert node.ridge == 1e-8

    node = Ridge(ridge=1e-8, output_dim=10)
    node.initialize(X)
    assert node.output_dim == 10
    assert node.input_dim == 100
    assert node.ridge == 1e-8

    with pytest.raises((ValueError, TypeError)):
        node = Ridge(ridge=1e-8, input_dim=100, output_dim=10)
        node(x)

    node = Ridge(ridge=1e-8, Wout=jnp.ones((100, 10)), bias=jnp.ones((10,)))
    node.run(X)
    assert node.output_dim == 10
    assert node.input_dim == 100
    assert node.ridge == 1e-8


def test_ridge_worker():
    X: jnp.ndarray = 2 * jnp.ones((120, 100))
    Y: jnp.ndarray = 2 * jnp.ones((120, 10))

    node = Ridge(1e-6)

    XXT, YXT, x_sum, y_sum, sample_size = node.worker(X, Y)
    assert XXT.shape == (100, 100)
    assert YXT.shape == (100, 10)
    assert_array_equal(x_sum, 2 * 120 * jnp.ones((100,)))
    assert_array_equal(y_sum, 2 * 120 * jnp.ones((10,)))
    assert_array_equal(sample_size, 120)


def test_ridge_fit():
    X: jnp.ndarray = jnp.ones((120, 100))
    Y: jnp.ndarray = jnp.ones((120, 10))

    node = Ridge(1e-6)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert node.bias.shape == (10,)
    assert jnp.any(node.bias != jnp.zeros((10,)))

    node = Ridge(1e-6, fit_bias=False)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert_array_equal(node.bias, jnp.zeros((10,)))


def test_ridge_fit_multiseries():
    rng = numpy.random.default_rng(seed=0)
    X: jnp.ndarray = rng.uniform(size=(15, 12, 100))
    Y: jnp.ndarray = X @ rng.uniform(size=(100, 10)) + rng.uniform(size=10)

    node = Ridge(1e-6)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert node.bias.shape == (10,)
    assert not jnp.all(node.Wout == jnp.zeros((100, 10)))
    assert not jnp.all(node.bias == jnp.zeros((10,)))
    node2 = Ridge(1e-6)
    node2.fit(X.reshape(15 * 12, 100), Y.reshape(15 * 12, 10))
    assert_array_almost_equal(node.Wout, node2.Wout)
    assert_array_almost_equal(node.bias, node2.bias)

    node = Ridge(1e-6, fit_bias=False)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert_array_equal(node.bias, jnp.zeros((10,)))


def test_ridge_fit_parallel():
    X: jnp.ndarray = jnp.ones((15, 12, 100))
    Y: jnp.ndarray = jnp.ones((15, 12, 10))

    node = Ridge(1e-6)
    node.fit(X, Y, workers=-1)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert node.bias.shape == (10,)
    assert jnp.any(node.bias != jnp.zeros((10,)))

    node = Ridge(1e-6, fit_bias=False)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert_array_equal(node.bias, jnp.zeros((10,)))


def test_parallel():
    process_count = 16

    rng = numpy.random.default_rng(seed=42)
    x = rng.random((400, 10))
    y = x[:, 2::-1] + rng.random((400, 3)) / 10
    x_run = rng.random((20, 10))

    def run_ridge(i):
        readout = Ridge(ridge=1e-8)
        return readout.fit(x, y).run(x_run)

    parallel = Parallel(n_jobs=process_count, return_as="generator")
    results = list(parallel(delayed(run_ridge)(i) for i in range(process_count)))

    for result in results:
        assert_array_almost_equal(result, results[0])
        # assert jnp.all(result == results[0])
