# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import numpy as np
import pytest
from joblib import Parallel, delayed
from numpy.testing import assert_array_almost_equal, assert_array_equal

from reservoirpy.nodes import Ridge


def test_ridge_init():
    x: np.ndarray = np.ones((100,))
    X: np.ndarray = np.ones((120, 100))
    Y: np.ndarray = np.ones((120, 10))

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

    with pytest.raises(ValueError):
        node = Ridge(ridge=1e-8, input_dim=100, output_dim=10)
        node(x)

    node = Ridge(ridge=1e-8, Wout=np.ones((100, 10)), bias=np.ones((10,)))
    node.run(X)
    assert node.output_dim == 10
    assert node.input_dim == 100
    assert node.ridge == 1e-8


def test_ridge_worker():
    X: np.ndarray = 2 * np.ones((120, 100))
    Y: np.ndarray = 2 * np.ones((120, 10))

    node = Ridge(1e-6)

    XXT, YXT, x_sum, y_sum, sample_size = node.worker(X, Y)
    assert XXT.shape == (100, 100)
    assert YXT.shape == (100, 10)
    assert_array_equal(x_sum, 2 * 120 * np.ones((100,)))
    assert_array_equal(y_sum, 2 * 120 * np.ones((10,)))
    assert_array_equal(sample_size, 120)


def test_ridge_fit():
    X: np.ndarray = np.ones((120, 100))
    Y: np.ndarray = np.ones((120, 10))

    node = Ridge(1e-6)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert node.bias.shape == (10,)
    assert np.any(node.bias != np.zeros((10,)))

    node = Ridge(1e-6, fit_bias=False)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert_array_equal(node.bias, np.zeros((10,)))


def test_ridge_fit_multiseries():
    rng = np.random.default_rng(seed=0)
    X: np.ndarray = rng.uniform(size=(15, 12, 100))
    Y: np.ndarray = X @ rng.uniform(size=(100, 10)) + rng.uniform(size=10)

    node = Ridge(1e-6)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert node.bias.shape == (10,)
    assert not np.all(node.Wout == np.zeros((100, 10)))
    assert not np.all(node.bias == np.zeros((10,)))
    node2 = Ridge(1e-6)
    node2.fit(X.reshape(15 * 12, 100), Y.reshape(15 * 12, 10))
    assert_array_almost_equal(node.Wout, node2.Wout)
    assert_array_almost_equal(node.bias, node2.bias)

    node = Ridge(1e-6, fit_bias=False)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert_array_equal(node.bias, np.zeros((10,)))


def test_ridge_fit_parallel():
    X: np.ndarray = np.ones((15, 12, 100))
    Y: np.ndarray = np.ones((15, 12, 10))

    node = Ridge(1e-6)
    node.fit(X, Y, workers=-1)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert node.bias.shape == (10,)
    assert np.any(node.bias != np.zeros((10,)))

    node = Ridge(1e-6, fit_bias=False)
    node.fit(X, Y)
    assert node.input_dim == 100
    assert node.output_dim == 10
    assert node.Wout.shape == (100, 10)
    assert_array_equal(node.bias, np.zeros((10,)))
