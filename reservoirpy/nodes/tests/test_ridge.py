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


@pytest.mark.parametrize("solver", ["normal", "cholesky", "qr", "svd", "lsqr"])
@pytest.mark.parametrize("fit_bias", [True, False])
def test_ridge_solvers(solver, fit_bias):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 50, 20)) + 3.0
    Y = X @ rng.normal(size=(20, 4)) + 1.0 + 0.1 * rng.normal(size=(8, 50, 4))
    # reference: the normal equations on one timeseries (well conditioned problem)
    reference = Ridge(1e-2, fit_bias=fit_bias, solver="normal").fit(X.reshape(-1, 20), Y.reshape(-1, 4))
    Wout, bias = reference.Wout, reference.bias

    # several timeseries, in parallel or not, and one timeseries
    for x, y, workers in [(X, Y, 1), (list(X), list(Y), -1), (X.reshape(-1, 20), Y.reshape(-1, 4), 1)]:
        node = Ridge(1e-2, fit_bias=fit_bias, solver=solver)
        node.fit(x, y, workers=workers)
        assert_array_almost_equal(node.Wout, Wout, decimal=6)
        assert_array_almost_equal(node.bias, bias, decimal=6)


@pytest.mark.parametrize("solver", ["cholesky", "qr", "svd"])
def test_ridge_solvers_ill_conditioned(solver):
    # inputs with a large mean and nearly dependent dimensions (as reservoir states), small ridge
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(10, 200, 5))
    X = 100.0 + latent @ rng.normal(size=(5, 40)) + 1e-6 * rng.normal(size=(10, 200, 40))
    Y = latent @ rng.normal(size=(5, 3))
    # reference (the normal equations fail here): least squares of the centered data augmented with sqrt(ridge) * Id
    x_mean, y_mean = X.reshape(-1, 40).mean(axis=0), Y.reshape(-1, 3).mean(axis=0)
    A = np.vstack([X.reshape(-1, 40) - x_mean, np.sqrt(1e-12) * np.eye(40)])
    B = np.vstack([Y.reshape(-1, 3) - y_mean, np.zeros((40, 3))])
    Wout = np.linalg.lstsq(A, B, rcond=None)[0]
    bias = y_mean - x_mean @ Wout

    node = Ridge(1e-12, solver=solver)
    node.fit(X, Y)
    prediction = node.run(X.reshape(-1, 40))
    reference = X.reshape(-1, 40) @ Wout + bias
    assert np.max(np.abs(prediction - reference)) < 1e-5 * np.max(np.abs(Y))


def test_ridge_unknown_solver():
    with pytest.raises(ValueError):
        Ridge(1e-6, solver="sag")


def test_ridge_svd_no_ridge_rank_deficient():
    # ridge=0 (default) with dependent inputs: minimum norm least squares, no NaN
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 3)) @ rng.normal(size=(3, 6))
    Y = X @ rng.normal(size=(6, 2))
    node = Ridge(solver="svd").fit(X, Y)
    assert np.all(np.isfinite(node.Wout))
    assert_array_almost_equal(node.run(X), Y)
