# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

"""Differential tests: the jax backend must produce the same results as the
numpy backend (the reference implementation), node by node and operation by
operation, for the same weights and inputs. The numpy backend is the oracle.
"""

from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

import reservoirpy.jax.nodes as jax_nodes
import reservoirpy.nodes as numpy_nodes
from reservoirpy.jax.mat_gen import uniform as jax_uniform
from reservoirpy.mat_gen import bernoulli, normal
from reservoirpy.mat_gen import uniform as numpy_uniform

# jax runs in float32 by default while the numpy backend runs in float64, so
# comparisons use a tolerance that accommodates single-precision rounding.
TOL = 1e-4

RESERVOIR_NODES = [
    ("Reservoir", {"lr": 0.3}),
    ("ES2N", {"proximity": 0.5}),
    ("IPReservoir", {"lr": 0.3}),
]


def _array_weights(node):
    """Extract the array-valued weights of an initialized node."""
    weights = {}
    for key in ["W", "Win", "O", "bias"]:
        if hasattr(node, key):
            value = getattr(node, key)
            array = np.asarray(value.todense() if hasattr(value, "todense") else value)
            if array.ndim >= 1:
                weights[key] = array
    return weights


def _reservoir_weights(numpy_cls, extra, x):
    """Draw a set of dense weights from a numpy-backend reservoir node so the
    exact same explicit weights can be fed to both backends."""
    seed_node = numpy_cls(
        units=20,
        W=partial(normal, sparsity_type="dense"),
        Win=partial(bernoulli, sparsity_type="dense"),
        seed=0,
        **extra,
    )
    seed_node.initialize(x)
    return _array_weights(seed_node)


@pytest.mark.parametrize("name,extra", RESERVOIR_NODES)
def test_reservoir_run_matches_numpy_backend(name, extra):
    jax_cls = getattr(jax_nodes, name)
    numpy_cls = getattr(numpy_nodes, name)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    weights = _reservoir_weights(numpy_cls, extra, x)

    jax_single = np.asarray(jax_cls(**weights, **extra).run(x))
    numpy_single = np.asarray(numpy_cls(**weights, **extra).run(x))
    assert not np.all(jax_single == 0.0)
    assert_allclose(jax_single, numpy_single, atol=TOL)

    # multi-series as a 3D array: each series is independent, so the reference
    # for series i is a fresh single-series numpy run
    x_multi = rng.normal(size=(4, 40, 3))
    jax_multi = np.asarray(jax_cls(**weights, **extra).run(x_multi))
    assert jax_multi.shape[0] == x_multi.shape[0]
    for i in range(x_multi.shape[0]):
        numpy_ref = np.asarray(numpy_cls(**weights, **extra).run(x_multi[i]))
        assert_allclose(jax_multi[i], numpy_ref, atol=TOL)


@pytest.mark.parametrize("name,extra", RESERVOIR_NODES)
def test_reservoir_run_list_matches_numpy_backend(name, extra):
    # run() also accepts a Python list of series, which is a distinct code path
    # from the 3D-array case
    jax_cls = getattr(jax_nodes, name)
    numpy_cls = getattr(numpy_nodes, name)
    rng = np.random.default_rng(1)
    weights = _reservoir_weights(numpy_cls, extra, rng.normal(size=(5, 3)))
    series = [rng.normal(size=(t, 3)) for t in (12, 18, 9)]

    jax_out = jax_cls(**weights, **extra).run(series)
    numpy_out = numpy_cls(**weights, **extra).run(series)
    assert len(jax_out) == len(series)
    for jax_series, numpy_series in zip(jax_out, numpy_out):
        assert_allclose(np.asarray(jax_series), np.asarray(numpy_series), atol=TOL)


@pytest.mark.parametrize("name,extra", RESERVOIR_NODES)
def test_reservoir_step_matches_numpy_backend(name, extra):
    jax_cls = getattr(jax_nodes, name)
    numpy_cls = getattr(numpy_nodes, name)
    rng = np.random.default_rng(2)
    weights = _reservoir_weights(numpy_cls, extra, rng.normal(size=(5, 3)))
    jax_node = jax_cls(**weights, **extra)
    numpy_node = numpy_cls(**weights, **extra)

    jax_out = numpy_out = None
    for step_input in rng.normal(size=(6, 3)):
        jax_out = np.asarray(jax_node.step(step_input))
        numpy_out = np.asarray(numpy_node.step(step_input))
    assert_allclose(jax_out, numpy_out, atol=TOL)


def test_ipreservoir_fit_matches_numpy_backend():
    # intrinsic plasticity updates the gains `a` and biases `b` during fit; the
    # learned values (and predictions afterwards) must match the numpy backend
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 3))
    extra = dict(lr=0.3, learning_rate=1e-3, epochs=2, mu=0.0, sigma=1.0)
    weights = _reservoir_weights(numpy_nodes.IPReservoir, extra, x)

    jax_node = jax_nodes.IPReservoir(**weights, **extra)
    numpy_node = numpy_nodes.IPReservoir(**weights, **extra)
    jax_node.fit(x)
    numpy_node.fit(x)

    assert not np.all(np.asarray(jax_node.a) == 1.0)
    assert_allclose(np.asarray(jax_node.a), np.asarray(numpy_node.a), atol=TOL)
    assert_allclose(np.asarray(jax_node.b), np.asarray(numpy_node.b), atol=TOL)
    assert_allclose(np.asarray(jax_node.run(x[:20])), np.asarray(numpy_node.run(x[:20])), atol=TOL)


@pytest.mark.parametrize("delay,order,strides", [(2, 2, 1), (3, 2, 2), (2, 3, 1)])
def test_nvar_run_matches_numpy_backend(delay, order, strides):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2))
    jax_out = np.asarray(jax_nodes.NVAR(delay, order, strides=strides).run(x))
    numpy_out = np.asarray(numpy_nodes.NVAR(delay, order, strides=strides).run(x))
    assert not np.all(numpy_out == 0.0)
    assert_allclose(jax_out, numpy_out, atol=TOL)


def test_nvar_multiseries_series_are_independent():
    # each series in a batched run must be processed from a fresh state; the
    # reference for series i is a fresh single-series numpy run
    rng = np.random.default_rng(0)
    x_multi = rng.normal(size=(4, 30, 2))
    jax_multi = np.asarray(jax_nodes.NVAR(2, 2).run(x_multi))
    for i in range(x_multi.shape[0]):
        numpy_ref = np.asarray(numpy_nodes.NVAR(2, 2).run(x_multi[i]))
        assert_allclose(jax_multi[i], numpy_ref, atol=TOL)


@pytest.mark.parametrize("shape", [(60, 3), (3, 40, 3)])
def test_lif_run_matches_numpy_backend(shape):
    rng = np.random.default_rng(0)
    x = rng.normal(size=shape)
    kwargs = dict(units=20, inhibitory=0.3, rc_connectivity=1.0, input_connectivity=1.0, seed=0)
    jax_out = np.asarray(
        jax_nodes.LIF(
            W=partial(jax_uniform, low=0.0, sparsity_type="dense"),
            Win=partial(jax_uniform, low=0.0, sparsity_type="dense"),
            **kwargs,
        ).run(x)
    )
    numpy_out = np.asarray(
        numpy_nodes.LIF(
            W=partial(numpy_uniform, low=0.0, sparsity_type="dense"),
            Win=partial(numpy_uniform, low=0.0, sparsity_type="dense"),
            **kwargs,
        ).run(x)
    )
    assert_allclose(jax_out, numpy_out, atol=TOL)


def test_ridge_fit_matches_numpy_backend():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 40, 8))
    y = x @ rng.normal(size=(8, 3)) + 0.5

    jax_node = jax_nodes.Ridge(1e-6)
    jax_node.fit(x, y)
    numpy_node = numpy_nodes.Ridge(1e-6)
    numpy_node.fit(x, y)
    assert_allclose(np.asarray(jax_node.Wout), np.asarray(numpy_node.Wout), atol=TOL)
    assert_allclose(np.asarray(jax_node.bias), np.asarray(numpy_node.bias), atol=TOL)


ONLINE_READOUTS = [("LMS", {"learning_rate": 0.02}), ("RLS", {"alpha": 0.1})]


@pytest.mark.parametrize("name,extra", ONLINE_READOUTS)
def test_online_readout_partial_fit_matches_numpy_backend(name, extra):
    jax_cls = getattr(jax_nodes, name)
    numpy_cls = getattr(numpy_nodes, name)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8))
    y = x @ rng.normal(size=(8, 3)) + 0.5

    jax_pred = np.asarray(jax_cls(**extra).partial_fit(x, y))
    numpy_pred = np.asarray(numpy_cls(**extra).partial_fit(x, y))
    assert not np.all(jax_pred == 0.0)
    assert_allclose(jax_pred, numpy_pred, atol=1e-3)


@pytest.mark.parametrize("name,extra", ONLINE_READOUTS)
def test_online_readout_fit_matches_numpy_backend(name, extra):
    jax_cls = getattr(jax_nodes, name)
    numpy_cls = getattr(numpy_nodes, name)
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(4, 40, 8))
    y_train = x_train @ rng.normal(size=(8, 3)) + 0.5
    x_test = rng.normal(size=(30, 8))

    jax_node = jax_cls(**extra)
    jax_node.fit(x_train, y_train)
    numpy_node = numpy_cls(**extra)
    numpy_node.fit(x_train, y_train)
    assert_allclose(np.asarray(jax_node.Wout), np.asarray(numpy_node.Wout), atol=1e-3)
    assert_allclose(np.asarray(jax_node.run(x_test)), np.asarray(numpy_node.run(x_test)), atol=1e-3)


@pytest.mark.parametrize("name", ["Tanh", "Sigmoid", "ReLU", "Softplus", "Softmax", "Identity"])
def test_activation_run_matches_numpy_backend(name):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(15, 5))
    jax_out = np.asarray(getattr(jax_nodes, name)().run(x))
    numpy_out = np.asarray(getattr(numpy_nodes, name)().run(x))
    assert_allclose(jax_out, numpy_out, atol=TOL)
