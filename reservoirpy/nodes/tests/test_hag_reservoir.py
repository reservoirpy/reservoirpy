# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import numpy as np
import pytest

from ..hag_reservoir import HAGReservoir

HAG_PARAMS = dict(target=0.6, spread=0.1, weight_increment=0.05, min_window=5, max_window=20)


def test_hag_init():
    x = np.ones((10, 5))
    res = HAGReservoir(100, input_dim=5, **HAG_PARAMS)

    res.initialize(x)

    assert res.W.shape == (100, 100)
    assert res.Win.shape == (100, 5)
    assert res.bias.shape == (100,)
    # HAG grows the connections from an empty matrix, each input feature drives a block of 20 units
    assert np.all(res.W == 0.0)
    assert np.all((res.Win != 0).sum(axis=0) == 20)
    assert np.all(res.bias > 0)

    res = HAGReservoir(100, **HAG_PARAMS)

    out = res.run(x)

    assert out.shape == (10, 100)
    assert res.Win.shape == (100, 5)

    with pytest.raises(ValueError):
        _ = HAGReservoir(**HAG_PARAMS)  # no units nor W
    with pytest.raises(ValueError):
        _ = HAGReservoir(100, target=0.6, spread=0.1, weight_increment=0.05)  # no min_window
    with pytest.raises(ValueError):
        _ = HAGReservoir(100, homeostasis="max", **HAG_PARAMS)
    with pytest.raises(ValueError):
        _ = HAGReservoir(100, **dict(HAG_PARAMS, weight_increment=-0.05))
    with pytest.raises(ValueError):
        _ = HAGReservoir(100, **dict(HAG_PARAMS, weight_increment=0.0))
    with pytest.raises(ValueError):
        _ = HAGReservoir(100, **dict(HAG_PARAMS, min_window=2))
    with pytest.raises(ValueError):
        _ = HAGReservoir(100, **dict(HAG_PARAMS, min_window=10, max_window=5))
    with pytest.raises(ValueError):
        _ = HAGReservoir(100, W=np.zeros((50, 50)), **HAG_PARAMS)
    with pytest.raises(ValueError):
        # units must be a multiple of the input dimension for the default input matrix
        HAGReservoir(101, **HAG_PARAMS).initialize(x)


@pytest.mark.parametrize("homeostasis", ["mean", "variance"])
def test_hag_fit(homeostasis):
    rng = np.random.default_rng(seed=0)
    x = rng.uniform(size=(500, 5))
    X = [rng.uniform(size=(30, 5)) for _ in range(20)]
    params = HAG_PARAMS if homeostasis == "mean" else dict(HAG_PARAMS, target=0.05, spread=0.01)

    res = HAGReservoir(100, homeostasis=homeostasis, input_scaling=0.5, seed=0, **params)
    res.fit(x)

    assert res.n_added > 0
    assert np.count_nonzero(res.W) > 0
    assert np.all(res.W >= 0.0)
    assert np.all(np.diag(res.W) == 0.0)  # no self-connection

    res.fit(X, warmup=2)
    assert res.run(X).__len__() == 20

    # one plasticity step per sequence
    res = HAGReservoir(100, homeostasis=homeostasis, use_full_instance=True, seed=0, **params)
    res.fit(X)
    assert np.count_nonzero(res.W) > 0


def test_hag_reproducibility():
    rng = np.random.default_rng(seed=1)
    X = [rng.uniform(size=(30, 5)) for _ in range(20)]

    res1 = HAGReservoir(100, seed=42, **HAG_PARAMS).fit(X)
    res2 = HAGReservoir(100, seed=42, **HAG_PARAMS).fit(X)
    res3 = HAGReservoir(100, seed=43, **HAG_PARAMS).fit(X)

    assert np.array_equal(res1.W, res2.W)
    assert not np.array_equal(res1.W, res3.W)


def test_hag_run_does_not_change_weights():
    rng = np.random.default_rng(seed=2)
    x = rng.uniform(size=(300, 5))

    res = HAGReservoir(100, seed=0, **HAG_PARAMS).fit(x)
    W = res.W.copy()
    res.reset()
    _ = res.run(x)

    assert np.array_equal(res.W, W)


def test_hag_matrices():
    rng = np.random.default_rng(seed=3)
    x = rng.uniform(size=(300, 4))
    W = np.zeros((40, 40))
    Win = rng.uniform(size=(40, 4))
    bias = np.full(40, 0.1)

    res = HAGReservoir(W=W, Win=Win, bias=bias, seed=0, **HAG_PARAMS)

    assert res.units == 40
    assert res.input_dim == 4

    res.fit(x)

    assert np.array_equal(res.Win, Win)
    assert np.array_equal(res.bias, bias)
    assert np.count_nonzero(res.W) > 0
    assert np.all(W == 0.0)  # the given matrix is not modified


def test_hag_undefined_correlations():
    # undefined correlations (windows of one timestep after the first one, or constant activity) must not stop fit:
    # the neurons concerned just get no new connection
    X = [np.random.default_rng(i).uniform(size=(2, 5)) for i in range(10)]
    res = HAGReservoir(100, use_full_instance=True, seed=0, **HAG_PARAMS)
    res.fit(X)
    assert np.all(res.W == 0.0)

    # constant inputs: constant activity once the initial transient is over
    res = HAGReservoir(100, seed=0, **HAG_PARAMS)
    res.fit(np.full((300, 5), 0.5))
    assert np.all(np.isfinite(res.W))
