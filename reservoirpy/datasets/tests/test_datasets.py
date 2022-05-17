import os
from contextlib import contextmanager

import numpy as np
import pytest
from joblib import Memory
from numpy.testing import assert_allclose

from reservoirpy import _TEMPDIR, datasets
from reservoirpy.datasets import to_forecasting


@contextmanager
def no_cache():
    """Disable caching temporarily when running tests"""
    datasets._chaos.memory = Memory(location=None)
    yield
    datasets._chaos.memory = Memory(os.path.join(_TEMPDIR, "datasets"), verbose=0)


@pytest.mark.parametrize(
    "dataset_func",
    [
        datasets.henon_map,
        datasets.logistic_map,
        datasets.lorenz,
        datasets.mackey_glass,
        datasets.multiscroll,
        datasets.doublescroll,
        datasets.rabinovich_fabrikant,
        datasets.narma,
        datasets.lorenz96,
        datasets.rossler,
        datasets.kuramoto_sivashinsky,
    ],
)
def test_generation(dataset_func):
    with no_cache():
        timesteps = 2000
        X = dataset_func(timesteps)

    assert isinstance(X, np.ndarray)
    assert len(X) == timesteps


@pytest.mark.parametrize(
    "dataset_func,kwargs,expected",
    [
        (datasets.logistic_map, {"r": -1}, ValueError),
        (datasets.logistic_map, {"x0": 1}, ValueError),
        (datasets.mackey_glass, {"seed": 1234}, None),
        (datasets.mackey_glass, {"seed": None}, None),
        (datasets.mackey_glass, {"tau": 0}, None),
        (datasets.narma, {"seed": 1234}, None),
        (datasets.lorenz96, {"N": 1}, ValueError),
        (datasets.lorenz96, {"x0": [0.1, 0.2, 0.3, 0.4, 0.5], "N": 4}, ValueError),
        (datasets.rossler, {"x0": [0.1, 0.2]}, ValueError),
        (
            datasets.kuramoto_sivashinsky,
            {"x0": np.random.normal(size=129), "N": 128},
            ValueError,
        ),
        (
            datasets.kuramoto_sivashinsky,
            {"x0": np.random.normal(size=128), "N": 128},
            None,
        ),
    ],
)
def test_kwargs(dataset_func, kwargs, expected):

    if expected is None:
        timesteps = 2000
        X = dataset_func(timesteps, **kwargs)

        assert isinstance(X, np.ndarray)
        assert len(X) == timesteps
    else:
        with pytest.raises(expected):
            timesteps = 2000
            dataset_func(timesteps, **kwargs)


@pytest.mark.parametrize("dataset_func", [datasets.mackey_glass])
def test_seed(dataset_func):
    x1 = dataset_func(200)
    x2 = dataset_func(200)

    assert_allclose(x1, x2)


@pytest.mark.parametrize("dataset_func", [datasets.mackey_glass])
def test_reseed(dataset_func):

    s = datasets.get_seed()
    assert s == datasets._seed._DEFAULT_SEED

    x1 = dataset_func(200)

    datasets.set_seed(1234)
    assert datasets._seed._DEFAULT_SEED == 1234
    assert datasets.get_seed() == 1234

    x2 = dataset_func(200)

    assert (np.abs(x1 - x2) > 1e-3).sum() > 0


@pytest.mark.parametrize("dataset_func", [datasets.mackey_glass, datasets.lorenz])
def test_to_forecasting(dataset_func):
    x = dataset_func(200)

    x, y = to_forecasting(x, forecast=5)

    assert x.shape[0] == 200 - 5
    assert y.shape[0] == 200 - 5
    assert x.shape[0] == y.shape[0]


@pytest.mark.parametrize("dataset_func", [datasets.mackey_glass, datasets.lorenz])
def test_to_forecasting_with_test(dataset_func):
    x = dataset_func(200)

    x, xt, y, yt = to_forecasting(x, forecast=5, test_size=10)

    assert x.shape[0] == 200 - 5 - 10
    assert y.shape[0] == 200 - 5 - 10
    assert x.shape[0] == y.shape[0]
    assert xt.shape[0] == yt.shape[0] == 10


def test_japanese_vowels():

    X, Y, X_test, Y_test = datasets.japanese_vowels()

    assert len(X) == 270 == len(Y)
    assert len(X_test) == 370 == len(Y_test)

    assert Y[0].shape == (1, 9)

    X, Y, X_test, Y_test = datasets.japanese_vowels(repeat_targets=True)

    assert Y[0].shape == (X[0].shape[0], 9)

    X, Y, X_test, Y_test = datasets.japanese_vowels(one_hot_encode=False)

    assert Y[0].shape == (1, 1)
