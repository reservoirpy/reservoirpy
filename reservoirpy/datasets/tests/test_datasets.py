# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import os
from contextlib import contextmanager

import numpy as np
import pytest
from joblib import Memory
from numpy.testing import assert_allclose

from reservoirpy import _TEMPDIR, datasets


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
        datasets.lorenz96,
        datasets.rossler,
        datasets.kuramoto_sivashinsky,
        datasets.mso2,
        datasets.mso8,
    ],
)
def test_generation(dataset_func):
    with no_cache():
        timesteps = 2000
        X = dataset_func(timesteps)

    assert isinstance(X, np.ndarray)
    assert len(X) == timesteps


def test_generation_narma():
    with no_cache():
        timesteps = 2000
        X = datasets.narma(timesteps)

    assert isinstance(X[0], np.ndarray)
    assert isinstance(X[1], np.ndarray)
    assert len(X[1]) == timesteps


@pytest.mark.parametrize(
    "dataset_func,kwargs,expected",
    [
        (datasets.logistic_map, {"r": -1}, ValueError),
        (datasets.logistic_map, {"x0": 1}, ValueError),
        (datasets.mackey_glass, {"history": np.ones((20,))}, np.ndarray),
        (datasets.mackey_glass, {"history": np.ones((10,))}, ValueError),
        (datasets.narma, {"seed": 1234}, tuple),
        (datasets.mackey_glass, {"seed": 1234}, np.ndarray),
        (datasets.mackey_glass, {"seed": None}, np.ndarray),
        (datasets.mackey_glass, {"tau": 0}, np.ndarray),
        (datasets.narma, {"seed": 1234}, tuple),
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
            np.ndarray,
        ),
        (datasets.mso, {"freqs": [0.1, 0.2, 0.3]}, np.ndarray),
        (datasets.mso, {"freqs": []}, np.ndarray),
        (datasets.mso2, {"normalize": False}, np.ndarray),
    ],
)
def test_kwargs(dataset_func, kwargs, expected):
    timesteps = 2000

    if not issubclass(expected, Exception):
        X = dataset_func(timesteps, **kwargs)
        assert isinstance(X, expected)
        if expected is np.ndarray:
            assert X.shape[-2] == timesteps
        if expected is tuple:
            assert X[1].shape[-2] == timesteps
    else:
        with pytest.raises(expected):
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

    x, y = datasets.to_forecasting(x, forecast=5)

    assert x.shape[0] == 200 - 5
    assert y.shape[0] == 200 - 5
    assert x.shape[0] == y.shape[0]


@pytest.mark.parametrize("dataset_func", [datasets.mackey_glass, datasets.lorenz])
def test_to_forecasting_with_test(dataset_func):
    x = dataset_func(200)

    x, xt, y, yt = datasets.to_forecasting(x, forecast=5, test_size=10)

    assert x.shape[0] == 200 - 5 - 10
    assert y.shape[0] == 200 - 5 - 10
    assert x.shape[0] == y.shape[0]
    assert xt.shape[0] == yt.shape[0] == 10


def test_japanese_vowels():
    X_train, X_test, Y_train, Y_test = datasets.japanese_vowels(reload=True)

    assert len(X_train) == 270 == len(Y_train)
    assert len(X_test) == 370 == len(Y_test)

    assert Y_train[0].shape == (1, 9)

    X_train, X_test, Y_train, Y_test = datasets.japanese_vowels(repeat_targets=True)

    assert Y_train[0].shape == (X_train[0].shape[0], 9)

    X, X_test, Y, Y_test = datasets.japanese_vowels(one_hot_encode=False)

    assert Y[0].shape == (1, 1)


def test_santafe_laser():
    timeseries = datasets.santafe_laser()

    assert timeseries.shape == (10_093, 1)


def test_one_hot_encode():
    classes = ["green", "blue", "black", "white", "purple"]
    n = 82
    m = 113
    rng = np.random.default_rng(seed=1)
    n_classes = len(classes)

    y = rng.choice(classes, size=(n,), replace=True)
    y_encoded, classes = datasets.one_hot_encode(y)
    assert len(classes) == n_classes
    assert isinstance(y_encoded, np.ndarray) and y_encoded.shape == (n, n_classes)

    y = rng.choice(classes, size=(n, 1), replace=True)
    y_encoded, classes = datasets.one_hot_encode(y)
    assert len(classes) == n_classes
    assert isinstance(y_encoded, np.ndarray) and y_encoded.shape == (n, n_classes)

    y = list(rng.choice(classes, size=(n,), replace=True))
    y_encoded, classes = datasets.one_hot_encode(y)
    assert len(classes) == n_classes
    assert isinstance(y_encoded, np.ndarray) and y_encoded.shape == (n, n_classes)

    y = rng.choice(classes, size=(n, m), replace=True)
    y_encoded, classes = datasets.one_hot_encode(y)
    assert len(classes) == n_classes
    assert isinstance(y_encoded, np.ndarray) and y_encoded.shape == (n, m, n_classes)

    y = rng.choice(classes, size=(n, m, 1), replace=True)
    y_encoded, classes = datasets.one_hot_encode(y)
    assert len(classes) == n_classes
    assert isinstance(y_encoded, np.ndarray) and y_encoded.shape == (n, m, n_classes)

    y = list([rng.choice(classes, size=(m + i, 1), replace=True) for i in range(n)])
    y_encoded, classes = datasets.one_hot_encode(y)
    assert len(classes) == n_classes
    assert isinstance(y_encoded, list)
    assert len(y_encoded) == n
    assert y_encoded[-1].shape == (m + n - 1, n_classes)

    y = list([rng.choice(classes, size=(m + i,), replace=True) for i in range(n)])
    y_encoded, classes = datasets.one_hot_encode(y)
    assert len(classes) == n_classes
    assert isinstance(y_encoded, list)
    assert len(y_encoded) == n
    assert y_encoded[-1].shape == (m + n - 1, n_classes)


def test_from_aeon_classification():
    n_timeseries = 10
    n_timesteps = 100
    n_dimensions = 3
    X_aeon = np.zeros((n_timeseries, n_dimensions, n_timesteps))
    X_aeon[0, 1, 2] = np.pi

    X_rpy = datasets.from_aeon_classification(X_aeon)

    assert X_rpy.shape == (n_timeseries, n_timesteps, n_dimensions)
    assert X_rpy[0, 2, 1] == np.pi

    # variable length collections
    X_aeon_list = [np.zeros((n_dimensions, 10 + i)) for i in range(10)]
    X_aeon_list[0][1, 2] = np.pi

    X_rpy_list = datasets.from_aeon_classification(X_aeon_list)

    assert len(X_rpy_list) == len(X_aeon_list)
    assert X_rpy_list[-1].shape == X_aeon_list[-1].shape[::-1]
    assert X_rpy[0][2, 1] == np.pi

    X_aeon_invalid = True
    with pytest.raises(TypeError):
        datasets.from_aeon_classification(X_aeon_invalid)

    X_aeon_array_like = range(10)
    with pytest.raises(ValueError):
        datasets.from_aeon_classification(X_aeon_array_like)
