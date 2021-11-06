import pytest
import numpy as np

from numpy.testing import assert_allclose

from reservoirpy import datasets
from reservoirpy.datasets import to_forecasting


@pytest.mark.parametrize("dataset_func", [
    datasets.henon_map,
    datasets.logistic_map,
    datasets.lorenz,
    datasets.mackey_glass,
    datasets.multiscroll,
    datasets.rabinovich_fabrikant,
    datasets.narma,
])
def test_generation(dataset_func):

    timesteps = 2000
    X = dataset_func(timesteps)

    assert isinstance(X, np.ndarray)
    assert len(X) == timesteps


@pytest.mark.parametrize("dataset_func", [
    datasets.mackey_glass
])
def test_seed(dataset_func):
    x1 = dataset_func(200)
    x2 = dataset_func(200)

    assert_allclose(x1, x2)


@pytest.mark.parametrize("dataset_func", [
    datasets.mackey_glass
])
def test_reseed(dataset_func):

    s = datasets.get_seed()
    assert s == datasets._seed._DEFAULT_SEED

    x1 = dataset_func(200)

    datasets.set_seed(1234)
    assert datasets._seed._DEFAULT_SEED == 1234
    assert datasets.get_seed() == 1234

    x2 = dataset_func(200)

    assert (np.abs(x1 - x2) > 1e-3).sum() > 0


@pytest.mark.parametrize("dataset_func", [
    datasets.mackey_glass,
    datasets.lorenz
])
def test_to_forecasting(dataset_func):
    x = dataset_func(200)

    x, y = to_forecasting(x, forecast=5)

    assert x.shape[0] == 200 - 5
    assert y.shape[0] == 200 - 5
    assert x.shape[0] == y.shape[0]


@pytest.mark.parametrize("dataset_func", [
    datasets.mackey_glass,
    datasets.lorenz
])
def test_to_forecasting_with_test(dataset_func):
    x = dataset_func(200)

    x, xt, y, yt = to_forecasting(x, forecast=5, test_size=10)

    assert x.shape[0] == 200 - 5 - 10
    assert y.shape[0] == 200 - 5 - 10
    assert x.shape[0] == y.shape[0]
    assert xt.shape[0] == yt.shape[0] == 10
