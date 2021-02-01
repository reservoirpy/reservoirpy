import pytest
import numpy as np

from numpy.testing import assert_allclose

from reservoirpy import datasets


@pytest.mark.parametrize("dataset_func", [
    datasets.henon_map,
    datasets.logistic_map,
    datasets.lorenz,
    datasets.mackey_glass,
    datasets.multiscroll,
    datasets.rabinovich_fabrikant
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
