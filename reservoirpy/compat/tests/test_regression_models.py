# Author: Nathan Trouvain at 18/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest

from ..regression_models import RidgeRegression


@pytest.fixture(scope="session")
def dummy_data():
    X = np.ones(shape=(200, 50))
    Y = np.ones(shape=(200, 5))

    return X, Y


@pytest.fixture(scope="session")
def bad_xdata():
    X = np.ones(shape=(200, 50))
    Y = np.ones(shape=(200, 5))

    bad_x = np.ones(shape=(199, 50))

    return [X, bad_x, X], [Y, Y, Y]


@pytest.fixture(scope="session")
def bad_ydata():
    X = np.ones(shape=(200, 50))
    Y = np.ones(shape=(200, 5))

    bad_y = np.ones(shape=(200, 4))

    return [X, X, X], [Y, bad_y, Y]


def test_ridge_regression(dummy_data):
    model = RidgeRegression(ridge=0.1)
    model.initialize(50, 5)

    X, Y = dummy_data
    for x, y in zip(X, Y):
        model.partial_fit(x, y)
        XXT = model._XXT.copy()
        YXT = model._YXT.copy()

    assert XXT.shape == (51, 51)
    assert YXT.shape == (5, 51)

    w = model.fit()

    assert w.shape == (5, 51)

    w = model.fit(X, Y)
    assert w.shape == (5, 51)

    for x, y in zip([X, X, X], [Y, Y, Y]):
        model.partial_fit(x, y)
        XXT = model._XXT.copy()
        YXT = model._YXT.copy()

    assert XXT.shape == (51, 51)
    assert YXT.shape == (5, 51)

    w = model.fit()

    assert w.shape == (5, 51)

    w = model.fit([X, X, X], [Y, Y, Y])
    assert w.shape == (5, 51)


def test_ridge_regression_raises(bad_xdata, bad_ydata, dummy_data):
    model = RidgeRegression(ridge=0.1)
    model.initialize(50, 5)

    X, Y = bad_xdata
    with pytest.raises(ValueError):
        for x, y in zip(X, Y):
            model.partial_fit(x, y)
            XXT = model._XXT.copy()
            YXT = model._YXT.copy()

    X, Y = bad_ydata
    with pytest.raises(ValueError):
        for x, y in zip(X, Y):
            model.partial_fit(x, y)
            XXT = model._XXT.copy()
            YXT = model._YXT.copy()

    X, Y = dummy_data
    with pytest.raises(RuntimeError):
        RidgeRegression(ridge=0.1).partial_fit(X, Y)


def test_ridge_setter():
    model = RidgeRegression(ridge=0.1)
    model.ridge = 1e-7
    assert model._ridge == 1e-7
    assert model._ridgeid is None

    model.initialize(50, 5)
    model.ridge = 1e-7
    assert model._ridge == 1e-7
    assert isinstance(model._ridgeid, np.ndarray)
    assert model._ridgeid.shape == (51, 51)


def test_input_preparation(dummy_data):
    x_not_prepared = [50 * [1] for _ in range(200)]
    y_not_prepared = [5 * [1] for _ in range(200)]

    model = RidgeRegression(ridge=0.1)
    model.initialize(50, 5)
    model.partial_fit(x_not_prepared, y_not_prepared)

    X, Y = dummy_data
    model2 = RidgeRegression(ridge=0.1)
    model2.initialize(50, 5)
    model2.partial_fit(X, Y)

    assert type(model._XXT) == type(model2._XXT)
    # numpy does not have any test equality for memmap
    # so use tolist to check that
    assert model._XXT.tolist() == model2._XXT.tolist()
