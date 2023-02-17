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


def test_ridge_regression_raises(bad_xdata, bad_ydata):
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
