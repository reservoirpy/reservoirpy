# Author: Nathan Trouvain at 18/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import pytest
import numpy as np

from ..regression_models import (RidgeRegression)


@pytest.fixture(scope="session")
def dummy_clf_data():
    Xn0 = np.array([[np.sin(x), np.cos(x)]
                    for x in np.linspace(0, 4*np.pi, 250)])
    Xn1 = np.array([[np.sin(10*x), np.cos(10*x)]
                   for x in np.linspace(np.pi/4, 4*np.pi+np.pi/4, 250)])
    X = np.vstack([Xn0, Xn1])
    y = np.r_[np.zeros(250), np.ones(250)].reshape(-1, 1)

    return X, y


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
