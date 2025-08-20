# Author: Nathan Trouvain at 17/05/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from reservoirpy.nodes import RLS


def test_rls_init():
    node = RLS(10, output_dim=3)

    data = np.ones((100,))
    # y = np.ones((1, 3))
    res = node(data)

    assert node.Wout.shape == (100, 3)
    assert node.bias.shape == (3,)
    assert node.alpha == 10

    data = np.ones((10000, 100))
    res = node.run(data)
    assert res.shape == (10000, 3)

    # with initialized Wout and bias
    Wout = np.ones((100, 3))
    bias = np.ones((3,))
    data = np.ones((100,))
    # unspecified dimensions
    node = RLS(10, Wout=Wout, bias=bias)
    res = node(data)
    # correct specified dimensions
    node = RLS(10, Wout=Wout, bias=bias, input_dim=100, output_dim=3)
    res = node(data)
    # incorrect specified dimensions
    with pytest.raises(ValueError):
        node = RLS(10, Wout=Wout, bias=bias, input_dim=101, output_dim=1)
    with pytest.raises(ValueError):
        bias = np.ones((10,))
        node = RLS(bias=bias, output_dim=1)


def test_rls_train_one_step():
    node = RLS(10)

    x = np.ones((10, 5, 2))
    y = np.ones((10, 5, 10))

    for x, y in zip(x, y):
        res = node.partial_fit(x, y)

    assert node.Wout.shape == (2, 10)
    assert node.bias.shape == (10,)
    assert node.alpha == 10

    data = np.ones((10000, 2))
    res = node.run(data)

    assert res.shape == (10000, 10)


def test_rls_train():
    node = RLS(alpha=1e-6)

    X, Y = np.ones((200, 100)), np.ones((200, 10))

    res = node.partial_fit(X, Y)

    assert res.shape == (200, 10)
    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.005, decimal=4)
    assert node.bias.shape == (10,)
    assert_array_almost_equal(node.bias, np.ones((10,)) * 0.5, decimal=4)

    node = RLS(1e-6, fit_bias=False)

    X, Y = np.ones((200, 100)), np.ones((200, 10))

    res = node.partial_fit(X, Y)

    assert res.shape == (200, 10)
    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
    assert node.bias.shape == (10,)
    assert_array_almost_equal(node.bias, np.ones((10,)) * 0.0, decimal=4)

    node = RLS(1e-6, fit_bias=True)

    X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

    for x, y in zip(X, Y):
        res = node.partial_fit(x, y)

    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.005, decimal=4)
    assert node.bias.shape == (10,)
    assert_array_almost_equal(node.bias, np.ones((10,)) * 0.5, decimal=4)

    data = np.ones((1000, 100))
    res = node.run(data)

    assert res.shape == (1000, 10)
