# Author: Nathan Trouvain at 06/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from reservoirpy.activationsfunc import relu
from reservoirpy.nodes import Reservoir


def test_reservoir_init():

    node = Reservoir(100, lr=0.8, input_bias=False)

    data = np.ones((1, 10))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert node.Win.shape == (100, 10)
    assert node.lr == 0.8
    assert node.units == 100

    data = np.ones((10000, 10))
    res = node.run(data)

    assert res.shape == (10000, 100)

    with pytest.raises(ValueError):
        Reservoir()

    with pytest.raises(ValueError):
        Reservoir(100, equation="foo")

    res = Reservoir(100, activation="relu", fb_activation="relu")
    assert id(res.activation) == id(relu)
    assert id(res.fb_activation) == id(relu)


def test_reservoir_init_from_matrices():

    Win = np.ones((100, 10))

    node = Reservoir(100, lr=0.8, Win=Win, input_bias=False)

    data = np.ones((1, 10))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert_array_equal(node.Win, Win)
    assert node.lr == 0.8
    assert node.units == 100

    data = np.ones((10000, 10))
    res = node.run(data)

    assert res.shape == (10000, 100)

    Win = np.ones((100, 11))

    node = Reservoir(100, lr=0.8, Win=Win, input_bias=True)

    data = np.ones((1, 10))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert_array_equal(np.c_[node.bias, node.Win], Win)
    assert node.lr == 0.8
    assert node.units == 100

    data = np.ones((10000, 10))
    res = node.run(data)

    assert res.shape == (10000, 100)

    # Shape override (matrix.shape > units parameter)
    data = np.ones((1, 10))
    W = np.ones((10, 10))
    res = Reservoir(100, W=W)
    _ = res(data)
    assert res.units == 10
    assert res.output_dim == 10

    with pytest.raises(ValueError):  # Bad matrix shape
        W = np.ones((10, 11))
        res = Reservoir(W=W)
        res(data)

    with pytest.raises(ValueError):  # Bad matrix format
        res = Reservoir(100, W=1.0)
        res(data)

    with pytest.raises(ValueError):  # Bias in Win but no bias accepted
        res = Reservoir(100, Win=np.ones((100, 11)), input_bias=False)
        res(data)

    with pytest.raises(ValueError):  # Bad Win shape
        res = Reservoir(100, Win=np.ones((100, 20)), input_bias=True)
        res(data)

    with pytest.raises(ValueError):  # Bad Win shape
        res = Reservoir(100, Win=np.ones((101, 10)), input_bias=True)
        res(data)

    with pytest.raises(ValueError):  # Bad matrix format
        res = Reservoir(100, Win=1.0)
        res(data)


def test_reservoir_bias():

    node = Reservoir(100, lr=0.8, input_bias=False)

    data = np.ones((1, 10))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert node.Win.shape == (100, 10)
    assert node.bias.shape == (100, 1)
    assert node.Wfb is None
    assert_array_equal(node.bias, np.zeros((100, 1)))
    assert node.lr == 0.8
    assert node.units == 100

    node = Reservoir(100, lr=0.8, input_bias=True)

    data = np.ones((1, 10))
    res = node(data)

    assert node.bias.shape == (100, 1)

    bias = np.ones((100, 1))
    node = Reservoir(100, bias=bias)
    res = node(data)

    assert_array_equal(node.bias, bias)

    bias = np.ones((100,))
    node = Reservoir(100, bias=bias)
    res = node(data)

    assert_array_equal(node.bias, bias)

    with pytest.raises(ValueError):
        bias = np.ones((101, 1))
        node = Reservoir(100, bias=bias)
        node(data)

    with pytest.raises(ValueError):
        bias = np.ones((101, 2))
        node = Reservoir(100, bias=bias)
        node(data)

    with pytest.raises(ValueError):
        node = Reservoir(100, bias=1.0)
        node(data)


def test_reservoir_run():
    x = np.ones((10, 5))

    res = Reservoir(100, equation="internal")
    out = res.run(x)
    assert out.shape == (10, 100)

    res = Reservoir(100, equation="external")
    out = res.run(x)
    assert out.shape == (10, 100)


def test_reservoir_chain():

    node1 = Reservoir(100, lr=0.8, input_bias=False)
    node2 = Reservoir(50, lr=1.0, input_bias=False)

    data = np.ones((1, 10))
    res = (node1 >> node2)(data)

    assert node1.W.shape == (100, 100)
    assert node1.Win.shape == (100, 10)
    assert node2.W.shape == (50, 50)
    assert node2.Win.shape == (50, 100)

    assert res.shape == (1, 50)


def test_reservoir_feedback():

    node1 = Reservoir(100, lr=0.8, input_bias=False)
    node2 = Reservoir(50, lr=1.0, input_bias=False)

    node1 <<= node2

    data = np.ones((1, 10))
    res = (node1 >> node2)(data)

    assert node1.W.shape == (100, 100)
    assert node1.Win.shape == (100, 10)
    assert node2.W.shape == (50, 50)
    assert node2.Win.shape == (50, 100)

    assert res.shape == (1, 50)

    assert node1.Wfb is not None
    assert node1.Wfb.shape == (100, 50)

    with pytest.raises(ValueError):
        Wfb = np.ones((100, 51))
        node1 = Reservoir(100, lr=0.8, Wfb=Wfb)
        node2 = Reservoir(50, lr=1.0)
        node1 <<= node2
        data = np.ones((1, 10))
        res = (node1 >> node2)(data)

    with pytest.raises(ValueError):
        Wfb = np.ones((101, 50))
        node1 = Reservoir(100, lr=0.8, Wfb=Wfb)
        node2 = Reservoir(50, lr=1.0)
        node1 <<= node2
        data = np.ones((1, 10))
        res = (node1 >> node2)(data)

    with pytest.raises(ValueError):
        node1 = Reservoir(100, lr=0.8, Wfb=1.0)
        node2 = Reservoir(50, lr=1.0)
        node1 <<= node2
        data = np.ones((1, 10))
        res = (node1 >> node2)(data)
