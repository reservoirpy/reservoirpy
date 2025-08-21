# Author: Nathan Trouvain at 06/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from reservoirpy.activationsfunc import relu
from reservoirpy.mat_gen import bernoulli, zeros
from reservoirpy.nodes import ES2N


def test_es2n_init():
    node = ES2N(100, proximity=0.8, bias=zeros)

    data = np.ones((10,))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert node.Win.shape == (100, 10)
    assert node.proximity == 0.8
    assert node.units == 100

    data = np.ones((10000, 10))
    res = node.run(data)

    assert res.shape == (10000, 100)

    with pytest.raises(ValueError):
        ES2N()

    res = ES2N(100, activation="relu")
    assert id(res.activation) == id(relu)


def test_es2n_init_from_proximity_is_arrays():
    proximity = np.ones((100,)) * 0.5
    input_scaling = np.ones((10,)) * 0.8
    node = ES2N(100, proximity=proximity, input_scaling=input_scaling)

    data = np.ones((2, 10))
    res = node.run(data)

    assert node.W.shape == (100, 100)
    assert node.Win.shape == (100, 10)
    assert_array_equal(node.proximity, np.ones(100) * 0.5)
    assert_array_equal(node.input_scaling, np.ones(10) * 0.8)


def test_es2n_init_from_matrices():
    Win = np.ones((100, 10))

    node = ES2N(100, proximity=0.8, Win=Win, bias=bernoulli)

    data = np.ones((10,))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert_array_equal(node.Win, Win)
    assert node.proximity == 0.8
    assert node.units == 100

    data = np.ones((10000, 10))
    res = node.run(data)

    assert res.shape == (10000, 100)

    Win = np.ones((100, 10))
    bias = np.ones((100,))

    node = ES2N(100, proximity=0.8, Win=Win, bias=bias)

    data = np.ones((10,))
    res = node(data)

    assert node.W.shape == (100, 100)
    assert node.proximity == 0.8
    assert node.units == 100
    assert node.output_dim == 100

    data = np.ones((10000, 10))
    res = node.run(data)

    assert res.shape == (10000, 100)

    with pytest.raises(ValueError):  # Shape override (matrix.shape > units parameter)
        data = np.ones((10))
        W = np.ones((10, 10))
        res = ES2N(100, W=W)
        _ = res(data)

    with pytest.raises(ValueError):  # Bad matrix shape
        W = np.ones((10, 11))
        res = ES2N(W=W)
        res(data)

    with pytest.raises(ValueError):  # Bad matrix format
        res = ES2N(100, W=1.0)
        res(data)

    with pytest.raises(ValueError):  # Bad Win shape
        res = ES2N(100, Win=np.ones((100, 20)))
        res(data)

    with pytest.raises(ValueError):  # Bad Win shape
        res = ES2N(100, Win=np.ones((101, 10)))
        res(data)

    with pytest.raises(ValueError):  # Bad matrix format
        res = ES2N(100, Win=1.0)
        res(data)


def test_es2n_bias():
    node = ES2N(100, proximity=0.8, bias=zeros)
    data = np.ones((10,))
    res = node(data)
    assert node.W.shape == (100, 100)
    assert node.Win.shape == (100, 10)
    assert node.bias.shape == (100,)
    assert_array_equal(node.bias, np.zeros((100,)))
    assert node.proximity == 0.8
    assert node.units == 100

    node = ES2N(100, proximity=0.8, bias=bernoulli)
    data = np.ones((10,))
    res = node(data)
    assert node.bias.shape == (100,)

    bias = np.ones((100,))
    node = ES2N(100, bias=bias)
    res = node(data)
    assert_array_equal(node.bias, bias)

    bias = np.ones((100,))
    node = ES2N(100, bias=bias)
    res = node(data)
    assert_array_equal(node.bias, bias)

    data = np.zeros((100,))
    node = ES2N(100, bias=1.0)
    res = node(data)
    assert_array_equal(res, np.tanh(np.ones((100,))))

    with pytest.raises(AssertionError):
        bias = np.ones((100, 1))
        node = ES2N(100, bias=bias)
        res = node(data)
        assert res.shape == (100,)

    with pytest.raises(ValueError):
        bias = np.ones((100, 2))
        node = ES2N(100, bias=bias)
        node(data)


def test_es2n_run():
    x = np.ones((10, 5))

    res = ES2N(100)
    out = res.run(x)
    assert out.shape == (10, 100)


def test_es2n_chain():
    node1 = ES2N(100, proximity=0.8)
    node2 = ES2N(50, proximity=1.0)

    data = np.ones((10,))
    res = (node1 >> node2)(data)

    assert node1.W.shape == (100, 100)
    assert node1.Win.shape == (100, 10)
    assert node2.W.shape == (50, 50)
    assert node2.Win.shape == (50, 100)

    assert res.shape == (50,)


def test_es2n_seed():
    node1 = ES2N(100, seed=123)
    node2 = ES2N(100, seed=123)

    data = np.ones((10, 10))

    assert_array_equal(node1.run(data), node2.run(data))

    node1 = ES2N(100, seed=123)
    node2 = ES2N(100, seed=123)

    data = np.ones((10, 10))

    assert_array_equal(node1.run(data), node2.run(data))
