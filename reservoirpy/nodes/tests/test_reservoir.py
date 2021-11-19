# Author: Nathan Trouvain at 06/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from numpy.testing import assert_array_equal

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
