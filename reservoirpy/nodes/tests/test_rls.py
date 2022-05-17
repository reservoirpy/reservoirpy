# Author: Nathan Trouvain at 17/05/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
from numpy.testing import assert_array_almost_equal

from reservoirpy.nodes import RLS, Reservoir


def test_rls_init():

    node = RLS(10)

    data = np.ones((1, 100))
    res = node(data)

    assert node.Wout.shape == (100, 10)
    assert node.bias.shape == (1, 10)
    assert node.alpha == 1e-6

    data = np.ones((10000, 100))
    res = node.run(data)

    assert res.shape == (10000, 10)


def test_rls_train_one_step():

    node = RLS(10)

    x = np.ones((5, 2))
    y = np.ones((5, 10))

    for x, y in zip(x, y):
        res = node.train(x, y)

    assert node.Wout.shape == (2, 10)
    assert node.bias.shape == (1, 10)
    assert node.alpha == 1e-6

    data = np.ones((10000, 2))
    res = node.run(data)

    assert res.shape == (10000, 10)


def test_rls_train():

    node = RLS(10)

    X, Y = np.ones((200, 100)), np.ones((200, 10))

    res = node.train(X, Y)

    assert res.shape == (200, 10)
    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
    assert node.bias.shape == (1, 10)
    assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

    node = RLS(10)

    X, Y = np.ones((200, 100)), np.ones((200, 10))

    res = node.train(X, Y)

    assert res.shape == (200, 10)
    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
    assert node.bias.shape == (1, 10)
    assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

    node = RLS(10)

    X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

    for x, y in zip(X, Y):
        res = node.train(x, y)

    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
    assert node.bias.shape == (1, 10)
    assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

    data = np.ones((1000, 100))
    res = node.run(data)

    assert res.shape == (1000, 10)


def test_esn_rls():

    readout = RLS(10)
    reservoir = Reservoir(100)

    esn = reservoir >> readout

    X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

    for x, y in zip(X, Y):
        res = esn.train(x, y)

    assert readout.Wout.shape == (100, 10)
    assert readout.bias.shape == (1, 10)

    data = np.ones((1000, 100))
    res = esn.run(data)

    assert res.shape == (1000, 10)


def test_rls_feedback():

    readout = RLS(10)
    reservoir = Reservoir(100)

    esn = reservoir >> readout

    reservoir <<= readout

    X, Y = np.ones((5, 200, 8)), np.ones((5, 200, 10))
    for x, y in zip(X, Y):
        res = esn.train(x, y)

    assert readout.Wout.shape == (100, 10)
    assert readout.bias.shape == (1, 10)
    assert reservoir.Wfb.shape == (100, 10)

    data = np.ones((1000, 8))
    res = esn.run(data)

    assert res.shape == (1000, 10)


def test_hierarchical_esn():

    readout1 = RLS(10, name="r1")
    reservoir1 = Reservoir(100)
    readout2 = RLS(3, name="r2")
    reservoir2 = Reservoir(100)

    esn = reservoir1 >> readout1 >> reservoir2 >> readout2

    X, Y = np.ones((200, 5)), {"r1": np.ones((200, 10)), "r2": np.ones((200, 3))}
    res = esn.train(X, Y)

    assert readout1.Wout.shape == (100, 10)
    assert readout1.bias.shape == (1, 10)

    assert readout2.Wout.shape == (100, 3)
    assert readout2.bias.shape == (1, 3)

    assert reservoir1.Win.shape == (100, 5)
    assert reservoir2.Win.shape == (100, 10)

    data = np.ones((10000, 5))
    res = esn.run(data)

    assert res.shape == (10000, 3)


def test_dummy_mutual_supervision():

    readout1 = RLS(1, name="r1")
    reservoir1 = Reservoir(100)
    readout2 = RLS(1, name="r2")
    reservoir2 = Reservoir(100)

    reservoir1 <<= readout1
    reservoir2 <<= readout2

    branch1 = reservoir1 >> readout1
    branch2 = reservoir2 >> readout2

    model = branch1 & branch2

    X = np.ones((200, 5))

    res = model.train(X, Y={"r1": readout2, "r2": readout1}, force_teachers=True)

    assert readout1.Wout.shape == (100, 1)
    assert readout1.bias.shape == (1, 1)

    assert readout2.Wout.shape == (100, 1)
    assert readout2.bias.shape == (1, 1)

    assert reservoir1.Win.shape == (100, 5)
    assert reservoir2.Win.shape == (100, 5)
