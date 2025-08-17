# Author: Nathan Trouvain at 08/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .dummy_nodes import AccumulateNode, Offline, OnlineUnsupervised, PlusNode


def test_pickling():
    plus_node = PlusNode(name="MyNode", h=12)
    pickled_node = pickle.dumps(plus_node)
    unpickled_node = pickle.loads(pickled_node)

    assert unpickled_node.name == "MyNode"
    assert plus_node.h == 12
    assert plus_node != unpickled_node


def test_node_init():
    plus_node = PlusNode()
    data = np.zeros((5,))

    res = plus_node(data)

    assert_array_equal(res, data + 1)
    assert plus_node.initialized
    assert plus_node.input_dim == 5
    assert plus_node.output_dim == 5
    assert plus_node.h == 1

    data = np.zeros((1, 8))

    with pytest.raises(ValueError):
        plus_node(data)


def test_node_call():
    plus_node = PlusNode(h=2)
    data = np.zeros((5,))
    res = plus_node(data)

    assert_array_equal(res, data + 2)
    assert plus_node.state is not None
    assert_array_equal(data + 2, plus_node.state["out"])


def test_node_dimensions():
    plus_node = PlusNode()
    data = np.zeros((5,))
    res = plus_node(data)

    # input size mismatch
    with pytest.raises(ValueError):
        data = np.zeros((6,))
        plus_node(data)

    # input size mismatch in run,
    # no matter how many timesteps are given
    with pytest.raises(ValueError):
        data = np.zeros((5, 6))
        plus_node.run(data)

    with pytest.raises(ValueError):
        data = np.zeros((1, 6))
        plus_node.run(data)

    # no timespans in call, only single timesteps
    with pytest.raises(ValueError):
        data = np.zeros((2, 5))
        plus_node(data)


def test_node_run():
    plus_node = AccumulateNode(h=2)
    data = np.zeros((3, 5))
    res = plus_node.run(data)
    expected = np.array([[2] * 5, [4] * 5, [6] * 5])

    assert_array_equal(res, expected)
    assert_array_equal(res[-1], plus_node.state["out"])

    # res2 = plus_node.run(data, stateful=False)
    # expected2 = np.array([[8] * 5, [10] * 5, [12] * 5])

    # assert_array_equal(res2, expected2)
    # assert_array_equal(res[-1][np.newaxis, :], plus_node.state["out"])

    # res3 = plus_node.run(data, reset=True)

    # assert_array_equal(res3, expected)
    # assert_array_equal(res[-1][np.newaxis, :], plus_node.state["out"])


def test_offline_fit():
    offline_node = Offline()
    X = np.ones((10, 5)) * 0.5
    Y = np.ones((10, 5))

    assert offline_node.b == 0

    offline_node.fit(X, Y)

    assert_allclose(offline_node.b, 50.0)

    Y = -np.ones((10, 5))

    offline_node.fit(X, Y)

    assert_allclose(offline_node.b, -50)

    X = [np.ones((10, 5)) * 2.0] * 3
    Y = [0.3 * np.ones((10, 5))] * 3

    offline_node.fit(X, Y)

    assert_allclose(offline_node.b, 0.3 * 10 * 5 * 3)

    offline_node.fit(np.array(X), np.array(Y))

    assert_allclose(offline_node.b, 0.3 * 10 * 5 * 3)


def test_unsupervised_fit():
    unsupervised_node = OnlineUnsupervised()
    X = np.ones((10, 5))

    assert unsupervised_node.b == 0

    unsupervised_node.fit(X)

    assert_allclose(unsupervised_node.b, 50.0)

    X = -np.ones((10, 5))

    unsupervised_node.fit(X)

    assert_allclose(unsupervised_node.b, -50)

    X = [0.3 * np.ones((10, 5))] * 3

    unsupervised_node.fit(X)

    assert_allclose(unsupervised_node.b, 0.3 * 10 * 5 * 3)

    unsupervised_node.fit(np.array(X))

    assert_allclose(unsupervised_node.b, 0.3 * 10 * 5 * 3)


def test_partial_fit_unsupervised():
    online_node = OnlineUnsupervised()
    X = np.ones((10, 5))

    assert online_node.b == 0

    online_node.partial_fit(X)

    assert_allclose(online_node.b, 50.0)

    X = np.ones((10, 5)) * 2.0

    online_node.partial_fit(X)

    assert_allclose(online_node.b, 150.0)

    X = [np.ones((10, 5)) * 2.0] * 3


# def test_train_raise():
#     online_node = OnlineUnsupervised()
#     X = [np.ones((10, 5)) * 2.0] * 3
#     Y = [np.ones((10, 5)) * 2.0] * 3

#     with pytest.raises(TypeError):
#         online_node.partial_fit(X, Y)


def test_node_bad_learning_method():
    online_node = OnlineUnsupervised()
    plus_node = PlusNode()
    offline_node = Offline()

    X = np.ones((10, 5))
    Y = np.ones((10, 5))

    with pytest.raises(AttributeError):
        plus_node.fit(X, Y)

    with pytest.raises(AttributeError):
        offline_node.partial_fit(X, Y)

    with pytest.raises(AttributeError):
        plus_node.partial_fit(X, Y)


# def test_offline_node_bad_warmup():
#     offline_node = Offline()
#     X = np.ones((10, 5))
#     Y = np.ones((10, 5))

#     with pytest.raises(ValueError):
#         offline_node.fit(X, Y, warmup=10)
