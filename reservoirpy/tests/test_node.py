# Author: Nathan Trouvain at 08/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

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
    data = np.zeros((1, 5))

    res = plus_node(data)

    assert_array_equal(res, data + 1)
    assert plus_node.initialized
    assert plus_node.input_dim == 5
    assert plus_node.output_dim == 5
    assert plus_node.h == 1

    data = np.zeros((1, 8))

    # TODO: uncomment when input check is implemented
    # with pytest.raises(ValueError):
    #     plus_node(data)


def test_node_call():
    plus_node = PlusNode(h=2)
    data = np.zeros((1, 5))
    res = plus_node(data)

    assert_array_equal(res, data + 2)
    assert plus_node.state is not None
    assert_array_equal(data + 2, plus_node.state["out"])


def test_node_dimensions():
    plus_node = PlusNode()
    data = np.zeros((5,))
    res = plus_node(data)

    # TODO: uncomment after type check
    # # input size mismatch
    # with pytest.raises(ValueError):
    #     data = np.zeros((6,))
    #     plus_node(data)

    # # input size mismatch in run,
    # # no matter how many timesteps are given
    # with pytest.raises(ValueError):
    #     data = np.zeros((5, 6))
    #     plus_node.run(data)

    # with pytest.raises(ValueError):
    #     data = np.zeros((1, 6))
    #     plus_node.run(data)

    # # no timespans in call, only single timesteps
    # with pytest.raises(ValueError):
    #     data = np.zeros((2, 5))
    #     plus_node(data)


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

    offline_node.partial_fit(X, Y)

    assert_array_equal(offline_node.get_buffer("b"), np.array([0.5]))

    offline_node.fit()

    assert_array_equal(offline_node.b, np.array([0.5]))

    X = np.ones((10, 5)) * 2.0
    Y = np.ones((10, 5))

    offline_node.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([1.0]))

    X = [np.ones((10, 5)) * 2.0] * 3
    Y = [np.ones((10, 5))] * 3

    offline_node.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([3.0]))

    offline_node.partial_fit(X, Y)

    assert_array_equal(offline_node.get_buffer("b"), np.array([3.0]))


def test_unsupervised_fit():
    unsupervised_node = OnlineUnsupervised()
    X = np.ones((10, 5))

    assert unsupervised_node.b == 0

    unsupervised_node.partial_fit(X)

    assert_array_equal(unsupervised_node.get_buffer("b"), np.array([1.0]))

    unsupervised_node.fit()

    assert_array_equal(unsupervised_node.b, np.array([1.0]))

    X = np.ones((10, 5)) * 2.0

    unsupervised_node.fit(X)

    assert_array_equal(unsupervised_node.b, np.array([2.0]))

    X = [np.ones((10, 5)) * 2.0] * 3

    unsupervised_node.fit(X)

    assert_array_equal(unsupervised_node.b, np.array([6.0]))

    unsupervised_node.partial_fit(X)

    assert_array_equal(unsupervised_node.get_buffer("b"), np.array([6.0]))


def test_train_unsupervised():
    online_node = OnlineUnsupervised()
    X = np.ones((10, 5))

    assert online_node.b == 0

    online_node.train(X)

    assert_array_equal(online_node.b, np.array([10.0]))

    X = np.ones((10, 5)) * 2.0

    online_node.train(X)

    assert_array_equal(online_node.b, np.array([30.0]))

    X = [np.ones((10, 5)) * 2.0] * 3

    with pytest.raises(TypeError):
        online_node.train(X)


def test_train():
    online_node = OnlineUnsupervised()
    X = np.ones((10, 5))
    Y = np.ones((10, 5))

    assert online_node.b == 0

    online_node.train(X, Y)

    assert_array_equal(online_node.b, np.array([20.0]))

    X = np.ones((10, 5)) * 2.0

    online_node.train(X, Y)

    assert_array_equal(online_node.b, np.array([50.0]))

    X = [np.ones((10, 5)) * 2.0] * 3

    with pytest.raises(TypeError):
        online_node.train(X, Y)


def test_train_raise():
    online_node = OnlineUnsupervised()
    X = [np.ones((10, 5)) * 2.0] * 3
    Y = [np.ones((10, 5)) * 2.0] * 3

    with pytest.raises(TypeError):
        online_node.train(X, Y)


def test_train_learn_every():
    online_node = OnlineUnsupervised()
    X = np.ones((10, 5))
    Y = np.ones((10, 5))

    assert online_node.b == 0

    online_node.train(X, Y, learn_every=2)

    assert_array_equal(online_node.b, np.array([10.0]))

    X = np.ones((10, 5)) * 2.0

    online_node.train(X, Y, learn_every=2)

    assert_array_equal(online_node.b, np.array([25.0]))


def test_train_supervised_by_teacher_node():
    online_node = OnlineUnsupervised()
    plus_node = PlusNode()

    X = np.ones((1, 5))

    # using not initialized node
    with pytest.raises(RuntimeError):
        online_node.train(X, plus_node)

    plus_node(np.ones((1, 5)))

    online_node.train(X, plus_node)

    assert_array_equal(online_node.b, np.array([4.0]))


def test_node_bad_learning_method():
    online_node = OnlineUnsupervised()
    plus_node = PlusNode()
    offline_node = Offline()

    X = np.ones((10, 5))
    Y = np.ones((10, 5))

    with pytest.raises(TypeError):
        online_node.fit(X, Y)

    with pytest.raises(TypeError):
        plus_node.fit(X, Y)

    with pytest.raises(TypeError):
        online_node.partial_fit(X, Y)

    with pytest.raises(TypeError):
        offline_node.train(X, Y)

    with pytest.raises(TypeError):
        plus_node.train(X, Y)


def test_offline_node_bad_warmup():
    offline_node = Offline()
    X = np.ones((10, 5))
    Y = np.ones((10, 5))

    with pytest.raises(ValueError):
        offline_node.fit(X, Y, warmup=10)


def test_offline_node_default_partial():
    basic_offline_node = Offline()
    X = np.ones((10, 5))
    Y = np.ones((10, 5))

    basic_offline_node.partial_fit(X, Y, warmup=2)
    assert_array_equal(basic_offline_node._X[0], X[2:])
