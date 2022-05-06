# Author: Nathan Trouvain at 08/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .dummy_nodes import *


def test_node_creation(plus_node):
    assert plus_node.name == "PlusNode-0"
    assert plus_node.params["c"] is None
    assert plus_node.hypers["h"] == 1
    assert plus_node.input_dim is None
    assert plus_node.output_dim is None
    assert not plus_node.is_initialized
    assert hasattr(plus_node, "c")
    assert hasattr(plus_node, "h")
    assert plus_node.state() is None


def test_node_attr(plus_node):
    assert plus_node.get_param("c") is None
    assert plus_node.get_param("h") == 1

    plus_node.set_param("c", 1)

    assert plus_node.get_param("c") == 1

    with pytest.raises(AttributeError):
        plus_node.get_param("foo")

    with pytest.raises(KeyError):
        plus_node.set_param("foo", 1)

    plus_node.params["a"] = 2

    assert plus_node.get_param("a") == 2
    plus_node.set_param("a", 3)
    assert plus_node.get_param("a") == 3
    assert plus_node.a == 3
    assert plus_node.c == 1
    assert plus_node.h == 1

    plus_node.h = 5
    assert plus_node.h == 5


def test_node_init(plus_node):

    data = np.zeros((1, 5))

    res = plus_node(data)

    assert_array_equal(res, data + 2)
    assert plus_node.is_initialized
    assert plus_node.input_dim == 5
    assert plus_node.output_dim == 5
    assert plus_node.c == 1

    data = np.zeros((1, 8))

    with pytest.raises(ValueError):
        plus_node(data)

    with pytest.raises(TypeError):
        plus_node.set_input_dim(9)
    with pytest.raises(TypeError):
        plus_node.set_output_dim(45)


def test_node_init_empty(plus_node):

    plus_noinit = PlusNode(input_dim=5)
    plus_noinit.initialize()

    assert plus_noinit.input_dim == 5
    assert plus_noinit.c == 1
    assert_array_equal(plus_noinit.state(), np.zeros((1, 5)))

    multiinput = MultiInput(input_dim=(5, 2))
    multiinput.initialize()

    assert multiinput.input_dim == (5, 2)

    with pytest.raises(RuntimeError):
        plus_noinit = PlusNode()
        plus_noinit.initialize()

    data = np.zeros((1, 5))

    plus_node.set_input_dim(5)

    plus_node.initialize()

    assert plus_node.is_initialized
    assert plus_node.input_dim == 5
    assert plus_node.output_dim == 5
    assert plus_node.c == 1

    data = np.zeros((1, 8))

    with pytest.raises(ValueError):
        plus_node(data)

    with pytest.raises(TypeError):
        plus_node.set_input_dim(9)
    with pytest.raises(TypeError):
        plus_node.set_output_dim(45)


def test_node_call(plus_node):
    data = np.zeros((1, 5))
    res = plus_node(data)

    assert_array_equal(res, data + 2)
    assert plus_node.state() is not None
    assert_array_equal(data + 2, plus_node.state())

    res2 = plus_node(data)
    assert_array_equal(res2, data + 4)
    assert_array_equal(plus_node.state(), data + 4)

    res3 = plus_node(data, stateful=False)
    assert_array_equal(res3, data + 6)
    assert_array_equal(plus_node.state(), data + 4)

    res4 = plus_node(data, reset=True)
    assert_array_equal(res4, res)
    assert_array_equal(plus_node.state(), data + 2)


def test_node_dimensions(plus_node):
    data = np.zeros((1, 5))
    res = plus_node(data)

    # input size mismatch
    with pytest.raises(ValueError):
        data = np.zeros((1, 6))
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


def test_node_state(plus_node):
    data = np.zeros((1, 5))

    with pytest.raises(RuntimeError):
        with plus_node.with_state(np.ones((1, 5))):
            plus_node(data)

    plus_node(data)
    assert_array_equal(plus_node.state(), data + 2)

    with plus_node.with_state(np.ones((1, 5))):
        res_w = plus_node(data)
        assert_array_equal(res_w, data + 3)
    assert_array_equal(plus_node.state(), data + 2)

    with plus_node.with_state(np.ones((1, 5)), stateful=True):
        res_w = plus_node(data)
        assert_array_equal(res_w, data + 3)
    assert_array_equal(plus_node.state(), data + 3)

    with plus_node.with_state(reset=True):
        res_w = plus_node(data)
        assert_array_equal(res_w, data + 2)
    assert_array_equal(plus_node.state(), data + 3)

    with pytest.raises(ValueError):
        with plus_node.with_state(np.ones((1, 8))):
            plus_node(data)


def test_node_run(plus_node):
    data = np.zeros((3, 5))
    res = plus_node.run(data)
    expected = np.array([[2] * 5, [4] * 5, [6] * 5])

    assert_array_equal(res, expected)
    assert_array_equal(res[-1][np.newaxis, :], plus_node.state())

    res2 = plus_node.run(data, stateful=False)
    expected2 = np.array([[8] * 5, [10] * 5, [12] * 5])

    assert_array_equal(res2, expected2)
    assert_array_equal(res[-1][np.newaxis, :], plus_node.state())

    res3 = plus_node.run(data, reset=True)

    assert_array_equal(res3, expected)
    assert_array_equal(res[-1][np.newaxis, :], plus_node.state())


def test_offline_fit(offline_node):
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


def test_unsupervised_fit(unsupervised_node):
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


def test_train_unsupervised(online_node):
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


def test_train(online_node):
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


def test_train_learn_every(online_node):
    X = np.ones((10, 5))
    Y = np.ones((10, 5))

    assert online_node.b == 0

    online_node.train(X, Y, learn_every=2)

    assert_array_equal(online_node.b, np.array([10.0]))

    X = np.ones((10, 5)) * 2.0

    online_node.train(X, Y, learn_every=2)

    assert_array_equal(online_node.b, np.array([25.0]))


def test_train_supervised_by_teacher_node(online_node, plus_node):

    X = np.ones((1, 5))

    # using not initialized node
    with pytest.raises(RuntimeError):
        online_node.train(X, plus_node)

    plus_node(np.ones((1, 5)))

    online_node.train(X, plus_node)

    assert_array_equal(online_node.b, np.array([4.0]))


def test_node_bad_learning_method(online_node, plus_node, offline_node):

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


def test_offline_node_bad_warmup(offline_node):

    X = np.ones((10, 5))
    Y = np.ones((10, 5))

    with pytest.raises(ValueError):
        offline_node.fit(X, Y, warmup=10)


def test_offline_node_default_partial(basic_offline_node):

    X = np.ones((10, 5))
    Y = np.ones((10, 5))

    basic_offline_node.partial_fit(X, Y, warmup=2)
    assert_array_equal(basic_offline_node._X[0], X[2:])


def test_multi_input(multiinput):

    multi_noinit = MultiInput(input_dim=(5, 2))
    multi_noinit.initialize()

    with pytest.raises(RuntimeError):
        multi_noinit = MultiInput()
        multi_noinit.initialize()

    x = [np.ones((1, 5)), np.ones((1, 2))]

    r = multiinput(x)

    assert r.shape == (1, 7)
    assert multiinput.input_dim == (5, 2)

    multi_noinit = MultiInput()
    x = [np.ones((2, 5)), np.ones((2, 2))]
    r = multi_noinit.run(x)

    assert multi_noinit.input_dim == (5, 2)


def test_feedback_noinit(feedback_node):

    with pytest.raises(RuntimeError):
        feedback_node.feedback()

    inv_notinit = Inverter(input_dim=5, output_dim=5)

    feedback_node <<= inv_notinit

    data = np.ones((1, 5))

    feedback_node(data)

    assert_array_equal(feedback_node.feedback(), inv_notinit.state())


def test_feedback_initialize_feedback(feedback_node):
    inv_notinit = Inverter(input_dim=5, output_dim=5)
    feedback_node <<= inv_notinit

    data = np.ones((1, 5))

    feedback_node.initialize_feedback()
    res = feedback_node(data)

    fb = feedback_node.feedback()
    inv_state = inv_notinit.state()

    assert_array_equal(inv_state, fb)

    inv_notinit = Inverter(input_dim=5, output_dim=5)
    plus_noinit = PlusNode(input_dim=5, output_dim=5)

    # default feedback initializer (plus_node is not supposed to handle feedback)
    plus_noinit <<= inv_notinit
    plus_noinit.initialize_feedback()
    res = plus_noinit(data)

    fb = plus_noinit.feedback()
    inv_state = inv_notinit.state()

    assert_array_equal(inv_state, fb)


def test_feedback_init_distant_model(feedback_node, plus_node, inverter_node):
    feedback_node <<= plus_node >> inverter_node

    with pytest.raises(RuntimeError):
        feedback_node.initialize_feedback()

    with pytest.raises(RuntimeError):
        data = np.ones((1, 5))
        feedback_node(data)

    plus_node.initialize(data)

    feedback_node.initialize_feedback()

    fb = feedback_node.feedback()
    inv_state = inverter_node.state()

    assert_array_equal(inv_state, fb)


def test_feedback_init_deep_distant_model(
    feedback_node, plus_node, minus_node, inverter_node
):

    feedback_node <<= plus_node >> minus_node >> inverter_node

    with pytest.raises(RuntimeError):
        feedback_node.initialize_feedback()

    with pytest.raises(RuntimeError):
        data = np.ones((1, 5))
        feedback_node(data)

    plus_node.initialize(data)
    feedback_node.initialize_feedback()

    fb = feedback_node.feedback()
    inv_state = inverter_node.state()

    assert_array_equal(inv_state, fb)
