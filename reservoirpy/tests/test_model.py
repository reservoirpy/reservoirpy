# Author: Nathan Trouvain at 10/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
from numpy.testing import assert_array_equal

from .dummy_nodes import *
from ..ops import merge
from ..model import Model
from reservoirpy.nodes.io import Input


def test_node_link(plus_node, minus_node):

    clean_registry(Model)

    model1 = plus_node >> minus_node
    model2 = minus_node >> plus_node

    assert model1.name == 'Model-0'
    assert model1.params["PlusNode-0"]["c"] is None
    assert model1.hypers["PlusNode-0"]["h"] == 1
    assert model1["PlusNode-0"].input_dim is None

    assert model2.name == 'Model-1'
    assert model2.params["PlusNode-0"]["c"] is None
    assert model2.hypers["PlusNode-0"]["h"] == 1
    assert model2["PlusNode-0"].input_dim is None

    assert model1.edges == [(plus_node, minus_node)]
    assert model2.edges == [(minus_node, plus_node)]
    assert set(model1.nodes) == set(model2.nodes)

    with pytest.raises(RuntimeError):
        model1 & model2

    with pytest.raises(RuntimeError):
        plus_node >> minus_node >> plus_node

    with pytest.raises(RuntimeError):
        plus_node >> plus_node


def test_model_call(plus_node, minus_node):

    model = plus_node >> minus_node

    data = np.zeros((1, 5))
    res = model(data)

    assert_array_equal(res, data)

    input_node = Input()
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node

    model = branch1 & branch2

    res = model(data)

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, data + 2)
        else:
            assert_array_equal(arr, data - 2)

    res = model(data)

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, data + 4)
        else:
            assert_array_equal(arr, data)

    res = model(data, reset=True)

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, data + 2)
        else:
            assert_array_equal(arr, data - 2)

    res = model(data, stateful=False)

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, data + 4)
        else:
            assert_array_equal(arr, data)

    for node in model.output_nodes:
        if node.name == "PlusNode-0":
            assert_array_equal(node.state(), data + 2)
        else:
            assert_array_equal(node.state(), data - 2)


def test_model_with_state(plus_node, minus_node):

    model = plus_node >> minus_node

    data = np.zeros((1, 5))
    res = model(data)

    assert_array_equal(res, data)

    input_node = Input()
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node

    model = branch1 & branch2

    res = model(data)

    with model.with_state(state={plus_node.name:
                                 np.zeros_like(plus_node.state())}):
        assert_array_equal(plus_node.state(),
                           np.zeros_like(plus_node.state()))

    with pytest.raises(TypeError):
        with model.with_state(state=np.zeros_like(plus_node.state())):
            pass


def test_model_run(plus_node, minus_node):

    input_node = Input()
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node

    model = merge(branch1, branch2)

    data = np.zeros((3, 5))
    res = model.run(data)

    expected_plus = np.array([[2] * 5, [4] * 5, [6] * 5])
    expected_minus = np.array([[-2] * 5, [0] * 5, [-2] * 5])

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, expected_plus)
            assert_array_equal(arr[-1][np.newaxis, :], plus_node.state())
        else:
            assert_array_equal(arr, expected_minus)
            assert_array_equal(arr[-1][np.newaxis, :], minus_node.state())

    res = model.run(data, reset=True)

    expected_plus = np.array([[2] * 5, [4] * 5, [6] * 5])
    expected_minus = np.array([[-2] * 5, [0] * 5, [-2] * 5])

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, expected_plus)
            assert_array_equal(arr[-1][np.newaxis, :], plus_node.state())
        else:
            assert_array_equal(arr, expected_minus)
            assert_array_equal(arr[-1][np.newaxis, :], minus_node.state())

    res = model.run(data, stateful=False)

    expected_plus2 = np.array([[8] * 5, [10] * 5, [12] * 5])
    expected_minus2 = np.array([[0] * 5, [-2] * 5, [0] * 5])

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, expected_plus2)
            assert_array_equal(expected_plus[-1][np.newaxis, :],
                               plus_node.state())
        else:
            assert_array_equal(arr, expected_minus2)
            assert_array_equal(expected_minus[-1][np.newaxis, :],
                               minus_node.state())


def test_model_feedback(plus_node, minus_node, feedback_node):

    model = plus_node >> feedback_node >> minus_node
    feedback_node <<= minus_node

    data = np.zeros((1, 5))
    res = model(data)

    assert_array_equal(res, data + 1)
    assert_array_equal(feedback_node.state(), data + 3)

    res = model(data)
    assert_array_equal(res, data + 3)
    assert_array_equal(feedback_node.state(), data + 6)


def test_model_feedback_run(plus_node, minus_node, feedback_node):

    model = plus_node >> feedback_node >> minus_node
    feedback_node <<= minus_node

    data = np.zeros((3, 5))
    res = model.run(data)

    expected = np.array([[1] * 5, [3] * 5, [5] * 5])

    assert_array_equal(res, expected)
    assert_array_equal(feedback_node.state(), data[0][np.newaxis, :] + 10)


def test_model_feedback_forcing_sender(plus_node, minus_node, feedback_node):

    model = plus_node >> feedback_node >> minus_node
    feedback_node <<= minus_node

    data = np.zeros((3, 5))
    res = model.run(data, forced_feedbacks={"MinusNode-0": data + 1},
                    shift_fb=False)
    expected = np.array([[2] * 5, [2] * 5, [4] * 5])

    assert_array_equal(res, expected)
    assert_array_equal(feedback_node.state(), data[0][np.newaxis, :] + 8)


def test_model_feedback_forcing_receiver(plus_node, minus_node, feedback_node):

    model = plus_node >> feedback_node >> minus_node
    feedback_node <<= minus_node

    data = np.zeros((3, 5))
    res = model.run(data, forced_feedbacks={"FBNode-0": data + 1},
                    shift_fb=False)
    expected = np.array([[2] * 5, [2] * 5, [4] * 5])

    assert_array_equal(res, expected)
    assert_array_equal(feedback_node.state(), data[0][np.newaxis, :] + 8)


def test_model_feedback_from_previous_node(plus_node,
                                           minus_node,
                                           feedback_node):

    model = plus_node >> feedback_node >> minus_node
    feedback_node <<= plus_node  # feedback in time, not in space anymore

    data = np.zeros((3, 5))
    res = model.run(data)

    expected = np.array([[1] * 5, [4] * 5, [5] * 5])

    assert_array_equal(res, expected)
    assert_array_equal(feedback_node.state(), data[0][np.newaxis, :] + 11)


def test_model_feedback_from_outsider(plus_node, feedback_node,
                                      inverter_node):

    model = plus_node >> feedback_node
    feedback_node <<= (plus_node >> inverter_node)

    data = np.zeros((1, 5))
    res = model(data)

    assert_array_equal(res, data + 3)
    assert_array_equal(plus_node.state(), data + 2)
    assert_array_equal(inverter_node.state(), data)

    res = model(data)
    assert_array_equal(res, data + 3)
    assert_array_equal(plus_node.state(), data + 4)
    assert_array_equal(inverter_node.state(), data - 2)


def test_model_feedback_from_outsider_complex(plus_node, feedback_node,
                                              inverter_node, minus_node):

    model = plus_node >> feedback_node
    fb_model = plus_node >> inverter_node >> minus_node
    feedback_node <<= fb_model

    data = np.zeros((1, 5))
    res = model(data)

    assert_array_equal(res, data + 1)
    assert_array_equal(plus_node.state(), data + 2)
    assert_array_equal(minus_node.state(), data - 2)

    res = model(data)

    assert_array_equal(res, data + 3)
    assert_array_equal(plus_node.state(), data + 4)
    assert_array_equal(minus_node.state(), data - 2)


def test_offline_fit_simple_model(offline_node, offline_node2,
                                  plus_node, minus_node):

    model = plus_node >> offline_node

    X = np.ones((5, 5)) * 0.5
    Y = np.ones((5, 5))

    model.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([6.5]))

    X = np.ones((3, 5, 5)) * 0.5
    Y = np.ones((3, 5, 5))

    model.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([94.5]))

    model.fit(X, Y, reset=True)

    assert_array_equal(offline_node.b, np.array([19.5]))

    res = model.run(X[0], reset=True)

    exp = np.tile(np.array([22., 24.5, 27., 29.5, 32.]), 5).reshape(5, 5).T

    assert_array_equal(exp, res)


def test_offline_fit_simple_model_fb(offline_node, offline_node2,
                                     plus_node, minus_node,
                                     feedback_node):

    model = plus_node >> feedback_node >> offline_node
    feedback_node <<= offline_node

    X = np.ones((5, 5)) * 0.5
    Y = np.ones((5, 5))

    model.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([7.5]))

    X = np.ones((3, 5, 5)) * 0.5
    Y = np.ones((3, 5, 5))

    model.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([97.5]))

    model.fit(X, Y, reset=True)

    assert_array_equal(offline_node.b, np.array([22.5]))

    res = model.run(X[0], reset=True)

    exp = np.tile(np.array([26, 54.5, 85.5, 119, 155]), 5).reshape(5, 5).T

    assert_array_equal(exp, res)
