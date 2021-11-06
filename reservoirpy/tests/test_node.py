# Author: Nathan Trouvain at 08/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import pytest
from copy import deepcopy
import numpy as np

from numpy.testing import assert_array_equal

#from ..node import Node, combine
from ..base import Node
from ..ops import merge
from ..nodes.io import Input


@pytest.fixture(scope="function")
def plus_node():

    def forward(node: Node, x: np.ndarray):
        return x + node.c + node.h + node.state()

    def initialize(node: Node, x=None, **kwargs):
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])
        node.set_param("c", 1)

    class PlusNode(Node):

        def __init__(self):
            super().__init__(params={"c": None}, hypers={"h": 1},
                             forward=forward, initializer=initialize)

    n = PlusNode()

    return n


@pytest.fixture(scope="function")
def minus_node():

    def forward(node: Node, x):
        return x - node.c - node.h - node.state()

    def initialize(node: Node, x=None, **kwargs):
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])
        node.set_param("c", 1)

    class MinusNode(Node):

        def __init__(self):
            super().__init__(params={"c": None}, hypers={"h": 1},
                             forward=forward, initializer=initialize)

    n = MinusNode()

    return n


@pytest.fixture(scope="function")
def feedback_node():

    def forward(node: Node, x):
        return node.feedback() + x + 1

    def initialize(node: Node, x=None, **kwargs):
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])

    def initialize_fb(node: Node):
        fb = node.feedback()
        node.set_feedback_dim(fb.shape[1])

    class FBNode(Node):

        def __init__(self):
            super().__init__(initializer=initialize,
                             fb_initializer=initialize_fb,
                             forward=forward)

    n = FBNode()

    return n


@pytest.fixture(scope="function")
def inverter_node():

    def forward(node: Node, x):
        return -x

    def initialize(node: Node, x=None, **kwarg):
        if x is not None:
            node.set_input_dim(x.shape[1])
            node.set_output_dim(x.shape[1])

    class Inverter(Node):

        def __init__(self):
            super(Inverter, self).__init__(initializer=initialize,
                                           forward=forward)

    n = Inverter()

    return n


@pytest.fixture(scope="function")
def offline_node():

    def forward(node: Node, x):
        return x + node.b

    def partial_backward(node: Node, X_batch, Y_batch=None):
        b = np.mean(np.abs(X_batch - Y_batch))
        node.set_buffer("b", node.get_buffer("b") + b)

    def backward(node: Node, X=None, Y=None):
        b = node.get_buffer("b")
        node.set_param("b", np.array(b).copy())
        initialize_buffers(node)

    def initialize(node: Node, x=None, y=None):
        if x is not None:
            node.set_input_dim(x.shape[1])
            node.set_output_dim(x.shape[1])

    def initialize_buffers(node: Node):
        node.create_buffer("b", (1,))

    class Offline(Node):

        def __init__(self):
            super(Offline, self).__init__(params={"b": 0},
                                          forward=forward,
                                          partial_backward=partial_backward,
                                          backward=backward,
                                          buffers_initializer=initialize_buffers,
                                          initializer=initialize)

    return Offline()


@pytest.fixture(scope="function")
def offline_node2():

    def forward(node: Node, x):
        return x + node.b

    def partial_backward(node: Node, X_batch, Y_batch=None):
        b = np.mean(np.abs(X_batch - Y_batch))
        node.set_buffer("b", node.get_buffer("b") + b)

    def backward(node: Node, X=None, Y=None):
        b = node.get_buffer("b")
        node.set_param("b", np.array(b).copy())
        initialize_buffers(node)

    def initialize(node: Node, x=None, y=None):
        if x is not None:
            node.set_input_dim(x.shape[1])
            node.set_output_dim(x.shape[1])

            initialize_buffers(node)

    def initialize_buffers(node: Node):
        node.create_buffer("b", (1,))

    class Offline2(Node):

        def __init__(self):
            super(Offline2, self).__init__(params={"b": 0},
                                          forward=forward,
                                          partial_backward=partial_backward,
                                          backward=backward,
                                          initializer=initialize)

    return Offline2()


@pytest.fixture(scope="function")
def sum_node():

    def forward(node: Node, x):
        if isinstance(x, list):
            x = np.concatenate(x, axis=0)
        return np.sum(x, axis=0)

    def initialize(node: Node, x=None, **kwargs):
        if x is not None:
            if isinstance(x, list):
                x = np.concatenate(x, axis=0)
            node.set_input_dim(x.shape[1])
            node.set_output_dim(x.shape[1])

    class Sum(Node):

        def __init__(self):
            super(Sum, self).__init__(forward=forward,
                                      initializer=initialize)

    return Sum()


def test_node_creation(plus_node):
    assert plus_node.name == 'PlusNode-0'
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


def test_node_link(plus_node, minus_node):

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


def test_offline_fit(offline_node):

    X = np.ones((10, 5)) * 0.5
    Y = np.ones((10, 5))

    assert offline_node.b == 0

    offline_node.partial_fit(X, Y)

    assert_array_equal(offline_node.get_buffer("b"), np.array([0.5]))

    offline_node.fit()

    assert_array_equal(offline_node.b, np.array([0.5]))

    X = np.ones((10, 5)) * 2.
    Y = np.ones((10, 5))

    offline_node.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([1.]))

    X = [np.ones((10, 5)) * 2.] * 3
    Y = [np.ones((10, 5))] * 3

    offline_node.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([3.]))

    offline_node.partial_fit(X, Y)

    assert_array_equal(offline_node.get_buffer("b"), np.array([3.]))


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

    assert_array_equal(offline_node.b, np.array([19.5]))

    model.fit(X, Y, stateful=True, reset=True)

    assert_array_equal(offline_node.b, np.array([57.0]))

    res = model.run(X[0])

    exp = np.tile(np.array([59.5, 62, 64.5, 67, 69.5]), 5).reshape(5, 5).T

    assert_array_equal(exp, res)


def test_offline_fit_complicated_model(offline_node, offline_node2,
                                       plus_node, minus_node, sum_node):

    input = Input()

    branch1 = input >> plus_node >> offline_node >> minus_node >> sum_node
    branch2 = input >> sum_node >> offline_node2

    model = merge(branch1, branch2)

    X = np.ones((5, 5)) * 0.5
    Y = np.ones((5, 5))

    model.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([6.5]))
    assert_array_equal(offline_node2.b, np.array([6.7]))

    X = np.ones((3, 5, 5)) * 0.5
    Y = np.ones((3, 5, 5))

    model.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([19.5]))
    assert_array_equal(offline_node2.b, np.array([43.5]))

    model.fit(X, Y, stateful=True, reset=True)

    assert_array_equal(offline_node.b, np.array([57.0]))
    assert_array_equal(offline_node2.b, np.array([118.5]))
