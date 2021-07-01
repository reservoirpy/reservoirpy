import pytest
from pytest import raises
import numpy as np
from numpy.testing import assert_array_almost_equal

from reservoirpy.model import Node, Model


def easy_forward(node, x):
    a = node.a
    s = node.state()

    return a*x + s


def easy_init(node,
              x=None):
    if x is not None:
        node.set_input_dim(x.shape[1])


class EasyNode(Node):

    def __init__(self, a, dim):
        super(EasyNode, self).__init__(hypers={"a": a},
                                       initializer=easy_init,
                                       forward=easy_forward,
                                       output_dim=dim)


@pytest.fixture
def data():
    return np.linspace(1, 10, 10).reshape(-1, 1)


def test_node_init(data):
    node = EasyNode(5, 1)

    assert node.output_dim == 1
    assert node.a == 5
    assert node.is_initialized is False

    s = node.call(data[0])

    assert node.is_initialized
    assert node.input_dim == 1
    assert node.state() == 5. * 1. + 0.


def test_node_set_get_params():
    node = EasyNode(5, 1)
    assert node.a == 5
    node.set_param("a", 10)
    assert node.a == 10

    with raises(KeyError):
        node.set_param("b", 4)

    assert node.b is None


def test_node_check_input(data):
    node = EasyNode(5, 1)

    s = node.call(data[0])

    with raises(ValueError):
        node.call(data.reshape(1, -1))


def test_node_set_dim(data):
    node = EasyNode(5, 4)
    assert node.output_dim == 4
    node.set_output_dim(1)
    assert node.output_dim == 1
    node.set_input_dim(8)
    assert node.input_dim == 8
    node.call(data[0])
    assert node.input_dim == 1
    with raises(TypeError):
        node.set_output_dim(2)
    with raises(TypeError):
        node.set_input_dim(5)


def test_state(data):
    node = EasyNode(5, 1)
    s = node.call(data[0])
    assert node.state().shape == (1, 1)
    assert node.state() == 5 * 1 + 0
    node.reset()
    assert node.state() == 0.
    node.reset(to_state=s)
    assert node.state() == 5.
    with raises(TypeError):
        node.reset(to_state="lala")
    with raises(ValueError):
        node.reset(to_state=np.array([[1, 2, 3], [1, 2, 3]]))


def test_node_call(data):
    node = EasyNode(5, 1)
    s = node.call(data[0])
    assert node.state() == s
    assert s.shape == (10, 1)
    with raises(TypeError):
        node.call(np.array([["abcd"]]))


def test_node_call_unstateful(data):
    node = EasyNode(5, 1)
    s = node.call(data[0], stateful=False)
    assert node.state() == node.zero_state()
    s0 = node.call(data[0])
    s1 = node.call(data[1])
    s01 = node.call(data[0], stateful=False)

    assert_array_almost_equal(s0, s01)

    with raises(AssertionError):
        assert_array_almost_equal(s0, s1)

    node.reset()

    s11 = node.call(data[1], from_state=s0)

    assert_array_almost_equal(s1, s11)

    s12 = node.call(data[1], stateful=None, from_state=s0)

    assert_array_almost_equal(s11, s12)
    assert node.state() == s11


def test_node_run(data):
    node = EasyNode(5, 1)
    s = node.run(data)
    assert node.state() == s[-1]
    assert s.shape == (10, 1)
    with raises(TypeError):
        node.run(np.array([["abcd"]]))
