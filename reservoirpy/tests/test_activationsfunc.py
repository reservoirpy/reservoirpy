import pytest
import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from reservoirpy.activationsfunc import softmax
from reservoirpy.activationsfunc import softplus
from reservoirpy.activationsfunc import identity
from reservoirpy.activationsfunc import tanh
from reservoirpy.activationsfunc import sigmoid
from reservoirpy.activationsfunc import relu


@pytest.mark.parametrize("value, expected", [
    ([1, 2, 3], np.exp([1, 2, 3]) / np.sum(np.exp([1, 2, 3]))),
    (1, np.exp(1) / np.sum(np.exp(1))),
    ([0, 0], [0.5, 0.5])
])
def test_softmax(value, expected):
    result = softmax(value)

    assert_almost_equal(np.sum(result), 1.0)
    assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("value, expected", [
    (0, np.log(1 + np.exp(0))),
    ([0, 1, 2], np.log(1 + np.exp([0, 1, 2]))),
    ([-2, -1], np.log(1 + np.exp([-2, -1])))
])
def test_softplus(value, expected):
    result = softplus(value)

    assert np.any(result > 0.0)
    assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("value", [
    ([1, 2, 3]),
    ([1]),
    (0),
    ([0.213565165, 0.1, 1.064598495615132]),
])
def test_identity(value):
    result = identity(value)
    val = np.asanyarray(value)

    assert np.any(result == val)


@pytest.mark.parametrize("value, expected", [
    ([1, 2, 3], np.tanh([1, 2, 3])),
    (0, np.tanh(0))
])
def test_tanh(value, expected):
    result = tanh(value)

    assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("value, expected", [
    ([1, 2, 3], 1 / (1 + np.exp(-np.array([1, 2, 3])))),
    (0, 1 / (1 + np.exp(0))),
    ([-1000, -2], [0.0, 1.35e-1])
])
def test_sigmoid(value, expected):
    result = sigmoid(value)
    assert_array_almost_equal(result, expected, decimal=1)


@pytest.mark.parametrize("value, expected", [
    ([1, 2, 3], np.array([1, 2, 3])),
    ([-1, -10, 5], np.array([0, 0, 5])),
    (0, np.array(0)),
    ([[1, 2, 3], [-1, 2, 3]], np.array([[1, 2, 3], [0, 2, 3]]))
])
def test_relu(value, expected):
    result = relu(value)
    assert_array_almost_equal(result, expected)
