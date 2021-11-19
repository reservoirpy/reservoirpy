from math import sin, cos

import pytest
import numpy as np

from .._esn_online import ESNOnline
from reservoirpy.datasets import lorenz


@pytest.fixture(scope="session")
def matrices():
    Win = np.array([[1, -1],
                    [-1, 1],
                    [1, -1],
                    [-1, -1]])
    W = np.array([[0.0, 0.1, -0.1, 0.0],
                  [0.2, 0.0, 0.0, -0.2],
                  [0.0, 0.2, 0.3,  0.1],
                  [-0.1, 0.0, 0.0, 0.0]])
    Wout = np.zeros((2, 4 + 1))

    return W, Win, Wout

@pytest.fixture(scope="session")
def matrices_fb():
    Win = np.array([[1, -1],
                    [-1, 1],
                    [1, -1],
                    [-1, -1]])
    W = np.array([[0.0, 0.1, -0.1, 0.0],
                  [0.2, 0.0, 0.0, -0.2],
                  [0.0, 0.2, 0.3,  0.1],
                  [-0.1, 0.0, 0.0, 0.0]])
    Wfb = np.array([[1, -1],
                    [-1, -1],
                    [1, 1],
                    [-1, 1]])
    Wout = np.zeros((2, 4 + 1))
    return W, Win, Wout, Wfb


@pytest.fixture(scope="session")
def dummy_data():
    Xn0 = np.array([[sin(x), cos(x)] for x in np.linspace(0, 4*np.pi, 500)])
    Xn1 = np.array([[sin(x), cos(x)]
                    for x in np.linspace(np.pi/4, 4*np.pi+np.pi/4, 500)])
    return Xn0, Xn1


def test_esn(matrices, dummy_data):
    W, Win, Wout = matrices
    esn = ESNOnline(lr=0.1, W=W, Win=Win, dim_out=2, input_bias=False)
    X, y = dummy_data
    states = esn.train([X], [y])

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X])

    assert states[0].shape[0] == X.shape[0]
    assert outputs[0].shape[1] == y.shape[1]

    states = esn.train([X, X, X], [y, y, y])

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X, X])

    assert len(states) == 2
    assert len(outputs) == 2


def test_esn_fb(matrices_fb, dummy_data):
    W, Win, Wout, Wfb = matrices_fb
    esn = ESNOnline(lr=0.1, W=W, Win=Win, Wfb=Wfb,
                    dim_out=2, input_bias=False,
                    fbfunc=np.tanh)
    X, y = dummy_data
    states = esn.train([X], [y])

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X])

    assert states[0].shape[0] == X.shape[0]
    assert outputs[0].shape[1] == y.shape[1]

    states = esn.train([X, X, X], [y, y, y])

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X, X])

    assert len(states) == 2
    assert len(outputs) == 2
