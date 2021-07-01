from math import sin, cos

import pytest
import numpy as np

from pytest import raises
from numpy.testing import assert_array_almost_equal
from numpy.random import default_rng

from reservoirpy.nodes.reservoir import Reservoir


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

    Wfb = np.array([[1, -1],
                    [1, -1],
                    [-1, 1],
                    [-1, 1]])

    return W, Win.T, Wfb.T


@pytest.fixture(scope="session")
def biased_matrices():
    Win = np.array([[1, -1, 1],
                    [-1, 1, -1],
                    [1, -1, 1],
                    [-1, -1, -1]])
    W = np.array([[0.0, 0.1, -0.1, 0.0],
                  [0.2, 0.0, 0.0, -0.2],
                  [0.0, 0.2, 0.3,  0.1],
                  [-0.1, 0.0, 0.0, 0.0]])
    Wfb = Win[:, :-1].copy()

    return W, Win, Wfb


@pytest.fixture(scope="session")
def dummy_data():
    Xn0 = np.array([[sin(x), cos(x)]
                    for x in np.linspace(0, 4*np.pi, 500)])
    Xn1 = np.array([[sin(x), cos(x)]
                    for x in np.linspace(np.pi/4, 4*np.pi+np.pi/4, 500)])
    return Xn0, Xn1


@pytest.fixture
def dummy_readout():

    def dummy_Wout(x, inputs=None):
        return np.array([[sin(x), cos(x)]
                         for x in np.linspace(0, 4*np.pi, x.shape[0])])
    return dummy_Wout


def test_reservoir_call(dummy_data, matrices):

    W, Win, _ = matrices
    res = Reservoir(lr=1.0, W=W, Win=Win)

    X, y = dummy_data
    states = res(X[0])

    assert res.is_initialized

    assert states.shape == (1, W.shape[0])
    assert res.output_dim == W.shape[0]


def test_reservoir_run(dummy_data, matrices):

    W, Win, _ = matrices
    res = Reservoir(lr=1.0, W=W, Win=Win)

    X, y = dummy_data
    states = res.run(X)

    assert states.shape == (X.shape[0], W.shape[0])
    assert res.output_dim == W.shape[0]


def test_reservoir_call_from_previous_state(dummy_data, matrices):

    W, Win, _ = matrices
    res = Reservoir(lr=1.0, W=W, Win=Win)

    X, y = dummy_data
    states = res(X[0])

    assert states.shape == (1, W.shape[0])
    assert res.output_dim == W.shape[0]

    states20 = res(X[0])
    states3 = res(X[0])

    states21 = res(X[0], from_state=states)

    assert_array_almost_equal(states20, states21)
    with pytest.raises(AssertionError):
        assert_array_almost_equal(states21, states3)


def test_reservoir_run_from_previous_state(dummy_data, matrices):

    W, Win, _ = matrices
    res = Reservoir(lr=1.0, W=W, Win=Win)

    X, y = dummy_data
    states = res.run(X)

    assert states.shape == (X.shape[0], W.shape[0])
    assert res.output_dim == W.shape[0]

    states20 = res.run(X)
    states3 = res.run(X, from_state=states[-5])

    states21 = res.run(X, from_state=states[-1])

    assert_array_almost_equal(states20, states21)
    with pytest.raises(AssertionError):
        assert_array_almost_equal(states21, states3)


def test_reservoir_call_no_stateful(dummy_data, matrices):

    W, Win, _ = matrices
    res = Reservoir(lr=0.1, W=W, Win=Win)

    X, y = dummy_data
    states = res(X[0], stateful=False)

    states2 = res(X[0])

    states3 = res(X[0], stateful=False)

    assert_array_almost_equal(states2, states)
    assert_array_almost_equal(states2, states3)


def test_reservoir_noisy(dummy_data, matrices):

    W, Win, _ = matrices
    res = Reservoir(lr=0.1, W=W, Win=Win, input_bias=False,
                    noise_in=0.1, noise_rc=0.5)

    rg = default_rng(123456789)

    X, y = dummy_data
    states = res(X, stateful=False, noise_generator=rg)

    rg = default_rng(123456789)
    states2 = res(X, noise_generator=rg)

    states3 = res(X, stateful=False)

    assert_array_almost_equal(states2, states)
    with raises(AssertionError):
        assert_array_almost_equal(states2, states3)


def test_reservoir_with_fb(dummy_data, dummy_readout, matrices):

    W, Win, Wfb = matrices
    res = Reservoir(lr=1.0, W=W, Win=Win, Wfb=Wfb, input_bias=False)

    assert res.shape == (Win.shape[1], W.shape[0], Wfb.shape[1])

    X, y = dummy_data
    states = res(X, dummy_readout)

    assert states.shape == (X.shape[0], W.shape[0])
    assert_array_almost_equal(states[-1, np.newaxis], res.state)


def test_reservoir_with_unused_fb(dummy_data, matrices):

    W, Win, Wfb = matrices
    res = Reservoir(lr=1.0, W=W, Win=Win, Wfb=Wfb, input_bias=False)

    assert res.shape == (Win.shape[1], W.shape[0], Wfb.shape[1])

    X, y = dummy_data
    states = res(X)

    assert states.shape == (X.shape[0], W.shape[0])
    assert_array_almost_equal(states[-1, np.newaxis], res.state)


def test_reservoir_isolated(matrices):

    W, Win, Wfb = matrices
    res = Reservoir(lr=1.0, W=W, Win=Win, Wfb=Wfb, input_bias=False)

    assert res.shape == (Win.shape[1], W.shape[0], Wfb.shape[1])

    states = res(n_steps=500)

    assert states.shape == (500, W.shape[0])
    assert_array_almost_equal(states[-1, np.newaxis], res.state)


def test_reservoir_fb_only(dummy_data, dummy_readout, matrices):

    W, Win, Wfb = matrices
    res = Reservoir(lr=1.0, W=W, Win=Win, Wfb=Wfb, input_bias=False)

    assert res.shape == (Win.shape[1], W.shape[0], Wfb.shape[1])

    X, y = dummy_data
    states = res(readout=dummy_readout, n_steps=500)

    assert states.shape == (500, W.shape[0])
    print(states[-1, np.newaxis])
    assert_array_almost_equal(states[-1, np.newaxis], res.state)


def test_reservoir_generative(dummy_data, dummy_readout, matrices):

    W, Win, _ = matrices
    res = Reservoir(lr=1.0, W=W, Win=Win, input_bias=False)

    assert res.shape == (Win.shape[1], W.shape[0], None)

    X, y = dummy_data
    states = res(readout=dummy_readout, n_steps=500, generative=True)

    assert states.shape == (500, W.shape[0])
    assert_array_almost_equal(states[-1, np.newaxis], res.state)


def test_reservoir_with_bias(dummy_data, dummy_readout, biased_matrices):

    W, Win, Wfb = biased_matrices
    res = Reservoir(lr=1.0, W=W, Win=Win, Wfb=Wfb,
                    input_bias=True)

    assert res.shape == (Win.shape[1], W.shape[0], Wfb.shape[1])

    X, y = dummy_data
    states = res(X, readout=dummy_readout)

    assert states.shape == (X.shape[0], W.shape[0])
    assert_array_almost_equal(states[-1, np.newaxis], res.state)
