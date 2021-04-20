import pytest
import numpy as np

from scipy import linalg

from .._readout import Readout


@pytest.fixture
def ridge_reg():

    def ridge_model_solving(X, Y):
        ridgeid = (1e-5 * np.eye(X.shape[1]))

        return np.dot(np.dot(Y.T, X), linalg.inv(np.dot(X.T, X) + ridgeid))

    return ridge_model_solving


@pytest.fixture(scope="session")
def states():
    return np.ones((200, 10))


@pytest.fixture(scope="session")
def teachers():
    return np.ones((200, 2))


@pytest.fixture(scope="session")
def inputs():
    return np.ones((200, 3))


def test_readout_fit_predict(states, teachers, ridge_reg):

    readout = Readout(2, reg_model=ridge_reg)

    readout.fit(states, teachers)

    assert readout.Wout.shape == (2, 11)

    outs = readout(states)

    assert outs.shape == (200, 2)

    outs = readout(states[-1])

    assert outs.shape == (1, 2)


def test_readout_fit_with_inputs(states, inputs, teachers, ridge_reg):

    readout = Readout(2, reg_model=ridge_reg, use_inputs=True)

    readout.fit(states, teachers, inputs)

    assert readout.Wout.shape == (2, 11 + 3)

    outs = readout(states, inputs)

    assert outs.shape == (200, 2)

    outs = readout(states[-1], inputs[-1])

    assert outs.shape == (1, 2)
