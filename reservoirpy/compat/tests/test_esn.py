from math import sin, cos
from tempfile import TemporaryDirectory

import pytest
import numpy as np

from scipy import sparse

from .._esn import ESN
from ..utils.save import load


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

    return W, Win


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
    return W, Win, Wfb


@pytest.fixture(scope="session")
def dummy_data():
    Xn0 = np.array([[sin(x), cos(x)] for x in np.linspace(0, 4*np.pi, 500)])
    Xn1 = np.array([[sin(x), cos(x)]
                    for x in np.linspace(np.pi/4, 4*np.pi+np.pi/4, 500)])
    return Xn0, Xn1


@pytest.fixture(scope="session")
def dummy_clf_data():
    Xn0 = np.array([[sin(x), cos(x)]
                    for x in np.linspace(0, 4*np.pi, 250)])
    Xn1 = np.array([[sin(10*x), cos(10*x)]
                   for x in np.linspace(np.pi/4, 4*np.pi+np.pi/4, 250)])
    X = np.vstack([Xn0, Xn1])
    y = np.r_[np.zeros(250), np.ones(250)].reshape(-1, 1)

    return X, y


def test_esn(matrices, dummy_data):
    W, Win = matrices
    esn = ESN(lr=0.1, W=W, Win=Win, input_bias=False)
    X, y = dummy_data
    states = esn.train([X], [y], return_states=True)

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X], return_states=True)

    assert states[0].shape[0] == X.shape[0]
    assert outputs[0].shape[1] == y.shape[1]
    assert outputs[0].shape[0] == states[0].shape[0]

    esn.train([X, X, X], [y, y, y])

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X, X], return_states=True)

    assert len(states) == 2
    assert len(outputs) == 2


def test_esn_compute_all_states(matrices, dummy_data):
    W, Win = matrices
    esn = ESN(lr=0.1, W=W, Win=Win, input_bias=False)
    X, y = dummy_data
    states = esn.compute_all_states([X])

    assert states[0].shape[0] == X.shape[0]
    assert states[0].shape[1] == esn.N

    states = esn.compute_all_states([X, X, X])

    assert len(states) == 3


def test_esn_generate(matrices, dummy_data):
    W, Win = matrices
    esn = ESN(lr=0.1, W=W, Win=Win, input_bias=False)

    X, y = dummy_data
    states = esn.train([X], [y])
    out, states, warm_out, warm_states = esn.generate(500,
                                                      warming_inputs=X)

    assert states.shape[0] == 500
    assert out.shape[0] == 500
    assert warm_states.shape[0] == X.shape[0]
    assert warm_out.shape[0] == X.shape[0]


def test_esn_ridge(matrices, dummy_data):
    W, Win = matrices
    esn = ESN(lr=0.1, W=W, Win=Win, input_bias=False, ridge=1e-4)

    X, y = dummy_data
    states = esn.train([X], [y])

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X], return_states=True)

    assert states[0].shape[0] == X.shape[0]
    assert outputs[0].shape[1] == 2

    states = esn.train([X, X, X], [y, y, y])

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X, X], return_states=True)

    assert len(states) == 2
    assert len(outputs) == 2



def test_esn_fb(matrices_fb, dummy_data):
    W, Win, Wfb = matrices_fb

    esn = ESN(lr=0.1, W=W, Win=Win, Wfb=Wfb,
              input_bias=False, fbfunc=np.tanh)
    X, y = dummy_data
    states = esn.train([X], [y])

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X], return_states=True)

    assert states[0].shape[0] == X.shape[0]
    assert outputs[0].shape[1] == y.shape[1]

    states = esn.train([X, X, X], [y, y, y])

    assert esn.Wout.shape == (2, 5)

    outputs, states = esn.run([X, X], return_states=True)

    assert len(states) == 2
    assert len(outputs) == 2


def test_esn_compute_all_states_fb(matrices_fb, dummy_data):
    W, Win, Wfb = matrices_fb
    esn = ESN(lr=0.1, W=W, Win=Win, Wfb=Wfb, input_bias=False,
              fbfunc=lambda x: x)
    X, y = dummy_data
    states = esn.compute_all_states([X], [y])

    assert states[0].shape[0] == X.shape[0]
    assert states[0].shape[1] == W.shape[0]

    states = esn.compute_all_states([X, X, X], [y, y, y])

    assert len(states) == 3


def test_esn_compute_all_states_fb_with_readout(matrices_fb, dummy_data):
    W, Win, Wfb = matrices_fb
    esn = ESN(lr=0.1, W=W, Win=Win, Wfb=Wfb, input_bias=False,
              fbfunc=lambda x: x)
    X, y = dummy_data
    esn.train([X], [y])
    states = esn.compute_all_states([X])

    assert states[0].shape[0] == X.shape[0]
    assert states[0].shape[1] == W.shape[0]

    states = esn.compute_all_states([X, X, X])

    assert len(states) == 3


def test_esn_generate_fb(matrices_fb, dummy_data):
    W, Win, Wfb = matrices_fb
    esn = ESN(lr=0.1, W=W, Win=Win, Wfb=Wfb, input_bias=False,
              fbfunc=lambda x: x)

    X, y = dummy_data
    states = esn.train([X], [y])
    out, states, warm_out, warm_states = esn.generate(500, warming_inputs=X,
                                                      init_fb=y[0])

    assert states.shape[0] == 500
    assert out.shape[0] == 500
    assert warm_states.shape[0] == X.shape[0]
    assert warm_out.shape[0] == X.shape[0]


def test_save(matrices_fb, dummy_data):
    W, Win, Wfb = matrices_fb
    esn = ESN(lr=0.1, W=W, Win=Win, Wfb=Wfb,
              input_bias=False, fbfunc=np.tanh,
              noise_rc=0.01)

    with TemporaryDirectory() as tempdir:

        esn.save(f"{tempdir}/esn")
        esn = load(f"{tempdir}/esn")

        X, y = dummy_data
        states = esn.train([X], [y])

        assert esn.Wout.shape == (2, 5)

        outputs, states = esn.run([X], return_states=True)

        assert states[0].shape[0] == X.shape[0]
        assert outputs[0].shape[1] == y.shape[1]

        esn.save(f"{tempdir}/esn_trained")
        esn = load(f"{tempdir}/esn_trained")

        outputs, states = esn.run([X, X], return_states=True)

        assert len(states) == 2
        assert len(outputs) == 2


def test_save_sparse(matrices_fb, dummy_data):
    W, Win, Wfb = matrices_fb
    W = sparse.csr_matrix(W)
    esn = ESN(lr=0.1, W=W, Win=Win, Wfb=Wfb,
              input_bias=False, fbfunc=np.tanh,
              noise_rc=0.01)

    with TemporaryDirectory() as tempdir:

        esn.save(f"{tempdir}/esn")
        esn = load(f"{tempdir}/esn")

        X, y = dummy_data
        states = esn.train([X], [y])

        assert esn.Wout.shape == (2, 5)

        outputs, states = esn.run([X], return_states=True)

        assert states[0].shape[0] == X.shape[0]
        assert outputs[0].shape[1] == y.shape[1]

        esn.save(f"{tempdir}/esn_trained")
        esn = load(f"{tempdir}/esn_trained")

        assert sparse.issparse(esn.W)

        outputs, states = esn.run([X, X], return_states=True)

        assert len(states) == 2
        assert len(outputs) == 2
