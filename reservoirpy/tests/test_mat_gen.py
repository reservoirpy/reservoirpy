import pytest
import numpy as np

from numpy.random import default_rng
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_raises
from scipy import linalg
from scipy import sparse

from reservoirpy.mat_gen import fast_spectral_initialization
from reservoirpy.mat_gen import generate_input_weights
from reservoirpy.mat_gen import generate_internal_weights


@pytest.mark.parametrize("N,dim_input,input_bias,expected", [
    (100, 20, False, (100, 20)),
    (100, 20, True, (100, 21)),
    (20, 100, True, (20, 101)),
])
def test_generate_inputs_shape(N, dim_input, input_bias, expected):

    Win = generate_input_weights(N,
                                 dim_input,
                                 input_bias=input_bias)

    assert Win.shape == expected


@pytest.mark.parametrize("N,dim_input,input_bias", [
    (-1, 10, True),
    (100, -5, False),
])
def test_generate_inputs_shape_exception(N, dim_input, input_bias):
    with pytest.raises(ValueError):
        generate_input_weights(N, dim_input, input_bias=input_bias)


@pytest.mark.parametrize("proba,iss", [
    (0.1, 0.1),
    (1.0, 0.5),
    (0.5, 2.0)
])
def test_generate_inputs_features(proba, iss):

    Win = generate_input_weights(100, 20, input_scaling=iss,
                                 proba=proba, seed=1234)
    Win_noiss = generate_input_weights(100, 20, input_scaling=1.0,
                                       proba=proba, seed=1234)

    result_proba = np.sum(Win != 0.0) / Win.size

    assert_almost_equal(result_proba, proba, decimal=1)

    assert_array_almost_equal(Win / iss, Win_noiss, decimal=3)


@pytest.mark.parametrize("proba,iss", [
    (-1, "foo"),
    (5, 1.0)
])
def test_generate_inputs_features_exception(proba, iss):
    with pytest.raises(Exception):
        generate_input_weights(100, 20, input_scaling=iss,
                               proba=proba)


@pytest.mark.parametrize("N,expected", [
    (100, (100, 100)),
    (-1, Exception),
    ("foo", Exception)
])
def test_generate_internal_shape(N, expected):
    if expected is Exception:
        with pytest.raises(expected):
            generate_internal_weights(N)
    else:
        W = generate_internal_weights(N)
        assert W.shape == expected


@pytest.mark.parametrize("sr,proba", [
    (0.5, 0.1),
    (2.0, 1.0),
])
def test_generate_internal_features(sr, proba):

    W = generate_internal_weights(100, sr=sr,
                                  proba=proba, seed=1234,
                                  sparsity_type='dense')

    assert_almost_equal(max(abs(linalg.eig(W)[0])), sr, decimal=2)
    assert_almost_equal(np.sum(W != 0.0) / W.size, proba, decimal=1)


@pytest.mark.parametrize("sr,proba", [
    (0.5, 0.1),
    (2.0, 1.0)
])
def test_generate_internal_sparse(sr, proba):

    W = generate_internal_weights(100, sr=sr,
                                  proba=proba, sparsity_type="csr")

    rho = max(abs(sparse.linalg.eigs(W, k=1, which='LM',
                                     maxiter=20*W.shape[0],
                                     return_eigenvectors=False)))
    assert_almost_equal(rho, sr, decimal=2)

    if sparse.issparse(W):
        assert_almost_equal(np.sum(W.toarray() != 0.0) / W.toarray().size,
                            proba, decimal=1)
    else:
        assert_almost_equal(np.sum(W != 0.0) / W.size, proba, decimal=1)


@pytest.mark.parametrize("sr,proba", [
    (1, -0.5),
    (1, 12),
    ("foo", 0.1)
])
def test_generate_internal_features_exception(sr, proba):
    with pytest.raises(Exception):
        generate_internal_weights(100, sr=sr,
                                  proba=proba)


@pytest.mark.parametrize("N,expected", [
    (100, (100, 100)),
    (-1, Exception),
    ("foo", Exception)
])
def test_fast_spectral_shape(N, expected):
    if expected is Exception:
        with pytest.raises(expected):
            fast_spectral_initialization(N)
    else:
        W = fast_spectral_initialization(N)
        assert W.shape == expected


@pytest.mark.parametrize("sr,proba", [
    (0.5, 0.1),
    (10., 0.5),
    (1., 1.0),
    (1., 0.0)
])
def test_fast_spectral_features(sr, proba):
    W = fast_spectral_initialization(1000, sr=sr,
                                     proba=proba, seed=1234)

    if sparse.issparse(W):
        rho = max(abs(sparse.linalg.eigs(W, k=1, which='LM',
                                         maxiter=20*W.shape[0],
                                         return_eigenvectors=False)))
    else:
        rho = max(abs(linalg.eig(W)[0]))

    assert_almost_equal(rho, sr, decimal=0)

    if 1. - proba < 1e-5:
        assert not sparse.issparse(W)
    if sparse.issparse(W):
        assert_almost_equal(np.sum(W.toarray() != 0.0) / W.toarray().size,
                            proba, decimal=1)
    else:
        assert_almost_equal(np.sum(W != 0.0) / W.size, proba, decimal=1)


@pytest.mark.parametrize("sr,proba", [
    (1, -0.5),
    (1, 12),
    ("foo", 0.1)
])
def test_fast_spectral_features_exception(sr, proba):
    with pytest.raises(Exception):
        fast_spectral_initialization(100, sr=sr,
                                     proba=proba)


def test_reproducibility_W():

    seed0 = default_rng(78946312)
    W0 = generate_internal_weights(N=100,
                                   sr=1.2,
                                   proba=0.4,
                                   dist="uniform",
                                   low=-1,
                                   high=1,
                                   seed=seed0).toarray()

    seed1 = default_rng(78946312)
    W1 = generate_internal_weights(N=100,
                                   sr=1.2,
                                   proba=0.4,
                                   dist="uniform",
                                   low=-1,
                                   high=1,
                                   seed=seed1).toarray()

    seed2 = default_rng(6135435)
    W2 = generate_internal_weights(N=100,
                                   sr=1.2,
                                   proba=0.4,
                                   dist="uniform",
                                   low=-1,
                                   high=1,
                                   seed=seed2).toarray()

    assert_array_almost_equal(W0, W1)
    assert_raises(AssertionError, assert_array_almost_equal, W0, W2)


def test_reproducibility_Win():

    seed0 = default_rng(78946312)
    W0 = generate_input_weights(100, 50,
                                input_scaling=1.2,
                                proba=0.4,
                                seed=seed0)

    seed1 = default_rng(78946312)
    W1 = generate_input_weights(100, 50,
                                input_scaling=1.2,
                                proba=0.4,
                                seed=seed1)

    seed2 = default_rng(6135435)
    W2 = generate_input_weights(100, 50,
                                input_scaling=1.2,
                                proba=0.4,
                                seed=seed2)

    assert_array_almost_equal(W0, W1)
    assert_raises(AssertionError, assert_array_almost_equal, W0, W2)


def test_reproducibility_fsi():

    seed0 = default_rng(78946312)
    W0 = fast_spectral_initialization(N=100,
                                      sr=1.2,
                                      proba=0.4,
                                      seed=seed0).toarray()

    seed1 = default_rng(78946312)
    W1 = fast_spectral_initialization(N=100,
                                      sr=1.2,
                                      proba=0.4,
                                      seed=seed1).toarray()

    seed2 = default_rng(6135435)
    W2 = fast_spectral_initialization(N=100,
                                      sr=1.2,
                                      proba=0.4,
                                      seed=seed2).toarray()

    assert_array_almost_equal(W0, W1)
    assert_raises(AssertionError, assert_array_almost_equal, W0, W2)
