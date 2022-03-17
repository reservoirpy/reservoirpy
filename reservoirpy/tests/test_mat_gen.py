import numpy as np
import pytest
from numpy.random import default_rng
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises,
)
from scipy import linalg, sparse

from reservoirpy.mat_gen import (
    bernoulli,
    fast_spectral_initialization,
    generate_input_weights,
    generate_internal_weights,
    normal,
    ones,
    random_sparse,
    uniform,
    zeros,
)


@pytest.mark.parametrize(
    "shape,dist,connectivity,kwargs,expects",
    [
        ((50, 50), "uniform", 0.1, {}, "sparse"),
        ((50, 50), "uniform", 0.1, {"loc": 5.0, "scale": 1.0}, "sparse"),
        ((50, 50), "uniform", 1.0, {}, "dense"),
        ((50, 50), "custom_bernoulli", 0.1, {}, "sparse"),
        ((50, 50, 50), "custom_bernoulli", 0.1, {"p": 0.9}, "dense"),
        ((50, 50), "custom_bernoulli", 1.0, {}, "dense"),
        ((50, 50), "foo", 0.1, {}, "raise"),
        ((50, 50), "uniform", 5.0, {}, "raise"),
        ((50, 50), "uniform", 0.1, {"p": 0.9}, "raise"),
    ],
)
def test_random_sparse(shape, dist, connectivity, kwargs, expects):

    if expects == "sparse":
        init = random_sparse(dist=dist, connectivity=connectivity, seed=42, **kwargs)
        w0 = init(*shape)
        w1 = random_sparse(
            *shape, dist=dist, connectivity=connectivity, seed=42, **kwargs
        )

        assert_array_equal(w1.toarray(), w0.toarray())
        assert_allclose(
            np.count_nonzero(w0.toarray()) / w0.toarray().size, connectivity, atol=1e-2
        )

    if expects == "dense":
        init = random_sparse(dist=dist, connectivity=connectivity, seed=42, **kwargs)
        w0 = init(*shape)
        w1 = random_sparse(
            *shape, dist=dist, connectivity=connectivity, seed=42, **kwargs
        )

        assert_array_equal(w1, w0)
        assert_allclose(np.count_nonzero(w0) / w0.size, connectivity, atol=1e-2)

    if expects == "raise":
        with pytest.raises(Exception):
            init = random_sparse(
                dist=dist, connectivity=connectivity, seed=42, **kwargs
            )
            w0 = init(5, 5)
            w1 = random_sparse(
                5, 5, dist="uniform", connectivity=0.2, seed=42, **kwargs
            )


@pytest.mark.parametrize(
    "shape,sr,input_scaling,kwargs,expects",
    [
        ((50, 50), 2.0, None, {"connectivity": 0.1}, "sparse"),
        ((50, 50), None, -2.0, {"connectivity": 1.0}, "dense"),
        ((50, 50), 2.0, None, {"connectivity": 1.0}, "dense"),
        ((50, 50), None, -2.0, {"connectivity": 1.0}, "dense"),
        ((50, 50), None, np.ones((50,)) * 0.1, {"connectivity": 1.0}, "dense"),
        ((50, 50), None, np.ones((50,)) * 0.1, {"connectivity": 0.1}, "sparse"),
        ((50, 50), 2.0, None, {"connectivity": 0.0}, "sparse"),
        ((50, 50), 2.0, -2.0, {"connectivity": 0.1}, "raise"),
    ],
)
def test_random_sparse_scalings(shape, sr, input_scaling, kwargs, expects):

    if expects == "sparse":
        init = random_sparse(
            dist="uniform", sr=sr, input_scaling=input_scaling, seed=42, **kwargs
        )
        w0 = init(*shape)
        w1 = random_sparse(
            *shape,
            dist="uniform",
            sr=sr,
            input_scaling=input_scaling,
            seed=42,
            **kwargs,
        )

        assert_allclose(w1.toarray(), w0.toarray(), atol=1e-12)

    if expects == "dense":
        init = random_sparse(
            dist="uniform", sr=sr, input_scaling=input_scaling, seed=42, **kwargs
        )
        w0 = init(*shape)
        w1 = random_sparse(
            *shape,
            dist="uniform",
            sr=sr,
            input_scaling=input_scaling,
            seed=42,
            **kwargs,
        )
        assert_allclose(w1, w0, atol=1e-12)

    if expects == "raise":
        with pytest.raises(Exception):
            init = random_sparse(
                dist="uniform", sr=sr, input_scaling=input_scaling, seed=42, **kwargs
            )
            w0 = init(*shape)
            w1 = random_sparse(
                *shape,
                dist="uniform",
                sr=sr,
                input_scaling=input_scaling,
                seed=42,
                **kwargs,
            )


@pytest.mark.parametrize(
    "shape,dtype,sparsity_type,kwargs,expects",
    [
        ((50, 50), np.float64, "csr", {"dist": "norm", "connectivity": 0.1}, "sparse"),
        ((50, 50), np.float32, "csc", {"dist": "norm", "connectivity": 0.1}, "sparse"),
        ((50, 50), np.int64, "coo", {"dist": "norm", "connectivity": 0.1}, "sparse"),
        ((50, 50), float, "dense", {"dist": "norm", "connectivity": 0.1}, "dense"),
    ],
)
def test_random_sparse_types(shape, dtype, sparsity_type, kwargs, expects):

    all_sparse_types = {
        "csr": sparse.isspmatrix_csr,
        "coo": sparse.isspmatrix_coo,
        "csc": sparse.isspmatrix_csc,
    }

    if expects == "sparse":
        init = random_sparse(
            dtype=dtype, sparsity_type=sparsity_type, seed=42, **kwargs
        )
        w0 = init(*shape)
        w1 = random_sparse(
            *shape, dtype=dtype, sparsity_type=sparsity_type, seed=42, **kwargs
        )

        assert_allclose(w1.toarray(), w0.toarray(), atol=1e-12)
        assert w0.dtype == dtype
        assert all_sparse_types[sparsity_type](w0)

    if expects == "dense":
        init = random_sparse(
            dtype=dtype, sparsity_type=sparsity_type, seed=42, **kwargs
        )
        w0 = init(*shape)
        w1 = random_sparse(
            *shape, dtype=dtype, sparsity_type=sparsity_type, seed=42, **kwargs
        )

        assert_allclose(w1, w0, atol=1e-12)
        assert w0.dtype == dtype


@pytest.mark.parametrize(
    "initializer,shape,kwargs,expects",
    [
        (uniform, (50, 50), {"connectivity": 0.1}, "sparse"),
        (uniform, (50, 50, 50), {"connectivity": 0.1}, "dense"),
        (uniform, (50, 50), {"connectivity": 0.1, "sparsity_type": "dense"}, "dense"),
        (uniform, (50, 50), {"connectivity": 0.1, "high": 5.0, "low": 2.0}, "sparse"),
        (uniform, (50, 50), {"connectivity": 0.1, "high": 5.0, "low": "a"}, "raise"),
        (normal, (50, 50), {"connectivity": 0.1}, "sparse"),
        (normal, (50, 50, 50), {"connectivity": 0.1}, "dense"),
        (normal, (50, 50), {"connectivity": 0.1, "sparsity_type": "dense"}, "dense"),
        (normal, (50, 50), {"connectivity": 0.1, "loc": 5.0, "scale": 2.0}, "sparse"),
        (normal, (50, 50), {"connectivity": 0.1, "loc": 5.0, "scale": "a"}, "raise"),
        (bernoulli, (50, 50), {"connectivity": 0.1}, "sparse"),
        (bernoulli, (50, 50, 50), {"connectivity": 0.1}, "dense"),
        (bernoulli, (50, 50), {"connectivity": 0.1, "sparsity_type": "dense"}, "dense"),
        (bernoulli, (50, 50), {"connectivity": 0.1, "p": 0.9}, "sparse"),
        (bernoulli, (50, 50), {"connectivity": 0.1, "p": 5.0}, "raise"),
    ],
)
def test_dists(initializer, shape, kwargs, expects):
    if expects == "sparse":
        init = initializer(seed=42, **kwargs)
        w0 = init(*shape)
        w1 = initializer(*shape, seed=42, **kwargs)

        assert_allclose(w1.toarray(), w0.toarray(), atol=1e-12)

    if expects == "dense":
        init = initializer(seed=42, **kwargs)
        w0 = init(*shape)
        w1 = initializer(*shape, seed=42, **kwargs)
        assert_allclose(w1, w0, atol=1e-12)

    if expects == "raise":
        with pytest.raises(Exception):
            init = initializer(seed=42, **kwargs)
            w0 = init(*shape)
            w1 = initializer(*shape, seed=42, **kwargs)


def test_ones():
    w = ones(50, 50)
    assert_allclose(w, 1.0)

    w = ones(50, 50, dtype=np.float32)
    assert_allclose(w, 1.0)
    assert w.dtype == np.float32


def test_zeros():
    w = zeros(50, 50)
    assert_allclose(w, 0.0)

    w = zeros(50, 50, dtype=np.float32)
    assert_allclose(w, 0.0)
    assert w.dtype == np.float32

    with pytest.raises(ValueError):
        w = zeros(50, 50, sr=2.0)


@pytest.mark.parametrize(
    "N,dim_input,input_bias,expected",
    [
        (100, 20, False, (100, 20)),
        (100, 20, True, (100, 21)),
        (20, 100, True, (20, 101)),
    ],
)
def test_generate_inputs_shape(N, dim_input, input_bias, expected):

    with pytest.warns(DeprecationWarning):
        Win = generate_input_weights(N, dim_input, input_bias=input_bias)

    assert Win.shape == expected


@pytest.mark.parametrize(
    "N,dim_input,input_bias",
    [
        (-1, 10, True),
        (100, -5, False),
    ],
)
def test_generate_inputs_shape_exception(N, dim_input, input_bias):
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError):
            generate_input_weights(N, dim_input, input_bias=input_bias)


@pytest.mark.parametrize("proba,iss", [(0.1, 0.1), (1.0, 0.5), (0.5, 2.0)])
def test_generate_inputs_features(proba, iss):

    with pytest.warns(DeprecationWarning):
        Win = generate_input_weights(100, 20, input_scaling=iss, proba=proba, seed=1234)

        with pytest.warns(DeprecationWarning):
            Win_noiss = generate_input_weights(
                100, 20, input_scaling=1.0, proba=proba, seed=1234
            )

            if sparse.issparse(Win):
                result_proba = np.count_nonzero(Win.toarray()) / Win.toarray().size
            else:
                result_proba = np.count_nonzero(Win) / Win.size

            assert_allclose(result_proba, proba, rtol=1e-2)

            if sparse.issparse(Win):
                assert_allclose(Win.toarray() / iss, Win_noiss.toarray(), rtol=1e-4)
            else:
                assert_allclose(Win / iss, Win_noiss, rtol=1e-4)


@pytest.mark.parametrize("proba,iss", [(-1, "foo"), (5, 1.0)])
def test_generate_inputs_features_exception(proba, iss):
    with pytest.warns(DeprecationWarning):
        with pytest.raises(Exception):
            generate_input_weights(100, 20, input_scaling=iss, proba=proba)


@pytest.mark.parametrize(
    "N,expected", [(100, (100, 100)), (-1, Exception), ("foo", Exception)]
)
def test_generate_internal_shape(N, expected):
    if expected is Exception:
        with pytest.raises(expected):
            with pytest.warns(DeprecationWarning):
                generate_internal_weights(N)
    else:
        with pytest.warns(DeprecationWarning):
            W = generate_internal_weights(N)
        assert W.shape == expected


@pytest.mark.parametrize(
    "sr,proba",
    [
        (0.5, 0.1),
        (2.0, 1.0),
    ],
)
def test_generate_internal_features(sr, proba):

    with pytest.warns(DeprecationWarning):
        W = generate_internal_weights(
            100, sr=sr, proba=proba, seed=1234, sparsity_type="dense"
        )

        assert_allclose(max(abs(linalg.eig(W)[0])), sr)
        assert_allclose(np.sum(W != 0.0) / W.size, proba)


@pytest.mark.parametrize("sr,proba", [(0.5, 0.1), (2.0, 1.0)])
def test_generate_internal_sparse(sr, proba):

    with pytest.warns(DeprecationWarning):
        W = generate_internal_weights(
            100, sr=sr, proba=proba, sparsity_type="csr", seed=42
        )

        rho = max(
            abs(
                sparse.linalg.eigs(
                    W,
                    k=1,
                    which="LM",
                    maxiter=20 * W.shape[0],
                    return_eigenvectors=False,
                )
            )
        )
        assert_allclose(rho, sr)

        if sparse.issparse(W):
            assert_allclose(np.sum(W.toarray() != 0.0) / W.toarray().size, proba)
        else:
            assert_allclose(np.sum(W != 0.0) / W.size, proba)


@pytest.mark.parametrize("sr,proba", [(1, -0.5), (1, 12), ("foo", 0.1)])
def test_generate_internal_features_exception(sr, proba):
    with pytest.warns(DeprecationWarning):
        with pytest.raises(Exception):
            generate_internal_weights(100, sr=sr, proba=proba)


@pytest.mark.parametrize(
    "N,expected", [(100, (100, 100)), (-1, Exception), ("foo", Exception)]
)
def test_fast_spectral_shape(N, expected):
    if expected is Exception:
        with pytest.raises(expected):
            fast_spectral_initialization(N)
    else:
        W = fast_spectral_initialization(N)
        assert W.shape == expected


@pytest.mark.parametrize("sr,proba", [(0.5, 0.1), (10.0, 0.5), (1.0, 1.0), (1.0, 0.0)])
def test_fast_spectral_features(sr, proba):
    W = fast_spectral_initialization(1000, sr=sr, connectivity=proba, seed=1234)

    if sparse.issparse(W):
        rho = max(
            abs(
                sparse.linalg.eigs(
                    W,
                    k=1,
                    which="LM",
                    maxiter=20 * W.shape[0],
                    return_eigenvectors=False,
                )
            )
        )
    else:
        rho = max(abs(linalg.eig(W)[0]))

    if proba == 0.0:
        assert_allclose(rho, 0.0)
    else:
        assert_allclose(rho, sr, rtol=1e-1)

    if 1.0 - proba < 1e-5:
        assert not sparse.issparse(W)
    if sparse.issparse(W):
        assert_allclose(
            np.count_nonzero(W.toarray()) / W.toarray().size, proba, rtol=1e-1
        )
    else:
        assert_allclose(np.count_nonzero(W) / W.size, proba, rtol=1e-1)


@pytest.mark.parametrize("sr,proba", [(1, -0.5), (1, 12), ("foo", 0.1)])
def test_fast_spectral_features_exception(sr, proba):
    with pytest.raises(Exception):
        fast_spectral_initialization(100, sr=sr, connectivity=proba)

    with pytest.raises(ValueError):
        fast_spectral_initialization(100, input_scaling=10.0, connectivity=proba)


def test_reproducibility_W():

    seed0 = default_rng(78946312)
    with pytest.warns(DeprecationWarning):
        W0 = generate_internal_weights(
            N=100, sr=1.2, proba=0.4, dist="uniform", loc=-1, scale=2, seed=seed0
        ).toarray()

    seed1 = default_rng(78946312)
    with pytest.warns(DeprecationWarning):
        W1 = generate_internal_weights(
            N=100, sr=1.2, proba=0.4, dist="uniform", loc=-1, scale=2, seed=seed1
        ).toarray()

    seed2 = default_rng(6135435)
    with pytest.warns(DeprecationWarning):
        W2 = generate_internal_weights(
            N=100, sr=1.2, proba=0.4, dist="uniform", loc=-1, scale=2, seed=seed2
        ).toarray()

    assert_array_almost_equal(W0, W1)
    assert_raises(AssertionError, assert_array_almost_equal, W0, W2)


def test_reproducibility_Win():

    seed0 = default_rng(78946312)
    with pytest.warns(DeprecationWarning):
        W0 = generate_input_weights(100, 50, input_scaling=1.2, proba=0.4, seed=seed0)

    seed1 = default_rng(78946312)
    with pytest.warns(DeprecationWarning):
        W1 = generate_input_weights(100, 50, input_scaling=1.2, proba=0.4, seed=seed1)

    seed2 = default_rng(6135435)
    with pytest.warns(DeprecationWarning):
        W2 = generate_input_weights(100, 50, input_scaling=1.2, proba=0.4, seed=seed2)

    assert_allclose(W0.toarray(), W1.toarray())

    with pytest.raises(AssertionError):
        assert_allclose(W0.toarray(), W2.toarray())


def test_reproducibility_fsi():

    seed0 = default_rng(78946312)
    W0 = fast_spectral_initialization(
        100, sr=1.2, connectivity=0.4, seed=seed0
    ).toarray()

    seed1 = default_rng(78946312)
    W1 = fast_spectral_initialization(
        100, sr=1.2, connectivity=0.4, seed=seed1
    ).toarray()

    seed2 = default_rng(6135435)
    W2 = fast_spectral_initialization(
        100, sr=1.2, connectivity=0.4, seed=seed2
    ).toarray()

    assert_array_almost_equal(W0, W1)
    assert_raises(AssertionError, assert_array_almost_equal, W0, W2)
