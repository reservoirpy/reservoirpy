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
    cluster,
    fast_spectral_initialization,
    line,
    normal,
    ones,
    orthogonal,
    random_sparse,
    ring,
    small_world,
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
        ((50, 5), "uniform", 0.1, {"degree": 23, "direction": "out"}, "sparse"),
        ((50, 5), "uniform", 0.1, {"degree": 3, "direction": "in"}, "sparse"),
        ((50, 5), "uniform", 0.1, {"degree": 6, "direction": "in"}, "raise"),
        ((50, 5), "uniform", 0.1, {"degree": -1000, "direction": "out"}, "raise"),
    ],
)
def test_random_sparse(shape, dist, connectivity, kwargs, expects):
    if expects == "sparse":
        init = random_sparse(dist=dist, connectivity=connectivity, seed=42, **kwargs)
        w0 = init(*shape)
        w1 = random_sparse(
            *shape, dist=dist, connectivity=connectivity, seed=42, **kwargs
        )

        w0 = w0.toarray()
        w1 = w1.toarray()

    if expects == "dense":
        init = random_sparse(dist=dist, connectivity=connectivity, seed=42, **kwargs)
        w0 = init(*shape)
        w1 = random_sparse(
            *shape, dist=dist, connectivity=connectivity, seed=42, **kwargs
        )

    if expects == "raise":
        with pytest.raises(Exception):
            init = random_sparse(
                dist=dist, connectivity=connectivity, seed=42, **kwargs
            )
            w0 = init(*shape)
        with pytest.raises(Exception):
            w1 = random_sparse(
                *shape, dist=dist, connectivity=connectivity, seed=42, **kwargs
            )
    else:
        assert_array_equal(w1, w0)
        if kwargs.get("degree") is None:
            assert_allclose(np.count_nonzero(w0) / w0.size, connectivity, atol=1e-2)
        else:
            dim_length = {"in": shape[0], "out": shape[1]}
            assert (
                np.count_nonzero(w0)
                == kwargs["degree"] * dim_length[kwargs["direction"]]
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
        ((50, 50), 2.0, -2.0, {"connectivity": 0.1}, "raise"),
        ((50, 50), None, 1e-12, {"connectivity": 0.1}, "sparse"),
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
        assert sparse.issparse(w0) and w0.format == sparsity_type

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
    "N,expected", [(100, (100, 100)), (-1, Exception), ("foo", Exception)]
)
def test_fast_spectral_shape(N, expected):
    if expected is Exception:
        with pytest.raises(expected):
            fast_spectral_initialization(N)
    else:
        W = fast_spectral_initialization(N)
        assert W.shape == expected


@pytest.mark.parametrize("sr,proba", [(0.5, 0.1), (10.0, 0.5), (1.0, 1.0)])
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


def test_sanity_checks():
    with pytest.raises(ValueError):
        _ = uniform(20, 20, degree=10, direction="all")
    with pytest.raises(ValueError):
        _ = uniform(30, degree=5, direction="in")
    with pytest.raises(ValueError):
        _ = uniform(30, 100, 10, degree=5, direction="in")
    with pytest.raises(ValueError):
        _ = bernoulli(30, 100, p=1.1)
    with pytest.raises(ValueError):
        _ = uniform(30, 100, low=1, high=0)


def test_ring_matrix():
    _ = ring(10, 10, weights=np.arange(1.0, 11.0), sr=1.0)
    W = ring(10, 10, input_scaling=2.0)

    assert W[1, 0] == 2.0 and W[0, -1] == 2.0

    # 1 on the 1st neuron, 0 elsewhere
    x0 = np.zeros((10, 1))
    x0[0, 0] = 1.0
    x = x0
    # loop all over the ring
    for i in range(10):
        x = W @ x

    assert np.all(x == 2**10 * x0)

    W_dense = ring(10, 10, input_scaling=2.0, sparsity_type="dense")
    assert_array_equal(W_dense, W.toarray())

    with pytest.raises(ValueError):
        _ = ring(10, 2, seed=1)


def test_line_matrix():
    _ = line(10, 10, weights=np.arange(1.0, 10.0), sr=1.0)
    W = line(10, 10, input_scaling=2.0)

    assert W[1, 0] == 2.0 and W[0, -1] == 0.0

    # 1 on the 1st neuron, 0 elsewhere
    x0 = np.zeros((10, 1))
    x0[0, 0] = 1.0
    x = x0
    # loop all over the line
    for i in range(10):
        x = W @ x

    assert np.all(x == 0.0)

    W_dense = line(10, 10, input_scaling=2.0, sparsity_type="dense")
    assert_array_equal(W_dense, W.toarray())

    with pytest.raises(ValueError):
        _ = line(10, 2, seed=1)


def test_orthogonal_matrix():
    W1 = orthogonal(10, 10, seed=1)
    W2 = orthogonal(10, 10, seed=1)

    assert np.all(np.isclose(W1, W2))

    assert np.all(np.isclose(W1 @ W1.T, np.eye(10)))

    with pytest.raises(ValueError):
        _ = orthogonal(10, 2, seed=1)

    with pytest.raises(ValueError):
        _ = orthogonal(10, 10, 10, seed=1)


def test_cluster_matrix():

    shape = 1000
    c = 10
    p_in = 0.1
    p_out = 0.01

    W1 = cluster(
        shape, shape, cluster=c, seed=1, p_in=p_in, p_out=p_out, distribution="normal"
    )
    W1 = W1.toarray()

    W2 = cluster(
        shape, shape, cluster=c, seed=1, p_in=p_in, p_out=p_out, distribution="normal"
    )
    W2 = W2.toarray()

    n_c = int(shape / c)

    # Check shape
    assert W1.shape == (shape, shape)

    # Assert that 2 matrices with same seed are equal
    assert_array_equal(W1, W2)

    # Check for cluster concentration
    def diagonal_concentration(band, w):
        n = w.shape[0]
        total = np.sum(np.abs(w))
        band_mask = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) <= band
        diag_sum = np.sum(np.abs(w[band_mask]))
        diag_concentration = diag_sum / total
        # diagonal concentration ratio
        return diag_concentration

    W1_diag = diagonal_concentration(n_c, W1)
    np.random.seed(42)
    np.random.shuffle(W2)
    W2_diag = diagonal_concentration(n_c, W2)

    assert W1_diag > W2_diag

    # Connectivity test

    # p_in & p_out are 0 --> matrix is full of 0
    w_min = cluster(shape, shape, cluster=c, seed=1, p_in=0, p_out=0)
    w_min = w_min.toarray()
    assert np.all(w_min == 0)

    # p_in & p_out are 1 --> no 0s in the matrix
    w_max = cluster(shape, shape, cluster=c, seed=1, p_in=1, p_out=1)
    assert np.all(w_max != 0)

    # Check cluster density
    w_cluster_check = cluster(shape, shape, cluster=c, seed=1, p_in=p_in, p_out=p_out)
    w_cluster_check = w_cluster_check.toarray()

    non_zeros_in_clusters = 0
    for i in range(0, c):
        current_cluster = w_cluster_check[
            i * n_c : i * n_c + n_c, i * n_c : i * n_c + n_c
        ]
        non_zeros_in_clusters += np.sum(current_cluster != 0)
    assert non_zeros_in_clusters == (p_in * n_c * n_c * c)

    # Check for incorrect distribution
    with pytest.raises(ValueError):
        _ = cluster(
            shape,
            shape,
            cluster=c,
            seed=1,
            p_in=1,
            p_out=1,
            distribution="not_a_distribution",
        )

    # Check for invalid cluster and shape size
    with pytest.raises(ValueError):
        _ = cluster(
            shape, shape, cluster=3, seed=1, p_in=1, p_out=1, distribution="normal"
        )


def test_small_world_matrix():
    W1 = small_world(10, 10, seed=1)
    W2 = small_world(10, 10, seed=1)

    assert np.all(np.isclose(W1.toarray(), W2.toarray()))

    W1_big = small_world(1000, 1000, seed=1)
    W2_big = small_world(1000, 1000, seed=1)
    assert np.all(np.isclose(W1_big.toarray(), W2_big.toarray()))
    assert W1_big.shape == (1000, 1000)

    nb_close_neighbours = 2
    W3 = small_world(
        10,
        10,
        seed=1,
        sparsity_type="dense",
        nb_close_neighbours=nb_close_neighbours,
        proba_rewire=0.0,
    )

    assert np.all(np.diag(W3) == 0)

    assert np.all(np.sum(W3 != 0, axis=0) == nb_close_neighbours)
    assert np.all(np.sum(W3 != 0, axis=1) == nb_close_neighbours)

    W4 = small_world(10, 10, seed=1, nb_close_neighbours=0)

    assert np.all(np.sum(W4, axis=0) == 0)
    assert np.all(np.sum(W4, axis=1) == 0)

    with pytest.raises(ValueError):
        _ = small_world(10, 2, seed=1)

    with pytest.raises(ValueError):
        _ = small_world(10, 10, 10, seed=1)
    with pytest.raises(ValueError):
        _ = small_world(10, 10, seed=1, nb_close_neighbours=3)
    with pytest.raises(ValueError):
        _ = small_world(10, 10, seed=1, proba_rewire=1.5)
    with pytest.raises(ValueError):
        _ = small_world(10, 10, seed=1, proba_rewire=-0.5)
    with pytest.raises(ValueError):
        _ = small_world(10, 10, seed=1, distribution="not_a_distribution")


def test_small_world_sparsity_type():
    W1 = small_world(100, 100, seed=1, sparsity_type="csr")
    assert isinstance(W1, sparse.sparray)
    assert W1.format == "csr"
