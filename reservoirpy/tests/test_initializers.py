import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ..initializers._base import _get_rvs
from ..initializers import RandomSparse
from ..initializers import FastSpectralScaling, SpectralScaling, UniformSpectralScaling, \
    NormalSpectralScaling, BimodalSpectralScaling, LogNormalSpectralScaling
from ..initializers._internal import _rescale_sr
from ..initializers import NormalScaling, UniformScaling, BimodalScaling
from ..observables import spectral_radius


def test_raises_not_a_scipy_dist():
    with pytest.raises(ValueError):
        rg = np.random.default_rng(123456789)
        _get_rvs("foo", rg)


@pytest.mark.parametrize("dist,connectivity,format,kwargs", [
    ("norm", 0.1, "csr", {}),
    ("uniform", 0.5, "coo", {"high": 3, "low": -3}),
    ("lognorm", 1, "csr", {"s": 2, "scale": 5}),
    ("norm", 0.4, "dense", {"scale": 2}),
    ("bimodal", 0.2, "csc", {"value": 2})
])
def test_random_sparse(dist, connectivity, format, kwargs):
    initializer = RandomSparse(connectivity=connectivity,
                               distribution=dist,
                               sparsity_type=format,
                               seed=123456789,
                               **kwargs)

    matrix = initializer((100, 50))

    if connectivity == 1 or format == "dense":
        assert type(matrix) is np.ndarray
    else:
        if format is not None:
            assert format in type(matrix).__name__
        else:
            assert "csr" in type(matrix).__name__
        matrix = matrix.toarray()

    assert_allclose(np.sum(matrix != 0.0) / matrix.size, connectivity)
    assert initializer.distribution == dist


def test_random_sparse_property_access():
    initializer = RandomSparse(connectivity=0.1, distribution="norm",
                               seed=123456789, loc=2, scale=0.5)

    assert initializer.loc == 2
    assert initializer._loc == 2
    assert initializer.scale == 0.5
    assert initializer._scale == 0.5

    with pytest.raises(AttributeError):
        initializer.loc = 5


def test_random_sparse_reseed():
    seed = 123456789

    initializer = RandomSparse(seed=seed)

    matrix_0 = initializer((100, 100)).toarray()
    matrix_1 = initializer((100, 100)).toarray()

    seed2 = 789456123
    initializer_foo = RandomSparse(seed=seed2)

    matrix_foo = initializer_foo((100, 100)).toarray()

    initializer.reset_seed()

    matrix_01 = initializer((100, 100)).toarray()
    matrix_11 = initializer((100, 100)).toarray()

    assert_array_equal(matrix_0, matrix_01)
    assert_array_equal(matrix_1, matrix_11)

    initializer_foo.reset_seed(seed)

    matrix_foo1 = initializer_foo((100, 100)).toarray()
    matrix_foo2 = initializer_foo((100, 100)).toarray()

    assert_array_equal(matrix_foo1, matrix_01)
    assert_array_equal(matrix_foo2, matrix_11)

    initializer_foo.reset_seed(seed2)

    matrix_foo3 = initializer_foo((100, 100)).toarray()

    assert_array_equal(matrix_foo3, matrix_foo)


@pytest.mark.parametrize("sr,dist,connectivity,format,kwargs", [
    (0.9, "norm", 0.1, "csr", {}),
    (1.5, "uniform", 0.5, "coo", {"high": 3, "low": -3}),
    (None, "lognorm", 1, "csr", {"s": 2, "scale": 5}),
    (None, "norm", 0.4, "dense", {"scale": 2}),
    (10, "bimodal", 0.2, "csc", {"value": 2})
])
def test_random_internal(sr, dist, connectivity, format, kwargs):
    initializer = SpectralScaling(connectivity=connectivity,
                                  sr=sr,
                                  distribution=dist,
                                  sparsity_type=format,
                                  seed=123456789,
                                  **kwargs)

    matrix = initializer(100)

    if connectivity == 1 or format == "dense":
        assert type(matrix) is np.ndarray
    else:
        if format is not None:
            assert format in type(matrix).__name__
        else:
            assert "csr" in type(matrix).__name__
        matrix = matrix.toarray()

    assert_allclose(np.sum(matrix != 0.0) / matrix.size, connectivity)
    assert initializer.distribution == dist


@pytest.mark.parametrize("sr", [0.1, 0.2, 0.5, 1.])
def test_spectral_rescaling(sr):
    m = SpectralScaling(sr=None, seed=123456789)(100)
    m = _rescale_sr(m, sr=sr)

    assert_allclose(spectral_radius(m), sr)


@pytest.mark.parametrize("sr", [0.1, 0.2, 0.5, 1.])
def test_fsi(sr):
    m = FastSpectralScaling(sr=sr, seed=123456789)(100)
    assert_allclose(spectral_radius(m), sr, rtol=1e-2)


def test_fsi_no_sr():
    initializer = FastSpectralScaling(seed=123456789)
    m = initializer(100)

    assert set(initializer._rvs_kwargs.values()) == {-1., 1.}


@pytest.mark.parametrize("initializer", [
    UniformSpectralScaling,
    NormalSpectralScaling,
    BimodalSpectralScaling,
    LogNormalSpectralScaling
])
def test_dist_internals(initializer):
    m = initializer(sr=0.5)(100)
    assert_allclose(spectral_radius(m), 0.5)


@pytest.mark.parametrize("initializer", [
    UniformScaling,
    NormalScaling,
    BimodalScaling
])
def test_dist_scaling(initializer):
    m = initializer(scaling=0.5)((100, 500))

    if initializer is NormalScaling:
        m = m.toarray()
        assert_allclose(np.std(m[m != 0.0]), 0.5, rtol=1e-1)
    else:
        assert_allclose(np.max(m.toarray()), 0.5, rtol=1e-1)


@pytest.mark.parametrize("loc,scaling", [
    (0., 2.),
    (5., 0.5)
])
def test_normal_scaling_variations(loc, scaling):
    init0 = NormalScaling(scaling=scaling, loc=loc, seed=123456789)
    m0 = init0((100, 500)).toarray()

    init1 = NormalScaling(scaling=scaling, loc=loc, seed=123456789)
    m1 = init1((100, 500)).toarray()

    assert_array_equal(m0, m1)

    assert_allclose(np.std(m0[m0 != 0.0]), scaling, rtol=1e-1)
    assert_allclose(np.mean(m0[m0 != 0.0]), loc, atol=1e-1)
