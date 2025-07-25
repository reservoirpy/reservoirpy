import numpy as np
import pytest
from scipy.sparse import csr_array

from ..nodes import Reservoir, Ridge
from ..observables import (
    effective_spectral_radius,
    memory_capacity,
    mse,
    nrmse,
    rmse,
    rsquare,
    spectral_radius,
)


@pytest.mark.parametrize(
    "obs,ytest,ypred,kwargs,expects",
    [
        (mse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {}, None),
        (rmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {}, None),
        (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {}, None),
        (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {"norm": "var"}, None),
        (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {"norm": "q1q3"}, None),
        (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {"norm": "foo"}, "raise"),
        (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {"norm_value": 3.0}, None),
        (rsquare, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {}, None),
        (mse, [1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5], {}, "raise"),
        (rmse, [[1.0, 2.0, 3.0]], [1.5, 2.5, 3.5], {}, "raise"),
        (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5, 4.2], {}, "raise"),
        (rsquare, [1.0, 2.0, 3.0, 0.0], [1.5, 2.5, 3.5], {}, "raise"),
    ],
)
def test_observable(obs, ytest, ypred, kwargs, expects):
    if expects == "raise":
        with pytest.raises(ValueError):
            obs(ytest, ypred, **kwargs)
    else:
        m = obs(ytest, ypred, **kwargs)
        assert isinstance(m, float)


def test_spectral_radius():
    rng = np.random.default_rng(1234)

    w = rng.uniform(size=(100, 100))

    rho = spectral_radius(w)

    assert isinstance(rho, float)

    idxs = rng.random(size=(100, 100))
    w[idxs < 0.5] = 0
    w = csr_array(w)

    rho = spectral_radius(w)

    assert isinstance(rho, float)

    rho = spectral_radius(w, maxiter=500)

    assert isinstance(rho, float)

    with pytest.raises(ValueError):  # not a square matrix
        w = rng.uniform(size=(5, 100))
        rho = spectral_radius(w)


@pytest.mark.parametrize("observable", [mse, rmse, nrmse, rsquare])
def test_dimensionwise(observable):
    rng = np.random.default_rng(1234)
    # single series
    y1 = rng.uniform(size=(100, 2))
    noise = rng.uniform(size=(100, 2))
    y2 = y1 + noise

    total_score = observable(y_true=y1, y_pred=y2)
    dimensionwise_score = observable(y_true=y1, y_pred=y2, dimensionwise=True)

    assert isinstance(total_score, float)
    assert isinstance(dimensionwise_score, np.ndarray)
    assert dimensionwise_score.shape == (2,)

    # multi-series
    y1 = rng.uniform(size=(3, 100, 2))
    noise = rng.uniform(size=(3, 100, 2))
    y2 = y1 + noise

    total_score = observable(y_true=y1, y_pred=y2)
    dimensionwise_score = observable(y_true=y1, y_pred=y2, dimensionwise=True)

    assert isinstance(total_score, float)
    assert isinstance(dimensionwise_score, np.ndarray)
    assert dimensionwise_score.shape == (2,)


def test_memory_capacity():
    N = 100
    k_max = 2 * N
    model = Reservoir(N, seed=1) >> Ridge(ridge=1e-4)
    mc = memory_capacity(model, k_max=k_max, seed=1)
    mcs = memory_capacity(model, k_max=k_max, as_list=True, seed=1)

    assert isinstance(mc, float)
    assert 0 < mc < k_max
    assert isinstance(mcs, np.ndarray)
    assert mcs.shape == (k_max,)
    assert np.abs(mc - np.sum(mcs)) < 1e-10
    for mc_k in mcs:
        assert 0 < mc_k < 1

    _ = memory_capacity(model, k_max=300, test_size=200)

    # longer lag than the series length
    with pytest.raises(ValueError):
        _ = memory_capacity(model, k_max=300, series=np.ones((100, 1)))
    # invalid test_size argument
    with pytest.raises(ValueError):
        _ = memory_capacity(model, k_max=300, test_size=23.41)
    with pytest.raises(ValueError):
        _ = memory_capacity(model, k_max=300, test_size=None)


def test_effective_spectral_radius():
    reservoir = Reservoir(200, sr=1.0, lr=0.3)
    reservoir.initialize(np.ones((1, 1)))

    esr = effective_spectral_radius(W=reservoir.W, lr=reservoir.lr)
    assert isinstance(esr, float)
