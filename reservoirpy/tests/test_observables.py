# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import numpy as np
import pytest
from scipy.sparse import csr_array

from ..nodes import Reservoir, Ridge
from ..observables import (
    _lyapunov,
    effective_spectral_radius,
    ky_dim,
    lyapunov,
    memory_capacity,
    mse,
    nrmse,
    rmse,
    rsquare,
    mae,
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
        (mae, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], {}, None),
        (mse, [1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5], {}, "raise"),
        (rmse, [[1.0, 2.0, 3.0]], [1.5, 2.5, 3.5], {}, "raise"),
        (nrmse, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5, 4.2], {}, "raise"),
        (rsquare, [1.0, 2.0, 3.0, 0.0], [1.5, 2.5, 3.5], {}, "raise"),
        (mae, [1.0, 2.0, 3.0, 0.0], [1.5, 2.5, 3.5], {}, "raise"),
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


@pytest.mark.parametrize("observable", [mse, rmse, nrmse, rsquare, mae])
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


def test_lyapunov_logistic_map():
    """_lyapunov recovers the Lyapunov exponent of the logistic map within tolerance.

    The logistic map x -> r*x*(1-x) with r=3.9 has a numerically known
    leading Lyapunov exponent of approximately 0.494.
    """
    r = 3.9

    class LogisticModel:
        def __init__(self, x):
            self.state = np.array([float(x)])
            self.t_step = 1.0
            self.stdev = 0.2  # rough std of logistic map attractor

        def run(self, steps):
            x = self.state[0]
            for _ in range(steps):
                x = r * x * (1.0 - x)
            self.state = np.array([x])

    model = LogisticModel(0.5)
    result = _lyapunov(
        model,
        k=1,
        cycle_length=1,
        breakin_cycles=500,
        min_cycles=500,
        max_cycles=5000,
    )

    assert "spectrum" in result
    assert "ky_dim" in result
    assert "n_cycles" in result
    lam = result["spectrum"][0]
    assert 0.4 < lam < 0.6, f"Expected λ_1 ≈ 0.494 for logistic r=3.9, got {lam:.4f}"


def test_ky_dim():
    """ky_dim returns correct dimension and CI for a simple known spectrum."""
    # Lorenz63 true spectrum is approximately [0.9, 0.0, -14.6]
    spec = np.array([0.9, 0.0, -14.6])
    # Without CI: returns a float
    d = ky_dim(spec)
    assert isinstance(d, float)
    assert 2.0 < d < 3.0

    # With CI: returns (dim, ci) tuple
    ci_half = np.array([0.01, 0.01, 0.1])
    d2, ci = ky_dim(spec, ci_half=ci_half)
    assert isinstance(d2, float)
    assert isinstance(ci, float)
    assert ci > 0
    assert abs(d2 - d) < 1e-10

    # Purely contracting spectrum → dimension 0
    assert ky_dim(np.array([-1.0, -2.0])) == 0.0

    # All-positive cumsum (dimension ≥ spectrum length) → float(len)
    with pytest.warns(RuntimeWarning):
        assert ky_dim(np.array([1.0, 1.0])) == 2.0


def test_lyapunov():
    """lyapunov() Model wrapper produces a finite, ordered Lyapunov spectrum.

    Smoke test: verifies the public wrapper runs to completion and returns
    the expected dict structure and array shape.  Sign convergence of the
    leading exponent is not asserted here because the small ESN and low
    cycle count used to keep the test fast may not converge to the true
    attractor dynamics within the allowed budget.
    """
    from ..datasets import lorenz

    data = lorenz(3000, seed=0)
    x_train, y_train = data[:2000], data[1:2001]
    esn = Reservoir(50, sr=0.95, seed=0) >> Ridge(ridge=1e-6)
    esn = esn.fit(x_train, y_train, warmup=100)

    result = lyapunov(esn, init_traj=x_train[:100], k=3, min_cycles=200, max_cycles=500)

    assert "spectrum" in result
    assert "ky_dim" in result
    assert "ky_dim_ci" in result
    assert "spectrum_ci" in result
    assert "n_cycles" in result
    assert "converged" in result
    assert result["spectrum"].shape == (3,)
    assert np.all(np.isfinite(result["spectrum"])), "Spectrum should not contain NaN or Inf"
    lam = result["spectrum"]
    assert lam[0] >= lam[1] >= lam[2], (
        f"Benettin algorithm should produce an ordered spectrum; got {lam}"
    )
