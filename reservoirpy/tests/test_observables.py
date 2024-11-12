import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ..observables import mse, nrmse, rmse, rsquare, spectral_radius


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
    w = csr_matrix(w)

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
    y1 = rng.uniform(size=(100, 2))
    noise = rng.uniform(size=(100, 2))
    y2 = y1 + noise

    total_score = observable(y_true=y1, y_pred=y2)
    dimensionwise_score = observable(y_true=y1, y_pred=y2, dimensionwise=True)

    assert isinstance(total_score, float)
    assert isinstance(dimensionwise_score, np.ndarray)
    assert dimensionwise_score.shape == (2,)
