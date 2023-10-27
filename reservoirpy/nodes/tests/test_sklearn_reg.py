import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from reservoirpy.nodes import Reservoir, Ridge, ScikitLearnNode
from reservoirpy.utils.sklearn_helper import (
    TransformInputSklearn,
    TransformOutputSklearn,
)

from ..concat import Concat


@pytest.mark.parametrize("linear_model", [("Ridge"), ("ElasticNet"), ("Lasso")])
def test_sklearn_timeseries(linear_model):
    pytest.importorskip("sklearn")
    node = ScikitLearnNode(method=linear_model, alpha=1e-3)
    X_ = np.sin(np.linspace(0, 6 * np.pi, 100)).reshape(-1, 1)
    X = X_[:50]
    y = X_[1:51]
    enocoder = TransformInputSklearn()
    X, y = enocoder(X, y, task="regression")
    res = node.fit(X, y)
    pred = node.run(X_[50:])
    real = X_[50:]
    decoder = TransformOutputSklearn()
    (pred, real) = decoder(pred, real)
    assert pred.shape == real.shape


@pytest.mark.parametrize("linear_model", [("Ridge"), ("ElasticNet"), ("Lasso")])
def test_sklearn_esn_timeseries(linear_model):
    pytest.importorskip("sklearn")
    readout = ScikitLearnNode(method=linear_model)
    reservoir = Reservoir(100)
    esn = reservoir >> readout
    X_ = np.sin(np.linspace(0, 6 * np.pi, 100)).reshape(-1, 1)
    X = X_[:50]
    y = X_[1:51]
    enocoder = TransformInputSklearn()
    X, y = enocoder(X, y, task="regression")
    res = esn.fit(X, y)
    pred = esn.run(X_[50:])
    real = X_[50:]
    decoder = TransformOutputSklearn()
    pred, real = decoder(pred, real)
    assert pred.shape == real.shape


@pytest.mark.parametrize("linear_model", [("Ridge"), ("ElasticNet"), ("Lasso")])
def test_sklearn_esn_feedback(linear_model):
    readout = ScikitLearnNode(method=linear_model)
    reservoir = Reservoir(100)

    esn = reservoir >> readout

    reservoir <<= readout

    X_ = np.sin(np.linspace(0, 6 * np.pi, 100)).reshape(-1, 1)
    X = X_[:50]
    y = X_[1:51]
    enocoder = TransformInputSklearn()
    X, y = enocoder(X, y, task="regression")
    res = esn.fit(X, y)
    pred = esn.run(X_[50:])
    real = X_[50:]
    decoder = TransformOutputSklearn()
    (pred, real) = decoder(pred, real)
    assert pred.shape == real.shape
