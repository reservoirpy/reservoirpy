import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from reservoirpy.nodes import Reservoir, Ridge, ScikitLearnNode
from reservoirpy.utils.sklearn_helper import (
    TransformInputSklearn,
    TransformOutputSklearn,
)

from ..concat import Concat


@pytest.mark.parametrize("linear_model", [("LogisticRegression"), ("RidgeClassifier")])
def test_sklearn_classification(linear_model):
    pytest.importorskip("sklearn")
    readout = ScikitLearnNode(method=linear_model)
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=150, n_features=4, n_classes=2, random_state=0)
    enocoder = TransformInputSklearn()
    X, y = enocoder(X, y, task="regression")
    X_train, y_train = X[:100], y[:100]
    res = readout.fit(X_train, y_train)
    pred = readout.run(X[100:])
    real = y[100:]
    decoder = TransformOutputSklearn()
    pred, real = decoder(pred, real)
    assert pred.shape == real.shape


@pytest.mark.parametrize("linear_model", [("LogisticRegression"), ("RidgeClassifier")])
def test_sklearn_esn_classification(linear_model):
    pytest.importorskip("sklearn")
    readout = ScikitLearnNode(method=linear_model)
    reservoir = Reservoir(100)
    esn = reservoir >> readout
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        n_samples=250, n_features=6, n_classes=3, random_state=0, n_informative=3
    )
    enocoder = TransformInputSklearn()
    X, y = enocoder(X, y, task="regression")
    X_train, y_train = X[:100], y[:100]
    res = readout.fit(X_train, y_train)
    pred = readout.run(X[100:])
    real = y[100:]
    decoder = TransformOutputSklearn()
    pred, real = decoder(pred, real)
    assert pred.shape == real.shape
