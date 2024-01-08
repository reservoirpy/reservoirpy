# Author: Paul BERNARD at 01/01/2024 <paul.bernard@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import numpy as np
import pytest
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    Ridge,
    RidgeClassifier,
)

from reservoirpy.nodes import ScikitLearnNode


@pytest.mark.parametrize(
    "model, model_hypers",
    [
        (LogisticRegression, {"random_state": 2341}),
        (PassiveAggressiveClassifier, {"random_state": 2341}),
        (Perceptron, {"random_state": 2341}),
        (RidgeClassifier, {"random_state": 2341}),
    ],
)
def test_scikitlearn_classifiers(model, model_hypers):
    pytest.importorskip("sklearn")

    rng = np.random.default_rng(seed=2341)

    X_train = rng.normal(0, 1, size=(10000, 2))
    y_train = (X_train[:, 0:1] > 0.0).astype(np.float16)
    X_test = rng.normal(0, 1, size=(100, 2))
    y_test = (X_test[:, 0:1] > 0.0).astype(np.float16)

    scikit_learn_node = ScikitLearnNode(model=model, model_hypers=model_hypers)

    scikit_learn_node.fit(X_train, y_train)
    y_pred = scikit_learn_node.run(X_test)
    assert y_pred.shape == y_test.shape
    assert np.all(y_pred == y_test)


@pytest.mark.parametrize(
    "model, model_hypers",
    [
        (LinearRegression, {}),
        (Ridge, {"random_state": 2341}),
    ],
)
def test_scikitlearn_regressors(model, model_hypers):
    pytest.importorskip("sklearn")
    seed = 2341
    rng = np.random.default_rng(seed)
    X_train = rng.normal(0, 1, size=(10000, 2))
    y_train = X_train[:, 0:1] + X_train[:, 1:2]
    y_train = y_train.astype(np.float16)
    X_test = rng.normal(0, 1, size=(100, 2))
    y_test = X_test[:, 0:1] + X_test[:, 1:2]
    y_test = y_test.astype(np.float16)

    scikit_learn_node = ScikitLearnNode(model=model, model_hypers=model_hypers)

    scikit_learn_node.fit(X_train, y_train)
    y_pred = scikit_learn_node.run(X_test)
    assert y_pred.shape == y_test.shape
    mse = np.mean(np.square(y_pred - y_test))
    assert mse < 1e-5
