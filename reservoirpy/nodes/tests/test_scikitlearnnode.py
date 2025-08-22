# Author: Paul BERNARD at 01/01/2024 <paul.bernard@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.linear_model import (
    ElasticNet,
    Lars,
    Lasso,
    LassoCV,
    LassoLars,
    LinearRegression,
    LogisticRegression,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveClassifier,
    Perceptron,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

import reservoirpy
from reservoirpy.nodes import ScikitLearnNode


def test_fail_non_predictors():
    with pytest.raises(AttributeError):
        _ = ScikitLearnNode(PCA)
    with pytest.raises(AttributeError):
        _ = ScikitLearnNode(DotProduct)


def test_scikitlearn_initializer():

    with pytest.raises(TypeError):
        _ = ScikitLearnNode(LinearRegression).initialize()

    with pytest.raises(TypeError):
        _ = ScikitLearnNode(LinearRegression).initialize(np.ones((100, 2)))

    _ = ScikitLearnNode(LinearRegression).initialize(np.ones((100, 2)), np.ones((100, 2)))

    linear_regressor = ScikitLearnNode(LinearRegression, output_dim=2, positive=False)
    linear_regressor.initialize(np.ones((100, 2)))

    assert linear_regressor.model_kwargs == {"positive": False}


# Note that a different seed may fail the tests
@pytest.mark.parametrize(
    "model, model_hypers",
    [
        (LogisticRegression, {"random_state": 2341}),
        (PassiveAggressiveClassifier, {"random_state": 2341}),
        (Perceptron, {"random_state": 2341}),
        (RidgeClassifier, {"random_state": 2341}),
        (SGDClassifier, {"random_state": 2341}),
        (MLPClassifier, {"tol": 1e-3, "random_state": 2341}),
    ],
)
def test_scikitlearn_classifiers(model, model_hypers):
    rng = np.random.default_rng(seed=2341)

    X_train = rng.normal(0, 1, size=(100, 2))
    y_train = (X_train[:, 0:1] > 0.0).astype(np.float16)
    X_test = rng.normal(0, 1, size=(10, 2))
    y_test = (X_test[:, 0:1] > 0.0).astype(np.float16)

    scikit_learn_node = ScikitLearnNode(model=model, **model_hypers)

    scikit_learn_node.fit(X_train, y_train)
    y_pred = scikit_learn_node.run(X_test)
    assert y_pred.shape == y_test.shape
    assert np.all(y_pred == y_test)


@pytest.mark.parametrize(
    "model, model_hypers",
    [
        (LinearRegression, {}),
        (Ridge, {"random_state": 2341}),
        (SGDRegressor, {"random_state": 2341}),
        (ElasticNet, {"alpha": 1e-4, "random_state": 2341}),
        (Lars, {"random_state": 2341}),
        (Lasso, {"alpha": 1e-4, "random_state": 2341}),
        (LassoLars, {"alpha": 1e-4, "random_state": 2341}),
        (OrthogonalMatchingPursuitCV, {}),
        (MLPRegressor, {"tol": 1e-4, "random_state": 2341}),
    ],
)
def test_scikitlearn_regressors_monooutput(model, model_hypers):
    rng = np.random.default_rng(seed=2341)
    X_train = list(rng.normal(0, 1, size=(30, 100, 2)))
    y_train = [(x[:, 0:1] + x[:, 1:2]).astype(np.float16) for x in X_train]
    X_test = rng.normal(0, 1, size=(100, 2))
    y_test = (X_test[:, 0:1] + X_test[:, 1:2]).astype(np.float16)

    scikit_learn_node = ScikitLearnNode(model=model, **model_hypers)

    scikit_learn_node.fit(X_train, y_train)
    y_pred = scikit_learn_node.run(X_test)
    assert y_pred.shape == y_test.shape
    mse = np.mean(np.square(y_pred - y_test))
    assert mse < 2e-3

    timestep = scikit_learn_node(X_test[0])
    assert timestep.ndim == 1


def test_scikitlearn_multioutput():
    rng = np.random.default_rng(seed=2341)
    X_train = rng.normal(0, 1, size=(200, 3))
    y_train = X_train @ np.array([[0, 1, 0], [0, 1, 1], [-1, 0, 1]])
    X_test = rng.normal(0, 1, size=(100, 3))

    lasso = ScikitLearnNode(model=LassoCV, random_state=2341).fit(X_train, y_train).fit(X_train, y_train)  # fit twice
    lasso_pred = lasso.run(X_test)

    mt_lasso = ScikitLearnNode(model=MultiTaskLassoCV, random_state=2341).fit(X_train, y_train)
    mt_lasso_pred = mt_lasso.run(X_test)

    assert type(lasso.instances) is list
    assert type(mt_lasso.instances) is not list

    coef_single = [
        lasso.instances[0].coef_,
        lasso.instances[1].coef_,
        lasso.instances[2].coef_,
    ]
    coef_multitask = mt_lasso.instances.coef_
    assert np.linalg.norm(coef_single[0] - coef_multitask[0]) < 1e-3
    assert np.linalg.norm(coef_single[1] - coef_multitask[1]) < 1e-3
    assert np.linalg.norm(coef_single[2] - coef_multitask[2]) < 1e-3

    assert lasso_pred.shape == mt_lasso_pred.shape == (100, 3)
    assert np.linalg.norm(mt_lasso_pred - lasso_pred) < 1e-2


def test_scikitlearn_multiseries():
    rng = np.random.default_rng(seed=2341)
    X_train = rng.normal(0, 1, size=(5, 50, 3))
    y_train = X_train @ np.array([[0, 1, 0], [0, 1, 1], [-1, 0, 1]])
    X_test = rng.normal(0, 1, size=(2, 100, 3))

    lasso = ScikitLearnNode(model=LassoCV, random_state=2341)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.run(X_test)

    assert lasso_pred.shape == (2, 100, 3)

    rng = np.random.default_rng(seed=2341)
    X_train = list(rng.normal(0, 1, size=(5, 200, 3)))
    y_train = X_train @ np.array([[0, 1, 0], [0, 1, 1], [-1, 0, 1]])
    X_test = list(rng.normal(0, 1, size=(2, 100, 3)))

    lasso = ScikitLearnNode(model=LassoCV, random_state=2341)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.run(X_test)

    assert isinstance(lasso_pred, list)
    assert len(lasso_pred) == 2
    assert lasso_pred[0].shape == (100, 3)


def test_scikitlearn_reproductibility_random_state():
    rng = np.random.default_rng(seed=2341)
    X_train = rng.normal(0, 1, size=(100, 3))
    y_train = (X_train @ np.array([0.5, 1, 2])).reshape(-1, 1)
    X_test = rng.normal(0, 1, size=(100, 3))

    # Different scikit-learn random_states
    reservoirpy.set_seed(0)
    y_pred1 = ScikitLearnNode(model=SGDRegressor, random_state=1).fit(X_train, y_train).run(X_test)

    reservoirpy.set_seed(0)
    y_pred2 = ScikitLearnNode(model=SGDRegressor, random_state=2).fit(X_train, y_train).run(X_test)

    assert not np.all(y_pred1 == y_pred2)

    # Same scikit-learn random_states
    reservoirpy.set_seed(0)
    y_pred1 = ScikitLearnNode(model=SGDRegressor, random_state=1).fit(X_train, y_train).run(X_test)

    reservoirpy.set_seed(0)
    y_pred2 = ScikitLearnNode(model=SGDRegressor, random_state=1).fit(X_train, y_train).run(X_test)

    assert np.all(y_pred1 == y_pred2)

    # Same scikit-learn random_states (call)
    reservoirpy.set_seed(0)
    y_pred1 = ScikitLearnNode(model=SGDRegressor, random_state=1).fit(X_train, y_train)(X_test[0])

    reservoirpy.set_seed(0)
    y_pred2 = ScikitLearnNode(model=SGDRegressor, random_state=1).fit(X_train, y_train)(X_test[0])

    assert np.all(y_pred1 == y_pred2)


def test_scikitlearn_reproductibility_rpy_seed():
    rng = np.random.default_rng(seed=2341)
    X_train = rng.normal(0, 1, size=(100, 3))
    y_train = (X_train @ np.array([0.5, 1, 2])).reshape(-1, 1)
    X_test = rng.normal(0, 1, size=(100, 3))

    # Different ReservoirPy random generator
    reservoirpy.set_seed(1)
    y_pred1 = ScikitLearnNode(model=SGDRegressor).fit(X_train, y_train).run(X_test)

    reservoirpy.set_seed(2)
    y_pred2 = ScikitLearnNode(model=SGDRegressor).fit(X_train, y_train).run(X_test)

    assert not np.all(y_pred1 == y_pred2)

    # Same ReservoirPy random generator
    reservoirpy.set_seed(0)
    y_pred1 = ScikitLearnNode(model=SGDRegressor).fit(X_train, y_train).run(X_test)

    reservoirpy.set_seed(0)
    y_pred2 = ScikitLearnNode(model=SGDRegressor).fit(X_train, y_train).run(X_test)

    assert np.all(y_pred1 == y_pred2)
