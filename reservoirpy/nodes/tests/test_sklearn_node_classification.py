import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from reservoirpy.nodes import SklearnNode, Ridge, Reservoir
from reservoirpy.utils.sklearn_helper import check_sklearn_dim
from ..concat import Concat
import pdb

@pytest.mark.parametrize(
	"linear_model",
	[("LogisticRegression"), ("RidgeClassifier"), ("SGDClassifier"), ("Perceptron")])

def test_sklearn_classification(linear_model):
	readout = SklearnNode(name=linear_model)
	from sklearn.datasets import make_classification
	from sklearn.metrics import accuracy_score
	X, y = make_classification(n_samples=150, n_features=4, 
                           n_classes=2, random_state=0)
	X_train, y_train = X[:100], y[:100]
	X_train, y_train = check_sklearn_dim(X_train, y_train, readout)
	res = readout.fit(X_train, y_train)
	pred = readout.run(X[100:])
	real = y[100:]
	assert pred.squeeze().shape == real.shape

@pytest.mark.parametrize(
	"linear_model",
	[("ElasticNet"), ("Ridge"), ("LinearRegression")]
)

def test_sklearn_classification_with_regressors(linear_model):
	readout = SklearnNode(name=linear_model)
	from sklearn.datasets import make_classification
	from sklearn.metrics import accuracy_score
	X, y = make_classification(n_samples=150, n_features=4, 
                           n_classes=2, random_state=0)
	X_train, y_train = X[:100], y[:100]
	X_train, y_train = check_sklearn_dim(X_train, y_train, readout)
	res = readout.fit(X_train, y_train)
	pred = readout.run(X[100:])
	pred = np.argmax(pred, axis=1)
	real = y[100:]
	assert pred.shape == real.shape


@pytest.mark.parametrize(
	"linear_model",
	[("LogisticRegression"), ("RidgeClassifier"), ("Perceptron"), ("SGDClassifier")]
)
def test_sklearn_esn_classification(linear_model):
	readout = SklearnNode(name=linear_model)
	reservoir = Reservoir(100)
	esn = reservoir >> readout
	from sklearn.datasets import make_classification
	from sklearn.metrics import accuracy_score
	X, y = make_classification(n_samples=250, n_features=6, 
                           n_classes=3, random_state=0, n_informative=3)
	X_train, y_train = X[:100], y[:100]
	X_train, y_train = check_sklearn_dim(X_train, y_train, readout)
	res = esn.fit(X_train, y_train)
	pred = esn.run(X[100:])
	real = y[100:]
	assert pred.squeeze().shape == real.shape


@pytest.mark.parametrize(
	"linear_model",
	[("ElasticNet"), ("Ridge"), ("LinearRegression")]
)
def test_sklearn_esn_classification_with_regressors(linear_model):
	readout = SklearnNode(name=linear_model)
	reservoir = Reservoir(100)
	esn = reservoir >> readout
	from sklearn.datasets import make_classification
	from sklearn.metrics import accuracy_score
	X, y = make_classification(n_samples=250, n_features=6, 
                           n_classes=3, random_state=0, n_informative=3)
	X_train, y_train = X[:200], y[:200]
	X_train, y_train = check_sklearn_dim(X_train, y_train, readout)
	res = esn.fit(X_train, y_train)
	pred = esn.run(X[200:])
	pred = np.argmax(pred, axis=1)
	real = y[200:]
	assert pred.shape == real.shape

# test_sklearn_esn_classification_with_regressors("Ridge")