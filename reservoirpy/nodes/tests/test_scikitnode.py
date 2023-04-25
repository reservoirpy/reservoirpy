import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from reservoirpy.nodes import ScikitNode, Ridge, Reservoir
from reservoirpy.scikit_helper import check_scikit_dim
from ..concat import Concat
import pdb

@pytest.mark.parametrize(
	"linear_model",
	[("Ridge"), ("ElasticNet"), ("Lasso")]
)
def test_scikit_regression(linear_model):
	node = ScikitNode(name=linear_model, alpha=1e-4)
	from sklearn.datasets import make_regression
	X, y = make_regression(n_samples=10, n_features=2, 
                       random_state=123)
	
	X, y = check_scikit_dim(X, y, node)
	res = node.fit(X[:-1], y[:-1])
	pred = node.run(X[-1])
	real = y[-1]
	assert_array_almost_equal(pred, real, decimal=2)


@pytest.mark.parametrize(
	"linear_model",
	[("Ridge"), ("ElasticNet"), ("Lasso")]
)
def test_scikit_timeseries(linear_model):
	node = ScikitNode(name=linear_model, alpha=1e-3)
	X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
	X_train = X[:50]
	y_train = X[1:51]
	X_train, y_train = check_scikit_dim(X_train, y_train, node)
	res = node.fit(X_train, y_train)
	pred = node.run(X[50:])
	real = X[50:]
	assert_array_almost_equal(pred, real, decimal=1)


@pytest.mark.parametrize(
	"linear_model",
	[("Ridge"), ("ElasticNet"), ("Lasso")]
)

def test_scikit_esn_regression(linear_model):
	readout = ScikitNode(name=linear_model)
	reservoir = Reservoir(100)
	esn = reservoir >> readout
	from sklearn.datasets import make_regression
	X, y = make_regression(n_samples=10, n_features=2, 
                       random_state=123)
	
	X, y = check_scikit_dim(X, y, readout)
	res = esn.fit(X[:-1], y[:-1])
	pred = esn.run(X[-1])
	assert pred.shape == y[-1].shape


@pytest.mark.parametrize(
	"linear_model",
	[("Ridge"), ("ElasticNet"), ("Lasso")]
)

def test_scikit_esn_timeseries(linear_model):
	readout = ScikitNode(name=linear_model)
	reservoir = Reservoir(100)
	esn = reservoir >> readout
	X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
	X_train = X[:50]
	y_train = X[1:51]
	X_train, y_train = check_scikit_dim(X_train, y_train, readout)
	res = esn.fit(X_train, y_train)
	pred = esn.run(X[50:])
	real =  X[50:]
	assert pred.shape == real.shape


@pytest.mark.parametrize(
	"linear_model",
	[("Ridge"), ("ElasticNet"), ("Lasso")]
)
def test_scikit_esn_feedback(linear_model):
	readout = ScikitNode(name=linear_model)
	reservoir = Reservoir(100)

	esn = reservoir >> readout

	reservoir <<= readout

	X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
	X_train = X[:50]
	y_train = X[1:51]
	X_train, y_train = check_scikit_dim(X_train, y_train, readout)
	res = esn.fit(X_train, y_train)
	pred = esn.run(X[50:])
	real =  X[50:]
	assert pred.shape == real.shape

@pytest.mark.parametrize(
	"linear_model",
	[("LogisticRegression"), ("RidgeClassifier"), ("SGDClassifier"), ("Perceptron")])

def test_scikit_classification(linear_model):
	readout = ScikitNode(name=linear_model)
	from sklearn.datasets import make_classification
	from sklearn.metrics import accuracy_score
	X, y = make_classification(n_samples=150, n_features=4, 
                           n_classes=2, random_state=0)
	X_train, y_train = X[:100], y[:100]
	X_train, y_train = check_scikit_dim(X_train, y_train, readout)
	res = readout.fit(X_train, y_train)
	pred = readout.run(X[100:])
	real = y[100:]
	assert pred.squeeze().shape == real.shape

@pytest.mark.parametrize(
	"linear_model",
	[("ElasticNet"), ("Lasso"), ("LinearRegression")]
)

def test_scikit_classification_with_regressors(linear_model):
	readout = ScikitNode(name=linear_model)
	from sklearn.datasets import make_classification
	from sklearn.metrics import accuracy_score
	X, y = make_classification(n_samples=150, n_features=4, 
                           n_classes=2, random_state=0)
	X_train, y_train = X[:100], y[:100]
	X_train, y_train = check_scikit_dim(X_train, y_train, readout)
	res = readout.fit(X_train, y_train)
	pred = readout.run(X[100:])
	pred = np.argmax(pred, axis=1)
	real = y[100:]
	assert pred.shape == real.shape


@pytest.mark.parametrize(
	"linear_model",
	[("LogisticRegression"), ("RidgeClassifier"), ("Perceptron"), ("SGDClassifier")]
)
def test_scikit_esn_classification(linear_model):
	readout = ScikitNode(name=linear_model)
	reservoir = Reservoir(100)
	esn = reservoir >> readout
	from sklearn.datasets import make_classification
	from sklearn.metrics import accuracy_score
	X, y = make_classification(n_samples=250, n_features=6, 
                           n_classes=3, random_state=0, n_informative=3)
	X_train, y_train = X[:100], y[:100]
	X_train, y_train = check_scikit_dim(X_train, y_train, readout)
	res = esn.fit(X_train, y_train)
	pred = esn.run(X[100:])
	real = y[100:]
	assert pred.squeeze().shape == real.shape


@pytest.mark.parametrize(
	"linear_model",
	[("ElasticNet"), ("Lasso"), ("LinearRegression")]
)
def test_scikit_esn_classification_with_regressors(linear_model):
	readout = ScikitNode(name=linear_model)
	reservoir = Reservoir(100)
	esn = reservoir >> readout
	from sklearn.datasets import make_classification
	from sklearn.metrics import accuracy_score
	X, y = make_classification(n_samples=250, n_features=6, 
                           n_classes=3, random_state=0, n_informative=3)
	X_train, y_train = X[:200], y[:200]
	X_train, y_train = check_scikit_dim(X_train, y_train, readout)
	res = esn.fit(X_train, y_train)
	pred = esn.run(X[200:])
	pred = np.argmax(pred, axis=1)
	real = y[200:]
	assert pred.shape == real.shape

# test_scikit_esn_classification_with_regressors("Ridge")