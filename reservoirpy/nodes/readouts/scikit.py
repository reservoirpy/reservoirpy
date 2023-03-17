# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial

import numpy as np


from ...node import Node
from ...type import global_dtype
from ...scikit_helper import get_linear

import pdb

def readout_forward(readout: Node, X):
	return readout.clf.predict(X)

def partial_backward(readout: Node, X_batch, Y_batch=None):
	"""Pre-compute XXt and YXt before final fit."""
	readout.clf.fit(X_batch, Y_batch.squeeze())

def backward(readout: Node, X, Y):
	pass


def initialize(readout: Node, x=None, y=None, *args, **kwargs):

	if x is not None:

		in_dim = x.shape[1]
		if readout.output_dim is not None:
			out_dim = readout.output_dim
		elif y is not None:
			out_dim = y.shape[1]
		else:
			raise RuntimeError(
				f"Impossible to initialize {readout.name}: "
				f"output dimension was not specified at "
				f"creation, and no teacher vector was given."
			)

		readout.set_input_dim(in_dim)
		readout.set_output_dim(out_dim)
		readout.clf = readout.f(**kwargs)

class LinearRegression(Node):
	def __init__(
		self,
		output_dim=None,
		fit_intercept=True,
		name=None,
	):
		super(LinearRegression, self).__init__(
			hypers={"f":get_linear("linear_regression")},
			forward=readout_forward,
			partial_backward=partial_backward,
			backward=backward,
			output_dim=output_dim,
			initializer=partial(initialize, fit_intercept=fit_intercept),
			name=name,
		)

class RidgeRegression(Node):
	def __init__(
		self,
		output_dim=None,
		alpha=1.0,
		fit_intercept=True,
		max_iter=None,
		tol=1e-4,
		name=None
	):
		super(RidgeRegression, self).__init__(
			hypers={"f":get_linear("ridge_regression"), "tol":tol, "alpha":alpha},
			forward=readout_forward,
			partial_backward=partial_backward,
			backward=backward,
			output_dim=output_dim,
			initializer=partial(initialize, 
				alpha=alpha,
				max_iter=max_iter,
				fit_intercept=fit_intercept,
				tol=tol),
			name=name,
		)

class ElasticNet(Node):
	def __init__(
		self,
		output_dim=None,
		penalty=None,
		alpha=1.0,
		l1_ratio=0.5,
		fit_intercept=True,
		max_iter=1000,
		tol=1e-4,
		warm_start=False,
		name=None
	):
		super(ElasticNet, self).__init__(
			hypers={"f":get_linear("elastic_net")},
				forward=readout_forward,
				partial_backward=partial_backward,
				backward=backward,
				output_dim=output_dim,
				initializer=partial(initialize,
					alpha=alpha,
					max_iter=max_iter,
					l1_ratio=l1_ratio,
					warm_start=warm_start,
					fit_intercept=fit_intercept,
					tol=tol),
				name=name,
			)

class Lasso(Node):
	def __init__(
		self,
		output_dim=None,
		penalty=None,
		alpha=1.0,
		fit_intercept=True,
		max_iter=1000,
		tol=1e-4,
		warm_start=False,
		name=None
	):
		super(Lasso, self).__init__(
			hypers={"f":get_linear("lasso")},
				forward=readout_forward,
				partial_backward=partial_backward,
				backward=backward,
				output_dim=output_dim,
				initializer=partial(initialize,
					alpha=alpha,
					max_iter=max_iter,
					warm_start=warm_start,
					fit_intercept=fit_intercept,
					tol=tol),
				name=name,
			)