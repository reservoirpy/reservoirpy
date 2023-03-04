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
	# readout.clf.partial_fit(X_batch.reshape(1, -1), Y_batch[0], classes=[0,1,2,3,4,5])

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
			hypers={"f":get_linear("ridge_regression")},
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

class Perceptron(Node):
	def __init__(
		self,
		output_dim=None,
		penalty=None,
		alpha=1e-4,
		l1_ratio=0.15,
		fit_intercept=True,
		max_iter=1000,
		tol=1e-3,
		eta0=1,
		name=None
	):
		super(Perceptron, self).__init__(
			hypers={"f":get_linear("perceptron")},
				forward=readout_forward,
				partial_backward=partial_backward,
				backward=backward,
				output_dim=output_dim,
				initializer=partial(initialize,
					penalty=penalty,
					alpha=alpha,
					max_iter=max_iter,
					l1_ratio=l1_ratio,
					eta0=eta0,
					fit_intercept=fit_intercept,
					tol=tol),
				name=name,
			)