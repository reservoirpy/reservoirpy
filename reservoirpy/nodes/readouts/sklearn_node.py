# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
# from functools import partial
import numpy as np

from ...node import Node
from ...type import global_dtype
from ...utils.sklearn_helper import get_linear

from functools import partial
import pdb
def readout_forward(readout: Node, X):
    pred = readout.clf.predict(X)
    return pred

def partial_backward(readout: Node, X_batch, Y_batch=None):
    readout.X_buff.append(X_batch)
    readout.Y_buff.append(Y_batch)

def initialize_buffers(readout):
    input_dim = readout.input_dim
    output_dim = readout.output_dim

def backward(readout: Node, X, Y):
    X, Y = np.array(readout.X_buff), np.array(readout.Y_buff)
    if readout.method_name in ["LogisticRegression", "RidgeClassifier", "Perceptron"]:
        X, Y = X[:, -1:, :], Y[:, -1, 0]
        N, T, D = X.shape
        X = np.reshape(X, (N*T, D))
    else:
        N, T, D = X.shape
        C = Y.shape[-1]
        X, Y = np.reshape(X, (N*T, D)), np.reshape(Y, (N*T, C))  # concating the 1st and 2nd dimis
    readout.clf.fit(X, Y)


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
        kwargs = {k:v for k,v in kwargs.items() if v}
        readout.clf = readout.f(**kwargs)

class SklearnNode(Node):
    """
    A node representing a sklearn linear model that learns the connections
    between input and output data.

    The Sklearn can take any sklearn linear model as input and create a node
    with the specified model.

    Currently we support Linear classifiers like LogisticRegression, Perceptron, 
    RidgeClassifiers, SGDClassifier and Linear regressors like Ridge, LinearRegression
    Lasso and ElastiNet.

    For more information on the above mentioned estimators, 
    please visit sklearn linear model API reference <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model>`_

    :py:attr:`SklearnNode.hypers` **list**

    ================== =================================================================
    ``f``              Function to get the sklearn linear model.
    ``X_buff``         Buffer to store input data.
    ``Y_buff``         Buffer to store output data.
    ================== =================================================================

    Parameters
    ----------
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    name : str, optional
        Node name.
    **kwargs
        Additional keyword arguments for the sklearn linear model.

    Example
    -------
    >>> from reservoirpy import SklearnNode
    >>> node = SklearnNode(name="Ridge", alpha=0.5)
    """

    def __init__(
        self,
        output_dim=None,
        method=None,
        **kwargs
    ):
        super(SklearnNode, self).__init__(
            hypers={
                "f": get_linear(method),
                "X_buff": list(),
                "Y_buff": list(),
                "method_name":method
            },
            forward=readout_forward,
            partial_backward=partial_backward,
            buffers_initializer=initialize_buffers,
            backward=backward,
            output_dim=output_dim,
            initializer=partial(initialize, **kwargs),
            method=method,
        )

