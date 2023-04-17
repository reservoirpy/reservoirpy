# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
# from functools import partial

import numpy as np


from ...node import Node
from ...type import global_dtype
from ...scikit_helper2 import get_linear

from functools import partial
import pdb

def readout_forward(readout: Node, X):
    pred = readout.clf.predict(X[None, :, -1])
    if readout.name in ['Perceptron', 'LogisticRegression', 
    'RidgeClassifier', 'SGDClassifier']:
        return pred[0]
    return np.argmax(pred)

def partial_backward(readout: Node, X_batch, Y_batch=None):
    readout.X_buff.append(X_batch[:, -1])
    readout.Y_buff.append(Y_batch[:, 0])
    

def initialize_buffers(readout):
    input_dim = readout.input_dim
    output_dim = readout.output_dim

def backward(readout: Node, X, Y):
    X, Y = np.array(readout.X_buff), np.array(readout.Y_buff)
    if readout.name in ['Perceptron', 
        'LogisticRegression', 'RidgeClassifier', 'SGDClassifier']:
        Y = np.argmax(np.array(Y), axis=1)
    # pdb.set_trace()
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

class ScikitNodes(Node):
    def __init__(
        self,
        output_dim=None,
        name=None,
        **kwargs
    ):
        super(ScikitNodes, self).__init__(
            hypers={"f":get_linear(name), 'X_buff':list(), 'Y_buff':list()},
            forward=readout_forward,
            partial_backward=partial_backward,
            buffers_initializer=initialize_buffers,
            backward=backward,
            output_dim=output_dim,
            initializer=partial(initialize,
                **kwargs),
            name=name,
            )

