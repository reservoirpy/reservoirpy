# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial

import numpy as np


from ...node import Node
from ...type import global_dtype
from ...scikit_helper2 import get_linear

import pdb

def readout_forward(readout: Node, X):
    pred = readout.clf.predict(X)
    if pred.ndim == 1:
        pred_onehot = np.zeros(readout.output_dim)
        pred_onehot[pred[0]] = 1
        return pred_onehot
    return pred

def partial_backward(readout: Node, X_batch, Y_batch=None):
    X_buff = readout.get_buffer("X_buff")
    Y_buff = readout.get_buffer("Y_buff")
    counter = readout.get_buffer("counter")
    index = int(counter[0])
    X_buff[index] = X_batch[:, -1] # last step
    Y_buff[index] = Y_batch[:, 0]
    index += 1
    counter[0] = index
    

def initialize_buffers(readout):
    input_dim = readout.input_dim
    output_dim = readout.output_dim
    readout.create_buffer("X_buff", (10000, 561))
    readout.create_buffer("Y_buff", (10000, 6))
    readout.create_buffer("counter", (1))

def backward(readout: Node, X, Y):
    X_buff = readout.get_buffer("X_buff")
    Y_buff = readout.get_buffer("Y_buff")
    counter = readout.get_buffer("counter")
    X, Y = X_buff[:int(counter[0])], Y_buff[:int(counter[0])]
    if readout.name in ['Perceptron', 
        'LogisticRegression', 'RidgeClassifier', 'SGDClassifier']:
        Y_batch = np.argmax(np.array(Y), axis=1)
    readout.clf.fit(X, Y)
    # pass


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
            hypers={"f":get_linear(name)},
            forward=readout_forward,
            partial_backward=partial_backward,
            buffers_initializer=initialize_buffers,
            backward=backward,
            output_dim=output_dim,
            initializer=partial(initialize,
                **kwargs),
            name=name,
            )
