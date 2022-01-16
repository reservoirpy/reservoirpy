# Author: Nathan Trouvain at 06/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from ..activationsfunc import get_function
from reservoirpy.node import Node


def forward(node: Node, x):
    return node.f(x)


def initialize(node: Node, x=None, *args, **kwargs):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])


class Softmax(Node):
    def __init__(self):
        super(Softmax, self).__init__(hypers={"f": get_function("softmax")},
                                      forward=forward,
                                      initializer=initialize)


class Softplus(Node):
    def __init__(self):
        super(Softplus, self).__init__(hypers={"f": get_function("softplus")},
                                       forward=forward,
                                       initializer=initialize)


class Sigmoid(Node):
    def __init__(self):
        super(Sigmoid, self).__init__(hypers={"f": get_function("sigmoid")},
                                      forward=forward,
                                      initializer=initialize)


class Tanh(Node):
    def __init__(self):
        super(Tanh, self).__init__(hypers={"f": get_function("tanh")},
                                   forward=forward,
                                   initializer=initialize)


class Identity(Node):
    def __init__(self):
        super(Identity, self).__init__(hypers={"f": get_function("identity")},
                                       forward=forward,
                                       initializer=initialize)


class ReLU(Node):
    def __init__(self):
        super(ReLU, self).__init__(hypers={"f": get_function("relu")},
                                   forward=forward,
                                   initializer=initialize)
