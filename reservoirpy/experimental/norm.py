# Author: Nathan Trouvain at 06/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from reservoirpy.node import Node
from ..activationsfunc import relu, tanh


def forward(node: Node, x: np.ndarray) -> np.ndarray:
    store = node.store
    beta = node.beta

    new_store = np.roll(store, -1)
    new_store[-1] = x

    node.set_param("store", new_store)

    sigma = np.std(new_store)

    if sigma < 1e-8:
        sigma = 1e-8

    x_norm = (x - np.mean(new_store)) / sigma

    return relu(tanh(x_norm / beta))


def initialize(node, x=None, *args, **kwargs):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])

        window = node.window

        node.set_param("store", np.zeros((window, node.output_dim)))


class AsabukiNorm(Node):

    def __init__(self, window, beta=3, name=None):
        super(AsabukiNorm, self).__init__(params={"store": None},
                                          hypers={"window": window,
                                                  "beta": beta},
                                          forward=forward,
                                          initializer=initialize,
                                          name=name)
