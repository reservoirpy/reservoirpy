# Author: Nathan Trouvain at 06/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from reservoirpy.node import Node
from ..utils.random import rand_generator


def forward(node, x):
    choice = node.choice
    return x[:, choice]


def initialize(node, x=None, *args, **kwargs):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(node.n)

        choice = rand_generator(node.seed).choice(np.arange(x.shape[1]),
                                                  node.n,
                                                  replace=False)

        node.set_param("choice", choice)


class RandomChoice(Node):

    def __init__(self, n, seed=None, name=None):
        super(RandomChoice, self).__init__(params={"choice": None},
                                           hypers={"n": n},
                                           forward=forward,
                                           initializer=initialize,
                                           name=name)
        self.seed = seed
