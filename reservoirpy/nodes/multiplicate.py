# Author: Nathan Trouvain at 09/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from ..model import Node


def forward(multiplicate, data):
    factor = multiplicate.factor
    return (factor * data).reshape(1, -1)


def initialize(multiplicate, x=None):
    if x is not None:
        # x is an array, add over axis 0
        if isinstance(x, np.ndarray):
            multiplicate.set_input_dim(x.shape)
            multiplicate.set_output_dim(x.shape)


class Multiplicate(Node):

    def __init__(self, factor, name=None):
        super(Multiplicate, self).__init__(params={"factor": factor},
                                           forward=forward,
                                           initializer=initialize,
                                           name=name)
