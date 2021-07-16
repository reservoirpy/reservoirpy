# Author: Nathan Trouvain at 12/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from ..node import Node


def _input_initialize(input_node: "Node", x=None):
    if x is not None:
        if input_node.input_dim is None:
            input_node.set_input_dim(x.shape[1])
            input_node.set_output_dim(x.shape[1])


class Input(Node):

    def __init__(self, input_dim=None, name=None):
        super(Input, self).__init__(forward=lambda inp, x: x,
                                    initializer=_input_initialize,
                                    input_dim=input_dim,
                                    output_dim=input_dim,
                                    name=name)


class Output(Node):

    def __init__(self, name=None):
        super(Output, self).__init__(forward=lambda inp, x: x,
                                     initializer=_input_initialize,
                                     name=name)
