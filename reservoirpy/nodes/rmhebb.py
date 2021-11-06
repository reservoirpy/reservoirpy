# Author: Nathan Trouvain at 27/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from ..base import Node
from .utils import readout_forward
from .utils import _initialize_readout

def train():
    ...




class RMHebb(Node):

    def __init__(self):
        super(RMHebb, self).__init__(forward=readout_forward)
