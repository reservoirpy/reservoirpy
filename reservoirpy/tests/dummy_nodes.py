# Author: Nathan Trouvain at 10/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Sequence

import numpy as np
import pytest

from reservoirpy.node import Node, OnlineNode, TrainableNode
from reservoirpy.type import is_multiseries


class IdentityNode(Node):
    def __init__(self, name=None):
        self.name = name
        self.initialized = False
        self.state = {}

    def _step(self, state, x):
        return {"out": x}

    def initialize(self, x, y=None):
        self.input_dim = x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
        self.output_dim = self.input_dim
        self.initialized = True


class PlusNode(Node):
    def __init__(self, h=1, name=None):
        self.h = h
        self.name = name
        self.initialized = False
        self.state = {}

    def _step(self, state, x):
        return {"out": x + self.h}

    def initialize(self, x, y=None):
        self.input_dim = x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
        self.output_dim = self.input_dim
        self.initialized = True


class AccumulateNode(Node):
    def __init__(self, h=0, name=None):
        self.h = h
        self.name = name
        self.initialized = False
        self.state = {}

    def _step(self, state, x):
        return {"out": x + state["out"] + self.h}

    def initialize(self, x, y=None):
        self.input_dim = x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
        self.output_dim = self.input_dim
        self.state = {
            "out": np.zeros(
                self.output_dim,
            )
        }
        self.initialized = True


class MinusNode(Node):
    def __init__(self, h=1, name=None):
        self.h = h
        self.name = name
        self.initialized = False
        self.state = {}

    def _step(self, state, x):
        return {"out": x - self.h}

    def initialize(self, x, y=None):
        self.input_dim = x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
        self.output_dim = self.input_dim
        self.initialized = True


class Inverter(Node):
    def __init__(self, name=None):
        self.name = name
        self.initialized = False
        self.state = {}

    def _step(self, state, x):
        return {"out": -x}

    def initialize(self, x, y=None):
        self.input_dim = x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
        self.output_dim = self.input_dim
        self.initialized = True


class Offline(TrainableNode):
    def __init__(self, name=None):
        self.name = name
        self.b = 0
        self.initialized = False
        self.state = {}

    def initialize(self, x, y=None):
        self.input_dim = x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
        self.output_dim = self.input_dim
        self.initialized = True

    def _step(self, state, x):
        return {"out": x + self.b}

    def fit(self, x, y, warmup=0):
        if not self.initialized:
            self.initialize(x, y)
        self.b = 0

        for el in y[warmup:]:
            self.b += np.sum(el)

        return self


class Unsupervised(TrainableNode):
    def __init__(self, name=None):
        self.name = name
        self.b = 0
        self.initialized = False
        self.state = {}

    def initialize(self, x, y=None):
        self.input_dim = x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
        self.output_dim = self.input_dim
        self.initialized = True

    def _step(self, x):
        return {"out": x + self.b}

    def fit(self, x, y=None, warmup=0):
        if not self.initialized:
            self.initialize(x)

        for el in x[warmup:]:
            self.b += np.sum(el)

        return self


class OnlineUnsupervised(OnlineNode):
    def __init__(self, name=None):
        self.name = name
        self.initialized = False
        self.b = 0
        self.state = {}

    def initialize(self, x, y=None):
        self.input_dim = x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
        self.output_dim = self.input_dim
        self.b = 0
        self.initialized = True

    def _step(self, state, x):
        return {"out": x + self.b}

    def _learning_step(self, x, y=None):
        self.b += np.sum(x)
        return x + self.b

    def partial_fit(self, x, y=None):
        if not self.initialized:
            self.initialize(x)

        self.b += np.sum(x)

        return self
