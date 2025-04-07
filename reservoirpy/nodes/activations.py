# Author: Nathan Trouvain at 06/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial
from typing import Callable, Optional, Tuple

from ..activationsfunc import identity, relu, sigmoid, softmax, softplus, tanh
from ..node import Node
from ..type import NodeInput, Timeseries, Timestep


class F(Node):
    def __init__(self, f: Callable, **kwargs):
        self.f = partial(f, **kwargs)
        self.initialized = False

    def initialize(
        self, x: Optional[NodeInput | Timestep], y: Optional[NodeInput | Timestep]
    ):
        self.input_dim = x.shape[-1] if not isinstance(x, list) else x[0].shape[-1]
        self.output_dim = self.input_dim
        self.initialized = True

    def _step(self, state: tuple, x: Timestep) -> Tuple[tuple, Timestep]:
        return (), self.f(x)

    def _run(self, state: tuple, x: Timeseries) -> Tuple[tuple, Timeseries]:
        return (), self.f(x)


class Softmax(F):
    """Softmax activation function.

    .. math::

        y_k = \\frac{e^{\\beta x_k}}{\\sum_{i=1}^{n} e^{\\beta x_i}}

    Parameters
    ----------
    beta: float, default to 1.0
        Beta parameter of softmax.
    """

    def __init__(self, beta=1.0):
        self.f = partial(softmax, beta=beta)
        self.initialized = False


class Softplus(F):
    """Softplus activation function.

    .. math::

        f(x) = \\mathrm{ln}(1 + e^{x})
    """

    def __init__(self):
        self.f = softplus
        self.initialized = False


class Sigmoid(F):
    """Sigmoid activation function.

    .. math::

        f(x) = \\frac{1}{1 + e^{-x}}
    """

    def __init__(self):
        self.f = sigmoid
        self.initialized = False


class Tanh(F):
    """Hyperbolic tangent activation function.

    .. math::

        f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}
    """

    def __init__(self):
        self.f = tanh
        self.initialized = False


class Identity(F):
    """Identity function.

    .. math::

        f(x) = x

    Provided for convenience.
    """

    def __init__(self):
        self.f = identity
        self.initialized = False


class ReLU(F):
    """ReLU activation function.

    .. math::

        f(x) = x ~~ \\mathrm{if} ~~ x > 0 ~~ \\mathrm{else} ~~ 0
    """

    def __init__(self):
        self.f = relu
        self.initialized = False
