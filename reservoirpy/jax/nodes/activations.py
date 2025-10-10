# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

from ...type import NodeInput, State, Timeseries, Timestep
from ..activationsfunc import identity, relu, sigmoid, softmax, softplus, tanh
from ..node import Node


class F(Node):
    """Generic elementwise and stateless Node

    This Node takes a function f: array -> array that can take both timeseries
    and timesteps.

    Parameters
    ----------
    f: callable
        Beta parameter of softmax.
    name : str, optional
        Node name.
    **kwargs :
        Additional arguments passed to f.
    """

    def __init__(self, f: Callable, name: Optional[str] = None, **kwargs):
        self.f = partial(f, **kwargs)
        self.state = {}
        self.name = name

    def initialize(
        self,
        x: Union[NodeInput, Timestep],
        y: None = None,
    ):
        self._set_input_dim(x)
        self.output_dim = self.input_dim
        self.state = {"out": jnp.zeros((self.output_dim,))}
        self.initialized = True

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state: State, x: Timestep) -> State:
        return {"out": self.f(x)}

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        out = self.f(x)
        return {"out": out[-1]}, out


class Softmax(F):
    """Softmax activation function.

    .. math::

        y_k = \\frac{e^{\\beta x_k}}{\\sum_{i=1}^{n} e^{\\beta x_i}}

    Parameters
    ----------
    beta: float, default to 1.0
        Beta parameter of softmax.
    name : str, optional
        Node name.
    """

    def __init__(self, beta=1.0, name: Optional[str] = None):
        self.f = lambda x: softmax(beta * x)
        self.state = {}
        self.name = name


class Softplus(F):
    """Softplus activation function.

    .. math::

        f(x) = \\mathrm{ln}(1 + e^{x})

    Parameters
    ----------
    name : str, optional
        Node name.
    """

    def __init__(self, name: Optional[str] = None):
        self.f = softplus
        self.state = {}
        self.name = name


class Sigmoid(F):
    """Sigmoid activation function.

    .. math::

        f(x) = \\frac{1}{1 + e^{-x}}

    Parameters
    ----------
    name : str, optional
        Node name.
    """

    def __init__(self, name: Optional[str] = None):
        self.f = sigmoid
        self.state = {}
        self.name = name


class Tanh(F):
    """Hyperbolic tangent activation function.

    .. math::

        f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}

    Parameters
    ----------
    name : str, optional
        Node name.
    """

    def __init__(self, name: Optional[str] = None):
        self.f = tanh
        self.state = {}
        self.name = name


class Identity(F):
    """Identity function.

    .. math::

        f(x) = x

    Provided for convenience.

    Parameters
    ----------
    name : str, optional
        Node name.
    """

    def __init__(self, name: Optional[str] = None):
        self.f = identity
        self.state = {}
        self.name = name


class ReLU(F):
    """ReLU activation function.

    .. math::

        f(x) = x ~~ \\mathrm{if} ~~ x > 0 ~~ \\mathrm{else} ~~ 0

    Parameters
    ----------
    name : str, optional
        Node name.
    """

    def __init__(self, name: Optional[str] = None):
        self.f = relu
        self.state = {}
        self.name = name
