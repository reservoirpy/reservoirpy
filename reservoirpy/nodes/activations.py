# Author: Nathan Trouvain at 06/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial

from ..activationsfunc import get_function
from ..node import Node


def forward(node: Node, x, **kwargs):
    return node.f(x, **kwargs)


def initialize(node: Node, x=None, *args, **kwargs):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])


class Softmax(Node):
    """Softmax activation function.

    .. math::

        y_k = \\frac{e^{\\beta x_k}}{\\sum_{i=1}^{n} e^{\\beta x_i}}

    :py:attr:`Softmax.hypers` **list**

    ============= ======================================================================
    ``f``         Activation function (:py:func:`reservoir.activationsfunc.softmax`).
    ``beta``      Softmax :math:`\\beta` parameter (1.0 by default).
    ============= ======================================================================

    Parameters
    ----------
    beta: float, default to 1.0
        Beta parameter of softmax.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    name : str, optional
        Node name.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    """

    def __init__(self, beta=1.0, **kwargs):
        super(Softmax, self).__init__(
            hypers={"f": get_function("softmax"), "beta": beta},
            forward=partial(forward, beta=beta),
            initializer=initialize,
            **kwargs,
        )


class Softplus(Node):
    """Softplus activation function.

    .. math::

        f(x) = \\mathrm{ln}(1 + e^{x})

    :py:attr:`Softplus.hypers` **list**

    ============= ======================================================================
    ``f``         Activation function (:py:func:`reservoir.activationsfunc.softplus`).
    ============= ======================================================================

    Parameters
    ----------
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    name : str, optional
        Node name.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    """

    def __init__(self, **kwargs):
        super(Softplus, self).__init__(
            hypers={"f": get_function("softplus")},
            forward=forward,
            initializer=initialize,
            **kwargs,
        )


class Sigmoid(Node):
    """Sigmoid activation function.

    .. math::

        f(x) = \\frac{1}{1 + e^{-x}}

    :py:attr:`Sigmoid.hypers` **list**

    ============= ======================================================================
    ``f``         Activation function (:py:func:`reservoir.activationsfunc.sigmoid`).
    ============= ======================================================================

    Parameters
    ----------
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    name : str, optional
        Node name.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    """

    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(
            hypers={"f": get_function("sigmoid")},
            forward=forward,
            initializer=initialize,
            **kwargs,
        )


class Tanh(Node):
    """Hyperbolic tangent activation function.

    .. math::

        f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}

    :py:attr:`Tanh.hypers` **list**

    ============= ======================================================================
    ``f``         Activation function (:py:func:`reservoir.activationsfunc.tanh`).
    ============= ======================================================================

    Parameters
    ----------
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    name : str, optional
        Node name.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    """

    def __init__(self, **kwargs):
        super(Tanh, self).__init__(
            hypers={"f": get_function("tanh")},
            forward=forward,
            initializer=initialize,
            **kwargs,
        )


class Identity(Node):
    """Identity function.

    .. math::

        f(x) = x

    Provided for convenience.

    :py:attr:`Identity.hypers` **list**

    ============= ======================================================================
    ``f``         Activation function (:py:func:`reservoir.activationsfunc.identity`).
    ============= ======================================================================

    Parameters
    ----------
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    name : str, optional
        Node name.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    """

    def __init__(self, **kwargs):
        super(Identity, self).__init__(
            hypers={"f": get_function("identity")},
            forward=forward,
            initializer=initialize,
            **kwargs,
        )


class ReLU(Node):
    """ReLU activation function.

    .. math::

        f(x) = x ~~ \\mathrm{if} ~~ x > 0 ~~ \\mathrm{else} ~~ 0

    :py:attr:`ReLU.hypers` **list**

    ============= ======================================================================
    ``f``         Activation function (:py:func:`reservoir.activationsfunc.relu`).
    ============= ======================================================================

    Parameters
    ----------
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    name : str, optional
        Node name.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    """

    def __init__(self, **kwargs):
        super(ReLU, self).__init__(
            hypers={"f": get_function("relu")},
            forward=forward,
            initializer=initialize,
            **kwargs,
        )
