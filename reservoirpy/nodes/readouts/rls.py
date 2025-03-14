# Author: Nathan Trouvain at 17/05/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial

import numpy as np

from ...mat_gen import zeros
from ...node import Node
from .base import (
    _assemble_wout,
    _compute_error,
    _initialize_readout,
    _prepare_inputs_for_learning,
    _split_and_save_wout,
    readout_forward,
)


def _rls(P, r, e, f):
    """Recursive Least Squares learning rule with forgetting factor."""
    f = float(f)    
    k = np.dot(P, r)
    rPr = np.dot(r.T, k).squeeze()
    c1 = float(1.0 / (f*(f + rPr)))
    c2 = float(1.0 / f)
    P = c2*P - c1 * np.outer(k, k)
    dw = -np.outer(e,np.dot(P, r))
    
    return dw, P


def train(node: "RLS", x, y=None):
    """Train a readout using RLS learning rule."""
    x, y = _prepare_inputs_for_learning(x, y, bias=node.input_bias, allow_reshape=True)

    error, r = _compute_error(node, x, y)

    P = node.P
    dw, P = _rls(P, r, error, node.forgetting)
    wo = _assemble_wout(node.Wout, node.bias, node.input_bias)
    wo = wo + dw.T

    _split_and_save_wout(node, wo)

    node.set_param("P", P)


def initialize(
    readout: "RLS", x=None, y=None, init_func=None, bias_init=None, bias=None
):

    _initialize_readout(readout, x, y, init_func, bias_init, bias)

    if x is not None:
        input_dim, alpha = readout.input_dim, readout.alpha

        if readout.input_bias:
            input_dim += 1

        P = np.eye(input_dim) / alpha

        readout.set_param("P", P)


class RLS(Node):
    """Single layer of neurons learning connections using Recursive Least Squares
    algorithm.

    The learning rules is well described in [1]_.
    The forgetting factor version of the RLS algorithm used here is described in [2]_.

    :py:attr:`RLS.params` **list**

    ================== =================================================================
    ``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
    ``bias``           Learned bias (:math:`\\mathbf{b}`).
    ``P``              Matrix :math:`\\mathbf{P}` of RLS rule.
    ================== =================================================================

    :py:attr:`RLS.hypers` **list**

    ================== =================================================================
    ``alpha``          Diagonal value of matrix P (:math:`\\alpha`) (:math:`1\\cdot 10^{-6}` by default).
    ``input_bias``     If True, learn a bias term (True by default).
    ``forgetting``     Forgetting factor (:math:`\\lambda`) (:math:`1` by default).  
    ================== =================================================================

    Parameters
    ----------
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    alpha : float or Python generator or iterable, default to 1e-6
        Diagonal value of matrix P.
    Wout : callable or array-like of shape (units, targets), default to :py:func:`~reservoirpy.mat_gen.zeros`
        Output weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.zeros`
        Bias weights vector or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    input_bias : bool, default to True
        If True, then a bias parameter will be learned along with output weights.
    forgetting : float, default to 1.0
        The forgetting factor controls the weight given to past observations in the RLS update.
        A value less than 1.0 gives more weight to recent observations.
    name : str, optional
        Node name.

    References
    ----------

    .. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
           Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
           https://doi.org/10.1016/j.neuron.2009.07.018

    .. [2] Waegeman, T., Wyffels, F., & Schrauwen, B. (2012). Feedback Control by Online
           Learning an Inverse Model. IEEE Transactions on Neural Networks and Learning
           Systems, 23(10), 1637–1648. https://doi.org/10.1109/TNNLS.2012.2208655

    Examples
    --------
    >>> x = np.random.normal(size=(100, 3))
    >>> noise = np.random.normal(scale=0.1, size=(100, 1))
    >>> y = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.

    >>> from reservoirpy.nodes import RLS
    >>> rls_node = RLS(alpha=1e-1)

    >>> _ = rls_node.train(x[:5], y[:5])
    >>> print(rls_node.Wout.T, rls_node.bias)
    [[ 9.90731641 -0.06884784  6.87944632]] [[12.07802068]]
    >>> _ = rls_node.train(x[5:], y[5:])
    >>> print(rls_node.Wout.T, rls_node.bias)
    [[ 9.99223366 -0.20499636  6.98924066]] [[12.01128622]]
    """

    def __init__(
        self,
        output_dim=None,
        alpha=1e-6,
        Wout=zeros,
        bias=zeros,
        input_bias=True,
        forgetting=1.0,
        name=None,
    ):
        super(RLS, self).__init__(
            params={"Wout": None, "bias": None, "P": None},
            hypers={
                "alpha": alpha,
                "input_bias": input_bias,
                "forgetting": forgetting,
            },
            forward=readout_forward,
            train=train,
            initializer=partial(
                initialize, init_func=Wout, bias_init=bias, bias=input_bias
            ),
            output_dim=output_dim,
            name=name,
        )
