# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import warnings
from functools import partial
from numbers import Number
from typing import Iterable

from ...mat_gen import zeros
from ...node import Node
from .base import readout_forward
from .lms import initialize as initialize_lms
from .lms import train as lms_like_train
from .rls import initialize as initialize_rls
from .rls import train as rls_like_train

RULES = ("lms", "rls")


class FORCE(Node):
    """Single layer of neurons learning connections through online learning rules.

    Warning
    -------

    This class is deprecated since v0.3.4 and will be removed in future versions.
    Please use :py:class:`~reservoirpy.LMS` or :py:class:`~reservoirpy.RLS` instead.

    The learning rules involved are similar to Recursive Least Squares (``rls`` rule)
    as described in [1]_ or Least Mean Squares (``lms`` rule, similar to Hebbian
    learning) as described in [2]_.

    "FORCE" name refers to the training paradigm described in [1]_.

    :py:attr:`FORCE.params` **list**

    ================== =================================================================
    ``Wout``           Learned output weights (:math:`\\mathbf{W}_{out}`).
    ``bias``           Learned bias (:math:`\\mathbf{b}`).
    ``P``              Matrix :math:`\\mathbf{P}` of RLS rule (optional).
    ================== =================================================================

    :py:attr:`FORCE.hypers` **list**

    ================== =================================================================
    ``alpha``          Learning rate (:math:`\\alpha`) (:math:`1\\cdot 10^{-6}` by
    default).
    ``input_bias``     If True, learn a bias term (True by default).
    ``rule``           One of RLS or LMS rule ("rls" by default).
    ================== =================================================================

    Parameters
    ----------
    output_dim : int, optional
        Number of units in the readout, can be inferred at first call.
    alpha : float or Python generator or iterable, default to 1e-6
        Learning rate. If an iterable or a generator is provided and the learning
        rule is "lms", then the learning rate can be changed at each timestep of
        training. A new learning rate will be drawn from the iterable or generator
        at each timestep.
    rule : {"rls", "lms"}, default to "rls"
        Learning rule applied for online training.
    Wout : callable or array-like of shape (units, targets), default to
    :py:func:`~reservoirpy.mat_gen.zeros`
        Output weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    bias : callable or array-like of shape (units, 1), default to
    :py:func:`~reservoirpy.mat_gen.zeros`
        Bias weights vector or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    input_bias : bool, default to True
        If True, then a bias parameter will be learned along with output weights.
    name : str, optional
        Node name.

    References
    ----------

    .. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
           Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
           https://doi.org/10.1016/j.neuron.2009.07.018

    .. [2] Hoerzer, G. M., Legenstein, R., & Maass, W. (2014). Emergence of Complex
           Computational Structures From Chaotic Neural Networks Through
           Reward-Modulated Hebbian Learning. Cerebral Cortex, 24(3), 677–690.
           https://doi.org/10.1093/cercor/bhs348
    """

    def __init__(
        self,
        output_dim=None,
        alpha=1e-6,
        rule="rls",
        Wout=zeros,
        bias=zeros,
        input_bias=True,
        name=None,
    ):

        warnings.warn(
            "'FORCE' is deprecated since v0.3.4 and will be removed "
            "in "
            "future versions. Consider using 'RLS' or 'LMS'.",
            DeprecationWarning,
        )

        params = {"Wout": None, "bias": None}

        if rule not in RULES:
            raise ValueError(
                f"Unknown rule for FORCE learning. "
                f"Available rules are {self._rules}."
            )
        else:
            if rule == "lms":
                train = lms_like_train
                initialize = initialize_lms
            else:
                train = rls_like_train
                initialize = initialize_rls
                params["P"] = None

        if isinstance(alpha, Number):

            def _alpha_gen():
                while True:
                    yield alpha

            alpha_gen = _alpha_gen()
        elif isinstance(alpha, Iterable):
            alpha_gen = alpha
        else:
            raise TypeError(
                "'alpha' parameter should be a float or an iterable yielding floats."
            )

        super(FORCE, self).__init__(
            params=params,
            hypers={
                "alpha": alpha,
                "_alpha_gen": alpha_gen,
                "input_bias": input_bias,
                "rule": rule,
            },
            forward=readout_forward,
            train=train,
            initializer=partial(
                initialize, init_func=Wout, bias_init=bias, bias=input_bias
            ),
            output_dim=output_dim,
            name=name,
        )
