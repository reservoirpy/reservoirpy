# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial
from typing import Callable, Optional, Union

import numpy as np

from ..activationsfunc import identity
from ..mat_gen import generate_input_weights, generate_internal_weights
from ..node import Node
from ..nodes.reservoir import _reservoir_kernel
from ..nodes.reservoir import initialize as initialize_reservoir
from ..nodes.reservoir import initialize_feedback
from ..types import Weights
from ..utils.model_utils import to_ragged_seq_set
from ..utils.random import noise
from ..utils.validation import is_array


def _gaussian_gradients(x, y, a, sigma, mu):
    """Compute weight deltas"""
    sig2 = sigma**2
    delta_b = (mu / sig2) - (y / sig2) * (2 * sig2 + 1 - y**2 + mu * y)
    delta_a = (1 / a) + delta_b * x
    return delta_a, delta_b


def _exp_gradients(x, y, a, mu):
    mu_inv = 1 / mu
    delta_b = 1 - (2 + mu_inv) * y + mu_inv * y**2
    delta_a = (1 / a) + delta_b * x
    return delta_a, delta_b


def _apply_gradients(a, b, delta_a, delta_b, eta):
    a2 = a + eta * delta_a
    b2 = b + eta * delta_b
    return a2, b2


def _ip(reservoir, X_seq, warmup=0):
    epochs = reservoir.epochs


def forward(reservoir: "IPReservoir", x: np.ndarray) -> np.ndarray:
    """Reservoir with internal activation function:

    .. math::

        r[n+1] = (1 - \\alpha) \\cdot r[t] + \\alpha
         \\cdot f (W_{in} \\cdot u[n] + W \\cdot r[t])


    where :math:`r[n]` is the state and the output of the reservoir."""
    f = reservoir.activation
    dist = reservoir.noise_type
    g_rc = reservoir.noise_rc
    a, b = reservoir.a, reservoir.b
    noise_gen = reservoir.noise_generator

    u = x.reshape(-1, 1)
    r = reservoir.state().T

    s_next = f(a * _reservoir_kernel(reservoir, u, r) + b) + noise_gen(
        dist, r.shape, g_rc
    )

    return s_next.T


def backward(reservoir: Node, X=None, Y=None, warmup=0):

    if X is None:
        X = reservoir._X

    X = to_ragged_seq_set(X)

    for e in range(reservoir.epochs):
        for x in enumerate(X):
            if t > warmup and e == 0:
                _ip(reservoir, x)
            else:
                reservoir(x)

    input_dim = readout.input_dim
    if readout.input_bias:
        input_dim += 1

    ridgeid = ridge * np.eye(input_dim, dtype=global_dtype)

    Wout_raw = _solve_ridge(XXT, YXT, ridgeid)

    if readout.input_bias:
        Wout, bias = Wout_raw[1:, :], Wout_raw[0, :][np.newaxis, :]
        readout.set_param("Wout", Wout)
        readout.set_param("bias", bias)
    else:
        readout.set_param("Wout", Wout_raw)


def initialize(reservoir, *args, **kwargs):

    initialize_reservoir(reservoir, *args, **kwargs)

    a = np.ones((1, reservoir.output_dim))
    b = np.zeros((1, reservoir.output_dim))

    reservoir.set_param("a", a)
    reservoir.set_param("b", b)


class IPReservoir(Node):
    """
    Pool of neurons with random recurrent connexions, tuned using Intrinsic
    Plasticity as described in [1]_ and [2]_.

    Parameters
    ----------
    units : int, optional
        Number of reservoir units, by default None
    lr : float, optional
        Neurons leak rate. Must be in [0; 1], by default 1.0
    sr : float, optional
        Spectral radius of recurrent weight matrix, by default None
    input_bias : bool, optional
        If ``False``, no bias is added to inputs, by default True
    noise_rc : float, optional
        Gain of noise applied to reservoir internal states, by default 0.0
    noise_in : float, optional
        Gain of noise applied to inputs, by default 0.0
    noise_fb : float, optional
        Gain of noise applied to feedback signal, by default 0.0
    noise_type : str, optional
        Distribution of noise. Must be a Numpy random variable generator
        distribution (see :py:class:`numpy.random.Generator`),
        by default "normal"
    input_scaling : float, optional
        Input gain, by default 1.0
    fb_scaling : float, optional
        Feedback gain, by default 1.0
    input_connectivity : float, optional
        Connectivity of input neurons, i.e. ratio of input neurons connected
        to reservoir neurons. Must be in ]0, 1], by default 0.1
    rc_connectivity : float, optional
        Connectivity of recurrent weights matrix, i.e. ratio of reservoir
        neurons connected to other reservoir neurons, including themselves.
        Must be in ]0, 1], by default 0.1
    fb_connectivity : float, optional
        Connectivity of feedback neurons, i.e. ratio of feedabck neurons
        connected to reservoir neurons. Must be in ]0, 1], by default 0.1
    Win : Union[Weights, Callable], optional
        Input weights matrix initializer.
        - If a :py:class:`numpy.ndarray` or :py:mod:`scipy.sparse` matrix,
        should be of shape ``(input_dim + bias, units)``.
        - If a callable, should accept keywords parameters such as ``N`` to
        specify number of reservoir units and ``dim_input`` to specify input
        dimension. By default :py:func:`mat_gen.generate_input_weights`.
    W : Union[Weights, Callable], optional
        Reccurent weights matrix initializer.
        - If a :py:class:`numpy.ndarray` or :py:mod:`scipy.sparse` matrix,
        should be of shape ``(units, units)``.
        - If a callable, should accept keywords parameters such as ``N`` to
        specify number of reservoir units.
        By default :py:func:`mat_gen.generate_internal_weights`
    Wfb : Union[Weights, Callable], optional
        Feedback weights matrix initializer.
        - If a :py:class:`numpy.ndarray` or :py:mod:`scipy.sparse` matrix,
        should be of shape ``(feedback_dim, units)``.
        - If a callable, should accept keywords parameters such as ``N`` to
        specify number of reservoir units and ``dim_input`` to specify feedback
        dimension. By default :py:func:`mat_gen.generate_input_weights`
    fb_dim : int, optional
        Feedback dimension, by default None
    fb_activation : Union[str, Callable], optional
        Feedback activation function.
        - If a str, should be a :py:mod:`reservoirpy.activationfunc`
        function name.
        - If a callable, should be an element-wise operator on ndarray.
        By default, :py:func:`activationfunc.identity`.
    activation : Union[str, Callable], optional
        Reservoir activation function.
        - If a str, should be a :py:mod:`reservoirpy.activationfunc`
        function name.
        - If a callable, should be an element-wise operator on ndarray.
        By default, :py:func:`activationfunc.tanh`.
    equation : {"internal", "external"}, optional
        - If "internal", then leaky integration happens on states transformed
        by the activation function:

        .. math::

            r[n+1] = (1 - \\alpha) \\cdot r[t] + \\alpha
             \\cdot f(W_{in} \\cdot u[n] + W \\cdot r[t])

        - If "external", then leaky integration happens on internal states of
        each neuron, stored in an ``internal_state`` parameter (:math:`x` in
        the equation below).
        A neuron internal state is the value of its state before applying
        the activation function :math:`f`:

        .. math::

            x[n+1] &= (1 - \\alpha) \\cdot x[t] \\\\
                   &+ \\alpha \\cdot f(W_{in} \\cdot u[n] + W \\cdot r[t]) \\\\
            r[n+1] &= f(x[n+1])

        By default, "internal".
    name : str, optional
        Node name, by default None
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for noise generation, by default None

    References
    ----------

        .. [1] Triesch, J. (2005). A Gradient Rule for the Plasticity of a
               Neuron’s Intrinsic Excitability. In W. Duch, J. Kacprzyk,
               E. Oja, & S. Zadrożny (Eds.), Artificial Neural Networks:
               Biological Inspirations – ICANN 2005 (pp. 65–70).
               Springer. https://doi.org/10.1007/11550822_11


        .. [2] Schrauwen, B., Wardermann, M., Verstraeten, D., Steil, J. J.,
               & Stroobandt, D. (2008). Improving reservoirs using intrinsic
               plasticity. Neurocomputing, 71(7), 1159–1171.
               https://doi.org/10.1016/j.neucom.2007.12.020

    """

    def __init__(
        self,
        units: int = None,
        sr: Optional[float] = None,
        input_bias: bool = True,
        noise_rc: float = 0.0,
        noise_in: float = 0.0,
        noise_fb: float = 0.0,
        noise_type: str = "normal",
        input_scaling: Optional[float] = 1.0,
        fb_scaling: Optional[float] = 1.0,
        input_connectivity: Optional[float] = 0.1,
        rc_connectivity: Optional[float] = 0.1,
        fb_connectivity: Optional[float] = 0.1,
        Win: Union[Weights, Callable] = generate_input_weights,
        W: Union[Weights, Callable] = generate_internal_weights,
        Wfb: Union[Weights, Callable] = generate_input_weights,
        fb_dim: int = None,
        fb_activation: Union[str, Callable] = identity,
        activation: str = "tanh",
        name=None,
        seed=None,
    ):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not "
                "a matrix."
            )

        if activation not in ["tanh", "sigmoid"]:
            raise ValueError(
                f"Activation '{activation}' must be 'tanh' or 'sigmoid' when "
                "appliying intrinsic plasticity."
            )

        super(IPReservoir, self).__init__(
            fb_initializer=partial(
                initialize_feedback,
                Wfb_init=Wfb,
                fb_scaling=fb_scaling,
                fb_connectivity=fb_connectivity,
                fb_dim=fb_dim,
                seed=seed,
            ),
            params={
                "W": None,
                "Win": None,
                "Wfb": None,
                "bias": None,
                "a": None,
                "b": None,
                "internal_state": None,
            },
            hypers={
                "sr": sr,
                "input_scaling": input_scaling,
                "fb_scaling": fb_scaling,
                "rc_connectivity": rc_connectivity,
                "input_connectivity": input_connectivity,
                "fb_connectivity": fb_connectivity,
                "noise_in": noise_in,
                "noise_rc": noise_rc,
                "noise_out": noise_fb,
                "noise_type": noise_type,
                "activation": activation,
                "fb_activation": fb_activation,
                "units": units,
                "noise_generator": partial(noise, seed=seed),
            },
            forward=forward,
            initializer=partial(
                initialize,
                sr=sr,
                input_scaling=input_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity,
                W_init=W,
                Win_init=Win,
                input_bias=input_bias,
                seed=seed,
            ),
            output_dim=units,
            name=name,
        )
