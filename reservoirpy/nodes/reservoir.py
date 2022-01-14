# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial
from typing import Callable, Optional, Union

import numpy as np

from ..activationsfunc import identity, tanh
from ..mat_gen import generate_input_weights, generate_internal_weights
from reservoirpy.node import Node
from ..types import Weights
from ..utils.validation import is_array
from ..utils.random import noise


def _reservoir_kernel(reservoir, u, r):
    """
    Reservoir base forward function. Computes:
    s[t+1] = W.r[t] + Win.(u[t] + noise) + Wfb.(y[t] + noise) + bias
    """
    W = reservoir.W
    Win = reservoir.Win
    bias = reservoir.bias

    g_in = reservoir.noise_in
    dist = reservoir.noise_type
    noise_gen = reservoir.noise_generator

    pre_s = W @ r + Win @ (u + noise(dist, u.shape, g_in)) + bias

    if reservoir.has_feedback:
        Wfb = reservoir.Wfb
        g_fb = reservoir.noise_out
        h = reservoir.fb_activation

        y = reservoir.feedback().reshape(-1, 1)
        y = h(y) + noise_gen(dist, y.shape, g_fb)

        pre_s += Wfb @ y

    return pre_s


def forward_internal(reservoir: "Reservoir", x: np.ndarray) -> np.ndarray:
    """Reservoir with internal activation function:

    .. math::

        r[n+1] = (1 - \\alpha) \\cdot r[t] + \\alpha
         \\cdot f (W_{in} \\cdot u[n] + W \\cdot r[t])


    where :math:`r[n]` is the state and the output of the reservoir."""
    lr = reservoir.lr
    f = reservoir.activation
    dist = reservoir.noise_type
    g_rc = reservoir.noise_rc
    noise_gen = reservoir.noise_generator

    u = x.reshape(-1, 1)
    r = reservoir.state().T

    s_next = (1 - lr) * r \
             + lr * f(_reservoir_kernel(reservoir, u, r)) \
             + noise_gen(dist, r.shape, g_rc)

    return s_next.T


def forward_external(reservoir: "Reservoir", x: np.ndarray) -> np.ndarray:
    """Reservoir with external activation function:

    .. math::

        x[n+1] = (1 - \\alpha) \\cdot x[t] + \\alpha
         \\cdot f (W_{in} \\cdot u[n] + W \\cdot r[t])

        r[n+1] = f(x[n+1])


    where :math:`x[n]` is the internal state of the reservoir and :math:`r[n]`
    is the response of the reservoir."""
    lr = reservoir.lr
    f = reservoir.activation
    dist = reservoir.noise_type
    g_rc = reservoir.noise_rc
    noise_gen = reservoir.noise_generator

    u = x.reshape(-1, 1)
    r = reservoir.state().T
    s = reservoir.internal_state

    s_next = (1 - lr) * s \
             + lr * _reservoir_kernel(reservoir, u, r) \
             + noise_gen(dist, r.shape, g_rc)

    reservoir.set_param("internal_state", s_next)

    return f(s_next).T


def initialize(reservoir,
               x=None,
               sr=None,
               input_scaling=None,
               input_connectivity=None,
               rc_connectivity=None,
               W_init=None,
               Win_init=None,
               bias_init=None,
               input_bias=None,
               seed=None,
               **kwargs):
    if x is not None:
        reservoir.set_input_dim(x.shape[1])

        if is_array(W_init):
            W = W_init
            if W.shape[0] != W.shape[1]:
                raise ValueError("Dimension mismatch inside W: "
                                 f"W is {W.shape} but should be "
                                 f"a square matrix.")

            if W.shape[0] != reservoir.output_dim:
                reservoir._output_dim = W.shape[0]

        elif callable(W_init):
            W = W_init(N=reservoir.output_dim, sr=sr,
                       proba=rc_connectivity, seed=seed)
        else:
            raise ValueError(f"Data type {type(W_init)} not "
                             f"understood for matrix initializer "
                             f"'W_init' in {reservoir.name}. W "
                             f"should be an array or a callable "
                             f"returning an array.")

        reservoir.set_param("units", W.shape[0])
        reservoir.set_param("W", W)

        out_dim = reservoir.output_dim

        if is_array(Win_init):
            Win = Win_init
            bias_dim = 1 if input_bias else 0
            bias_msg = "+ 1 (bias)" if input_bias else ""
            if Win.shape[1] != x.shape[1] + bias_dim:
                raise ValueError("Dimension mismatch between Win and input "
                                 f"vector in {reservoir.name}: Win is "
                                 f"{Win.shape} "
                                 f"and input is {x.shape} ({x.shape[1]} "
                                 f"{bias_msg} "
                                 f"!= {Win.shape[0] - bias_dim} {bias_msg})")

            if Win.shape[0] != out_dim:
                raise ValueError(f"Dimension mismatch between Win and W in "
                                 f"{reservoir.name}: "
                                 f"Win is {Win.shape} and W is "
                                 f"{(out_dim, out_dim)}"
                                 f" ({Win.shape[1]} != {out_dim})")

        elif callable(Win_init):
            Win = Win_init(N=reservoir.output_dim, dim_input=x.shape[1],
                           input_bias=False, input_scaling=input_scaling,
                           proba=input_connectivity, seed=seed)
        else:
            raise ValueError(f"Data type {type(Win_init)} not "
                             f"understood for matrix initializer "
                             f"'Win_init' in {reservoir.name}. Win "
                             f"should be an array or a callable returning "
                             f"an array.")

        if input_bias:
            if callable(Win_init):  # TODO: allow bias initializers
                bias = Win_init(N=reservoir.output_dim, dim_input=1,
                                input_bias=False, input_scaling=input_scaling,
                                proba=input_connectivity, seed=seed)
            elif is_array(Win_init):
                bias = Win[:, :1]
                Win = Win[:, 1:]
        else:
            bias = np.zeros((reservoir.output_dim, 1))

        reservoir.set_param("Win", Win)
        reservoir.set_param("bias", bias)
        reservoir.set_param("internal_state", reservoir.zero_state().T)


def initialize_feedback(reservoir,
                        feedback=None,
                        Wfb_init=None,
                        fb_scaling=None,
                        fb_connectivity=None,
                        fb_dim: int = None,
                        seed=None):
    if reservoir.has_feedback:
        fb_dim = feedback.shape[1]
        reservoir.set_feedback_dim(fb_dim)
    elif fb_dim is not None:
        reservoir.set_feedback_dim(fb_dim)
    else:
        reservoir.set_feedback_dim(0)

    if fb_dim is not None:
        if is_array(Wfb_init):
            Wfb = Wfb_init
            if not fb_dim == Wfb.shape[1]:
                raise ValueError("Dimension mismatch between Wfb and feedback "
                                 f"vector in {reservoir.name}: Wfb is "
                                 f"{Wfb.shape} "
                                 f"and feedback is {(1, fb_dim)} "
                                 f"({fb_dim} != {Wfb.shape[0]})")

            if not Wfb.shape[0] == reservoir.output_dim:
                raise ValueError(f"Dimension mismatch between Wfb and W in "
                                 f"{reservoir.name}: Wfb is {Wfb.shape} and "
                                 f"W is "
                                 f"{reservoir.W.shape} ({Wfb.shape[1]} "
                                 f"!= {reservoir.output_dim})")

        elif callable(Wfb_init):
            Wfb = Wfb_init(N=reservoir.output_dim, dim_input=fb_dim,
                           input_bias=False, input_scaling=fb_scaling,
                           proba=fb_connectivity, seed=seed)
        else:
            raise ValueError(f"Data type {type(Wfb_init)} not understood "
                             f"for matrix initializer 'Wfb_init' in "
                             f"{reservoir.name}. Wfb should be an array "
                             f"or a callable returning an array.")

        reservoir.set_param("Wfb", Wfb)


class Reservoir(Node):
    """
    Pool of leaky-integrator neurons with random recurrent connexions.

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

            x[n+1] &= (1 - \\alpha) \\cdot x[t] + \\alpha \\cdot f(W_{in} \\cdot u[n] + W \\cdot r[t]) \\\\
            r[n+1] &= f(x[n+1])

        By default, "internal".
    name : str, optional
        Node name, by default None
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for noise generation, by default None
    """

    def __init__(self,
                 units: int = None,
                 lr: float = 1.0,
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
                 activation: Union[str, Callable] = tanh,
                 equation: str = "internal",
                 name=None,
                 seed=None):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not "
                "a matrix.")

        if equation == "internal":
            forward = forward_internal
        elif equation == "external":
            forward = forward_external
        else:
            raise ValueError("'equation' parameter must be either 'internal' "
                             "or 'external'.")

        super(Reservoir, self).__init__(
            fb_initializer=partial(initialize_feedback,
                                   Wfb_init=Wfb,
                                   fb_scaling=fb_scaling,
                                   fb_connectivity=fb_connectivity,
                                   fb_dim=fb_dim,
                                   seed=seed),
            params={"W": None, "Win": None, "Wfb": None, "bias": None,
                    "internal_state": None},
            hypers={"lr": lr,
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
                    "noise_generator": partial(noise, seed=seed)},
            forward=forward,
            initializer=partial(initialize,
                                sr=sr,
                                input_scaling=input_scaling,
                                input_connectivity=input_connectivity,
                                rc_connectivity=rc_connectivity,
                                W_init=W,
                                Win_init=Win,
                                input_bias=input_bias,
                                seed=seed),
            output_dim=units,
            name=name)
