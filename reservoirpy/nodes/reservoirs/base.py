# Author: Nathan Trouvain at 08/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from ...mat_gen import zeros
from ...utils.random import noise
from ...utils.validation import is_array


def reservoir_kernel(reservoir, u, r):
    """Reservoir base forward function.

    Computes: s[t+1] = W.r[t] + Win.(u[t] + noise) + Wfb.(y[t] + noise) + bias
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

    return np.array(pre_s)


def forward_internal(reservoir, x: np.ndarray) -> np.ndarray:
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

    s_next = (
        (1 - lr) * r
        + lr * f(reservoir_kernel(reservoir, u, r))
        + noise_gen(dist, r.shape, g_rc)
    )

    return s_next.T


def forward_external(reservoir, x: np.ndarray) -> np.ndarray:
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
    s = reservoir.internal_state.T

    s_next = (
        (1 - lr) * s
        + lr * reservoir_kernel(reservoir, u, r)
        + noise_gen(dist, r.shape, g_rc)
    )

    reservoir.set_param("internal_state", s_next.T)

    return f(s_next).T


def initialize(
    reservoir,
    x=None,
    sr=None,
    input_scaling=None,
    bias_scaling=None,
    input_connectivity=None,
    rc_connectivity=None,
    W_init=None,
    Win_init=None,
    bias_init=None,
    input_bias=None,
    seed=None,
    **kwargs,
):
    if x is not None:
        reservoir.set_input_dim(x.shape[1])

        dtype = reservoir.dtype
        dtype_msg = (
            "Data type {} not understood in {}. {} should be an array or a "
            "callable returning an array."
        )

        if is_array(W_init):
            W = W_init
            if W.shape[0] != W.shape[1]:
                raise ValueError(
                    "Dimension mismatch inside W: "
                    f"W is {W.shape} but should be "
                    f"a square matrix."
                )

            if W.shape[0] != reservoir.output_dim:
                reservoir._output_dim = W.shape[0]
                reservoir.hypers["units"] = W.shape[0]

        elif callable(W_init):
            W = W_init(
                reservoir.output_dim,
                reservoir.output_dim,
                sr=sr,
                proba=rc_connectivity,
                dtype=dtype,
                seed=seed,
            )
        else:
            raise ValueError(dtype_msg.format(str(type(W_init)), reservoir.name, "W"))

        reservoir.set_param("units", W.shape[0])
        reservoir.set_param("W", W.astype(dtype))

        out_dim = reservoir.output_dim

        Win_has_bias = False
        if is_array(Win_init):
            Win = Win_init

            msg = (
                f"Dimension mismatch in {reservoir.name}: Win input dimension is "
                f"{Win.shape[1]} but input dimension is {x.shape[1]}."
            )

            # is bias vector inside Win ?
            if Win.shape[1] == x.shape[1] + 1:
                if input_bias:
                    Win_has_bias = True
                else:
                    bias_msg = (
                        " It seems Win has a bias column, but 'input_bias' is False."
                    )
                    raise ValueError(msg + bias_msg)
            elif Win.shape[1] != x.shape[1]:
                raise ValueError(msg)

            if Win.shape[0] != out_dim:
                raise ValueError(
                    f"Dimension mismatch in {reservoir.name}: Win internal dimension "
                    f"is {Win.shape[0]} but reservoir dimension is {out_dim}"
                )

        elif callable(Win_init):
            Win = Win_init(
                reservoir.output_dim,
                x.shape[1],
                input_scaling=input_scaling,
                proba=input_connectivity,
                dtype=dtype,
                seed=seed,
            )
        else:
            raise ValueError(
                dtype_msg.format(str(type(Win_init)), reservoir.name, "Win")
            )

        if input_bias:
            if not Win_has_bias:
                if callable(bias_init):
                    bias = bias_init(
                        reservoir.output_dim,
                        1,
                        input_scaling=bias_scaling,
                        proba=input_connectivity,
                        dtype=dtype,
                        seed=seed,
                    )
                elif is_array(bias_init):
                    bias = bias_init
                    if bias.shape[0] != reservoir.output_dim or (
                        bias.ndim > 1 and bias.shape[1] != 1
                    ):
                        raise ValueError(
                            f"Dimension mismatch in {reservoir.name}: bias shape is "
                            f"{bias.shape} but should be {(reservoir.output_dim, 1)}"
                        )
                else:
                    raise ValueError(
                        dtype_msg.format(str(type(bias_init)), reservoir.name, "bias")
                    )
            else:
                bias = Win[:, :1]
                Win = Win[:, 1:]
        else:
            bias = zeros(reservoir.output_dim, 1, dtype=dtype)

        reservoir.set_param("Win", Win.astype(dtype))
        reservoir.set_param("bias", bias.astype(dtype))
        reservoir.set_param("internal_state", reservoir.zero_state())


def initialize_feedback(
    reservoir,
    feedback=None,
    Wfb_init=None,
    fb_scaling=None,
    fb_connectivity=None,
    seed=None,
):
    if reservoir.has_feedback:
        fb_dim = feedback.shape[1]
        reservoir.set_feedback_dim(fb_dim)

        if is_array(Wfb_init):
            Wfb = Wfb_init
            if not fb_dim == Wfb.shape[1]:
                raise ValueError(
                    "Dimension mismatch between Wfb and feedback "
                    f"vector in {reservoir.name}: Wfb is "
                    f"{Wfb.shape} "
                    f"and feedback is {(1, fb_dim)} "
                    f"({fb_dim} != {Wfb.shape[0]})"
                )

            if not Wfb.shape[0] == reservoir.output_dim:
                raise ValueError(
                    f"Dimension mismatch between Wfb and W in "
                    f"{reservoir.name}: Wfb is {Wfb.shape} and "
                    f"W is "
                    f"{reservoir.W.shape} ({Wfb.shape[1]} "
                    f"!= {reservoir.output_dim})"
                )

        elif callable(Wfb_init):
            Wfb = Wfb_init(
                reservoir.output_dim,
                fb_dim,
                input_scaling=fb_scaling,
                proba=fb_connectivity,
                seed=seed,
                dtype=reservoir.dtype,
            )
        else:
            raise ValueError(
                f"Data type {type(Wfb_init)} not understood "
                f"for matrix initializer 'Wfb_init' in "
                f"{reservoir.name}. Wfb should be an array "
                f"or a callable returning an array."
            )

        reservoir.set_param("Wfb", Wfb)
