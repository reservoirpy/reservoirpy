# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

from functools import partial
from typing import Callable, Optional, Sequence, Union

import numpy as np

from ..._base import check_xy
from ...activationsfunc import get_function, identity
from ...mat_gen import bernoulli, uniform
from ...node import Unsupervised, _init_with_sequences
from ...type import Weights
from ...utils.random import noise
from ...utils.validation import is_array
from .base import forward_external
from .base import initialize as initialize_base
from .base import initialize_feedback


def gaussian_gradients(x, y, a, mu, sigma, eta):
    """KL loss gradients of neurons with tanh activation (~ Normal(mu, sigma))."""
    sig2 = sigma**2
    delta_b = -eta * (-(mu / sig2) + (y / sig2) * (2 * sig2 + 1 - y**2 + mu * y))
    delta_a = (eta / a) + delta_b * x
    return delta_a, delta_b


def exp_gradients(x, y, a, mu, eta):
    """KL loss gradients of neurons with sigmoid activation
    (~ Exponential(lambda=1/mu))."""
    delta_b = eta * (1 - (2 + (1 / mu)) * y + (y**2) / mu)
    delta_a = (eta / a) + delta_b * x
    return delta_a, delta_b


def apply_gradients(a, b, delta_a, delta_b):
    """Apply gradients on a and b parameters of intrinsic plasticity."""
    a2 = a + delta_a
    b2 = b + delta_b
    return a2, b2


def ip(reservoir, pre_state, post_state):
    """Perform one step of intrinsic plasticity.

    Optimize a and b such that
    post_state = f(a*pre_state+b) ~ Dist(params) where Dist can be normal or
    exponential."""
    a = reservoir.a
    b = reservoir.b
    mu = reservoir.mu
    eta = reservoir.learning_rate

    if reservoir.activation_type == "tanh":
        sigma = reservoir.sigma
        delta_a, delta_b = gaussian_gradients(
            x=pre_state.T, y=post_state.T, a=a, mu=mu, sigma=sigma, eta=eta
        )
    else:  # sigmoid
        delta_a, delta_b = exp_gradients(
            x=pre_state.T, y=post_state.T, a=a, mu=mu, eta=eta
        )

    return apply_gradients(a=a, b=b, delta_a=delta_a, delta_b=delta_b)


def ip_activation(state, *, reservoir, f):
    """Activation of neurons f(a*x+b) where a and b are intrinsic plasticity
    parameters."""
    a, b = reservoir.a, reservoir.b
    return f(a * state + b)


def backward(reservoir: "IPReservoir", X=None, *args, **kwargs):
    for e in range(reservoir.epochs):
        for seq in X:
            for u in seq:
                post_state = reservoir.call(u.reshape(1, -1))
                pre_state = reservoir.internal_state

                a, b = ip(reservoir, pre_state, post_state)

                reservoir.set_param("a", a)
                reservoir.set_param("b", b)


def initialize(reservoir, *args, **kwargs):

    initialize_base(reservoir, *args, **kwargs)

    a = np.ones((reservoir.output_dim, 1))
    b = np.zeros((reservoir.output_dim, 1))

    reservoir.set_param("a", a)
    reservoir.set_param("b", b)


class IPReservoir(Unsupervised):
    """Pool of neurons with random recurrent connexions, tuned using Intrinsic
    Plasticity.

    Intrinisc Plasticity is applied as described in [1]_ and [2]_.

    Reservoir neurons states, gathered in a vector :math:`\\mathbf{x}`, follow
    the update rule below:

    .. math::


        \\mathbf{r}[t+1] = (1 - \\mathrm{lr}) * \\mathbf{r}[t] + \\mathrm{lr}
        * (\\mathbf{W}_{in} \\cdot (\\mathbf{u}[t+1]+c_{in}*\\xi)
         + \\mathbf{W} \\cdot \\mathbf{x}[t]
        + \\mathbf{W}_{fb} \\cdot (g(\\mathbf{y}[t])+c_{fb}*\\xi) + \\mathbf{b}_{in})

    .. math::

        \\mathbf{x}[t+1] = f(\\mathbf{a}*\\mathbf{r}[t+1]+\\mathbf{b}) + c * \\xi

    Parameters :math:`\\mathbf{a}` and :math:`\\mathbf{b}` are updated following two
    different rules:

    - **1.** Neuron activation is tanh:

    In that case, output distribution should be a Gaussian distribution of parameters
    (:math:`\\mu`, :math:`\\sigma`). The learning rule to obtain this output
    distribution is described in [2]_.

    - **2.** Neuron activation is sigmoid:

    In that case, output distribution should be an exponential distribution of
    parameter :math:`\\mu = \\frac{1}{\\lambda}`.
    The learning rule to obtain this output distribution is described in [1]_ and [2]_.

    where:
        - :math:`\\mathbf{x}` is the output activation vector of the reservoir;
        - :math:`\\mathbf{r}` is the internal activation vector of the reservoir;
        - :math:`\\mathbf{u}` is the input timeseries;
        - :math:`\\mathbf{y}` is a feedback vector;
        - :math:`\\xi` is a random noise;
        - :math:`f` and :math:`g` are activation functions.

    :py:attr:`IPReservoir.params` **list:**

    ================== =================================================================
    ``W``              Recurrent weights matrix (:math:`\\mathbf{W}`).
    ``Win``            Input weights matrix (:math:`\\mathbf{W}_{in}`).
    ``Wfb``            Feedback weights matrix (:math:`\\mathbf{W}_{fb}`).
    ``bias``           Input bias vector (:math:`\\mathbf{b}_{in}`).
    ``inernal_state``  Internal state (:math:`\\mathbf{r}`).
    ``a``              Gain of reservoir activation (:math:`\\mathbf{a}`).
    ``b``              Bias of reservoir activation (:math:`\\mathbf{b}`).
    ================== =================================================================

    :py:attr:`IPReservoir.hypers` **list:**

    ======================= ========================================================
    ``lr``                  Leaking rate (1.0 by default) (:math:`\\mathrm{lr}`).
    ``sr``                  Spectral radius of ``W`` (optional).
    ``mu``                  Mean of the target distribution (0.0 by default) (:math:`\\mu`).
    ``sigma``               Variance of the target distribution (1.0 by default) (:math:`\\sigma`).
    ``learning_rate``       Learning rate (5e-4 by default).
    ``epochs``              Number of epochs for training (1 by default).
    ``input_scaling``       Input scaling (float or array) (1.0 by default).
    ``fb_scaling``          Feedback scaling (float or array) (1.0 by default).
    ``rc_connectivity``     Connectivity (or density) of ``W`` (0.1 by default).
    ``input_connectivity``  Connectivity (or density) of ``Win`` (0.1 by default).
    ``fb_connectivity``     Connectivity (or density) of ``Wfb`` (0.1 by default).
    ``noise_in``            Input noise gain (0 by default) (:math:`c_{in} * \\xi`).
    ``noise_rc``            Reservoir state noise gain (0 by default) (:math:`c*\\xi`).
    ``noise_fb``            Feedback noise gain (0 by default) (:math:`c_{fb}*\\xi`).
    ``noise_type``          Distribution of noise (normal by default) (:math:`\\xi\\sim\\mathrm{Noise~type}`).
    ``activation``          Activation of the reservoir units (tanh by default) (:math:`f`).
    ``fb_activation``       Activation of the feedback units (identity by default) (:math:`g`).
    ``units``               Number of neuronal units in the reservoir.
    ``noise_generator``     A random state generator.
    ======================= ========================================================

    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be infered from
        the ``W`` matrix shape.
    lr : float, default to 1.0
        Neurons leak rate. Must be in :math:`[0, 1]`.
    sr : float, optional
        Spectral radius of recurrent weight matrix.
    mu : float, default to 0.0
        Mean of the target distribution.
    sigma : float, default to 1.0
        Variance of the target distribution.
    learning_rate : float, default to 5e-4
        Learning rate.
    epochs : int, default to 1
        Number of training iterations.
    input_bias : bool, default to True
        If False, no bias is added to inputs.
    noise_rc : float, default to 0.0
        Gain of noise applied to reservoir activations.
    noise_in : float, default to 0.0
        Gain of noise applied to input inputs.
    noise_fb : float, default to 0.0
        Gain of noise applied to feedback signal.
    noise_type : str, default to "normal"
        Distribution of noise. Must be a Numpy random variable generator
        distribution (see :py:class:`numpy.random.Generator`).
    input_scaling : float or array-like of shape (features,), default to 1.0.
        Input gain. An array of the same dimension as the inputs can be used to
        set up different input scaling for each feature.
    bias_scaling: float, default to 1.0
        Bias gain.
    fb_scaling : float or array-like of shape (features,), default to 1.0
        Feedback gain. An array of the same dimension as the feedback can be used to
        set up different feedback scaling for each feature.
    input_connectivity : float, default to 0.1
        Connectivity of input neurons, i.e. ratio of input neurons connected
        to reservoir neurons. Must be in :math:`]0, 1]`.
    rc_connectivity : float, default to 0.1
        Connectivity of recurrent weight matrix, i.e. ratio of reservoir
        neurons connected to other reservoir neurons, including themselves.
        Must be in :math:`]0, 1]`.
    fb_connectivity : float, default to 0.1
        Connectivity of feedback neurons, i.e. ratio of feedabck neurons
        connected to reservoir neurons. Must be in :math:`]0, 1]`.
    Win : callable or array-like of shape (units, features), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
        Input weights matrix or initializer. If a callable (like a function) is
        used,
        then this function should accept any keywords
        parameters and at least two parameters that will be used to define the
        shape of
        the returned weight matrix.
    W : callable or array-like of shape (units, units), default to :py:func:`~reservoirpy.mat_gen.uniform`
        Recurrent weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the
        shape of
        the returned weight matrix.
    bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
        Bias weights vector or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the
        shape of
        the returned weight matrix.
    Wfb : callable or array-like of shape (units, feedback), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
        Feedback weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the
        shape of
        the returned weight matrix.
    fb_activation : str or callable, default to :py:func:`~reservoirpy.activationsfunc.identity`
        Feedback activation function.
        - If a str, should be a :py:mod:`~reservoirpy.activationsfunc`
        function name.
        - If a callable, should be an element-wise operator on arrays.
    activation : {"tanh", "sigmoid"}, default to "tanh"
        Reservoir units activation function.
    feedback_dim : int, optional
        Feedback dimension. Can be inferred at first call.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    name : str, optional
        Node name.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for noise generation.

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

    Example
    -------
    >>> from reservoirpy.nodes import IPReservoir
    >>> reservoir = IPReservoir(
    ...                 100, mu=0.0, sigma=0.1, sr=0.95, activation="tanh", epochs=10)

    We can fit the intrinsic plasticity parameters to reach a normal distribution
    of the reservoir activations.
    Using the :py:func:`~reservoirpy.datasets.narma` timeseries:

    >>> from reservoirpy.datasets import narma
    >>> x = narma(1000)
    >>> reservoir.fit(x, warmup=100)
    >>> states = reservoir.run(x)

    .. plot:: ./api/generated/intrinsic_plasticity_example.py

    """

    def __init__(
        self,
        units: int = None,
        sr: Optional[float] = None,
        lr: float = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        learning_rate: float = 5e-4,
        epochs: int = 1,
        input_bias: bool = True,
        noise_rc: float = 0.0,
        noise_in: float = 0.0,
        noise_fb: float = 0.0,
        noise_type: str = "normal",
        input_scaling: Union[float, Sequence] = 1.0,
        bias_scaling: float = 1.0,
        fb_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: Optional[float] = 0.1,
        rc_connectivity: Optional[float] = 0.1,
        fb_connectivity: Optional[float] = 0.1,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = uniform,
        Wfb: Union[Weights, Callable] = bernoulli,
        bias: Union[Weights, Callable] = bernoulli,
        feedback_dim: int = None,
        fb_activation: Union[str, Callable] = identity,
        activation: Literal["tanh", "sigmoid"] = "tanh",
        name=None,
        seed=None,
        **kwargs,
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
                "lr": lr,
                "mu": mu,
                "sigma": sigma,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "input_bias": input_bias,
                "input_scaling": input_scaling,
                "fb_scaling": fb_scaling,
                "rc_connectivity": rc_connectivity,
                "input_connectivity": input_connectivity,
                "fb_connectivity": fb_connectivity,
                "noise_in": noise_in,
                "noise_rc": noise_rc,
                "noise_out": noise_fb,
                "noise_type": noise_type,
                "activation_type": activation,
                "activation": partial(
                    ip_activation, reservoir=self, f=get_function(activation)
                ),
                "fb_activation": fb_activation,
                "units": units,
                "noise_generator": partial(noise, seed=seed),
            },
            forward=forward_external,
            initializer=partial(
                initialize,
                input_bias=input_bias,
                bias_scaling=bias_scaling,
                sr=sr,
                input_scaling=input_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity,
                W_init=W,
                Win_init=Win,
                bias_init=bias,
                seed=seed,
            ),
            backward=backward,
            output_dim=units,
            feedback_dim=feedback_dim,
            name=name,
            **kwargs,
        )

    # TODO: handle unsupervised learners with a specific attribute
    @property
    def fitted(self):
        return True

    def partial_fit(self, X_batch, Y_batch=None, warmup=0, **kwargs) -> "Node":
        """Partial offline fitting method of a Node.
        Can be used to perform batched fitting or to precompute some variables
        used by the fitting method.

        Parameters
        ----------
        X_batch : array-like of shape ([series], timesteps, features)
            A sequence or a batch of sequence of input data.
        Y_batch : array-like of shape ([series], timesteps, features), optional
            A sequence or a batch of sequence of teacher signals.
        warmup : int, default to 0
            Number of timesteps to consider as warmup and
            discard at the begining of each timeseries before training.

        Returns
        -------
            Node
                Partially fitted Node.
        """
        X, _ = check_xy(self, X_batch, allow_n_inputs=False)

        X, _ = _init_with_sequences(self, X)

        self.initialize_buffers()

        for i in range(len(X)):
            X_seq = X[i]

            if X_seq.shape[0] <= warmup:
                raise ValueError(
                    f"Warmup set to {warmup} timesteps, but one timeseries is only "
                    f"{X_seq.shape[0]} long."
                )

            if warmup > 0:
                self.run(X_seq[:warmup])

            self._partial_backward(self, X_seq[warmup:])

        return self
