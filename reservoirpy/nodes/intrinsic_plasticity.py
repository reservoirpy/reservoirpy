# Author: Nathan Trouvain at 16/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from functools import partial
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from reservoirpy.utils.data_validation import check_node_input

from ..activationsfunc import get_function
from ..mat_gen import bernoulli, uniform
from ..node import TrainableNode
from ..type import (
    NodeInput,
    State,
    Timeseries,
    Timestep,
    Weights,
    is_array,
    is_multiseries,
)
from ..utils.random import rand_generator


class IPReservoir(TrainableNode):
    """Pool of neurons with random recurrent connexions, tuned using Intrinsic
    Plasticity.

    Intrinsic Plasticity is applied as described in [1]_ and [2]_.

    Reservoir neurons states, gathered in a vector :math:`\\mathbf{x}`, follow
    the update rule below:

    .. math::


        \\mathbf{r}[t+1] = (1 - \\mathrm{lr}) * \\mathbf{r}[t] + \\mathrm{lr}
        * (\\mathbf{W}_{in} \\cdot \\mathbf{u}[t+1]
         + \\mathbf{W} \\cdot \\mathbf{x}[t]
        + \\mathbf{b}_{in})

    .. math::

        \\mathbf{x}[t+1] = f(\\mathbf{a}*\\mathbf{r}[t+1]+\\mathbf{b})

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
        - :math:`f` and :math:`g` are activation functions.

    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be inferred from
        the ``W`` matrix shape.
    sr : float, optional
        Spectral radius of recurrent weight matrix.
    lr : float or array-like of shape (units,), default to 1.0
        Neurons leak rate. Must be in :math:`[0, 1]`.
    mu : float, default to 0.0
        Mean of the target distribution.
    sigma : float, default to 1.0
        Variance of the target distribution.
    learning_rate : float, default to 5e-4
        Learning rate.
    epochs : int, default to 1
        Number of training iterations.
    input_scaling : float or array-like of shape (features,), default to 1.0.
        Input gain. An array of the same dimension as the inputs can be used to
        set up different input scaling for each feature.
    input_connectivity : float, default to 0.1
        Connectivity of input neurons, i.e. ratio of input neurons connected
        to reservoir neurons. Must be in :math:`]0, 1]`.
    rc_connectivity : float, default to 0.1
        Connectivity of recurrent weight matrix, i.e. ratio of reservoir
        neurons connected to other reservoir neurons, including themselves.
        Must be in :math:`]0, 1]`.
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
    activation : {"tanh", "sigmoid"}, default to "tanh"
        Reservoir units activation function.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for noise generation.
    name : str, optional
        Node name.

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
    >>> _ = reservoir.fit(x, warmup=100)
    >>> states = reservoir.run(x)

    .. plot:: ./api/intrinsic_plasticity_example.py

    """

    #: Recurrent weights matrix (:math:`\mathbf{W}`).
    W: Weights
    #: Input weights matrix (:math:`\mathbf{W}_{in}`).
    Win: Weights
    #: Bias vector (:math:`\mathbf{bias}`).
    bias: Weights
    #: Gain of reservoir activation (:math:`\mathbf{a}`).
    a: float
    #: Bias of reservoir activation (:math:`\mathbf{b}`).
    b: float
    #: Leaking rate (1.0 by default) (:math:`\mathrm{lr}`).
    lr: Union[float, np.ndarray]
    #: Spectral radius of W.
    sr: float
    #: Mean of the target distribution (0.0 by default) (:math:`\mu`).
    mu: float
    #: Variance of the target distribution (1.0 by default) (:math:`\sigma`).
    sigma: float
    #: Learning rate (5e-4 by default).
    learning_rate: float
    #: Number of epochs for training (1 by default).
    epochs: int
    #: Input scaling (float or array) (1.0 by default).
    input_scaling: Union[float, Sequence]
    #: Connectivity (or density) of W (0.1 by default).
    rc_connectivity: float
    #: Connectivity (or density) of Win (0.1 by default).
    input_connectivity: float
    #: Activation of the reservoir units (tanh by default) (:math:`f`).
    activation: Literal["tanh", "sigmoid"]
    #: Number of neuronal units in the reservoir.
    units: int
    #: A random state generator.
    rng: Generator

    def __init__(
        self,
        units: Optional[int] = None,
        sr: Optional[float] = None,
        lr: Union[float, np.ndarray] = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        learning_rate: float = 5e-4,
        epochs: int = 1,
        input_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = uniform,
        bias: Union[Weights, Callable] = bernoulli,
        activation: Literal["tanh", "sigmoid"] = "tanh",
        input_dim: Optional[int] = None,
        dtype: type = np.float64,
        seed: Optional[Union[int, np.random.Generator]] = None,
        name=None,
    ):

        self.units = units
        self.sr = sr
        self.lr = lr
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.Win = Win
        self.W = W
        self.bias = bias
        self.activation = get_function(activation)

        if activation == "tanh":
            self.gradient = partial(IPReservoir.gaussian_gradients, mu=mu, sigma=sigma)
        elif activation == "sigmoid":
            self.gradient = partial(IPReservoir.exp_gradients, mu=mu)
        else:
            raise ValueError(
                f"Activation '{activation}' must be 'tanh' or 'sigmoid' when " "applying intrinsic plasticity."
            )

        # set input_dim (if possible)
        if input_dim is not None and is_array(Win) and Win.shape[-1] != input_dim:
            raise ValueError(
                f"Both 'input_dim' and 'Win' are set but their dimensions doesn't "
                f"match: {input_dim} != {Win.shape[-1]}."
            )
        self.input_dim = input_dim
        if is_array(Win):
            self.input_dim = np.shape(Win)[-1]

        # set output_dim
        if units is None and not is_array(W):
            raise ValueError("'units' parameter must not be None if 'W' parameter is not " "a matrix.")
        if units is not None and is_array(W) and W.shape[-1] != units:
            raise ValueError(
                f"Both 'units' and 'W' are set but their dimensions doesn't match: " f"{units} != {W.shape[-1]}."
            )
        self.output_dim = units
        if is_array(W):
            self.output_dim = W.shape[-1]
            self.units = W.shape[-1]

        self.dtype = dtype
        self.rng = rand_generator(seed=seed)
        self.name = name

        self.a: np.ndarray
        self.b: np.ndarray
        self.state: State
        self.initialized = False

    def initialize(self, x: Union[NodeInput, Timestep], y: None = None):

        self._set_input_dim(x)

        [Win_rng, W_rng, bias_rng] = self.rng.spawn(3)

        if callable(self.Win):
            self.Win = self.Win(
                self.units,
                self.input_dim,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=Win_rng,
            )

        if callable(self.W):
            self.W = self.W(
                self.units,
                self.units,
                sr=self.sr,
                connectivity=self.rc_connectivity,
                dtype=self.dtype,
                seed=W_rng,
            )

        if callable(self.bias):
            self.bias = self.bias(
                self.units,
                connectivity=1.0,
                dtype=self.dtype,
                seed=bias_rng,
            )

        self.a = np.ones((self.output_dim,))
        self.b = np.zeros((self.output_dim,))

        self.state = dict(internal=np.zeros((self.output_dim,)), out=np.zeros((self.output_dim,)))

        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        W = self.W  # NxN
        Win = self.Win  # NxI
        bias = self.bias  # N or float
        f = self.activation
        lr = self.lr
        (internal, external) = state["internal"], state["out"]

        next_state = W @ external + Win @ x + bias
        next_state = (1 - lr) * internal + lr * next_state

        next_external = f(self.a * next_state + self.b)

        return {"internal": next_state, "out": next_external}

    def fit(self, x: NodeInput, y: None = None, warmup: int = 0) -> "IPReservoir":
        check_node_input(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x, y)

        for _epoch in range(self.epochs):
            if is_multiseries(x):
                for seq in x:
                    self.partial_fit(seq[warmup:])
            else:
                self.partial_fit(x[warmup:])

        return self

    def partial_fit(self, x: Timeseries):
        for u in x:
            post_state = self.step(u)
            pre_state = self.state["internal"]

            delta_a, delta_b = self.gradient(x=pre_state.T, y=post_state.T, a=self.a)
            self.a += self.learning_rate * delta_a
            self.b += self.learning_rate * delta_b

    def gaussian_gradients(x, y, a, mu, sigma):
        """KL loss gradients of neurons with tanh activation (~ Normal(mu, sigma))."""
        sig2 = sigma**2
        delta_b = -(-(mu / sig2) + (y / sig2) * (2 * sig2 + 1 - y**2 + mu * y))
        delta_a = (1 / a) + delta_b * x
        return delta_a, delta_b

    def exp_gradients(x, y, a, mu):
        """KL loss gradients of neurons with sigmoid activation
        (~ Exponential(lambda=1/mu))."""
        delta_b = 1 - (2 + (1 / mu)) * y + (y**2) / mu
        delta_a = (1 / a) + delta_b * x
        return delta_a, delta_b
