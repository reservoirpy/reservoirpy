from typing import Callable, Optional, Sequence, Union

import numpy as np

from ..activationsfunc import get_function, tanh
from ..mat_gen import bernoulli, normal, orthogonal
from ..node import Node
from ..type import NodeInput, State, Timestep, Weights, is_array
from ..utils import random


class ES2N(Node):
    """Edge of Stability Echo State Network.

    As first described in [1]_.

    The Edge of Stability Echo State Network (:math:`ES^2N`) model is similar to the reservoir
    equation, but adds an orthogonal transformation to the recurrent part of the equation.

    Reservoir neurons states, gathered in a vector :math:`\\mathbf{x}`, follow
    the update rule below:

    .. math::

        \\mathbf{x}[t+1] = (1 - \\mathrm{\\beta}) \\mathbf{O} * \\mathbf{x}[t] + \\mathrm{\\beta}
         * f(\\mathbf{W}_{in} \\cdot \\mathbf{u}[t+1]
          + \\mathbf{W} \\cdot \\mathbf{x}[t])

    where:
        - :math:`\\mathbf{x}` is the output activation vector of the reservoir;
        - :math:`\\mathbf{u}` is the input timeseries;
        - :math:`f` and :math:`g` are activation functions.
        - :math:`\\mathrm{\\beta}` is the *proximity* hyperparameter.
        - :math:`\\mathbf{O}` is a randomly generated orthogonal matrix.


    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be inferred from
        the ``W`` matrix shape.
    proximity : float or array-like of shape (units,), default to 1.0
        Proximity parameter. Must be in :math:`[0, 1]`.
    sr : float, optional
        Spectral radius of recurrent weight matrix.
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
        Input weights matrix or initializer. If a callable (like a function) is used,
        then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    W : callable or array-like of shape (units, units), default to :py:func:`~reservoirpy.mat_gen.normal`
        Recurrent weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    O : callable or array-like of shape (units, units), default to :py:func:`~reservoirpy.mat_gen.orthogonal`
        Orthogonal matrix. If a callable (like a function) is used, then this function should
        accept any keywords parameters and at least two parameters that will be used to define
        the shape of the returned weight matrix.
    bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
        Bias weights vector or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    activation : str or callable, default to :py:func:`~reservoirpy.activationsfunc.tanh`
        Reservoir units activation function.
        - If a str, should be a :py:mod:`~reservoirpy.activationsfunc`
        function name.
        - If a callable, should be an element-wise operator on arrays.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for noise generation.
    name : str, optional
        Node name.

    Note
    ----

    If W, Win, bias or Wfb are initialized with an array-like matrix, then all
    initializers parameters such as spectral radius (``sr``) or input scaling
    (``input_scaling``) are ignored.
    See :py:mod:`~reservoirpy.mat_gen` for more information.

    Example
    -------

    >>> from reservoirpy.nodes import ES2N, Ridge
    >>> es2n = ES2N(100, proximity=0.2, sr=0.8) >> Ridge(ridge=1e-6)

    Using the :py:func:`~reservoirpy.datasets.mackey_glass` timeseries:

    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x, y = to_forecasting(mackey_glass(200), forecast=10)
    >>> states = es2n.fit(x, y)

    .. plot::

        from reservoirpy.nodes import ES2N
        reservoir = ES2N(100, proximity=0.2, sr=0.8)
        from reservoirpy.datasets import mackey_glass
        x = mackey_glass(200)
        states = reservoir.run(x)
        fig, ax = plt.subplots(6, 1, figsize=(7, 10), sharex=True)
        ax[0].plot(x)
        ax[0].grid()
        ax[0].set_title("Neuron states (on Mackey-Glass)")
        for i in range(1, 6):
            ax[i].plot(states[:, i], label=f"Neuron {i}")
            ax[i].legend()
            ax[i].grid()
        ax[-1].set_xlabel("Timesteps")

    References
    ==========

        .. [1] Ceni, A., & Gallicchio, C. (2024). Edge of stability echo state network.
               IEEE Transactions on Neural Networks and Learning Systems, 36(4), 7555-7564.
    """

    initialized: bool
    input_dim: Optional[int]
    output_dim: int
    name: Optional[str]
    state: State

    # params
    #: Number of neuronal units in the reservoir.
    units: int
    #: Leaking rate (1.0 by default) (:math:`\mathrm{lr}`).
    proximity: Union[float, np.ndarray]
    #: Spectral radius of ``W`` (optional).
    sr: float
    #: Input scaling (float or array) (1.0 by default).
    input_scaling: Union[float, Sequence]
    #: Connectivity (or density) of ``Win`` (0.1 by default).
    input_connectivity: float
    #: Connectivity (or density) of ``Wfb`` (0.1 by default).
    rc_connectivity: float
    #: Input weights matrix (:math:`\mathbf{W}_{in}`).
    Win: Union["Weights", Callable]
    #: Recurrent weights matrix (:math:`\mathbf{W}`).
    W: Union["Weights", Callable]
    #: Orthogonal matrix (:math:`\mathbf{O}`).
    O: Union["Weights", Callable]
    #: Bias vector (:math:`\mathbf{b}`).
    bias: Union["Weights", Callable, float]
    #: Type of matrices elements. By default, ``np.float64``.
    dtype: type
    #: Activation of the reservoir units (tanh by default) (:math:`f`).
    activation: Callable
    #: A random state generator. Used for generating Win and W.
    rng: np.random.Generator

    def __init__(
        self,
        units: Optional[int] = None,
        proximity: Union[float, np.ndarray] = 1.0,
        sr: float = 1.0,
        input_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        Win: Union["Weights", Callable] = bernoulli,
        W: Union["Weights", Callable] = normal,
        O: Union["Weights", Callable] = orthogonal,
        bias: Union["Weights", Callable, float] = 0.0,
        activation: Union[str, Callable] = tanh,
        input_dim: Optional[int] = None,
        dtype: type = np.float64,
        seed: Optional[Union[int, np.random.Generator]] = None,
        name: Optional[str] = None,
    ):

        self.proximity = proximity
        self.sr = sr
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.Win = Win
        self.W = W
        self.O = O
        self.bias = bias
        self.activation = get_function(activation)
        self.dtype = dtype
        self.rng = random.rand_generator(seed=seed)
        self.initialized = False
        self.name = name

        # set units / output_dim
        if units is None and not is_array(W):
            raise ValueError("'units' parameter must not be None if 'W' parameter is not a matrix.")
        if units is not None and is_array(W) and W.shape[-1] != units:
            raise ValueError(
                f"Both 'units' and 'W' are set but their dimensions doesn't match: " f"{units} != {W.shape[-1]}."
            )
        self.units = units if units is not None else W.shape[-1]
        self.output_dim = self.units

        # set input_dim (if possible)
        if input_dim is not None and is_array(Win) and Win.shape[-1] != input_dim:
            raise ValueError(
                f"Both 'input_dim' and 'Win' are set but their dimensions doesn't "
                f"match: {input_dim} != {Win.shape[-1]}."
            )
        self.input_dim = Win.shape[-1] if is_array(Win) else input_dim

    def initialize(self, x: Optional[Union[NodeInput, Timestep]]):

        # set input_dim
        self._set_input_dim(x)

        [Win_rng, W_rng, bias_rng, O_rng] = self.rng.spawn(4)

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

        if callable(self.O):
            self.O = self.O(
                self.units,
                self.units,
                dtype=self.dtype,
                seed=O_rng,
            )

        if callable(self.bias):
            self.bias = self.bias(
                self.units,
                connectivity=1.0,
                dtype=self.dtype,
                seed=bias_rng,
            )

        self.state = {"out": np.zeros((self.units,))}

        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        W = self.W  # NxN
        O = self.O  # NxN
        Win = self.Win  # NxI
        bias = self.bias  # N or float
        f = self.activation
        beta = self.proximity
        s = state["out"]

        next_state = f(W @ s + Win @ x + bias)
        next_state = (1 - beta) * O @ s + beta * next_state

        return {"out": next_state}
