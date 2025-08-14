from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
from numpy.random import Generator

from reservoirpy.utils.data_validation import check_node_input

from ..activationsfunc import get_function, tanh
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


class LocalPlasticityReservoir(TrainableNode):
    """
    A reservoir that learns its recurrent weights W through a local
    learning rule selected by the ``learning_rule`` hyperparameter.

    Reservoir states are updated with the external equation:

    .. math::

        & r[t+1] = (1 - lr)*r[t] + lr*(W r[t] + W_{in} u[t+1] + bias) \\\\
        & x[t+1] = f(r[t+1])

    Where :math:`f` is the activation function.

    Then the local rule is applied each timestep to update W.

    .. math::

        W_{ij} \\leftarrow W_{ij} + \\Delta W_{ij}

    Supported rules:
      ``oja``:
        :math:`\\Delta W_{ij} = \\eta y (x - y W_{ij})`
      ``anti-oja`` [1]_ [2]_ [3]_ :
        :math:`\\Delta W_{ij} = - \\eta y (x - y W_{ij})`
      ``hebbian`` [4]_ :
        :math:`\\Delta W_{ij} = \\eta x y`
      ``anti-hebbian``:
        :math:`\\Delta W_{ij} = - \\eta x y`
      ``bcm`` [2]_ :
        :math:`\\Delta W_{ij} = \\eta x y (y - \\theta_{BCM})`

    Where :math:`x` represents the pre-synaptic state and :math:`y` represents
    the post-synaptic state of the neuron.

    For "`bcm`", you can set a threshold ``bcm_theta`` (default `0.0`).

    If ``synapse_normalization=True``, then after each local-rule update
    on a row i of W, the row is rescaled to unit L2 norm. [4]_



    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be inferred from
        the ``W`` matrix shape.
    local_rule : str, default to `oja`
        One of `"oja"`, `"anti-oja"`, `"hebbian"`, `"anti-hebbian"`, `"bcm"`.
    eta : float, default to 1e-3.
        Local learning rate for the weight update.
    bcm_theta : float, default to 0.0
        The threshold used in the "bcm" rule.
    synapse_normalization : bool, default to True
        If True, L2-normalize each row of W after its update.
    epochs : int, default to 1
        Number of training iterations.
    sr : float, default to 1.0
        Spectral radius of recurrent weight matrix.
    lr : float or array-like of shape (units,), default to 1.0
        Neurons leak rate. Must be in :math:`[0, 1]`.
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
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for noise generation.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    name : str, optional
        Node name.


    References
    ----------

    .. [1] Babinec, Š., & Pospíchal, J. (2007). Improving the prediction
           accuracy of echo state neural networks by anti-Oja’s learning.
           In International Conference on Artificial Neural Networks (pp. 19-28).
           Berlin, Heidelberg: Springer Berlin Heidelberg.
           https://doi.org/10.1007/978-3-540-74690-4_3

    .. [2] Yusoff, M. H., Chrol-Cannon, J., & Jin, Y. (2016). Modeling neural
           plasticity in echo state networks for classification and regression.
           Information Sciences, 364, 184-196.
           https://doi.org/10.1016/j.ins.2015.11.017

    .. [3] Morales, G. B., Mirasso, C. R., & Soriano, M. C. (2021). Unveiling
           the role of plasticity rules in reservoir computing. Neurocomputing,
           461, 705-715. https://doi.org/10.1016/j.neucom.2020.05.127

    .. [4] Wang, X., Jin, Y., & Hao, K. (2021). Synergies between synaptic and
           intrinsic plasticity in echo state networks. Neurocomputing,
           432, 32-43. https://doi.org/10.1016/j.neucom.2020.12.007

    Example
    -------
    >>> reservoir = LocalPlasticityReservoir(
    ...     units=100, sr=0.9, local_rule="hebbian",
    ...     eta=1e-3, epochs=5, synapse_normalization=True
    ... )
    >>> # Fit on data timeseries
    >>> reservoir.fit(X_data, warmup=10)
    >>> # Then run
    >>> states = reservoir.run(X_data)
    """

    #: Number of neuronal units in the reservoir.
    units: int
    #: Local learning rate for the weight update.
    eta: float
    #: The threshold used in the "bcm" rule.
    bcm_theta: float
    #: If True, L2-normalize each row of W after its update.
    synapse_normalization: bool
    #: Number of training iterations.
    epochs: int
    #: Leaking rate (1.0 by default) (:math:`\mathrm{lr}`).
    lr: float
    #: Spectral radius of ``W`` (optional).
    sr: float
    #: Input scaling (float or array) (1.0 by default).
    input_scaling: Union[float, Sequence]
    #: Connectivity (or density) of ``Win`` (0.1 by default).
    input_connectivity: float
    #: Connectivity (or density) of ``Wfb`` (0.1 by default).
    rc_connectivity: float
    #: Input weights matrix (:math:`\mathbf{W}_{in}`).
    Win: Weights
    #: Recurrent weights matrix (:math:`\mathbf{W}`).
    W: Weights
    #: Bias vector (:math:`\mathbf{b}`).
    bias: Weights
    #: Activation of the reservoir units (tanh by default) (:math:`f`).
    activation: Callable
    #: Type of matrices elements. By default, ``np.float64``.
    dtype: type
    #: A random state generator. Used for generating Win and W.
    rng: Generator

    def __init__(
        self,
        units: Optional[int] = None,
        # local rule choice
        local_rule: Literal[
            "oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"
        ] = "oja",
        eta: float = 1e-3,
        bcm_theta: float = 0.0,
        synapse_normalization: bool = False,
        epochs: int = 1,
        # standard reservoir params
        sr: float = 1.0,
        lr: float = 1.0,
        input_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = uniform,
        bias: Union[Weights, Callable] = bernoulli,
        activation: Union[str, Callable] = tanh,
        input_dim: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        dtype: type = np.float64,
        name: Optional[str] = None,
    ):
        self.units = units
        self.eta = eta
        self.bcm_theta = bcm_theta
        self.synapse_normalization = synapse_normalization
        self.epochs = epochs
        self.lr = lr
        self.sr = sr
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.Win = Win
        self.W = W
        self.bias = bias
        self.activation = get_function(activation)
        self.dtype = dtype
        self.rng = rand_generator(seed=seed)
        self.initialized = False
        self.name = name

        # set units / output_dim
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not a matrix."
            )
        if units is not None and is_array(W) and W.shape[-1] != units:
            raise ValueError(
                f"Both 'units' and 'W' are set but their dimensions doesn't match: "
                f"{units} != {W.shape[-1]}."
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

        # Validate local rule name
        local_rule = local_rule.lower()

        def oja(weights, pre_state, post_state):
            return eta * post_state * (pre_state - post_state * weights)

        def anti_oja(weights, pre_state, post_state):
            return -eta * post_state * (pre_state - post_state * weights)

        def hebbian(weights, pre_state, post_state):
            return eta * post_state * pre_state

        def anti_hebbian(weights, pre_state, post_state):
            return -eta * post_state * pre_state

        def bcm(weights, pre_state, post_state):
            return eta * post_state * (post_state - bcm_theta) * pre_state

        rules = {
            "oja": oja,
            "anti-oja": anti_oja,
            "hebbian": hebbian,
            "anti-hebbian": anti_hebbian,
            "bcm": bcm,
        }
        if not local_rule in rules:
            raise ValueError(
                f"Unknown learning rule '{local_rule}'. Choose from: "
                "['oja', 'anti-oja', 'hebbian', 'anti-hebbian', 'bcm']."
            )

        self.increment = rules[local_rule]

    def initialize(self, x: Optional[Union[NodeInput, Timestep]]):

        # set input_dim
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

        self.state = {
            "out": np.zeros((self.units,)),
            "internal": np.zeros((self.units,)),
        }

        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        W = self.W  # NxN
        Win = self.Win  # NxI
        bias = self.bias  # N or float
        f = self.activation
        lr = self.lr
        s = state["out"]

        next_state = W @ s + Win @ x + bias
        next_state = (1 - lr) * s + lr * next_state

        return {"internal": next_state, "out": f(next_state)}

    def fit(
        self, x: NodeInput, y: None = None, warmup: int = 0
    ) -> "LocalPlasticityReservoir":
        check_node_input(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x)

        increment = self.increment
        do_norm = self.synapse_normalization

        def _local_synaptic_plasticity(seq: Timeseries):
            """
            Apply the local learning rule (Oja, Anti-Oja, Hebbian, Anti-Hebbian, BCM)
            to update the recurrent weight matrix W.

            If `synapse_normalization=True`, then each row of W is L2-normalized
            immediately after the local rule update.
            """
            for u in seq:
                pre_state = self.state["internal"]  # (units,)
                post_state = self._step(self.state, u)["internal"]  # (units,)
                # Vectorized update of nonzero elements based on the chosen rule.
                (rows, cols, data) = sp.find(self.W)
                self.W[rows, cols] += increment(data, pre_state[cols], post_state[rows])
                # Optionally normalize each row.
                if do_norm:
                    # Compute the L2 norm per row for the updated data.
                    row_norms = np.sqrt(np.sum(self.W**2, axis=1)).reshape(-1, 1)
                    safe_norms = np.where(row_norms > 0, row_norms, 1)
                    self.W[:] /= safe_norms[:]

        for _epoch in range(self.epochs):
            if is_multiseries(x):
                for seq in x:
                    _local_synaptic_plasticity(seq[warmup:])
            else:
                _local_synaptic_plasticity(x[warmup:])
        return self
