from functools import partial
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp

from ..._base import check_xy
from ...activationsfunc import get_function, identity, tanh
from ...mat_gen import bernoulli, uniform
from ...node import Unsupervised, _init_with_sequences
from ...type import Weights
from ...utils.random import noise, rand_generator
from ...utils.validation import is_array
from .base import forward_external
from .base import initialize as initialize_base
from .base import initialize_feedback


def local_synaptic_plasticity(reservoir, pre_state, post_state):
    """
    Apply the local learning rule (Oja, Anti-Oja, Hebbian, Anti-Hebbian, BCM)
    to update the recurrent weight matrix W.

    If `synapse_normalization=True`, then each row of W is L2-normalized
    immediately after the local rule update.

    This version supports both dense and sparse matrices. For sparse matrices,
    the weight matrix is converted to LIL format for efficient row modifications.
    """

    W = reservoir.W  # Expecting W to be in CSR format.
    increment = reservoir.increment
    do_norm = reservoir.synapse_normalization

    # Extract presynaptic and postsynaptic vectors.
    x = pre_state[0]  # shape: (units,)
    y = post_state[0]  # shape: (units,)

    # Ensure W is in CSR format.
    if not sp.isspmatrix_csr(W):
        W = sp.csr_matrix(W)

    # Compute the row index for each nonzero element using np.repeat.
    # np.diff(W.indptr) gives the count of nonzeros in each row.
    rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    cols = W.indices
    data = W.data  # Update in place.

    # Vectorized update of nonzero elements based on the chosen rule.
    data += increment(data, x[cols], y[rows])

    # Optionally normalize each row.
    if do_norm:
        # Compute the L2 norm per row for the updated data.
        row_sums = np.bincount(rows, weights=data**2, minlength=W.shape[0])
        row_norms = np.sqrt(row_sums)
        safe_norms = np.where(row_norms > 0, row_norms, 1)
        data /= safe_norms[rows]

    return W


def sp_backward(reservoir, X=None, *args, **kwargs):
    """
    Offline learning method for the local-rule-based reservoir.
    """
    for epoch in range(reservoir.epochs):
        for seq in X:
            for u in seq:
                pre_state = reservoir.internal_state  # shape (1, units)
                post_state = reservoir.call(u.reshape(1, -1))  # shape (1, units)

                # Update W with the chosen local rule
                W_new = local_synaptic_plasticity(reservoir, pre_state, post_state)
                reservoir.set_param("W", W_new)


def initialize_synaptic_plasticity(reservoir, *args, **kwargs):
    """
    Custom initializer for the LocalRuleReservoir.
    Reuses the ESN-like initialization and sets the reservoir internal state to zeros.
    """

    initialize_base(reservoir, *args, **kwargs)


class LocalPlasticityReservoir(Unsupervised):
    """
    A reservoir that learns its recurrent weights W through a local
    learning rule selected by the 'learning_rule' hyperparameter.

    Reservoir states are updated with the external equation:

    .. math::

        r[t+1] = (1 - lr)*r[t] + lr*(W r[t] + Win u[t+1] + Wfb fb[t] + bias)
        x[t+1] = activation(r[t+1])

    Then the local rule is applied each timestep to update W.

    .. math::

        W_{ij} \\leftarrow W_{ij} + \\Delta W_{ij}

    Supported rules:
      `oja`:
        :math:`\\Delta W_{ij} = \\eta y (x - y W_{ij})`
      `anti-oja` [1]_ [2]_ [3]_ :
        :math:`\\Delta W_{ij} = - \\eta y (x - y W_{ij})`
      `hebbian` [4]_ :
        :math:`\\Delta W_{ij} = \\eta x y`
      `anti-hebbian`:
        :math:`\\Delta W_{ij} = - \\eta x y`
      `bcm` [2]_ :
        :math:`\\Delta W_{ij} = \\eta x y (y - \\theta_{BCM})`

    Where :math:`x` represents the pre-synaptic state and :math:`y` represents
    the post-synaptic state of the neuron.

    For "bcm", you can set a threshold 'bcm_theta' (default `0.0`).

    If `synapse_normalization=True`, then after each local-rule update
    on a row i of W, the row is rescaled to unit L2 norm. [4]_



    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be inferred from
        the ``W`` matrix shape.
    local_rule : str, default to `oja`
        One of `"oja"`, `"anti-oja"`, `"hebbian"`, `"anti-hebbian"`, `"bcm"`.
    bcm_theta : float, default to 0.0
        The threshold used in the "bcm" rule.
    eta : float, default to 1e-3.
        Local learning rate for the weight update.
    synapse_normalization : bool, default to True
        If True, L2-normalize each row of W after its update.

    Other standard reservoir parameters:
      - units, sr, lr, epochs, ...
      - input_bias, noise_in, noise_rc, ...
      - input_scaling, rc_connectivity, ...
      - W, Win, Wfb initializers, etc.

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

    def __init__(
        self,
        units: Optional[int] = None,
        # local rule choice
        local_rule: str = "oja",
        eta: float = 1e-3,
        bcm_theta: float = 0.0,
        synapse_normalization: bool = False,
        epochs: int = 1,
        # standard reservoir params
        sr: Optional[float] = None,
        lr: float = 1.0,
        input_bias: bool = True,
        noise_rc: float = 0.0,
        noise_in: float = 0.0,
        noise_fb: float = 0.0,
        noise_type: str = "normal",
        noise_kwargs: Optional[Dict] = None,
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
        feedback_dim: Optional[int] = None,
        fb_activation: Union[str, Callable] = identity,
        activation: Union[str, Callable] = tanh,
        name=None,
        seed=None,
        **kwargs,
    ):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not a matrix."
            )

        rng = rand_generator(seed=seed)
        noise_kwargs = dict() if noise_kwargs is None else noise_kwargs

        # Validate local rule name
        local_rule = local_rule.lower()

        if local_rule == "oja":

            def increment(weights, pre_state, post_state):
                return eta * post_state * (pre_state - post_state * weights)

        elif local_rule == "anti-oja":

            def increment(weights, pre_state, post_state):
                return -eta * post_state * (pre_state - post_state * weights)

        elif local_rule == "hebbian":

            def increment(weights, pre_state, post_state):
                return eta * post_state * pre_state

        elif local_rule == "anti-hebbian":

            def increment(weights, pre_state, post_state):
                return -eta * post_state * pre_state

        elif local_rule == "bcm":

            def increment(weights, pre_state, post_state):
                return eta * post_state * (post_state - bcm_theta) * pre_state

        else:
            raise ValueError(
                f"Unknown learning rule '{local_rule}'. Choose from: "
                "['oja', 'anti-oja', 'hebbian', 'anti-hebbian', 'bcm']."
            )

        self.increment = increment

        super(LocalPlasticityReservoir, self).__init__(
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
                "internal_state": None,
            },
            hypers={
                "local_rule": local_rule,
                "bcm_theta": bcm_theta,
                "eta": eta,
                "synapse_normalization": synapse_normalization,
                "sr": sr,
                "lr": lr,
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
                "activation": get_function(activation)
                if isinstance(activation, str)
                else activation,
                "fb_activation": get_function(fb_activation)
                if isinstance(fb_activation, str)
                else fb_activation,
                "units": units,
                "noise_generator": partial(noise, rng=rng, **noise_kwargs),
            },
            forward=forward_external,
            initializer=partial(
                initialize_synaptic_plasticity,
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
            backward=sp_backward,
            output_dim=units,
            feedback_dim=feedback_dim,
            name=name,
            **kwargs,
        )

    @property
    def fitted(self) -> bool:
        # For an unsupervised node that can always be updated,
        # we set `fitted = True` after first initialization/training.
        return True

    def partial_fit(
        self, X_batch, Y_batch=None, warmup=0, **kwargs
    ) -> "LocalPlasticityReservoir":
        """Partial offline fitting method (for batch training)."""
        X, _ = check_xy(self, X_batch, allow_n_inputs=False)
        X, _ = _init_with_sequences(self, X)

        self.initialize_buffers()

        for i in range(len(X)):
            X_seq = X[i]

            if X_seq.shape[0] <= warmup:
                raise ValueError(
                    f"Warmup set to {warmup} timesteps, "
                    f"but one timeseries is only {X_seq.shape[0]} long."
                )

            # Run warmup if specified
            if warmup > 0:
                self.run(X_seq[:warmup])

            self._partial_backward(self, X_seq[warmup:])

        return self
