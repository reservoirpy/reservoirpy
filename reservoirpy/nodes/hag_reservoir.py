# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np

from ..activationsfunc import get_function, tanh
from ..mat_gen import block_input, random_sparse, zeros
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
from ..utils.data_validation import check_node_input
from ..utils.random import rand_generator


class HAGReservoir(TrainableNode):
    """
    A reservoir that learns its recurrent connections through HAG [1]_, a
    homeostatic structural plasticity rule [2]_: during ``fit``, the neurons
    whose activity is too high lose incoming connections and the ones whose
    activity is too low gain incoming connections from the neurons most
    correlated with them.

    Reservoir states are updated with the equation:

    .. math::

        x[t+1] = (1 - lr) x[t] + lr f(W x[t] + W_{in} u[t+1] + b)

    where :math:`f` is the activation function. The weights are only changed
    by ``fit``: running the reservoir never changes them.

    During ``fit``, the reservoir is run on the inputs from a random state
    (uniform in :math:`[0, 1]`). After each window of :math:`T` timesteps
    (:math:`T` drawn among ``min_window`` ... ``max_window``, log-spaced, or one
    whole sequence if ``use_full_instance``), the activity of each neuron on the
    window is compared to the target:

    .. math::

        \\Delta z_i = (a_i - \\mathrm{target}) / \\mathrm{spread}

    where :math:`a_i` is the mean (``homeostasis="mean"``) or the standard
    deviation (``homeostasis="variance"``) of the state of neuron :math:`i` over
    the window. Then:

    - :math:`\\Delta z_i \\geq 1` (too active): one incoming connection of
      :math:`i`, drawn at random, is decreased by ``weight_increment`` (and
      removed when it reaches 0),
    - :math:`\\Delta z_i \\leq -1` (not active enough): the connection from the
      neuron :math:`j` most correlated with :math:`i` over the window (Pearson
      correlation), among the other neurons not active enough, is increased by
      ``weight_increment`` (created if absent).

    With ``homeostasis="variance"``, the incoming weights of the neurons
    saturated (state :math:`\\geq` ``intrinsic_saturation``) on the whole window
    are also multiplied by ``intrinsic_coef``.

    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be inferred from
        the ``W`` matrix shape.
    homeostasis : {"mean", "variance"}, default to "mean"
        Activity regulated by the plasticity: mean (mean HAG) or standard deviation
        (variance HAG) of the states.
    target : float
        Target activity (mean or standard deviation of the states).
    spread : float
        Tolerance around the target: connections change when the activity is more
        than ``spread`` away from it.
    weight_increment : float
        Weight added to (or removed from) a connection at each change.
    min_window : int
        Minimal number of timesteps between two plasticity steps.
    max_window : int, optional
        Maximal number of timesteps between two plasticity steps. If None,
        ``min_window`` is used.
    use_full_instance : bool, default to False
        If True and the data is a list of sequences, one plasticity step per
        sequence instead of windows of random lengths.
    max_partners : int, default to infinity
        Maximal number of incoming connections of a neuron: beyond it, only its
        existing connections are strengthened.
    intrinsic_saturation : float, default to 0.9
        Saturation threshold of the states (``homeostasis="variance"`` only).
    intrinsic_coef : float, default to 0.9
        Factor applied to the incoming weights of saturated neurons
        (``homeostasis="variance"`` only).
    lr : float or array-like of shape (units,), default to 1.0
        Neurons leak rate. Must be in :math:`[0, 1]`.
    input_scaling : float or array-like of shape (features,), default to 1.0.
        Input gain, used by the default ``Win`` initializer.
    bias_scaling : float, default to 1.0
        Bias gain, used by the default ``bias`` initializer.
    W : callable or array-like of shape (units, units), default to :py:func:`~reservoirpy.mat_gen.zeros`
        Initial recurrent weights matrix or initializer (HAG grows the connections
        from an empty matrix by default).
    Win : callable or array-like of shape (units, features), default to :py:func:`~reservoirpy.mat_gen.block_input`
        Input weights matrix or initializer. By default, each input feature drives
        its own block of ``units // features`` neurons.
    bias : callable or array-like of shape (units,), default to ``random_sparse(dist="foldnorm", c=1.0, scale=0.1)``
        Bias weights vector or initializer. By default, :math:`|\\mathcal{N}(0.1, 0.1)|`.
    activation : str or callable, default to :py:func:`~reservoirpy.activationsfunc.tanh`
        Reservoir units activation function.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for the initializers and the random choices of ``fit``.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    name : str, optional
        Node name.

    Notes
    -----
    **Reproducing the results of** [1]_. The experiments of [1]_ use the following
    configuration, the hyperparameters being optimized with Optuna (TPE sampler,
    cross-validation on the training set) within the given ranges:

    - Initial matrices (the default initializers): ``W``: :py:func:`~reservoirpy.mat_gen.zeros`
      (HAG grows the recurrent connections from an empty matrix); ``Win``:
      :py:func:`~reservoirpy.mat_gen.block_input` (uniform in :math:`[0, 1)` within each block), with
      ``input_scaling`` in :math:`[0.01, 0.2]`; ``bias``:
      ``random_sparse(dist="foldnorm", c=1.0, scale=0.1)``, i.e.
      :math:`|\\mathcal{N}(0.1, 0.1)|`, with ``bias_scaling`` in :math:`[0, 0.2]`.
    - Windows: ``min_window`` between 3 and the length :math:`L` of the longest
      training sequence (500 for timeseries prediction), ``max_window`` between
      ``min_window`` and :math:`5L`, and ``use_full_instance`` True or False
      (classification only).
    - Data: HAG is always given multivariate inputs, each feature driving its own
      block of neurons. Univariate signals are first decomposed into frequency
      bands: MFCCs for audio classification (hop length 50, window length 100) and
      short-time Fourier transform magnitudes for timeseries prediction (hop length
      1, Gaussian window of length 100). Inputs are then scaled to :math:`[0, 1]`
      (MinMax scaling fitted on the training set). HAG is fitted on up to 500
      training sequences drawn at random (on the whole training series for
      prediction).

    The reference implementation of HAG and the experiments of [1]_ are available at
    https://github.com/Finebouche/HAG.

    References
    ----------

    .. [1] Cazalets, T., & Dambre, J. (2026). Reshaping reservoirs with
           unsupervised Hebbian adaptation. Nature Communications, 17, 450.
           https://doi.org/10.1038/s41467-025-67137-1

    .. [2] Cazalets, T., & Dambre, J. (2023). An homeostatic activity-dependent
           structural plasticity algorithm for richer input combination.
           In 2023 International Joint Conference on Neural Networks (IJCNN)
           (pp. 1-8). IEEE. https://doi.org/10.1109/IJCNN54540.2023.10191230

    Example
    -------
    >>> from reservoirpy.nodes import HAGReservoir, Ridge
    >>> reservoir = HAGReservoir(
    ...     units=120, homeostasis="mean", target=0.8, spread=0.1,
    ...     weight_increment=0.05, min_window=5, max_window=50,
    ...     input_scaling=0.1, bias_scaling=0.1, seed=0,
    ... )
    >>> # Grow the connections on input timeseries (unsupervised)
    >>> reservoir.fit(X_data)
    >>> # Then run, each sequence starting from a zero state
    >>> reservoir.reset()
    >>> states = reservoir.run(X_train)
    >>> readout = Ridge(ridge=1e-6).fit(states, Y_train)
    """

    #: Number of neuronal units in the reservoir.
    units: int
    #: Activity regulated by the plasticity ("mean" or "variance").
    homeostasis: str
    #: Target activity (mean or standard deviation of the states).
    target: float
    #: Tolerance around the target activity.
    spread: float
    #: Weight added to (or removed from) a connection at each change.
    weight_increment: float
    #: Minimal number of timesteps between two plasticity steps.
    min_window: int
    #: Maximal number of timesteps between two plasticity steps.
    max_window: int
    #: If True, one plasticity step per sequence.
    use_full_instance: bool
    #: Maximal number of incoming connections of a neuron.
    max_partners: float
    #: Saturation threshold of the states (variance HAG).
    intrinsic_saturation: float
    #: Factor applied to the incoming weights of saturated neurons (variance HAG).
    intrinsic_coef: float
    #: Leaking rate (1.0 by default) (:math:`\mathrm{lr}`).
    lr: float
    #: Input scaling (float or array) (1.0 by default).
    input_scaling: Union[float, Sequence]
    #: Bias scaling (1.0 by default).
    bias_scaling: float
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
    #: A random state generator. Used for generating Win, W, bias and the random choices of ``fit``.
    rng: np.random.Generator
    #: Number of connections added by the last ``fit``.
    n_added: int
    #: Number of connections removed (weakened) by the last ``fit``.
    n_pruned: int

    def __init__(
        self,
        units: Optional[int] = None,
        # HAG
        homeostasis: Literal["mean", "variance"] = "mean",
        target: float = None,
        spread: float = None,
        weight_increment: float = None,
        min_window: int = None,
        max_window: Optional[int] = None,
        use_full_instance: bool = False,
        max_partners: float = np.inf,
        intrinsic_saturation: float = 0.9,
        intrinsic_coef: float = 0.9,
        # standard reservoir params
        lr: Union[float, np.ndarray] = 1.0,
        input_scaling: Union[float, Sequence] = 1.0,
        bias_scaling: float = 1.0,
        W: Union[Weights, Callable] = zeros,
        Win: Union[Weights, Callable] = block_input,
        bias: Union[Weights, Callable] = random_sparse(dist="foldnorm", c=1.0, scale=0.1),
        activation: Union[str, Callable] = tanh,
        input_dim: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        dtype: type = np.float64,
        name: Optional[str] = None,
    ):
        if homeostasis not in ("mean", "variance"):
            raise ValueError(f"Unknown homeostasis '{homeostasis}'. Choose from: ['mean', 'variance'].")
        missing = [
            name_
            for name_, value in dict(
                target=target, spread=spread, weight_increment=weight_increment, min_window=min_window
            ).items()
            if value is None
        ]
        if missing:
            raise ValueError(f"HAGReservoir needs {', '.join(missing)}.")

        self.homeostasis = homeostasis
        self.target = target
        self.spread = spread
        self.weight_increment = weight_increment
        self.min_window = min_window
        self.max_window = min_window if max_window is None else max_window
        self.use_full_instance = use_full_instance
        self.max_partners = max_partners
        self.intrinsic_saturation = intrinsic_saturation
        self.intrinsic_coef = intrinsic_coef
        self.lr = lr
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.Win = Win
        self.W = W
        self.bias = bias
        self.activation = get_function(activation) if isinstance(activation, str) else activation
        self.dtype = dtype
        self.rng = rand_generator(seed=seed)
        self.name = name
        self.n_added = 0
        self.n_pruned = 0

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

    def initialize(self, x: Optional[Union[NodeInput, Timestep]], y: None = None):

        # set input_dim
        self._set_input_dim(x)

        [Win_rng, W_rng, bias_rng, plasticity_rng] = self.rng.spawn(4)

        if callable(self.Win):
            self.Win = self.Win(
                self.units,
                self.input_dim,
                input_scaling=self.input_scaling,
                dtype=self.dtype,
                seed=Win_rng,
            )

        if callable(self.W):
            self.W = self.W(
                self.units,
                self.units,
                dtype=self.dtype,
                seed=W_rng,
            )

        if callable(self.bias):
            self.bias = self.bias(
                self.units,
                input_scaling=self.bias_scaling,
                dtype=self.dtype,
                seed=bias_rng,
            )

        # dense matrices: the plasticity changes W in place
        self.W = np.array(self.W.toarray() if hasattr(self.W, "toarray") else self.W, dtype=self.dtype)
        self.Win = np.asarray(self.Win.toarray() if hasattr(self.Win, "toarray") else self.Win, dtype=self.dtype)
        self.bias = np.ravel(self.bias.toarray() if hasattr(self.bias, "toarray") else self.bias).astype(self.dtype)

        # random choices of fit
        self._plasticity_rng = plasticity_rng

        self.state = {"out": np.zeros((self.units,))}

        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        W = self.W  # NxN
        Win = self.Win  # NxI
        bias = self.bias  # N
        f = self.activation
        lr = self.lr
        s = state["out"]

        next_state = f(W @ s + Win @ x + bias)
        next_state = (1 - lr) * s + lr * next_state

        return {"out": next_state}

    def _run_window(self, state: State, inputs: Timeseries) -> tuple[State, np.ndarray]:
        """Run the reservoir on a window of inputs: last state and states of the window, (timesteps, units)."""
        states = np.empty((len(inputs), self.units))
        for t, u in enumerate(inputs):
            state = self._step(state, u)
            states[t] = state["out"]
        return state, states

    def _activity_error(self, states: np.ndarray) -> np.ndarray:
        """:math:`\\Delta z` of each neuron: (activity - target) / spread, the activity being the mean or the
        standard deviation of its states over the window."""
        if self.homeostasis == "mean":
            return np.mean((states - self.target) / self.spread, axis=0)
        return (np.std(states, axis=0) - self.target) / self.spread

    def _new_partners(self, neurons: np.ndarray, states: np.ndarray) -> list:
        """(neuron, presynaptic neuron) of the new connections of the neurons not active enough, chosen among the
        other neurons not active enough (states: (units, timesteps)). Computed for all the neurons before any
        change."""
        pool = list(neurons)
        if len(pool) <= 1:
            return []
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.corrcoef(states[:, 1:])
        pairs = []
        for neuron in neurons:
            partners = self.W[neuron].nonzero()[0]
            # beyond max_partners, only the existing connections are strengthened
            available = partners if len(partners) >= self.max_partners else [n for n in pool if n != neuron]
            # the most correlated neurons (ties drawn at random)
            scores = correlations[neuron, available]
            candidates = np.array(available)[np.isclose(scores, np.nanmax(scores))]
            if candidates.size == 0:
                raise ValueError(f"No candidate presynaptic neuron for neuron {neuron}: undefined correlations.")
            pairs.append((neuron, self._plasticity_rng.choice(candidates)))
        return pairs

    def _plasticity(self, states: np.ndarray):
        """One plasticity step on the states (timesteps, units) of a window."""
        delta_z = self._activity_error(states)
        neurons = np.arange(self.units)
        # too active: one incoming connection, drawn at random, is weakened
        for neuron in neurons[delta_z >= 1]:
            partners = self.W[neuron].nonzero()[0]
            if len(partners) > 0:
                partner = self._plasticity_rng.choice(partners)
                # weights never become negative
                self.W[neuron, partner] = max(self.W[neuron, partner] - self.weight_increment, 0)
                self.n_pruned += 1
        # not active enough: one incoming connection from a correlated neuron is strengthened
        for neuron, partner in self._new_partners(neurons[delta_z <= -1], states.T):
            self.W[neuron, partner] = self.W[neuron, partner] + self.weight_increment
            self.n_added += 1
        if self.homeostasis == "variance":
            # intrinsic homeostatic plasticity: weaker inputs for the neurons saturated on the whole window
            self.W[np.all(states >= self.intrinsic_saturation, axis=0)] *= self.intrinsic_coef

    def fit(self, x: NodeInput, y: None = None, warmup: int = 0) -> "HAGReservoir":
        """Offline fitting method of the HAG reservoir: grows and prunes the recurrent connections on the input
        timeseries (unsupervised). Leaves the node in its last state.

        Parameters
        ----------
        x : list or array-like of shape ([series, ] timesteps, input_dim)
            Input sequences dataset.
        y : None
            Not used, HAG is unsupervised.
        warmup : int, default to 0
            Number of timesteps to discard at the beginning of each timeseries before training.

        Returns
        -------
        HAGReservoir
            Node trained offline.
        """
        check_node_input(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x)

        multiple = is_multiseries(x)
        sequences = [np.asarray(seq, dtype=self.dtype)[warmup:] for seq in (x if multiple else [x])]
        self.n_added = self.n_pruned = 0

        state = {"out": self._plasticity_rng.uniform(0, 1, self.units)}
        if self.use_full_instance and multiple:
            # Initialization
            # the first 3 sequences initialize the state, then one plasticity step per sequence
            state, _ = self._run_window(state, np.concatenate(sequences[:3]))
            # Fit
            for seq in sequences[3:]:
                state, states = self._run_window(state, seq)
                self._plasticity(states)
        else:
            # Initialization
            # sequences concatenated, initialization on 5 * min_window steps, then windows of random lengths
            inputs = np.concatenate(sequences)
            init_length = 5 * self.min_window
            state, _ = self._run_window(state, inputs[:init_length])
            # Fit
            inputs = inputs[init_length:]
            lengths = np.unique(
                np.round(np.logspace(np.log10(self.min_window), np.log10(self.max_window), num=10)).astype(int)
            )
            while len(inputs) > self.max_window:
                T = self._plasticity_rng.choice(lengths)
                state, states = self._run_window(state, inputs[:T])
                inputs = inputs[T:]
                self._plasticity(states)

        self.state = state
        return self
