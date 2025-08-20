from functools import partial
from typing import Callable, Optional, Sequence, Union

import numpy as np
from numpy.random import Generator

from ..mat_gen import uniform
from ..node import Node
from ..type import NodeInput, State, Timestep, Weights, is_array
from ..utils.random import rand_generator


class LIF(Node):
    """Pool of leaky integrate and fire (LIF) spiking neurons with random recurrent connexions.

    This node is similar to a reservoir (large pool of recurrent, randomly connected neurons),
    but the neurons follows a leaky integrate and fire activity rule.


    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be inferred from
        the ``W`` matrix shape.
    inhibitory : float, defaults to 0.0
        Proportion of neurons that have an inhibitory behavior (i.e. negative outgoing
        connections). Must be in :math:`[0, 1]`
    threshold : float, defaults to 1.0
        Limits above which the neurons spikes and returns to zero.
    lr : float or array-like of shape (units,), default to 1.0
        Neurons leak rate. Must be in :math:`[0, 1]`.
    sr : float, defaults to 1.0
        Spectral radius of recurrent weight matrix.
    input_scaling : float or array-like of shape (features,), default to 1.0.
        Input gain. An array of the same dimension as the inputs can be used to
        set up different input scaling for each feature.
    rc_connectivity : float, defaults to 0.1
        Connectivity of recurrent weight matrix, i.e. ratio of reservoir
        neurons connected to other reservoir neurons, including themselves.
        Must be in :math:`]0, 1]`.
    input_connectivity : float, default to 0.1
        Connectivity of input neurons, i.e. ratio of input neurons connected
        to reservoir neurons. Must be in :math:`]0, 1]`.
    Win : callable or array-like of shape (units, features), default to :py:func:`~reservoirpy.mat_gen.uniform` with a
        lower bound of 0.0.
        Input weights matrix or initializer. If a callable (like a function) is used,
        then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    W : callable or array-like of shape (units, units), defaults to :py:func:`~reservoirpy.mat_gen.uniform` with
        a lower bound of 0.0.
        Recurrent weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
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

    >>> from reservoirpy.nodes import LIF
    >>> liquid = LIF(
    ...     units=100,
    ...     inhibitory=0.1,
    ...     sr=1.0,
    ...     lr=0.2,
    ...     input_scaling=0.5,
    ...     rc_connectivity=1.0,
    ...     input_connectivity=1.0,
    ...     seed=0,
    ... )

    Using the :py:func:`~reservoirpy.datasets.mackey_glass` timeseries:

    >>> from reservoirpy.datasets import mackey_glass
    >>> x = mackey_glass(1000)
    >>> spikes = liquid.run(x)

    .. plot::

        from reservoirpy.nodes import LIF
        liquid = LIF(
            units=100,
            inhibitory=0.1,
            sr=0.5,
            lr=0.2,
            input_scaling=0.5,
            rc_connectivity=1.0,
            input_connectivity=1.0,
        )
        from reservoirpy.datasets import mackey_glass
        x = mackey_glass(1000)
        states = liquid.run(x)
        fig, ax = plt.subplots(6, 1, figsize=(7, 10), sharex=True)
        ax[0].plot(x)
        ax[0].grid()
        ax[0].set_title("Neuron spikes (on Mackey-Glass)")
        for i in range(1, 6):
            ax[i].plot(states[:, i], label=f"Neuron {i}")
            ax[i].legend()
            ax[i].grid()
        ax[-1].set_xlabel("Timesteps")
    """

    #: Number of neuronal units in the reservoir.
    units: int
    #: Proportion of inhibitory neurons. (0.0 by default)
    inhibitory: float
    #: Spike threshold. (1.0 by default)
    threshold: float
    #: Type of matrices elements. By default, ``np.float64``.
    dtype: type
    #: Leaking rate (1.0 by default) (:math:`\mathrm{lr}`).
    lr: Union[float, np.ndarray]
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
    #: A random state generator. Used for generating Win and W.
    rng: np.random.Generator

    def __init__(
        self,
        units: Optional[int] = None,
        inhibitory: float = 0.0,
        threshold: float = 1.0,
        lr: float = 0.0,
        sr: float = 1.0,
        input_scaling: Union[float, Sequence] = 1.0,
        rc_connectivity: float = 0.1,
        input_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = partial(uniform, low=0.0),
        W: Union[Weights, Callable] = partial(uniform, low=0.0),
        input_dim: Optional[int] = None,
        dtype: type = np.float64,
        seed: Optional[Union[int, Generator]] = None,
        name: Optional[str] = None,
    ):
        self.inhibitory = inhibitory
        self.threshold = threshold
        self.sr = sr
        self.input_scaling = input_scaling
        self.lr = lr
        self.rc_connectivity = rc_connectivity
        self.input_connectivity = input_connectivity
        self.Win = Win
        self.W = W
        self.dtype = dtype
        self.rng = rand_generator(seed=seed)
        self.name = name
        self.initialized = False

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

    def _step(self, state: State, x: Timestep):
        v = state["internal"].copy()
        threshold = self.threshold
        lr = self.lr
        W = self.W
        Win = self.Win
        # leak
        v *= 1 - lr
        # fire
        spikes = (v > threshold).astype(self.dtype)  # threshold
        v[v > threshold] = 0.0
        # integrate
        v += (W @ spikes.T).T
        v += (Win @ x.T).T

        # return spikes
        return {"internal": v, "out": spikes}

    def initialize(
        self,
        x: Optional[Union[NodeInput, Timestep]],
    ):
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
            n_inhibitory = int(self.inhibitory * self.units)
            self.W[:, :n_inhibitory] *= -1

        self.state = {
            "internal": np.zeros((self.units,)),
            "out": np.zeros((self.units,)),
        }

        self.initialized = True
