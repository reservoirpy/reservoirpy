import sys
from functools import partial

if sys.version_info < (3, 8):
    from typing_extensions import Callable, Optional, Sequence, Union
else:
    from typing import Optional, Union, Sequence, Callable

from ...mat_gen import uniform
from ...node import Node
from ...type import Weights
from ...utils.random import rand_generator
from ...utils.validation import is_array


def forward(lif, x):
    v = lif.get_param("internal_state").copy()
    threshold = lif.get_param("threshold")
    # leak
    v *= 1 - lif.lr
    # fire
    spikes = (v > threshold).astype(lif.dtype)  # threshold
    v[v > threshold] = 0.0
    # integrate
    v += (lif.W @ spikes.T).T
    v += (lif.Win @ x.T).T

    lif.set_param("internal_state", v)
    # return spikes
    return spikes


def initialize(
    lif,
    x=None,
    y=None,
    seed=None,
    input_scaling=None,
    input_connectivity=None,
    rc_connectivity=None,
    inhibitory=None,
    W_init=None,
    Win_init=None,
    sr=None,
):
    dtype = lif.dtype

    lif.set_input_dim(x.shape[-1])

    rng = rand_generator(seed)

    if is_array(W_init):
        W = W_init
        if W.shape[0] != W.shape[1]:
            raise ValueError(
                "Dimension mismatch inside W: "
                f"W is {W.shape} but should be "
                f"a square matrix."
            )

        if W.shape[0] != lif.output_dim:
            lif._output_dim = W.shape[0]
            lif.hypers["units"] = W.shape[0]

    elif callable(W_init):
        W = W_init(
            lif.output_dim,
            lif.output_dim,
            sr=sr,
            connectivity=rc_connectivity,
            dtype=dtype,
            seed=rng,
        )
        n_inhibitory = int(inhibitory * lif.output_dim)
        W[:, :n_inhibitory] *= -1

    lif.set_param("units", W.shape[0])
    lif.set_param("W", W.astype(dtype))

    out_dim = lif.output_dim

    if is_array(Win_init):
        Win = Win_init

        if Win.shape[1] != x.shape[1]:
            raise ValueError(
                f"Dimension mismatch in {lif.name}: Win input dimension is "
                f"{Win.shape[1]} but input dimension is {x.shape[1]}."
            )

        if Win.shape[0] != out_dim:
            raise ValueError(
                f"Dimension mismatch in {lif.name}: Win internal dimension "
                f"is {Win.shape[0]} but the liquid dimension is {out_dim}"
            )

    elif callable(Win_init):
        Win = Win_init(
            lif.output_dim,
            x.shape[1],
            input_scaling=input_scaling,
            connectivity=input_connectivity,
            dtype=dtype,
            seed=seed,
        )
    else:
        dtype = lif.dtype
        dtype_msg = (
            "Data type {} not understood in {}. {} should be an array or a "
            "callable returning an array."
        )
        raise ValueError(dtype_msg.format(str(type(Win_init)), lif.name, "Win"))

    lif.set_param("W", W.astype(dtype))
    lif.set_param("Win", Win.astype(dtype))
    lif.set_param("internal_state", lif.zero_state())


class LIF(Node):
    """Pool of leaky integrate and fire (LIF) spiking neurons with random recurrent connexions.

    This node is similar to a reservoir (large pool of recurrent, randomly connected neurons),
    but the neurons follows a leaky integrate and fire activity rule.

    This is a first version of a Liquid State Machine implementation. More models are expected
    to come in future versions of ReservoirPy.


    :py:attr:`LIF.params` **list:**

    ================== ===================================================================
    ``W``              Recurrent weights matrix (:math:`\\mathbf{W}`).
    ``Win``            Input weights matrix (:math:`\\mathbf{W}_{in}`).
    ``internal_state`` Internal state of the neurons.
    ================== ===================================================================

    :py:attr:`LIF.hypers` **list:**

    ======================= ========================================================
    ``lr``                  Leaking rate (1.0 by default) (:math:`\\mathrm{lr}`).
    ``sr``                  Spectral radius of ``W`` (optional).
    ``input_scaling``       Input scaling (float or array) (1.0 by default).
    ``rc_connectivity``     Connectivity (or density) of ``W`` (0.1 by default).
    ``input_connectivity``  Connectivity (or density) of ``Win`` (0.1 by default).
    ``units``               Number of neuronal units in the liquid.
    ``inhibitory``          Proportion of inhibitory neurons. (0.0 by default)
    ``threshold``           Spike threshold. (1.0 by default)
    ======================= ========================================================

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
    sr : float, optional
        Spectral radius of recurrent weight matrix.
    input_scaling : float or array-like of shape (features,), default to 1.0.
        Input gain. An array of the same dimension as the inputs can be used to
        set up different input scaling for each feature.
    input_connectivity : float, default to 0.1
        Connectivity of input neurons, i.e. ratio of input neurons connected
        to reservoir neurons. Must be in :math:`]0, 1]`.
    rc_connectivity : float, defaults to 0.1
        Connectivity of recurrent weight matrix, i.e. ratio of reservoir
        neurons connected to other reservoir neurons, including themselves.
        Must be in :math:`]0, 1]`.
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
    name : str, optional
        Node name.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for noise generation.

    Note
    ----

    If W, Win, bias or Wfb are initialized with an array-like matrix, then all
    initializers parameters such as spectral radius (``sr``) or input scaling
    (``input_scaling``) are ignored.
    See :py:mod:`~reservoirpy.mat_gen` for more information.

    Example
    -------

    >>> from reservoirpy.experimental import LIF
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

        from reservoirpy.experimental import LIF
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

    def __init__(
        self,
        units: Optional[int] = None,
        inhibitory: float = 0.0,
        threshold: float = 1.0,
        input_dim: Optional[int] = None,
        sr: Optional[float] = None,
        input_scaling: Union[float, Sequence] = 1.0,
        lr: float = 0.0,
        rc_connectivity: float = 0.1,
        input_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = partial(uniform, low=0.0),
        W: Union[Weights, Callable] = partial(uniform, low=0.0),
        seed=None,
        **kwargs,
    ):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not "
                "a matrix."
            )

        super(LIF, self).__init__(
            params={
                "W": None,
                "Win": None,
                "internal_state": None,
            },
            hypers={
                "units": units,
                "inhibitory": inhibitory,
                "threshold": threshold,
                "lr": lr,
                "rc_connectivity": rc_connectivity,
                "input_connectivity": input_connectivity,
                "input_scaling": input_scaling,
                "sr": sr,
            },
            forward=forward,
            initializer=partial(
                initialize,
                sr=sr,
                input_scaling=input_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity,
                inhibitory=inhibitory,
                W_init=W,
                Win_init=Win,
                seed=seed,
            ),
            input_dim=input_dim,
            output_dim=units,
            **kwargs,
        )
