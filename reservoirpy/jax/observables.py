# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Optional, Union

import jax
import jax.numpy as jnp

from .node import Node


def maximal_lyapunov_exponent(
    node: Node,
    X: jax.Array,
    *,
    warmup: int = 0,
    seed: int = 0,
) -> float:
    """Maximal Lyapunov Exponent (MLE) of a JAX Node driven by input X.

    Computes the MLE using the Benettin algorithm: a random unit tangent
    vector is evolved through the Jacobian of the node's one-step map at
    each timestep (via :func:`jax.jvp`), with Gram-Schmidt renormalization
    after each step to prevent overflow. The average log-growth rate
    converges to the MLE.

    A positive MLE indicates chaotic (diverging) dynamics; a negative MLE
    indicates stable (contracting) dynamics; MLE near zero is the edge of
    chaos.

    Parameters
    ----------
    node : Node
        An initialized JAX Node. Must be stateful (e.g. Reservoir, ES2N,
        LIF, IPReservoir, NVAR). Stateless readout nodes (Ridge, RLS, LMS)
        are accepted but the result is not meaningful.
    X : jax.Array of shape (T, input_dim)
        Input timeseries driving the node. Must have at least ``warmup + 1``
        timesteps.
    warmup : int, default to 0
        Number of initial timesteps used to warm up the node state before
        measuring divergence. These steps are not counted in the MLE average.
    seed : int, default to 0
        Seed for the initial random tangent vector.

    Returns
    -------
    float
        The estimated Maximal Lyapunov Exponent.

    Raises
    ------
    ValueError
        If the node is not initialized, or if X does not have enough
        timesteps given the warmup parameter.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from reservoirpy.jax.nodes import Reservoir
    >>> from reservoirpy.jax.observables import maximal_lyapunov_exponent
    >>> reservoir = Reservoir(100, sr=0.9)
    >>> X = jnp.ones((500, 1))
    >>> _ = reservoir.run(X[:1])  # initialize
    >>> reservoir.reset()
    >>> mle = maximal_lyapunov_exponent(reservoir, X, warmup=100)
    """
    if not node.initialized:
        raise ValueError(
            "Node must be initialized before computing the Lyapunov exponent. "
            "Call node.initialize(x) or run node(x) once first."
        )

    T = X.shape[0]
    if T - warmup < 1:
        raise ValueError(
            f"X must have more than {warmup} timesteps (warmup) but has {T}."
        )

    # Warmup: evolve state without tracking tangent
    def warmup_body(state, x_t):
        return node._step(state, x_t), None

    state, _ = jax.lax.scan(warmup_body, node.state, X[:warmup]) if warmup > 0 else (node.state, None)

    # Initialize a random unit tangent vector (same pytree as state)
    key = jax.random.PRNGKey(seed)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    keys = jax.random.split(key, len(leaves))
    tangent_leaves = [jax.random.normal(k, shape=a.shape, dtype=a.dtype) for k, a in zip(keys, leaves)]

    flat_v = jnp.concatenate([jnp.ravel(a) for a in tangent_leaves])
    norm = jnp.linalg.norm(flat_v)
    tangent_leaves = [a / norm for a in tangent_leaves]
    v_init = jax.tree_util.tree_unflatten(treedef, tangent_leaves)

    # Benettin scan: JVP + Gram-Schmidt renormalization at each step
    def scan_body(carry, x_t):
        state, v = carry
        new_state, Jv = jax.jvp(lambda s: node._step(s, x_t), (state,), (v,))

        flat_Jv = jnp.concatenate([jnp.ravel(a) for a in jax.tree_util.tree_leaves(Jv)])
        norm = jnp.linalg.norm(flat_Jv)
        v_next = jax.tree_util.tree_map(lambda a: a / norm, Jv)

        return (new_state, v_next), jnp.log(norm)

    _, log_growths = jax.lax.scan(scan_body, (state, v_init), X[warmup:])

    return float(jnp.mean(log_growths))
