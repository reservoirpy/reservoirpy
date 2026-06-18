# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax.numpy as jnp
import pytest

from reservoirpy.datasets import mackey_glass
from reservoirpy.jax.nodes import ES2N, LIF, IPReservoir, Reservoir
from reservoirpy.jax.observables import maximal_lyapunov_exponent


@pytest.fixture
def X():
    return jnp.ones((500, 1))


def initialized_reservoir(sr=0.9, units=50, seed=0):
    res = Reservoir(units, sr=sr, seed=seed)
    res.initialize(jnp.ones((1,)))
    return res


# Basic correctness

def test_mle_returns_float(X):
    res = initialized_reservoir()
    mle = maximal_lyapunov_exponent(res, X)
    assert isinstance(mle, float)


def test_mle_with_warmup(X):
    res = initialized_reservoir()
    mle = maximal_lyapunov_exponent(res, X, warmup=100)
    assert isinstance(mle, float)


def test_mle_stable_reservoir(X):
    """A reservoir with very small sr should have a negative MLE (contracting)."""
    res = initialized_reservoir(sr=0.1)
    mle = maximal_lyapunov_exponent(res, X, warmup=50)
    assert mle < 0.0


def test_mle_zero_input_matches_log_sr():
    """Under zero input the MLE has a closed form, used here as ground truth.

    The JAX ``Reservoir`` defaults to ``bias=0.0`` and ``lr=1.0``, so with
    zero input the state stays at zero, f'(0)=1 for tanh, the Jacobian is
    exactly ``W``, and the MLE converges to ``log(rho(W)) = log(sr)``.
    """
    sr = 3.0
    X_zero = jnp.zeros((1500, 1))
    res = initialized_reservoir(sr=sr)
    mle = maximal_lyapunov_exponent(res, X_zero, warmup=500)
    assert mle > 0.0
    assert mle == pytest.approx(jnp.log(sr), abs=0.05)


def test_mle_chaotic_driving_signal():
    """Driven by a chaotic signal, the conditional MLE is finite and the
    function runs end-to-end on a non-trivial (non-constant) input."""
    X = jnp.asarray(mackey_glass(1500))
    res = Reservoir(50, sr=1.2, seed=0)
    res.initialize(X[:1])
    mle = maximal_lyapunov_exponent(res, X, warmup=500)
    assert isinstance(mle, float)
    assert jnp.isfinite(mle)


def test_mle_different_seeds_same_result(X):
    """Same node and input must give the same MLE regardless of tangent seed."""
    res = initialized_reservoir()
    mle_0 = maximal_lyapunov_exponent(res, X, seed=0)
    mle_1 = maximal_lyapunov_exponent(res, X, seed=42)
    assert abs(mle_0 - mle_1) < 0.05


# Node compatibility

def test_mle_es2n():
    es2n = ES2N(50, seed=0)
    X = jnp.ones((300, 1))
    es2n.initialize(jnp.ones((1,)))
    mle = maximal_lyapunov_exponent(es2n, X, warmup=50)
    assert isinstance(mle, float)


def test_mle_lif():
    """LIF has {"internal", "out"} state — full state dict carry must be used."""
    lif = LIF(
        units=50,
        sr=0.5,
        lr=0.2,
        input_scaling=0.5,
        rc_connectivity=1.0,
        input_connectivity=1.0,
        seed=0,
    )
    X = jnp.ones((300, 1))
    lif.initialize(jnp.ones((1,)))
    mle = maximal_lyapunov_exponent(lif, X, warmup=50)
    assert isinstance(mle, float)


def test_mle_ipreservoir():
    """IPReservoir has {"internal", "out"} state."""
    ip = IPReservoir(50, sr=0.9, seed=0)
    X = jnp.ones((300, 1))
    ip.initialize(jnp.ones((1,)))
    mle = maximal_lyapunov_exponent(ip, X, warmup=50)
    assert isinstance(mle, float)


# Error handling

def test_mle_uninitialized_node_raises():
    res = Reservoir(50)
    X = jnp.ones((100, 1))
    with pytest.raises(ValueError, match="initialized"):
        maximal_lyapunov_exponent(res, X)


def test_mle_too_few_timesteps_raises():
    res = initialized_reservoir()
    X = jnp.ones((10, 1))
    with pytest.raises(ValueError):
        maximal_lyapunov_exponent(res, X, warmup=10)
