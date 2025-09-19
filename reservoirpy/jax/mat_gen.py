import jax.numpy as jnp
import scipy.sparse
from jax.experimental import sparse as jsparse

from .. import mat_gen


class JaxInitializer(mat_gen.Initializer):
    """Initializer for matrix generation for the Jax backend

    This inherits from the regular numpy/scipy Initializer class.
    The only difference is that the resulting array is converted to Jax
    (a regular :py:class:`jax.Array` or a :py:class:`jax.experimental.sparse.BCOO` array).
    """

    def __call__(self, *shape, **kwargs):
        call_result = mat_gen.Initializer.__call__(self, *shape, **kwargs)

        # Not initialized yet, only kwargs set
        if isinstance(call_result, mat_gen.Initializer):
            return call_result
        # Initialized, call_result is a sparse or dense array to convert to Jax
        else:
            if scipy.sparse.issparse(call_result):
                return jsparse.BCOO.from_scipy_sparse(call_result)
            else:
                return jnp.array(call_result)


bernoulli = JaxInitializer(mat_gen._bernoulli)
cluster = JaxInitializer(mat_gen._cluster)
line = JaxInitializer(mat_gen._line)
normal = JaxInitializer(mat_gen._normal)
orthogonal = JaxInitializer(mat_gen._orthogonal)
ring = JaxInitializer(mat_gen._ring)
small_world = JaxInitializer(mat_gen._small_world)
uniform = JaxInitializer(mat_gen._uniform)
zeros = JaxInitializer(mat_gen._zeros)
ones = JaxInitializer(mat_gen._ones)


def __getattr__(attr):
    if attr in ["random_sparse", "fast_spectral_initialization"]:
        raise AttributeError(f"{attr} does not have a Jax implementation. Please use reservoir.mat_gen.{attr} instead.")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attr}")
