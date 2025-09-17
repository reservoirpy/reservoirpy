import jax.numpy as jnp
import scipy.sparse
from jax.experimental import sparse as jsparse

from .. import mat_gen

INITIALIZER_NAMES = [
    "bernoulli",
    "cluster",
    "line",
    "normal",
    "orthogonal",
    "ring",
    "small_world",
    "uniform",
    "zeros",
]

INITIALIZER_MAPPING = {name: getattr(mat_gen, name) for name in INITIALIZER_NAMES}


class JaxInitializer(mat_gen.Initializer):
    """Initializer for matrix generation for the Jax backend

    This inherits from the regular numpy/scipy Initializer class.
    The only difference is that the resulting array is converted to Jax
    (a regular :py:class:`jax.Array` or a :py:class:`jax.experimental.sparse.BCOO` array).
    """

    def __call__(self, *shape, **kwargs):
        call_result = super(JaxInitializer, self).__call__(*shape, **kwargs)

        # Not initialized yet, only kwargs set
        if isinstance(call_result, mat_gen.Initializer):
            return call_result
        # Initialized, call_result is a sparse or dense array to convert to Jax
        else:
            if scipy.sparse.issparse(call_result):
                return jsparse.BCOO.from_scipy_sparse(call_result)
            else:
                return jnp.array(call_result)


def __getattr__(name):
    if name in INITIALIZER_MAPPING:
        initializer = INITIALIZER_MAPPING[name]
        # This is a very bad idea :)
        # But attributes are kept, only the methods are modified, which is what we want
        # so it should be okay
        initializer.__class__ = JaxInitializer
        return initializer
    else:
        raise AttributeError(
            f"Cannot import {name} from this Jax module. Available elements are {list(INITIALIZER_MAPPING.keys())}."
        )
