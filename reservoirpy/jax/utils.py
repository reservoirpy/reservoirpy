# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Optional, Union

import jax
import numpy as np

from ..utils.random import rand_generator


def rng_key(seed: Optional[Union[int, np.random.Generator]] = None) -> "jax.random.PRNGKey":
    if seed is None or isinstance(seed, np.random.Generator):
        return rand_generator(seed).integers(1 << 64 - 1)
    return jax.random.key(seed)
