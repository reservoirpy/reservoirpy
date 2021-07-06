# Author: Nathan Trouvain at 06/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Union

from numpy.random import RandomState, MT19937, Generator, default_rng


def get_generator(seed: Union[int, Generator, RandomState]) -> Generator:
    # provided to support legacy RandomState generator
    # of Numpy. It is not the best thing to do however
    # and recommend the user to keep using integer seeds
    # and proper Numpŷ Generator API.
    if isinstance(seed, RandomState):
        mt19937 = MT19937()
        mt19937.state = seed.get_state()
        return Generator(mt19937)

    if isinstance(seed, Generator):
        return seed
    else:
        return default_rng(seed)
