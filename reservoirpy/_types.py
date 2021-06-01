from typing import Union, Callable, List, Tuple, Optional

import numpy as np

from scipy import sparse

Weights = Union[np.ndarray,
                sparse.csr_matrix,
                sparse.csc_matrix,
                sparse.coo_matrix]

Numerical = Union[np.ndarray, int, float,
                  List[Union[int, float]],
                  Tuple[Union[int, float], ...]]

Activation = Callable[[Numerical], Numerical]

AnonymousReadout = Callable[[Numerical, Optional[Numerical]], Numerical]

AnyReadout = Union[AnonymousReadout,
                   Callable[..., Numerical]]

RandomSeed = Union[np.random.Generator, np.random.RandomState, int]


def is_iterable(obj):
    return hasattr(obj, "__iter__") or hasattr(obj, "__getitem__")
