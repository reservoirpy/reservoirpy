# Author: Nathan Trouvain at 22/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Callable, Union, Iterable

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix


Weights = Union[np.ndarray, csc_matrix, csc_matrix, coo_matrix]
Data = Union[Iterable[np.ndarray], np.ndarray]
Activation = Callable[[np.ndarray], np.ndarray]
