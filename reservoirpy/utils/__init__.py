# Author: Nathan Trouvain at 19/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from inspect import signature


def _obj_from_kwargs(klas, kwargs):
    sig = signature(klas.__init__)
    params = list(sig.parameters.keys())
    klas_kwargs = {n: v for n, v in kwargs.items() if n in params}
    return klas(**klas_kwargs)
