# Author: Nathan Trouvain at 19/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import Iterable, defaultdict


def safe_defaultdict_copy(d):
    new_d = defaultdict(list)
    for key, item in d.items():
        if isinstance(item, Iterable):
            new_d[key] = list(item)
        else:
            new_d[key] += item
    return new_d
