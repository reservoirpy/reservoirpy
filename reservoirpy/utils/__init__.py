# Author: Nathan Trouvain at 19/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import Iterable, defaultdict

from .validation import is_mapping, is_sequence_set


def safe_defaultdict_copy(d):
    new_d = defaultdict(list)
    for key, item in d.items():
        if isinstance(item, Iterable):
            new_d[key] = list(item)
        else:
            new_d[key] += item
    return new_d


def to_ragged_seq_set(data):
    if is_mapping(data):
        new_data = {}
        for name, datum in data.items():
            if not is_sequence_set(datum):
                new_datum = [datum]
            else:
                new_datum = datum
            new_data[name] = new_datum
        return new_data
    else:
        if not is_sequence_set(data):
            return [data]
        else:
            return data
