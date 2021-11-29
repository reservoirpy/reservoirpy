# Author: Nathan Trouvain at 19/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import defaultdict
from typing import Iterable
from inspect import signature

import numpy as np

from tqdm import tqdm

from .validation import is_mapping, is_sequence_set


VERBOSITY = 1


def verbosity(level=None):
    global VERBOSITY
    if isinstance(level, int) and VERBOSITY != level:
        VERBOSITY = level
    return VERBOSITY


def progress(it, *args, **kwargs):
    if VERBOSITY > 0:
        return tqdm(it, *args, **kwargs)
    else:
        return it


def safe_defaultdict_copy(d):
    new_d = defaultdict(list)
    for key, item in d.items():
        if isinstance(item, Iterable):
            new_d[key] = list(item)
        else:
            new_d[key] += item
    return new_d


def to_ragged_seq_set(data):
    # data is a dict
    if is_mapping(data):
        new_data = {}
        for name, datum in data.items():
            if not is_sequence_set(datum):
                # all sequences must at least be 2D (seq length, num features)
                # 1D sequences are converted to (1, num features) by default.
                new_datum = [np.atleast_2d(datum)]
            else:
                new_datum = datum
            new_data[name] = new_datum
        return new_data
    # data is an array or a list
    else:
        if not is_sequence_set(data):
            return [np.atleast_2d(data)]
        else:
            return data


def _obj_from_kwargs(klas, kwargs):
    sig = signature(klas.__init__)
    params = list(sig.parameters.keys())
    klas_kwargs = {n: v for n, v in kwargs.items() if n in params}
    return klas(**klas_kwargs)
