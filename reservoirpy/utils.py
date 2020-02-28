from typing import List, Union, Any

import numpy as np


def check_values(array_or_list: Union[List, np.array], value: Any):
    """ Check if the given array or list contains the given value. """
    if value == np.nan:
        assert np.isnan(array_or_list).any() == False, f"{array_or_list} should not contain NaN values."
    if value == None:
        if type(array_or_list) is list:
            assert np.count_nonzero(array_or_list == None) == 0, f"{array_or_list} should not contain None values."
        elif type(array_or_list) is np.array:
            # None is transformed to np.nan when it is in an array
            assert np.isnan(array_or_list).any() == False, f"{array_or_list} should not contain NaN values."