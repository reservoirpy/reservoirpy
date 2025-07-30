from typing import Mapping, Sequence

from reservoirpy.type import is_array


def check_timestep(x, *, expected_dim=None):
    if not is_array(x):
        raise TypeError(f"Input but be an array, got a {type(x)}.")
    if not x.ndim == 1:
        raise ValueError(
            f"Input but be a 1-dimensional array. Received an array of shape {x.shape}."
        )
    if expected_dim is not None and not x.shape == (expected_dim,):
        raise ValueError(f"Expected input of shape {(expected_dim,)}, got {x.shape}.")


def is_timestep(x):
    try:
        check_timestep(x)
        return True
    except:
        return False


def check_timeseries(x, *, expected_dim=None, expected_length=None):
    if not is_array(x):
        raise TypeError(f"Input but be an array, got a {type(x)}.")
    if not x.ndim == 2:
        raise ValueError(
            f"Input but be a 2-dimensional array. Got array of shape {x.shape}."
        )
    if expected_length is not None and not x.shape[0] == expected_length:
        raise ValueError(
            f"Expected timeseries of length {expected_length}, got {x.shape[0]}."
        )
    if expected_dim is not None and not x.shape[1] == expected_dim:
        raise ValueError(
            f"Expected feature dimension to be {expected_dim}, got {x.shape[1]}."
        )


def is_timeseries(x):
    try:
        check_timeseries(x)
        return True
    except:
        return False


def check_multiseries(x, *, expected_dim=None, expected_length=None):
    if isinstance(x, Sequence):
        for ts in x:
            check_timeseries(
                ts, expected_dim=expected_dim, expected_length=expected_length
            )
            if expected_dim is None:
                expected_dim = ts.shape[1]
    elif is_array(x):
        if not x.ndim == 3:
            raise ValueError(
                f"Input but be a 3-dimensional array. Got array of shape {x.shape}."
            )
        if expected_length is not None and not x.shape[1] == expected_length:
            raise ValueError(
                f"Expected timeseries of length {expected_length}, got {x.shape[1]}."
            )
        if expected_dim is not None and not x.shape[2] == expected_dim:
            raise ValueError(
                f"Expected feature dimension to be {expected_dim}, got {x.shape[2]}."
            )
    else:
        raise TypeError(
            f"Expected a 3-dimensional array or a sequence of array, got {type(x)}."
        )


def is_multiseries(x):
    try:
        check_multiseries(x)
        return True
    except:
        return False


def check_node_input(x, *, expected_dim=None, expected_length=None):
    if isinstance(x, Sequence):
        for ts in x:
            check_timeseries(
                ts, expected_dim=expected_dim, expected_length=expected_length
            )
            if (
                expected_dim is None
            ):  # Ensure all series have the same number of features
                expected_dim = ts.shape[1]
    elif is_array(x):
        if x.ndim == 3:
            check_multiseries(
                x, expected_dim=expected_dim, expected_length=expected_length
            )
        elif x.ndim == 2:
            check_timeseries(
                x, expected_dim=expected_dim, expected_length=expected_length
            )
        else:
            raise ValueError(
                f"Input but be a (2 or 3)-dimensional array. Got array of shape {x.shape}."
            )
    else:
        raise TypeError(f"Expected an array or a sequence of array, got {type(x)}.")


def is_node_input(x):
    try:
        check_node_input(x)
        return True
    except:
        return False


def check_model_timestep(x, *, expected_inputs=None, expected_dim=None):
    if isinstance(x, Mapping):
        if expected_inputs is not None and set(expected_inputs) != set(x.keys()):
            raise ValueError(
                f"Input does not match expected node names: {set(expected_inputs)} != {set(x.keys())}"
            )
        if isinstance(expected_dim, Mapping):
            for name in x:
                check_timestep(x[name], expected_dim=expected_dim[name])
        else:
            for name in x:
                check_timestep(x[name], expected_dim=expected_dim)
    elif is_array(x):
        if isinstance(expected_dim, Mapping):
            check_timestep(x, expected_dim=next(iter(expected_dim.values())))
        else:
            check_timestep(x, expected_dim=expected_dim)
    else:
        raise TypeError(f"Expected an array or a mapping of arrays, but got {type(x)}")


def is_model_timestep(x):
    try:
        check_model_timestep(x)
        return True
    except:
        return False


def check_model_input(x, *, expected_dim=None, expected_length=None):
    if isinstance(x, Mapping):
        for name in x:
            node_dim = expected_dim[name] if expected_dim is not None else None
            check_node_input(
                x[name], expected_dim=node_dim, expected_length=expected_length
            )
    elif is_array(x) or isinstance(x, Sequence):
        if isinstance(expected_dim, Mapping):
            if len(expected_dim) == 1:
                expected_dim = next(iter(expected_dim.values()))
            else:
                raise TypeError(
                    f"Expected a mapping of node inputs ({list(expected_dim.keys())}), but got {type(x)}."
                )
        check_node_input(x, expected_dim=expected_dim, expected_length=expected_length)
    else:
        raise TypeError(f"Expected an array or a mapping of arrays, but got {type(x)}")


def is_model_input(x):
    try:
        check_model_input(x)
        return True
    except:
        return False
