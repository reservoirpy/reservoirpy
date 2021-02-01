_DEFAULT_SEED = 5555


def get_seed():
    """Return the current random state seed used for dataset
    generation.

    Returns
    -------
    int
        Current seed value.
    """
    global _DEFAULT_SEED
    return _DEFAULT_SEED


def set_seed(s: int):
    """Change the default random seed value.

    This will change the behaviour of the Mackey-Glass
    timeseries generator (see :py:func:`mackey_glass`).

    Parameters
    ----------
    s : int
        A random state generator numerical seed.
    """
    global _DEFAULT_SEED
    _DEFAULT_SEED = s
