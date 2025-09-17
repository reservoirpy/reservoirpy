# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from .random import rand_generator

EXCLUDED_PARAMETERS = ["Win", "W", "bias", "Wout"]


def get_non_defaults(instance, constructor=None):
    """
    Return a dictionary of key: values where the key is both an attribute of ``instance`` and a named parameter of its
    constructor (or a parameter of ``constructor`` if specified), but the attribute value is different from the default
    value of the constructor. For the sake of readability, some parameters names are excluded, such as ``W`` or ``bias``.

    This method is used for the ``__str__`` and ``__repr__`` methods.

    Parameters
    ----------
    instance : object
    constructor : class, optional

    Returns
    -------
    dict[str, Any]
    """
    if constructor is None:
        constructor = type(instance)
    import inspect

    signature = inspect.signature(constructor)
    defaults = {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
    return {
        k: getattr(instance, k)
        for k, v in defaults.items()
        if hasattr(instance, k) and k not in EXCLUDED_PARAMETERS and getattr(instance, k) != v
    }
