import pickle
from os import PathLike, fspath
from pathlib import Path
from typing import Union


def save(model: Union["Node", "Model"], path: Union[str, PathLike, Path]):
    """Save a ReservoirPy model to a binary file"""
    path = fspath(path)
    if not path.endswith(".rpy"):
        path += ".rpy"
    with open(path, "wb+") as file:
        pickle.dump(model, file)
    return path


def load(path: Union[str, Path, PathLike]) -> Union["Node", "Model"]:
    """Load a ReservoirPy model from a binary file"""
    path = fspath(path)
    if not path.endswith(".rpy"):
        path += ".rpy"
    with open(path, "rb") as file:
        content = pickle.load(file)
    return content
