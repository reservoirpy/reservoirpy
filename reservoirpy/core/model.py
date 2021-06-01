from typing import Union, Callable
from abc import ABC
from collections import OrderedDict

from combinators import link


class Node:

    _forward_links: OrderedDict
    _delayed_links: OrderedDict
    registry: OrderedDict

    serial: int = 0

    @classmethod
    def _give_name(cls):
        if cls.serial is None:
            cls.serial = 1
        else:
            cls.serial += 1
        return f"{cls.__name__}_{cls.serial}"

    def __init__(self, name: str = None):
        self.name = name or self._get_serial_name()

    def __getitem__(self, item):
        if type(item) is int:
            key = list(self.registry.keys())[item]
            return self.registry[key]
        return self.registry[item]

    def __iter__(self):
        yield from self.registry.values()

    def __len__(self):
        return len(self.registry)

    def __rrshift__(self, other: "Model"):
        return self.link(other)

    def _get_serial_name(self):
        self.name = self.__class__._give_name()

    def link(self, other: "Model"):
        if isinstance(other, Model):
            return link(self, other)
        else:
            raise ValueError("Impossible to chain model with object"
                             f"of type {type(other)}.")

    def delayed_link(self, rule: "Model"):
        ...

    def run(self, inputs):
        raise NotImplementedError()

    def fit(self, inputs, targets):
        raise NotImplementedError()

