# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from collections import deque
from typing import Any, Optional, Sequence, TypeVar

T = TypeVar("T")


def unique_ordered(x: Sequence[T]) -> list[T]:
    """Returns a list with the same elements as x occuring only once, and in the same order"""
    return list(dict.fromkeys(x))


def find_parents_and_children(nodes: list[T], edges: list[tuple[T, int, T]]):
    """Returns two dicts linking nodes to their parents and children in the graph."""
    parents = {child: unique_ordered([p for p, d, c in edges if c is child and d == 0]) for child in nodes}
    children = {parent: unique_ordered([c for p, d, c in edges if p is parent and d == 0]) for parent in nodes}

    return parents, children


def find_indirect_children(nodes: list[T], edges: list[tuple[T, int, T]]):
    """Returns a dict linking nodes to their children in the graph, regardless of the delay."""

    children = {parent: unique_ordered([c for p, d, c in edges if p is parent]) for parent in nodes}

    return children


def topological_sort(nodes: list[T], edges: list[tuple[T, int, T]], inputs: Optional[list[T]] = None) -> list[T]:
    """Topological sort of nodes in a Model, to determine execution order."""
    if inputs is None:
        inputs = find_inputs(nodes, edges)

    parents, children = find_parents_and_children(nodes, edges)

    # using Kahn's algorithm
    ordered_nodes = []
    edges_set = set(edges)
    inputs_deque = deque(inputs)
    while len(inputs_deque) > 0:
        n = inputs_deque.pop()
        ordered_nodes.append(n)
        for m in children.get(n, ()):
            edges_set.remove((n, 0, m))
            parents[m].remove(n)
            if parents.get(m) is None or len(parents[m]) < 1:
                inputs_deque.append(m)
    if len(edges_set) > 0:
        raise RuntimeError(
            "Model has a cycle: impossible " "to automatically determine operations " "order in the model."
        )
    else:
        return ordered_nodes


def find_pseudo_inputs(nodes: list[T], edges: list[tuple[T, int, T]], y_mapping: dict[T, Any]) -> list[T]:
    """
    Find all nodes that are not receivers or that only receive forced feedback.
    Guaranteed to preserve order.
    """
    unsupervised_receivers: set[T] = set([n for c, d, n in edges if d == 0 and c not in y_mapping.keys()])
    sources = [node for node in nodes if node not in unsupervised_receivers]
    return sources


def find_inputs(nodes: list[T], edges: list[tuple[T, int, T]]) -> list[T]:
    """
    Find all nodes that are not receivers (without direct incoming connections).
    Guaranteed to preserve order.
    """
    receivers: set[T] = set([n for _, d, n in edges if d == 0])
    sources = [node for node in nodes if node not in receivers]
    return sources


def find_outputs(nodes: list[T], edges: list[tuple[T, int, T]]) -> list[T]:
    """
    Find all nodes that are not senders (no direct out-going connections).
    Guaranteed to preserve order.
    """
    senders = set([n for n, d, _ in edges if d == 0])
    sinks = [node for node in nodes if node not in senders]
    return sinks
