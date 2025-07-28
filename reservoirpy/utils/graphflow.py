# Author: Nathan Trouvain at 12/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import deque
from typing import Optional, Sequence, TypeVar

T = TypeVar("T")


def unique_ordered(x: Sequence[T]) -> list[T]:
    """Returns a list with the same elements as x occuring only once, and in the same order"""
    return list(dict.fromkeys(x))


def find_parents_and_children(nodes: list[T], edges: list[tuple[T, T]]):
    """Returns two dicts linking nodes to their parents and children in the graph."""
    # TODO: more efficient method, not in O(#nodes * #edges)
    parents = {
        child: unique_ordered([p for p, c in edges if c is child]) for child in nodes
    }
    children = {
        parent: unique_ordered([c for p, c in edges if p is parent]) for parent in nodes
    }

    return parents, children


def topological_sort(
    nodes: list[T], edges: list[tuple[T, T]], inputs: Optional[list[T]] = None
) -> list[T]:
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
            edges_set.remove((n, m))
            parents[m].remove(n)
            if parents.get(m) is None or len(parents[m]) < 1:
                inputs_deque.append(m)
    if len(edges_set) > 0:
        raise RuntimeError(
            "Model has a cycle: impossible "
            "to automatically determine operations "
            "order in the model."
        )
    else:
        return ordered_nodes


def get_offline_subgraphs(nodes, edges):
    """Cut a graph into several subgraphs where output nodes are untrained offline
    learner nodes."""
    inputs = find_inputs(nodes, edges)
    outputs = find_outputs(nodes, edges)
    parents, children = find_parents_and_children(nodes, edges)

    offlines = set(
        [n for n in nodes if n.is_trained_offline and not n.is_trained_online]
    )
    included, trained = set(), set()
    subgraphs, required = [], []
    _nodes = nodes.copy()
    while trained != offlines:
        subnodes, subedges = [], []
        for node in _nodes:
            if node in inputs or all([p in included for p in parents[node]]):
                if node.is_trained_offline and node not in trained:
                    trained.add(node)
                    subnodes.append(node)
                else:
                    if node not in outputs:
                        subnodes.append(node)
                    included.add(node)

        subedges = [
            edge for edge in edges if edge[0] in subnodes and edge[1] in subnodes
        ]
        subgraphs.append((subnodes, subedges))
        _nodes = [n for n in nodes if n not in included]

    required = _get_required_nodes(subgraphs, children)

    return list(zip(subgraphs, required))


def _get_required_nodes(subgraphs, children):
    """Get nodes whose outputs are required to run/fit children nodes."""
    req = []
    fitted = set()
    for i in range(1, len(subgraphs)):
        currs = set(subgraphs[i - 1][0])
        nexts = set(subgraphs[i][0])

        req.append(_get_links(currs, nexts, children))

        fitted |= set([node for node in currs if node.is_trained_offline])

    nexts = set(
        [n for n in subgraphs[-1][0] if n.is_trained_offline and n not in fitted]
    )
    currs = set(
        [n for n in subgraphs[-1][0] if not n.is_trained_offline or n in fitted]
    )

    req.append(_get_links(currs, nexts, children))

    return req


def _get_links(previous, nexts, children):
    """Returns graphs edges between two subgraphs."""
    links = {}
    for n in previous:
        next_children = []
        if n not in nexts:
            next_children = [c.name for c in children.get(n, []) if c in nexts]

        if len(next_children) > 0:
            links[n.name] = next_children

    return links


def find_inputs(nodes: list[T], edges: list[tuple[T, T]]) -> list[T]:
    """
    Find all nodes that are not receivers (without incoming connections).
    Guaranteed to preserve order.
    """
    receivers: set[T] = set([n for _, n in edges])
    sources = [node for node in nodes if node not in receivers]
    return sources


def find_outputs(nodes: list[T], edges: list[tuple[T, T]]) -> list[T]:
    """
    Find all nodes that are not senders (no out-going connections).
    Guaranteed to preserve order.
    """
    senders = set([n for n, _ in edges])
    sinks = [node for node in nodes if node not in senders]
    return sinks
