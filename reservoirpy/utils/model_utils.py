# Author: Nathan Trouvain at 19/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import defaultdict
from typing import Any, Generator, Iterable, Mapping, TypeVar

import numpy as np

from reservoirpy.type import MultiTimeseries, Timeseries, Timestep

from ..node import Node
from ..nodes import Input, Output


def build_forward_submodels(nodes, edges, already_trained):
    """Separate unfitted offline nodes from fitted nodes and gather all fitted
    nodes in submodels."""
    from ..model import Model

    offline_nodes = [
        n for n in nodes if n.is_trained_offline and n not in already_trained
    ]

    forward_nodes = list(set(nodes) - set(offline_nodes))
    forward_edges = [edge for edge in edges if edge[1] not in offline_nodes]

    submodel = Model(forward_nodes, forward_edges)

    return submodel, offline_nodes


def dist_states_to_next_subgraph(states, relations):
    """Map submodel output state vectors to input nodes of next submodel.

    Edges between first and second submodel are stored in 'relations'.
    """
    dist_states = {}
    for curr_node, next_nodes in relations.items():
        if len(next_nodes) > 1:
            for next_node in next_nodes:
                if dist_states.get(next_node) is None:
                    dist_states[next_node] = list()
                dist_states[next_node].append(states[curr_node])
        else:
            dist_states[next_nodes[0]] = states[curr_node]

    return dist_states


def allocate_returned_states(model, inputs, return_states=None):
    """Allocate output states matrices."""
    seq_len = inputs[list(inputs.keys())[0]].shape[0]

    # pre-allocate states
    if return_states == "all":
        states = {n.name: np.zeros((seq_len, n.output_dim)) for n in model.nodes}
    elif isinstance(return_states, Iterable):
        states = {
            n.name: np.zeros((seq_len, n.output_dim))
            for n in [model[name] for name in return_states]
        }
    else:
        states = {n.name: np.zeros((seq_len, n.output_dim)) for n in model.output_nodes}

    return states


def unfold_mapping(
    data_map: dict[Node, MultiTimeseries]
) -> list[dict[Node, Timeseries]]:
    """Convert a mapping of sequence lists into a list of sequence to nodes mappings."""
    # TODO: extensively test this
    seq_numbers = [len(data_map[n]) for n in data_map.keys()]
    if len(np.unique(seq_numbers)) > 1:
        seq_numbers = {n: len(data_map[n]) for n in data_map.keys()}
        raise ValueError(
            f"Found dataset with inconsistent number of sequences for each node. "
            f"Current number of sequences per node: {seq_numbers}"
        )

    # select an input dataset and check
    n_sequences = len(data_map[list(data_map.keys())[0]])

    mapped_sequences: list[dict[str, Timeseries]] = []
    for i in range(n_sequences):
        sequence = {n: data_map[n][i] for n in data_map.keys()}
        mapped_sequences.append(sequence)

    return mapped_sequences


def fold_mapping(model, states, return_states):
    """Convert a list of sequence to nodes mappings into a mapping of lists or a
    simple array if possible."""
    n_sequences = len(states)
    if n_sequences == 1:
        states_map = states[0]
    else:
        states_map = defaultdict(list)
        for i in range(n_sequences):
            for node_name, seq in states[i].items():
                states_map[node_name] += [seq]

    if len(states_map) == 1 and return_states is None:
        return states_map[model.output_nodes[0].name]

    return states_map


T = TypeVar("T")


def mapping_iterator(
    *x: Mapping[T, Timeseries]
) -> Generator[list[dict[T, Timestep]], Any, None]:
    n_timesteps = x[0][list(x[0].keys())[0]].shape[0]

    for i in range(n_timesteps):
        yield [{k: v[i] for k, v in data.items()} for data in x]


def check_input_output_connections(edges: list[tuple[Node, int, Node]]):
    """Raise a warning if an Input node has an incoming connection or if an
    Output node has an outgoing connection."""
    output_children = [c for p, d, c in edges if d == 0 and isinstance(p, Output)]
    if len(output_children) > 0:
        raise ValueError("An Output node is connected to another Node.")

    input_children = [p for p, d, c in edges if d == 0 and isinstance(c, Input)]
    if len(input_children) > 0:
        raise ValueError("A Node is connected to an Input node.")


def check_unnamed_in_out(model):
    """Raise ValueError if the model has multiple inputs but an input node is not named,
    or if the model has multiple outputs but an output node is not named
    """
    unnamed_inputs = [n for n in model.inputs if n.name is None]
    if len(model.inputs) > 1 and len(unnamed_inputs) > 0:
        raise ValueError(
            f"Model has multiple input nodes but at least one"
            f" of them is not named: {unnamed_inputs[1:-1]}."
        )
    unnamed_outputs = [n for n in model.outputs if n.name is None]
    if len(model.outputs) > 1 and len(unnamed_outputs) > 0:
        raise ValueError(
            f"Model has multiple input nodes but at least one"
            f" of them is not named: {unnamed_outputs[1:-1]}."
        )


def check_unnamed_trainable(model):
    """Raise ValueError if the model has multiple trainable nodes but one of
    them is not named.
    """
    unnamed_nodes = [n for n in model.trainable_nodes if n.name is None]
    if len(model.trainable_nodes) > 1 and len(unnamed_nodes) > 0:
        raise ValueError(
            f"Model has multiple trainable nodes but at least one"
            f" of them is not named: {unnamed_nodes[1:-1]}."
        )
