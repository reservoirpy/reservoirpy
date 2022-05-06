# Author: Nathan Trouvain at 19/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import defaultdict
from typing import Iterable
from uuid import uuid4

import numpy as np

from .._base import check_xy
from .validation import is_mapping, is_sequence_set


def build_forward_sumodels(nodes, edges, already_trained):
    """Separate unfitted offline nodes from fitted nodes and gather all fitted
    nodes in submodels."""
    from ..model import Model

    offline_nodes = [
        n for n in nodes if n.is_trained_offline and n not in already_trained
    ]

    forward_nodes = list(set(nodes) - set(offline_nodes))
    forward_edges = [edge for edge in edges if edge[1] not in offline_nodes]

    submodel = Model(forward_nodes, forward_edges, name=f"SubModel-{uuid4()}")

    submodel.already_trained = already_trained

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


def to_ragged_seq_set(data):
    """Convert dataset from mapping/array of sequences
    to lists of mappings of sequences."""
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
            if data.ndim < 3:
                return [np.atleast_2d(data)]
            else:
                return data
        else:
            return data


def build_mapping(nodes, data, io_type="input"):
    """Map input/target data to input/trainable nodes in the model."""
    data = to_ragged_seq_set(data)
    if not is_mapping(data):
        if io_type == "input":
            data_map = {n.name: data for n in nodes}
        elif io_type == "target":
            # Remove unsupervised or already fitted nodes from the mapping
            data_map = {n.name: data for n in nodes if not n.unsupervised}
        else:
            raise ValueError(
                f"Unknown io_type: '{io_type}'. "
                f"Accepted io_types are 'input' and 'target'."
            )
    else:
        data_map = data.copy()

    return data_map


def unfold_mapping(data_map):
    """Convert a mapping of sequence lists into a list of sequence to nodes mappings."""
    seq_numbers = [len(data_map[n]) for n in data_map.keys()]
    if len(np.unique(seq_numbers)) > 1:
        seq_numbers = {n: len(data_map[n]) for n in data_map.keys()}
        raise ValueError(
            f"Found dataset with inconsistent number of sequences for each node. "
            f"Current number of sequences per node: {seq_numbers}"
        )

    # select an input dataset and check
    n_sequences = len(data_map[list(data_map.keys())[0]])

    mapped_sequences = []
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


def to_data_mapping(model, X, Y=None):
    """Map dataset to input/target nodes in the model."""
    X_map = build_mapping(model.input_nodes, X, io_type="input")

    Y_map = None
    if Y is not None:
        Y_map = build_mapping(model.trainable_nodes, Y, io_type="target")

    X_map, Y_map = check_xy(model, x=X_map, y=Y_map)

    X_sequences = unfold_mapping(X_map)

    if Y_map is None:
        n_sequences = len(X_sequences)
        Y_sequences = [None] * n_sequences
    else:
        Y_sequences = unfold_mapping(Y_map)

    return X_sequences, Y_sequences
