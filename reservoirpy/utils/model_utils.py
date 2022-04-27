# Author: Nathan Trouvain at 19/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Iterable
from uuid import uuid4

import numpy as np

from .validation import is_mapping, is_sequence_set


def build_forward_sumodels(nodes, edges, already_trained):
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
    dist_states = {}
    for curr_node, next_nodes in relations.items():
        for next_node in next_nodes:
            dist_states[next_node] = states[curr_node]
    return dist_states


def allocate_returned_states(model, inputs, return_states=None):

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


def sort_and_unpack(states, return_states=None):
    """Maintain input order (even with parallelization on)"""
    states = sorted(states, key=lambda s: s[0])
    states = {n: [s[1][n] for s in states] for n in states[0][1].keys()}

    for n, s in states.items():
        if len(s) == 1:
            states[n] = s[0]

    if len(states) == 1 and return_states is None:
        states = states[0]

    return states


def pack_sequences(sequences, return_states=None, sort_idxs=False):
    if is_mapping(sequences) or isinstance(sequences, np.ndarray):
        return sequences

    if hasattr(sequences, "__iter__"):
        if sort_idxs:
            sequences = sorted(sequences, key=lambda s: s[0])
            sequences = [s[1] for s in sequences]  # remove index

        if len(sequences) == 1:
            return sequences[0]

        if is_mapping(sequences[0]):
            packed = {k: [] for k in sequences[0].keys()}
            for seq_map in sequences:
                for node_name, sequence in seq_map.items():
                    packed[node_name].append(sequence)

            if len(packed) == 1 and return_states is None:  # only one output node
                packed = packed[list(packed.keys())[0]]
        else:
            packed = sequences
            if len(packed) == 1:  # only one sequence
                return packed[0]

        return packed
    else:
        return sequences
