# Author: Nathan Trouvain at 19/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from uuid import uuid4
from typing import Iterable

import numpy as np

from ..utils.validation import is_mapping


def _build_forward_sumodels(nodes, edges, already_trained):
    from .model import Model

    offline_nodes = [n for n in nodes
                     if n.is_trained_offline and n not in already_trained]

    forward_nodes = list(set(nodes) - set(offline_nodes))
    forward_edges = [edge for edge in edges if edge[1] not in offline_nodes]

    submodel = Model(forward_nodes, forward_edges, name=f"SubModel-{uuid4()}")

    submodel.already_trained = already_trained

    return submodel, offline_nodes


def _dist_states_to_next_subgraph(states, relations):
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


def _allocate_returned_states(model, inputs=None, return_states=None):

    if inputs is not None:
        if is_mapping(inputs):
            seq_len = inputs[list(inputs.keys())[0]].shape[0]
        else:
            seq_len = inputs.shape[0]
    else:
        raise ValueError("'X' and 'n' parameters can't be None at the "
                         "same time.")

    # pre-allocate states
    if return_states == "all":
        states = {n.name: np.zeros((seq_len, n.output_dim))
                  for n in model.nodes}
    elif isinstance(return_states, Iterable):
        states = {n.name: np.zeros((seq_len, n.output_dim))
                  for n in [model[name]
                            for name in return_states]}
    else:
        states = {n.name: np.zeros((seq_len, n.output_dim))
                  for n in model.output_nodes}

    return states
