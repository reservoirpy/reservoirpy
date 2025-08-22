# Author: Nathan Trouvain at 19/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from inspect import signature
from typing import Any, Generator, Mapping, Sequence, TypeVar

import numpy as np

from reservoirpy.type import MultiTimeseries, NodeInput, Timeseries, Timestep

from ..node import Node
from ..nodes import Input, Output


def unfold_mapping(
    data_map: dict[Node, MultiTimeseries],
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


T = TypeVar("T")


def mapping_iterator(*x: Mapping[T, Timeseries]) -> Generator[list[dict[T, Timestep]], Any, None]:
    n_timesteps = x[0][list(x[0].keys())[0]].shape[0]

    for i in range(n_timesteps):
        yield [{k: v[i] for k, v in data.items()} for data in x]


def join_data(*xs: NodeInput) -> NodeInput:
    if isinstance(xs[0], Sequence):
        return [np.concatenate(elements, axis=-1) for elements in zip(*xs)]
    else:
        return np.concatenate(xs, axis=-1)


def data_from_buffer(buffer: np.ndarray, x: NodeInput) -> tuple[np.ndarray, NodeInput]:
    delay = buffer.shape[0]
    if isinstance(x, Sequence):
        buffer_and_x = [np.concatenate((buffer, series), axis=0) for series in x]
        return buffer_and_x[-1][-delay:], [ts[:-delay] for ts in buffer_and_x]
    if x.ndim == 3:
        duplicated_buffer = np.tile(buffer, (x.shape[0], 1, 1))
        buffer_and_x = np.concatenate((duplicated_buffer, x), axis=1)
        return buffer_and_x[-1][-delay:], buffer_and_x[:, :-delay]
    else:
        buffer_and_x = np.concatenate((buffer, x), axis=0)
        return buffer_and_x[-delay:], buffer_and_x[:-delay]


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
            f"Model has multiple input nodes but at least one" f" of them is not named: {unnamed_inputs[1:-1]}."
        )
    unnamed_outputs = [n for n in model.outputs if n.name is None]
    if len(model.outputs) > 1 and len(unnamed_outputs) > 0:
        raise ValueError(
            f"Model has multiple output nodes but at least one" f" of them is not named: {unnamed_outputs[1:-1]}."
        )


def check_unnamed_trainable(model):
    """Raise ValueError if the model has multiple trainable nodes but one of
    them is not named.
    """
    unnamed_nodes = [n for n in model.trainable_nodes if n.name is None]
    if len(model.trainable_nodes) > 1 and len(unnamed_nodes) > 0:
        raise ValueError(
            f"Model has multiple trainable nodes but at least one" f" of them is not named: {unnamed_nodes[1:-1]}."
        )


def obj_from_kwargs(klas, kwargs):
    sig = signature(klas.__init__)
    params = list(sig.parameters.keys())
    klas_kwargs = {n: v for n, v in kwargs.items() if n in params}
    return klas(**klas_kwargs)
