from abc import ABC, abstractmethod
from typing import Any, Generator, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from .type import NodeInput, Timeseries, Timestep, is_multiseries


class Node(ABC):
    initialized: bool
    input_dim: int
    output_dim: int
    state: tuple

    @abstractmethod
    def initialize(
        self, x: Optional[NodeInput | Timestep], y: Optional[NodeInput | Timestep]
    ):
        ...

    def step(self, x: Optional[Timestep]) -> Timestep:
        if not self.initialized:
            self.initialize(x)

        # Auto-regressive mode
        if x is None:
            x = np.empty((0,))

        new_state, output = self._step(self.state, x)

        self.state = new_state
        return output

    def run(self, x: Optional[NodeInput], length: Optional[int] = None):
        if not self.initialized:
            self.initialize(x)

        # Auto-regressive mode
        if x is None:
            x = np.empty((length, 0))

        initial_state = self.state

        if is_multiseries(x):
            result = []
            # TODO: parallelization
            for timeseries in x:
                final_state, output = self._run(initial_state, timeseries)
                result.append(output)

            if isinstance(x, np.ndarray):
                result = np.array(result)
        else:
            final_state, result = self._run(initial_state, x)

        self.state = final_state
        return result

    def _run(self, state: tuple, x: Timeseries) -> Tuple[tuple, Timeseries]:
        current_state = state
        n_timesteps = x.shape[-2]

        output = np.empty((n_timesteps, self.output_dim))
        for i, x_step in enumerate(x):
            current_state, output_step = self._step(state=current_state, x=x_step)
            output[i] = output_step

        return current_state, output

    @abstractmethod
    def _step(self, state: tuple, x: Timestep) -> Tuple[tuple, Timestep]:
        ...


class TrainableNode(Node):
    @abstractmethod
    def fit(self, x: NodeInput, y: Optional[NodeInput]) -> Node:
        ...


class OnlineNode(TrainableNode):
    @abstractmethod
    def partial_fit(self, x: Timeseries, y: Optional[Timeseries]):
        ...

    @abstractmethod
    def fit(self, x: NodeInput, y: Optional[NodeInput]):
        ...


class ParallelNode(TrainableNode, ABC):
    @abstractmethod
    def worker(self, x: Timeseries, y: Optional[Timeseries]):
        ...

    @abstractmethod
    def master(self, generator: Generator):
        ...

    def fit(self, x: NodeInput, y: Optional[NodeInput], workers=1):
        if not self.initialized:
            self.initialize(x, y)

        # Multi-series
        if is_multiseries(x):
            parallel_operator = Parallel(n_jobs=workers, return_as="generator")
            if y is None:
                results = parallel_operator(self.worker(x_ts) for x_ts in x)
            else:
                results = parallel_operator(
                    self.worker(x_ts, y_ts) for x_ts, y_ts in zip(x, y)
                )

        # Single timeseries
        else:
            results = (self.worker(x, y) for _ in range(1))

        self.master(results)
