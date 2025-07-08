from abc import ABC, abstractmethod
from itertools import repeat
from typing import Iterable, Optional, Union

import numpy as np
from joblib import Parallel, delayed

from .type import NodeInput, Timeseries, Timestep, is_multiseries


class Node(ABC):
    initialized: bool
    input_dim: int
    output_dim: int
    state: tuple
    name: Optional[str] = None

    @abstractmethod
    def initialize(
        self,
        x: Union[NodeInput, Timestep],
        y: Optional[Union[NodeInput, Timestep]] = None,
    ):
        ...

    @abstractmethod
    def _step(self, state: tuple, x: Timestep) -> tuple[tuple, Timestep]:
        ...

    def step(self, x: Optional[Timestep]) -> Timestep:
        # TODO: check input_dim==x.shape for all public functions

        # Auto-regressive mode
        if x is None:
            x = np.empty((0,))

        if not self.initialized:
            self.initialize(x)

        new_state, output = self._step(self.state, x)

        self.state = new_state
        return output

    def run(self, x: Optional[NodeInput], iters: Optional[int] = None):
        # Auto-regressive mode
        if x is None:
            x = np.empty((iters, 0))

        if not self.initialized:
            self.initialize(x)

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

    def _run(self, state: tuple, x: Timeseries) -> tuple[tuple, Timeseries]:
        current_state = state
        n_timesteps = x.shape[-2]

        output = np.empty((n_timesteps, self.output_dim))
        for i, x_step in enumerate(x):
            current_state, output_step = self._step(state=current_state, x=x_step)
            output[i] = output_step

        return current_state, output

    def __call__(self, x: Optional[Timestep]) -> Timestep:
        return self.step(x)


class TrainableNode(Node):
    # TODO: warmup
    @abstractmethod
    def fit(self, x: NodeInput, y: Optional[NodeInput]) -> "TrainableNode":
        ...


class OnlineNode(TrainableNode):
    @abstractmethod
    def partial_fit(self, x: Timeseries, y: Optional[Timeseries]) -> Timeseries:
        ...

    def fit(self, x: NodeInput, y: Optional[NodeInput]) -> "OnlineNode":
        if not self.initialized:
            self.initialize(x, y)

        # TODO: reset interface

        if is_multiseries(x):
            if y is None:
                y = repeat(None)

            for x_ts, y_ts in zip(x, y):
                _y_pred_current = self.partial_fit(x_ts, y_ts)
        else:
            _y_pred = self.partial_fit(x, y)

        return self


class ParallelNode(TrainableNode, ABC):
    @abstractmethod
    def worker(self, x: Timeseries, y: Optional[Timeseries]):
        ...

    @abstractmethod
    def master(self, generator: Iterable):
        ...

    def fit(
        self, x: NodeInput, y: Optional[NodeInput], workers: int = 1
    ) -> "ParallelNode":
        if not self.initialized:
            self.initialize(x, y)

        # Multi-series
        if is_multiseries(x):
            parallel_operator = Parallel(n_jobs=workers, return_as="generator")
            if y is None:
                results = parallel_operator(delayed(self.worker)(x_ts) for x_ts in x)
            else:
                results = parallel_operator(
                    delayed(self.worker)(x_ts, y_ts) for x_ts, y_ts in zip(x, y)
                )

        # Single timeseries
        else:
            results = (self.worker(x, y) for _ in range(1))

        self.master(results)

        return self
