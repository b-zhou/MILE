"""Schedulers."""
from typing import Callable

import jax.nn.initializers as jinit
import jax.numpy as jnp

from src.config.base import BaseStrEnum
from src.types import ScheduleState

class Scheduler(BaseStrEnum):
    CONSTANT = 'Constant'
    CYCLICAL = 'Cyclical'

    def get_scheduler(
            self,
            n_steps: int = 1000,
            step_size_init: float = 1e-3,
            **kwargs
        ) -> Callable[[int], ScheduleState]:
        """Return a scheduler function."""

        n_cycles = kwargs.get('n_cycles', 1)
        exploration_ratio = kwargs.get('exploration_ratio', 0.0)
        cycle_length = n_steps // n_cycles

        def _explore(step_count: int) -> bool:
            if exploration_ratio == 0.0:
                return False
            return (step_count % cycle_length) / cycle_length <= exploration_ratio
            
        if self.value == Scheduler.CONSTANT:
            step_size_sampling = kwargs.get('step_size_sampling', step_size_init)
            def _scheduler_fn(step_count: int) -> ScheduleState:
                """Constant step size scheduler."""
                explore = _explore(step_count)
                step_size = step_size_init if explore else step_size_sampling
                return ScheduleState(
                    step_size=step_size,
                    explore=explore
                )
        elif self.value == Scheduler.CYCLICAL:
            # https://blackjax-devs.github.io/sampling-book/algorithms/cyclical_sgld.html#id2
            def _scheduler_fn(step_count: int) -> ScheduleState:
                cos_out = jnp.cos(jnp.pi * (step_count % cycle_length) / cycle_length) + 1
                step_size = 0.5 * cos_out * step_size_init
                return ScheduleState(
                    step_size=step_size,
                    explore=_explore(step_count)
                )

        return _scheduler_fn
