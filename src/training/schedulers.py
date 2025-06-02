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
        if self.value == self.CONSTANT:
            return lambda step_count: ScheduleState(step_size=step_size_init, explore=False)

        elif self.value == self.CYCLICAL:
            n_cycles = kwargs.get('n_cycles', 4)
            exploration_ratio = kwargs.get('exploration_ratio', 0.1)

            cycle_length = n_steps // n_cycles
            # https://blackjax-devs.github.io/sampling-book/algorithms/cyclical_sgld.html#id2
            def _scheduler_fn(step_count):
                do_sample = False
                if ((step_count % cycle_length)/cycle_length) >= exploration_ratio:
                    do_sample = True

                cos_out = jnp.cos(jnp.pi * (step_count % cycle_length) / cycle_length) + 1
                step_size = 0.5 * cos_out * step_size_init

                return ScheduleState(step_size=step_size, explore=do_sample)

            return _scheduler_fn
        
        else:
            raise NotImplementedError(f"Scheduler {self.value} is not implemented.")
        