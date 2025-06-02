"""Kernels used for training in the src."""
from typing import Callable

from blackjax import (
    hmc,
    mclmc,
    nuts,
    sgld
)
from blackjax.base import SamplingAlgorithm

__all__ = ['nuts', 'hmc', 'mclmc', 'sgld']


KERNELS: dict[str, Callable[..., SamplingAlgorithm]] = {
    'nuts': nuts,
    'hmc': hmc,
    'mclmc': mclmc,
    'sgld': sgld
}

WARMUP_KERNELS: dict[str, Callable[..., SamplingAlgorithm]] = {}
