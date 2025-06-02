"""Type definitions for the mile module."""
import typing
from pathlib import Path
from typing import (
    NamedTuple,
    Protocol,
    TypeVar,
)

import jax
import jax.numpy as jnp
from blackjax.base import SamplingAlgorithm
import optax

ParamTree: typing.TypeAlias = dict[str, typing.Union[jax.Array, 'ParamTree']]
FileTree: typing.TypeAlias = dict[str, typing.Union[Path, 'FileTree']]
PRNGKey = jax.Array


# State in our context is a NamedTuple at least containing a position.
class State(NamedTuple):
    """Base `Sampler State` class for Probabilistic ML.

    Notes
    -----
    - The `Sampler State` is a NamedTuple that contains at least the `position`
        of the sampler.
    """

    position: ParamTree

# cSGLD stuff
class SamplerState(NamedTuple):
    """Sampler state for cSGLD.

    Notes
    -----
    - TODO
    """

    position: ParamTree
    opt_state: optax.OptState

class ScheduleState(NamedTuple):
    """Scheduler state for cSGLD.

    Notes
    -----
    - TODO
    """

    step_size: float | jax.Array
    explore: bool | jax.Array = False

class Scheduler(Protocol):
    """Protocol for Scheduler function.

    Signature:
    ---------
    `(step_count) -> ScheduleState

    Notes
    -----
    - We define some protocols to bring some structure in developing new
        samplers and gradient estimators.
    """

    def __call__(
        self, step_count: int
    ) -> ScheduleState:
        """Scheduler function for warmup."""
        ...

# PosteriorFunction is used in full-batch sampling it only requires position.


class PosteriorFunction(Protocol):
    """Protocol for Posterior Function used in full-batch sampling.

    Signature:
    ---------
    `(position: ParamTree) -> jnp.ndarray`

    Notes
    -----
    - We define some protocols to bring some structure in developing new
        samplers and gradient estimators.
    """

    def __call__(self, position: ParamTree) -> jnp.ndarray:
        """Posterior Function for full-batch sampling."""
        ...


class GradEstimator(Protocol):
    """Protocol for Gradient Estimator function used in mini-batch sampling.

    Signature:
    ---------
    `(position: ParamTree, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray`

    Notes
    -----
    - We define some protocols to bring some structure in developing new
        samplers and gradient estimators.
    - The `batch` is a tuple of (inputs, targets) for the mini-batch sampling.
      (This protocol is used by `blackjax.sgmcmc.gradients.grad_estimator`.)
    """

    def __call__(
        self, position: ParamTree, batch: typing.Tuple[jnp.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Gradient Estimator function for mini-batch sampling."""
        ...


S = TypeVar('S', bound=State)


class Runner(Protocol):
    """Protocol to describe the Runner callables that are used in MCMC sampling.

    Signature:
    ---------
    `(rng_key: jnp.ndarray, state: S, batch: tuple[jnp.ndarray, jnp.ndarray], *args)
    -> S`

    Notes
    -----
        - it simply describes the signature of the callable (the runner) should have.
    """

    def __call__(
        self,
        rng_key: jnp.ndarray,
        state: S,
        batch: tuple[jnp.ndarray, jnp.ndarray],
        *args
    ) -> S:
        """Runner callable for MCMC sampling."""
        ...


Kernel = typing.Callable[..., SamplingAlgorithm]

# Warmup result in mini-batch sampling.
# Warmup Functions must return warmup state and tuned parameters as a dictionary
WarmupResult = tuple[State, dict[str, typing.Any]]
