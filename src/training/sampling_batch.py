"""Toolbox for Handling a (flax) BNN with Blackjax."""
import logging
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from optax import GradientTransformation
from flax.jax_utils import replicate
import flax.linen as nn
from flax.training.train_state import TrainState

from src.config.base import JsonSerializableDict
from src.config.sampler import SamplerConfig, Scheduler
from src.training.callbacks import (
    save_position,
)
from src.types import (
    ParamTree,
)
from typing import Callable
from src.dataset.base import BaseLoader

from tqdm import tqdm

import time

logger = logging.getLogger(__name__)


def inference_loop_batch(
    model: nn.Module,
    grad_estimator: Callable,
    loader: BaseLoader,
    config: SamplerConfig,
    rng_key: jax.Array,
    init_params: ParamTree,
    step_ids: jnp.ndarray,
    saving_path: Path,
):
    """Blackjax inference loop for mini-batch sampling.

    Parameters
    ----------
    model : nn.Module
        Flax model.
    grad_estimator : Callable
        Gradient estimator function.
    loader : BaseLoader
        Data loader to generate batches.
    config : SamplerConfig
        Sampler configuration.
    rng_key : jnp.ndarray
        PRNG key.
    init_params : ParamTree
        Initial parameters to start the sampling from.
    step_ids : jnp.ndarray
        Step ids of the chain to be sampled.
    saving_path : Path
        Path to save the samples.

    Note:
        - Currently only supports the `sgld` sampler.
    """
    start_time = time.perf_counter()

    info: JsonSerializableDict = {
        # no info for now
    }

    n_devices = len(step_ids)
    n_warmup = config.warmup_steps
    n_samples = config.n_samples
    n_thinning = config.n_thinning
    batch_size = config.batch_size
    step_size_init = config.step_size_init

    # epoch-wise sampling is not supported
    if config.epoch_wise_sampling:
        raise NotImplementedError(
            "Epoch-wise sampling is not supported."
        )
    #     n_train = len(loader.data_train[0])
    #     batch_size = config.batch_size or n_train
    #     n_batches = n_train // batch_size
    #     n_warmup = n_warmup * n_batches
    #     n_samples = n_samples * n_batches
    #     n_thinning = n_thinning * n_batches

    # Cyclical scheduler
    schedule_config = config.scheduler_config
    n_cycles = schedule_config.parameters.get('n_cycles', 1)
    cycle_length = n_samples // n_cycles
    n_samples_per_cycle = schedule_config.n_samples_per_cycle or None
    exploration_ratio = schedule_config.exploration_ratio or 0.0
    step_size_sampling = schedule_config.parameters.get('step_size_sampling', step_size_init)

    if n_samples_per_cycle is not None:
        def _explore(step_count: int) -> bool:
            return (step_count % cycle_length) < cycle_length - n_samples_per_cycle
    else:
        def _explore(step_count: int) -> bool:
            return (step_count % cycle_length) / cycle_length < exploration_ratio

    if schedule_config.name == Scheduler.CONSTANT:
        @jax.jit
        def _scheduler_fn(step_count: int) -> float:
            return jax.lax.cond(
                _explore(step_count),
                lambda x: step_size_init,
                lambda x: step_size_sampling,
                1  # dummy operand
            )
    elif schedule_config.name == Scheduler.CYCLICAL:
        # https://blackjax-devs.github.io/sampling-book/algorithms/cyclical_sgld.html#id2
        @jax.jit
        def _scheduler_fn(step_count: int) -> float:
            cos_out = jnp.cos(jnp.pi * (step_count % cycle_length) / cycle_length) + 1
            step_size = 0.5 * cos_out * step_size_init
            return step_size.item()
    else:
        raise ValueError(f"Unsupported scheduler: {schedule_config.name}")

    if config.optimizer_name == "adam":
        logger.info("Using Adam optimizer.")
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=_scheduler_fn)
    elif config.optimizer_name == "sgd":
        logger.info("Using SGD optimizer.")
        optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=_scheduler_fn)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")

    sampler = config.kernel(grad_estimator=grad_estimator)

    # Use TrainState as sampler state, since it will be used in exploration/optimization phases as well.
    # Learing rate scheduling will be handled by the apply_gradients function.
    def _get_init_state(
        apply_fn: Callable,
        params: ParamTree,
        tx: GradientTransformation,
    ) -> TrainState:
        return TrainState.create(
            apply_fn=apply_fn,
            params=params,
            tx=tx
        )
    if n_devices > 1:
        get_init_state = jax.pmap(_get_init_state, static_broadcasted_argnums=(0,2))
    else:
        get_init_state = jax.jit(_get_init_state, static_argnums=(0,2))
    # Initialize sampler state
    state = get_init_state(
        model.apply,
        init_params,
        optimizer
    )

    logger.info(
        f"> Running sampling with the following configuration:"
        f"\n\t- Warmup steps: {n_warmup}"
        f"\n\t- Sampling steps: {n_samples}"
        f"\n\t- Thinning: {n_thinning}"
        f"\n\t- Scheduler: {schedule_config.name.value}"
    )
    logger.info(f"> Starting {config.name.value} Sampling...")

    def one_step(rng_key, state, batch, step_size, explore):
        def sample_step(current):
            rng_key, state, batch, step_size = current
            new_position = sampler.step(rng_key, state.params, batch, step_size) # type: ignore
            return state.replace(params=new_position)

        def explore_step(current):
            _, state, batch, step_size = current
            grads = grad_estimator(state.params, batch)
            grads = jax.tree_map(lambda x: -1. * x, grads)
            return state.apply_gradients(grads=grads)

        new_state = jax.lax.cond(
            explore,
            explore_step,
            sample_step,
            (rng_key, state, batch, step_size) # = current
        )

        return new_state
    

    if n_devices > 1:
        one_step_fn = jax.pmap(one_step)
        schedule_fn = lambda x: replicate(_scheduler_fn(x))
        explore_fn = lambda x: replicate(_explore(x))
    else:
        one_step_fn = jax.jit(one_step)
        schedule_fn = jax.jit(_scheduler_fn)
        explore_fn = jax.jit(_explore)

    cycle_time = []
    with tqdm(total=n_samples, desc="Sampling") as progress_bar:
        step_count = 0
        while step_count < n_samples:
            for batch in loader.iter(split="train", batch_size=batch_size, n_devices=n_devices):
                batch = (batch['feature'], batch['label'])  # sgld.step requires a tuple (X, y)
                step_size = schedule_fn(step_count)
                explore = explore_fn(step_count)

                rng_key, _ = jax.random.split(rng_key)
                if n_devices > 1:
                    sample_key = jax.random.split(rng_key, n_devices)
                else:
                    sample_key = rng_key

                state = one_step_fn(
                    sample_key,
                    state,
                    batch,
                    step_size,
                    explore
                )

                if saving_path and (step_count % n_thinning == 0) \
                        and (_explore(step_count) is not True):
                    if n_devices > 1:
                        # Save each chain's parameters separately
                        for i, chain_id in enumerate(step_ids):
                            save_position(
                                position=jax.tree.map(
                                    lambda x: x[i], state.params
                                ),
                                base=saving_path,
                                idx=chain_id,
                                n=step_count,
                            )
                    else:
                        save_position(
                            position=state.params,
                            base=saving_path,
                            idx=step_ids[0],
                            n=step_count,
                        )

                step_count += 1
                progress_bar.update(1)

                if step_count == n_samples:
                    break

                # Re-init optimizer when we reach the end of a cycle.
                # Resetting the step count to 0 for the optimizer state is not a problem,
                # since the schedule is cyclical anyway.
                if step_count % cycle_length == 0:
                    cycle_time.append(time.perf_counter() - start_time)
                    state = get_init_state(
                        model.apply,
                        state.params,
                        optimizer
                    )

            loader.shuffle()

    jax.block_until_ready(state)
    logger.info(f"> {config.name.value} Sampling completed successfully.")

    info.update({
        'total_time': time.perf_counter() - start_time,
        'cycle_time': cycle_time  # cycle time is not used in current analysis
    })

    # Dump Information
    with open(saving_path / "info.pkl", "wb") as f:
        pickle.dump(info, f)
