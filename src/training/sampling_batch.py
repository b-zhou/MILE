"""Toolbox for Handling a (flax) BNN with Blackjax."""
import logging
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.experimental
import jax.numpy as jnp
from blackjax.mcmc.mclmc import MCLMCInfo
from blackjax.mcmc.nuts import NUTSInfo
import optax
from optax import GradientTransformation
from flax.jax_utils import replicate
import flax.linen as nn
from flax.training.train_state import TrainState

from src.config.base import JsonSerializableDict
from src.config.sampler import Sampler, SamplerConfig
from src.training.callbacks import (
    progress_bar_scan,
    save_position,
)
from src.training.warmup import custom_mclmc_warmup, custom_window_adaptation
from src.types import (
    GradEstimator,
    Kernel,
    NamedTuple,
    ParamTree,
    PosteriorFunction,
    SamplerState,
    ScheduleState,
    WarmupResult,
)
from src.training.schedulers import Scheduler
from typing import Callable, Optional
from src.dataset.base import BaseLoader

from tqdm import tqdm

logger = logging.getLogger(__name__)


def inference_loop_batch(
    model: nn.Module,
    unnorm_log_posterior: Callable,
    grad_estimator: Callable,
    loader: BaseLoader,
    config: SamplerConfig,
    rng_key: jax.Array,
    init_params: ParamTree,
    step_ids: jnp.ndarray,
    saving_path: Path,
    saving_path_warmup: Path | None = None,
):
    """Blackjax inference loop for mini-batch sampling.

    Args:
        grad_estimator: Grad estimator for mini-batch sampling.
        loader: Data loader for the dataset.
        config: Sampler configuration.
        rng_key: Random key.
        init_params: Initial parameters to start the sampling from.
        step_ids: Step ids of the chain to be sampled.
        saving_path: Path to save the samples.
        saving_path_warmup: Path to save the warmup samples, by default None (no saving).

    Note:
        - Currently only supports the `sgld` sampler.
    """
    info: JsonSerializableDict = {
        'scheduler_config': config.scheduler_config.to_dict(),
    }  # Put any information you might need later for analysis

    n_devices = len(step_ids)
    n_warmup = config.warmup_steps
    n_samples = config.n_samples
    n_thinning = config.n_thinning
    batch_size = config.batch_size
    step_size_init = config.step_size_init

    # if config.epoch_wise_sampling:
    #     n_train = len(loader.data_train[0])
    #     batch_size = config.batch_size or n_train
    #     n_batches = n_train // batch_size
    #     n_warmup = n_warmup * n_batches
    #     n_samples = n_samples * n_batches
    #     n_thinning = n_thinning * n_batches

    # Cyclic learning rate scheduler
    schedule_config = config.scheduler_config
    n_cycles = schedule_config.parameters.get('n_cycles', 1)
    cycle_length = n_samples // n_cycles
    n_samples_per_cycle = schedule_config.n_samples_per_cycle or None
    exploration_ratio = schedule_config.exploration_ratio or 0.0
    step_size_sampling = schedule_config.parameters.get('step_size_sampling', step_size_init)

    def _explore(step_count: int) -> bool:
        """Determine if the current step is in the exploration phase."""
        if n_samples_per_cycle is not None:
            return (step_count % cycle_length) < cycle_length - n_samples_per_cycle
        else:
            return (step_count % cycle_length) / cycle_length < exploration_ratio

    if schedule_config.name == Scheduler.CONSTANT:
        def _scheduler_fn(step_count: int) -> jax.Array:
            """Constant step size scheduler."""
            explore = _explore(step_count)
            return jnp.where(explore, step_size_init, step_size_sampling) # type: ignore
    elif schedule_config.name == Scheduler.CYCLICAL:
        # https://blackjax-devs.github.io/sampling-book/algorithms/cyclical_sgld.html#id2
        def _scheduler_fn(step_count: int) -> jax.Array:
            cos_out = jnp.cos(jnp.pi * (step_count % cycle_length) / cycle_length) + 1
            step_size = 0.5 * cos_out * step_size_init
            return step_size
    else:
        raise ValueError(f"Unsupported scheduler: {schedule_config.name}")
    
    # store schedule states in info
    info.update({
        'step_size': [_scheduler_fn(step) for step in range(n_samples)],
        'explore': [_explore(step) for step in range(n_samples)]
    }) # type: ignore
    total_samples = jnp.sum(~jnp.array(info['explore']))
    logger.info(f"Will produce {total_samples} samples.")

    # TODO: use optimizer config
    if config.optimizer_name == "adam":
        logger.info("Using Adam optimizer.")
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=_scheduler_fn)
    elif config.optimizer_name == "sgd":
        logger.info("Using SGD optimizer.")
        optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=_scheduler_fn)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")

    sampler = config.kernel(grad_estimator=grad_estimator)

    def _get_init_state(
        apply_fn: PosteriorFunction,
        params: ParamTree,
        tx: GradientTransformation,
    ) -> TrainState:
        """Initialize the sampler state."""
        return TrainState.create(
            apply_fn=apply_fn,
            params=params,
            tx=tx
        )

    state = jax.pmap(_get_init_state, static_broadcasted_argnums=(0,2))(
        model.apply,
        init_params,
        optimizer
    )

    logger.info(
        f"> Running sampling with the following configuration:"
        f"\n\t- Warmup steps: {n_warmup}"
        f"\n\t- Sampling steps: {n_samples}"
        f"\n\t- Thinning: {n_thinning}"
        # f"\n\t- Scheduler: {scheduler}"
    )
    logger.info(f"> Starting {config.name.value} Sampling...")

    # def log_to_file(mean_grad, is_nan):
    #     with open("nan.log", "a") as f:
    #         f.write(f"Gradient: {mean_grad}\n")
    #         f.write(f"Is NaN: {is_nan}\n")
    #         f.write("=" * 20 + "\n")

    def _get_nan(position, step_size, train=True):
        """Check if the position contains NaN values."""
        is_nan = jax.tree_map(lambda x: jnp.isnan(x).any(), position)
        is_nan, _ = jax.tree_flatten(is_nan)
        if any(is_nan):
            logger.info(f"NaN detected. (train={train}, step_size={step_size})")

    def one_step(rng_key, state, batch, step_size, explore):
        def sample_step(current):
            rng_key, state, batch, step_size = current
            new_position = sampler.step(rng_key, state.params, batch, step_size) # type: ignore
            jax.debug.callback(_get_nan, new_position, step_size, train=False)
            return state.replace(params=new_position)

        def explore_step(current):
            _, state, batch, step_size = current
            grads = grad_estimator(state.params, batch)
            grads = jax.tree_map(lambda x: -1. * x, grads)
            # jax.debug.callback(_get_nan, grads, step_size, train=True)
            return state.apply_gradients(grads=grads)

        new_state = jax.lax.cond(
            explore,
            explore_step,
            sample_step,
            (rng_key, state, batch, step_size) # = current
        )

        return new_state

    def _rmse(pred, truth):
        """Root Mean Squared Error."""
        return jnp.sqrt(jnp.mean((pred - truth) ** 2))
    
    def _log_metrics(step_count, state, batch):
        """Compute and log metrics."""
        curr_logpost = jax.pmap(unnorm_log_posterior)(state.params, batch[0], batch[1])
        preds = jax.pmap(model.apply)({'params': state.params}, batch[0])[..., 0]
        curr_rmse = jax.pmap(_rmse)(preds, batch[1])
        file_path = saving_path.parent / "metrics.csv"
        if not file_path.exists():
            with open(file_path, "w") as f:
                f.write("step_count,log_posterior,rmse,chain\n")
        n_chains = curr_logpost.shape[-1]
        for i in range(n_chains):
            curr_logpost_i = curr_logpost[..., i]
            curr_rmse_i = curr_rmse[..., i]
            with open(file_path, "a") as f:
                f.write(f"{step_count},{curr_logpost_i},{curr_rmse_i},{i}\n")

    with tqdm(total=n_samples, desc="Sampling") as progress_bar:
        step_count = 0
        while step_count < n_samples:
            for batch in loader.iter(
                split="train",
                batch_size=batch_size,
                n_devices=n_devices
            ):
                batch = (batch['feature'], batch['label'])
                # jax.debug.print("\nBatch shape: {batch_shape}\n", batch_shape=batch[0].shape)
                # explore = _explore(step_count)
                step_size = _scheduler_fn(step_count)
                explore = _explore(step_count)
                rng_key, _ = jax.random.split(rng_key)
                
                state = jax.pmap(one_step)(
                    jax.random.split(rng_key, n_devices),
                    state,
                    batch,
                    replicate(step_size),
                    replicate(explore)
                )
                
                # compute metrics
                if saving_path and (step_count % 10 == 0):
                    _log_metrics(step_count, state, batch)

                if saving_path and (step_count % n_thinning == 0) \
                        and (explore is not True):
                    for i, chain_id in enumerate(step_ids):
                        save_position(
                            position=jax.tree.map(
                                lambda x: x[i], state.params
                            ),
                            base=saving_path,
                            idx=chain_id,
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
                    state = jax.pmap(_get_init_state, static_broadcasted_argnums=(0,2))(
                        model.apply,
                        state.params,
                        optimizer
                    )

            loader.shuffle()

    jax.block_until_ready(state)
    logger.info(f"> {config.name.value} Sampling completed successfully.")

    # Dump Information
    with open(saving_path / "info.pkl", "wb") as f:
        pickle.dump(info, f)
