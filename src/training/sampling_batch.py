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
    State,
    Scheduler,
    WarmupResult,
)
from typing import Callable, Optional
from src.dataset.base import BaseLoader

from tqdm import tqdm

logger = logging.getLogger(__name__)


def inference_loop_batch(
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
        rng_key: Random chainwise key.
        init_params: Initial parameters to start the sampling from.
        step_ids: Step ids of the chain to be sampled.
        saving_path: Path to save the samples.
        saving_path_warmup: Path to save the warmup samples, by default None (no saving).

    Note:
        - Currently only supports the `sgld` sampler.
    """
    info: JsonSerializableDict = {}  # Put any information you might need later for analysis

    n_devices = len(step_ids)
    n_warmup = config.warmup_steps
    n_samples = config.n_samples
    n_thinning = config.n_thinning

    if config.epoch_wise_sampling:
        n_train = len(loader.data_train[0])
        batch_size = config.batch_size or n_train
        n_batches = n_train // batch_size
        n_warmup = n_warmup * n_batches
        n_samples = n_samples * n_batches
        n_thinning = n_thinning * n_batches

    # Cyclic learning rate scheduler
    scheduler = config.scheduler
    optimizer = optax.sgd(1.0)

    logger.info(
        f"> Running sampling with the following configuration:"
        f"\n\t- Warmup steps: {n_warmup}"
        f"\n\t- Sampling steps: {n_samples}"
        f"\n\t- Thinning: {n_thinning}"
        f"\n\t- Scheduler: {scheduler}"
    )

    keys = jax.vmap(jax.random.split)(rng_key)

    sampler_warmup = config.warmup_kernel(
        grad_estimator=grad_estimator,
        position=init_params,
    )
    if sampler_warmup is not None:
        raise NotImplementedError(
            "cSGLD has no warm-up."
        )
        # logger.info(f"> Starting {config.name.value} Warmup sampling...")
        # _inference_loop_batch(
        #     rng_key=keys[..., 0],
        #     sampler=sampler_warmup,
        #     loader=loader,
        #     n_samples=n_warmup,
        #     batch_size=config.batch_size,
        #     n_thinning=n_thinning,
        #     step_size=jnp.array(config.step_size),
        #     n_devices=n_devices,
        #     step_ids=step_ids,
        #     saving_path=saving_path_warmup,
        #     scheduler=None,
        #     optimizer=None,
        #     grad_estimator=None,
        #     desc="Warmup",
        # )
        # state_warmup = jax.block_until_ready(sampler_warmup.state)
        # logger.info(f"> {config.name.value} Warmup sampling completed successfully.")

        # if config.name == GetSampler.ADASGHMC:
        #     logger.info(f"Checking balance condition for chains {step_ids}...")
        #     balance_check = jax.vmap(check_balance_condition, in_axes=(0, None, None))(
        #         sampler_warmup.state, config.step_size, config.mdecay
        #     )
        #     if jnp.all(balance_check):
        #         logger.info("Warmup converged to a balanced state.")
        #     else:
        #         logger.warning(
        #             "Warmup didn't converge to a balanced state for chains "
        #             f"{step_ids[~balance_check]}."
        #         )
        # if config.name == GetSampler.SMILE:
        #     logger.info(f"Step sizes from warmup: {state_warmup.step_size} ")

        # sampler = config.kernel(grad_estimator=grad_estimator, state=state_warmup)
    else:
        sampler = config.kernel(grad_estimator=grad_estimator)

    logger.info(f"> Starting {config.name.value} Sampling...")
    _inference_loop_batch(
        rng_key=keys[..., 1],
        sampler=sampler,
        loader=loader,
        n_samples=n_samples,
        batch_size=config.batch_size,
        n_thinning=n_thinning,
        step_size=jnp.array(config.step_size),
        n_devices=n_devices,
        step_ids=step_ids,
        saving_path=saving_path,
        scheduler=None,
        optimizer=None,
        grad_estimator=None,
        desc="Sampling",
    )
    jax.block_until_ready(sampler.state)
    logger.info(f"> {config.name.value} Sampling completed successfully.")

    # Dump Information
    with open(saving_path / "info.pkl", "wb") as f:
        pickle.dump(info, f)


def _inference_loop_batch(
    rng_key: jax.Array,
    sampler: Sampler,
    loader: BaseLoader,
    n_samples: int,
    batch_size: int | None,
    n_thinning: int,
    step_size: float,
    # n_devices: int,
    step_ids: jax.Array,
    saving_path: Optional[Path] = None,
    scheduler: Optional[Scheduler] = None,
    optimizer: Optional[GradientTransformation] = None,
    grad_estimator: Optional[GradEstimator] = None,
    desc: Optional[str] = "Sampling",
):
    """Blackjax inference loop for mini-batch sampling.

    Args:
        rng_key: Random chainwise keys.
        sampler: The sampler to use.
        loader: Data loader for the dataset.
        saving_path: Path to save the samples.
        n_samples: The number of samples.
        batch_size: The batch size.
        n_thinning: How to thin the samples.
        step_size: The step size.
        n_devices: The number of devices.
        step_ids: Chain IDs.
        scheduler: tbd.
        optimizer: tbd.
        grad_estimator: Grad estimator for mini-batch sampling.
        desc: Name of the sampling loop.
    """
    if optimizer is not None:
        opt_state = optimizer.init(sampler.state.position)

    keys = jax.vmap(jax.random.split, in_axes=(0, None))(rng_key, n_samples)
    with tqdm(total=n_samples, desc=desc) as progress_bar:
        _step_count = 0
        while _step_count < n_samples:
            for batch in loader.iter(
                split="train",
                batch_size=batch_size,
                chains=step_ids,
            ):
                if scheduler is not None:
                    scheduler_state = scheduler(_step_count)
                    step_size = jnp.array(scheduler_state.lr)

                    if scheduler_state.explore:
                        logger.debug(f"Exploring with step size: {step_size}")
                        sampler.state, opt_state = one_sgd_step(
                            sampler.state,
                            batch,
                            step_size,
                            grad_estimator,
                            opt_state,
                            optimizer,
                        )
                        _step_count += 1
                        progress_bar.update(1)
                        if _step_count == n_samples:
                            break
                        continue  # Skip the rest of the loop

                logger.debug(f"Sampling with step size: {step_size}")

                sampler.update_state(keys[:, _step_count], batch, step_size)

                if saving_path and (_step_count % n_thinning == 0):
                    for i, chain_id in enumerate(step_ids):
                        save_position(
                            position=jax.tree.map(
                                lambda x: x[i], sampler.state.position
                            ),
                            base=saving_path,
                            idx=chain_id,
                            n=_step_count,
                        )
                progress_bar.update(1)
                _step_count += 1
                if _step_count == n_samples:
                    break


def one_sgd_step(
    state: State,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    step_size: jnp.ndarray,
    grad_estimator: GradEstimator,
    opt_state: Optional[NamedTuple],
    optimizer: Optional[GradientTransformation],
):
    grads = grad_estimator(state.position, batch)
    grads = - 1. * step_size * grads
    updates, new_opt_state = optimizer.update(
        grads, opt_state, state.position
    )
    new_position = optax.apply_updates(state.position, updates)
    new_state = state.replace(position=new_position)
    return new_state, new_opt_state
