"""Sampler Configuration."""
import warnings
from dataclasses import dataclass, field
from typing import Any

from src.config.base import BaseConfig, BaseStrEnum
from src.config.warmstart import OptimizerConfig
from src.training.priors import PriorDist
from src.training.schedulers import Scheduler


class Sampler(BaseStrEnum):
    """Sampler Names.

    Notes
    -----
    - This enum class defines the samplers that can be used for Bayesian Inference.
    - To extend the possible samplers, add a new value to the `Sampler` enum.
    - The `get_kernel` and `get_warmup_kernel` methods are used to get the kernel
        and warmup kernel for the sampler respectively.

    - NOTE: The `get_kernel` and `get_warmup_kernel` methods return the kernel
        not the sampler itself. The sampler is later initialized in the
        `src.training.trainer` module.
    """

    NUTS = 'nuts'
    MCLMC = 'mclmc'
    HMC = 'hmc'
    SGLD = 'sgld'

    def get_kernel(self):
        """Get sampling kernel."""
        try:
            from src.training.kernels import KERNELS
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'For Bayesian Inference, samplers are required: '
                'Please install "blackjax" module'
            )
        if self.value not in KERNELS:
            raise NotImplementedError(
                f'Sampler for {self.value} is not yet implemented.'
            )
        return KERNELS[self.value]

    def get_warmup_kernel(self):
        """Get warmup kernel."""
        try:
            from src.training.kernels import WARMUP_KERNELS
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'For Bayesian Inference, samplers are required: '
                'Please install "blackjax" module'
            )
        if self.value not in WARMUP_KERNELS:
            raise NotImplementedError(
                f'Warmup Kernel for {self.value} is not yet implemented.'
            )
        return WARMUP_KERNELS[self.value]


@dataclass(frozen=True)
class PriorConfig(BaseConfig):
    """Configuration for the prior distribution on the model parameters.

    Notes
    -----
    - The `name` should be a `PriorDist` enum value which defines the complete
        prior distribution, it can be a general distribution or a pre-defined one.
        To extend the possible priors, add a new value to the `PriorDist` enum.
        and extend the `get_prior` method accordingly. Through `parameters` field
        the user can pass as many keyword arguments from the configuration file
        as needed for the initialization of the prior distribution.
    """

    name: PriorDist = field(
        default=PriorDist.StandardNormal,
        metadata={'description': 'Prior to Use', 'searchable': True},
    )
    parameters: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            'description': 'Parameters for the prior distribution.',
            'searchable': True,
        },
    )

    def get_prior(self):
        """Get the prior distribution.

        Notes
        -----
        - Get the prior by passing the parameters from the config to `get_prior`
        method of the `PriorDist` enum. See the `PriorDist` enum for more details.
        """
        return self.name.get_prior(**self.parameters)


@dataclass(frozen=True)
class SchedulerConfig(BaseConfig):
    """Scheduler Configuration for the step size."""

    name: Scheduler = field(
        default=Scheduler.CONSTANT,
        metadata={
            'description': 'Scheduler to use for the step size.',
            'searchable': True,
        },
    )
    n_samples_per_cycle: int | None = field(
        default=None,
        metadata={
            'description': (
                'Number of samples per cycle for cyclical schedulers. '
                'If not None, the scheduler will use this value to determine the exploration ratio.'
            ),
            'searchable': True,
        },
    )
    exploration_ratio: float | None = field(
        default=0.0,
        metadata={
            'description': 'Exploration ratio for the scheduler.',
            'searchable': True,
        },
    )
    parameters: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            'description': 'Parameters for the scheduler.',
            'searchable': True,
        },
    )

    def get_scheduler(self, n_steps, step_size_init, **kwargs):
        """Get the scheduler function."""
        return self.name.get_scheduler(
            n_steps=n_steps,
            step_size_init=step_size_init,
            n_samples_per_cycle=self.n_samples_per_cycle,
            exploration_ratio=self.exploration_ratio,
            **kwargs
        )


@dataclass(frozen=True)
class SamplerConfig(BaseConfig):
    """Sampler Configuration."""

    name: Sampler = field(
        default=Sampler.NUTS, metadata={'description': 'Sampler to Use.'}
    )
    epoch_wise_sampling: bool = field(
        default=False,
        metadata={
            'description': 'Perform epoch-wise or batch-wise in minibatch sampling.'
        },
    )
    params_frozen: list[str] = field(
        default_factory=list,
        metadata={
            'description': (
                'Point delimited parameter names in pytree to freeze.'
                '(not yet fully implemented)'
            )
        },
    )
    warmup_steps: int = field(
        default=50,
        metadata={'description': 'Number of warmup steps.', 'searchable': True},
    )
    n_chains: int = field(
        default=2,
        metadata={'description': 'Number of chains to run.', 'searchable': True},
    )
    n_samples: int = field(
        default=1000,
        metadata={'description': 'Number of samples to draw.', 'searchable': True},
    )
    use_warmup_as_init: bool = field(
        default=True,
        metadata={
            'description': 'Use params resulting from warmup as initial for sampling.'
        },
    )
    n_thinning: int = field(
        default=1, metadata={'description': 'Thinning.', 'searchable': True}
    )
    diagonal_preconditioning: bool = field(
        default=False,
        metadata={
            'description': 'Use Diagonal Preconditioning (MCLMC).',
            'searchable': True,
        },
    )
    desired_energy_var_start: float = field(
        default=5e-4,
        metadata={
            'description': 'Desired Energy Variance (MCLMC) at start of lin. decay.',
            'searchable': True,
        },
    )
    desired_energy_var_end: float = field(
        default=1e-4,
        metadata={
            'description': 'Desired Energy Variance (MCLMC) at end of lin. decay.',
            'searchable': True,
        },
    )
    trust_in_estimate: float = field(
        default=1.5,
        metadata={'description': 'Trust in Estimate (MCLMC).', 'searchable': True},
    )
    num_effective_samples: int = field(
        default=100,
        metadata={
            'description': 'Number of Effective Samples (MCLMC).',
            'searchable': True,
        },
    )
    step_size_init: float = field(
        default=0.005,
        metadata={'description': 'Initial Step Size (MCLMC).', 'searchable': True},
    )

    keep_warmup: bool = field(
        default=False, metadata={'description': 'Keep warmup samples.'}
    )
    prior_config: PriorConfig = field(
        default_factory=PriorConfig,
        metadata={'description': 'Prior configuration for the model.'},
    )
    # cSGLD stuff
    scheduler_config: SchedulerConfig = field(
        default_factory=SchedulerConfig,
        metadata={
            'description': 'Scheduler to use for the step size.',
            'searchable': True,
        },
    )
    batch_size: int = field(
        default=128,
        metadata={
            'description': 'Batch size for (c)SGLD sampling.',
            'searchable': True,
        },
    )
    # optimizer_config: OptimizerConfig = field(
    #     default_factory=OptimizerConfig,
    #     metadata={
    #         'description': 'Optimizer configuration for (c)SGLD sampling.',
    #         'searchable': True,
    #     },
    # )

    def __post_init__(self):
        """Post Initialization for the Sampler Configuration."""
        super().__post_init__()
    
    @property
    def prior(self):
        """Get the prior."""
        return self.prior_config.get_prior()
    
    @property
    def scheduler(self):
        """Get the scheduler function."""
        return self.scheduler_config.get_scheduler(
            n_steps=self.n_samples,
            step_size_init=self.step_size_init,
            **self.scheduler_config.parameters
        )
    
    # @property
    # def optimizer(self):
    #     """Get the optimizer for cSGLD exploration."""
    #     try:
    #         import optax  # type: ignore
    #     except ModuleNotFoundError:
    #         raise ModuleNotFoundError(
    #             'For Warmstart Training, optimizers are required: '
    #             'Please install "optax" module'
    #         )
    #     parameters = self.optimizer_config.parameters
    #     parameters.update({'learning_rate': 1.0})  # will be overwritten by scheduler anyway
    #     op = getattr(optax, self.optimizer_config.name.value)(**parameters)
    #     return op

    @property
    def kernel(self):
        """Returns the kernel: see src.training.kernels for more details."""
        return self.name.get_kernel()

    @property
    def warmup_kernel(self):
        """Returns the warmup kernel."""
        return self.name.get_warmup_kernel()

    @property
    def _warmup_dir_name(self):
        """Return the directory name for saving warmup samples."""
        return 'sampling_warmup'

    @property
    def _dir_name(self):
        """Return the directory name for saving samples."""
        return 'samples'
