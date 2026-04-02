"""Generation package

Utilities and solvers for generative modeling, unconditional and conditional.
"""

from .data_utils import get_raw_datasets, get_dataset, get_train_test
from .sampler_utils import sample_unconditional, sample_conditional
from .denoiser_utils import (
    create_denoiser_model,
    create_diffusion_scheme,
    restore_denoise_fn,
    build_model,
    build_trainer,
    run_training,
    vp_linear_beta_schedule,
)
from .metrics_utils import (
    calculate_constraint_rmse,
    calculate_sample_variability,
    calculate_kld_pooled,
    calculate_wass1_pooled,
    evaluate_sample,
    evaluate_all_samples,
    plot_marginal_densities,
)

__all__ = [
    # Data
    "get_raw_datasets",
    "get_dataset",
    "get_train_test",
    # Sampling
    "sample_unconditional",
    "sample_conditional",
    # Model construction
    "create_denoiser_model",
    "create_diffusion_scheme",
    "restore_denoise_fn",
    "build_model",
    "build_trainer",
    "run_training",
    "vp_linear_beta_schedule",
    # Metrics
    "calculate_constraint_rmse",
    "calculate_sample_variability",
    "calculate_kld_pooled",
    "calculate_wass1_pooled",
    "evaluate_sample",
    "evaluate_all_samples",
    "plot_marginal_densities",
]
