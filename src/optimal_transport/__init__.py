"""Optimal Transport package

Models, algorithms, and evaluation utilities for optimal transport-based
statistical downscaling.
"""

from .alg1_OT import (
    sinusoidal_time_embedding,
    ConditionerMLP,
    ConditionalSplineCouplingFlow,
    RhoNet,
    NormalizingFlowModel,
    PolicyGradient,
)
from .dgp_OT import (
    TrueDataModelUnimodal,
    TrueDataModelBimodal,
    RobustHedgingModel,
    KSTrueDataModel,
    ARTrueDataModel,
)
from .preprocessing_OT import DataNormalizer
from .utils_OT import (
    EvalBatch,
    save_eval_batch_npz,
    load_eval_batch_npz,
    build_eval_batch,
    compute_traj_dist_metrics_from_batch,
    compute_dist_metrics_all_times_from_batch,
    compute_transition_marginal_kl_from_batch,
    save_transition_kl_to_csv,
    plot_transition_kl,
    compute_adjacent_corr_from_batch,
    plot_adjacent_corrs,
    plot_hist_from_batch,
    run_adjacent_corr_permutation_tests_from_batch,
    evaluate_all_with_one_batch,
)

__all__ = [
    # Algorithm
    "sinusoidal_time_embedding",
    "ConditionerMLP",
    "ConditionalSplineCouplingFlow",
    "RhoNet",
    "NormalizingFlowModel",
    "PolicyGradient",
    # Data generating processes
    "TrueDataModelUnimodal",
    "TrueDataModelBimodal",
    "RobustHedgingModel",
    "KSTrueDataModel",
    "ARTrueDataModel",
    # Preprocessing
    "DataNormalizer",
    # Evaluation utilities
    "EvalBatch",
    "save_eval_batch_npz",
    "load_eval_batch_npz",
    "build_eval_batch",
    "compute_traj_dist_metrics_from_batch",
    "compute_dist_metrics_all_times_from_batch",
    "compute_transition_marginal_kl_from_batch",
    "save_transition_kl_to_csv",
    "plot_transition_kl",
    "compute_adjacent_corr_from_batch",
    "plot_adjacent_corrs",
    "plot_hist_from_batch",
    "run_adjacent_corr_permutation_tests_from_batch",
    "evaluate_all_with_one_batch",
]
