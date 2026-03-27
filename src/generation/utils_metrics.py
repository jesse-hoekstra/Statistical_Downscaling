"""Metrics used to evaluate generated KS samples against reference data.

Provided metrics (pooled over samples/conditions where applicable):
- constraint RMSE: relative error of C x vs. y' per condition, averaged.
- sample variability: sqrt(mean pixel-wise variance) across generated samples.
- KLD (sum over dimensions) via KDE on a fixed grid.
- MELR: mismatch of log energy spectra (weighted or unweighted).
- 1-Wasserstein (per-dimension, histogram-based) averaged over dimensions.
"""

import os
import h5py
import yaml
import numpy as np
from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from jax.scipy.integrate import trapezoid


def _single_calculate_constraint_rmse(
    predicted_samples: jnp.ndarray, condition_reference_samples: jnp.ndarray
) -> float:
    """Relative RMSE for one condition.

    Args:
        predicted_samples: Array `(N, n_y, d)` predicted at LR for one condition.
        condition_reference_samples: Array `(n_y, d)` reference y' for one condition.

    Returns:
        Scalar mean relative RMSE over N samples.
    """
    diff_norm = jnp.linalg.norm(
        predicted_samples - condition_reference_samples[None], axis=(1, 2)
    )
    predicted_norm = jnp.linalg.norm(predicted_samples, axis=(1, 2))
    relative_errors = jnp.where(predicted_norm != 0, diff_norm / predicted_norm, 0.0)
    return jnp.mean(relative_errors)


def calculate_constraint_rmse(
    predicted_samples: jnp.ndarray,
    condition_reference_samples: jnp.ndarray,
    downsampling_type: str = "select",
) -> float:
    """Compute constraint RMSE pooled over conditions.

    Args:
        predicted_samples: Array `(N, C, n_x, d)` of HF predictions.
        condition_reference_samples: Array `(C, n_y, d)` of LR y' per condition.
        downsampling_type: `"select"` (every k-th point, KS) or `"average"`
            (block mean, AR).

    Returns:
        Scalar RMSE (mean, std) averaged over conditions.
    """
    N, C, n_x, d = predicted_samples.shape
    n_y = condition_reference_samples.shape[1]
    step = n_x // n_y
    if downsampling_type == "average":
        x_lr = predicted_samples.reshape(N, C, n_y, step, d).mean(axis=3)
    else:
        x_lr = predicted_samples[:, :, ::step, :]
    vec_c = jax.vmap(_single_calculate_constraint_rmse, in_axes=(1, 0), out_axes=0)(
        x_lr, condition_reference_samples
    )
    return jnp.mean(vec_c), jnp.std(vec_c)


def _single_calculate_sample_variability(generated_samples: jnp.ndarray) -> float:
    """Compute variability for one condition by aggregating across samples.

    Args:
        generated_samples: Array `(N, d, 1)` for a fixed condition.

    Returns:
        Scalar sqrt(mean variance) across spatial positions.
    """
    pixel_wise_variances = jnp.var(generated_samples, axis=0)
    mean_variance = jnp.mean(pixel_wise_variances)
    sample_variability = jnp.sqrt(mean_variance)

    return sample_variability


@jax.jit
def calculate_sample_variability(generated_samples: jnp.ndarray) -> float:
    """Average sample variability across conditions.

    Args:
        generated_samples: Array `(N, C, d, 1)`.

    Returns:
        Scalar mean of variability over conditions.
    """
    vec_c = jax.vmap(_single_calculate_sample_variability, in_axes=(1,), out_axes=0)(
        generated_samples
    )
    return jnp.mean(vec_c), jnp.std(vec_c)


def _single_dimension_calculate_kld(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """KL divergence for one spatial dimension using KDE and trapezoidal rule.

    Args:
        predicted_samples: Array `(N, 1)` for one dimension.
        reference_samples: Array `(M, 1)` for one dimension.
        epsilon: Small positive value to avoid division by zero.

    Returns:
        Scalar KLD D_KL(ref || pred) on a common grid.
    """

    pred_data = jnp.squeeze(predicted_samples)
    ref_data = jnp.squeeze(reference_samples)

    kde_pred = gaussian_kde(pred_data, bw_method="scott")
    kde_ref = gaussian_kde(ref_data, bw_method="scott")

    min_val = jnp.minimum(jnp.min(pred_data), jnp.min(ref_data))
    max_val = jnp.maximum(jnp.max(pred_data), jnp.max(ref_data))
    max_val = jnp.where(max_val == min_val, min_val + 1e-6, max_val)
    grid = jnp.linspace(min_val, max_val, 256)

    pdf_pred = kde_pred(grid)
    pdf_ref = kde_ref(grid)

    mask = pdf_ref > epsilon
    integrand = jnp.where(mask, pdf_ref * jnp.log(pdf_ref / (pdf_pred + epsilon)), 0.0)

    kld_m = trapezoid(integrand, x=grid)

    return kld_m


@jax.jit
def _single_calculate_kld(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Sum of 1D KLD over spatial dimensions for a single pool of samples.

    Args:
        predicted_samples: Array `(N, d, 1)`.
        reference_samples: Array `(M, d, 1)`.
        epsilon: Stability constant.

    Returns:
        Scalar sum of per-dimension KLD.
    """
    if predicted_samples.shape[1] != reference_samples.shape[1]:
        raise ValueError(
            "Predicted and reference samples must have the same number of dimensions (columns)."
        )

    kld_vec = jax.vmap(
        _single_dimension_calculate_kld, in_axes=(1, 1, None), out_axes=0
    )(predicted_samples, reference_samples, epsilon)
    total_kld = jnp.sum(kld_vec)

    return total_kld, jnp.std(kld_vec)


@jax.jit
def calculate_kld_pooled(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """KLD pooled across samples and conditions.

    Pools the (N, C) axes of predictions into a single batch and computes KLD.

    Args:
        predicted_samples: `(N, C, d, 1)`.
        reference_samples: `(M, d, 1)`.
        epsilon: Stability constant.

    Returns:
        Scalar KLD (sum over dimensions).
    """
    num_pooled_samples = predicted_samples.shape[0] * predicted_samples.shape[1]
    num_dimensions = predicted_samples.shape[2]
    pooled_predicted_samples = jnp.reshape(
        predicted_samples,
        (num_pooled_samples, num_dimensions, predicted_samples.shape[3]),
    )
    kld = _single_calculate_kld(pooled_predicted_samples, reference_samples, epsilon)
    return kld


@partial(jax.jit, static_argnames="sample_shape")
def _get_k_grids(sample_shape: tuple):
    """Build FFT frequency grids and radial histogram bins for MELR."""
    freqs = [jnp.fft.fftfreq(n, d=1.0 / n) for n in sample_shape]
    k_grids = jnp.meshgrid(*freqs, indexing="ij")
    k_magnitude = jnp.sqrt(sum(k**2 for k in k_grids))
    # Use a fixed, large number of bins to make it JIT-compatible.
    max_bins = max(sample_shape) // 2 + 1
    k_bins = jnp.arange(0.5, max_bins)

    counts, _ = jnp.histogram(k_magnitude.flatten(), bins=k_bins)

    return k_magnitude, k_bins, counts


def _get_energy_spectrum_for_one_sample(
    sample: jnp.ndarray,
    sample_shape: tuple,
    k_magnitude: jnp.ndarray,
    k_bins: jnp.ndarray,
) -> jnp.ndarray:
    """Compute binned radial energy spectrum for a single sample."""
    sample_reshaped = sample.reshape(sample_shape)

    fft_coeffs = jnp.fft.fftn(sample_reshaped)
    power_spectrum = jnp.abs(fft_coeffs) ** 2

    energy_spectrum, _ = jnp.histogram(
        k_magnitude.flatten(), bins=k_bins, weights=power_spectrum.flatten()
    )

    return energy_spectrum


@partial(jax.jit, static_argnames=["sample_shape", "weighted"])
def calculate_melr_pooled(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    sample_shape: tuple,
    weighted: bool,
    epsilon: float = 1e-10,
) -> jnp.ndarray:
    """Mean energy log-ratio discrepancy (weighted or unweighted) pooled.

    Args:
        predicted_samples: `(N, C, d, 1)` or `(N, d, 1)`.
        reference_samples: `(M, d, 1)`.
        sample_shape: Spatial shape, e.g., `(d,)` for 1D.
        weighted: If True, weight by normalized reference spectrum.
        epsilon: Stability constant for log-ratio.

    Returns:
        Scalar MELR value.
    """
    if predicted_samples.ndim == 3:
        predicted_samples = predicted_samples[:, None, :, :]
    num_pooled_samples = predicted_samples.shape[0] * predicted_samples.shape[1]
    num_dimensions = predicted_samples.shape[2]
    pooled_predicted_samples = jnp.reshape(
        predicted_samples,
        (num_pooled_samples, num_dimensions, predicted_samples.shape[3]),
    )

    pred_clean = jnp.squeeze(pooled_predicted_samples, axis=-1)
    ref_clean = jnp.squeeze(reference_samples, axis=-1)

    k_magnitude, k_bins, counts = _get_k_grids(sample_shape)

    vmapped_spectrum_fn = jax.vmap(
        _get_energy_spectrum_for_one_sample, in_axes=(0, None, None, None)
    )
    E_pred_batch = vmapped_spectrum_fn(pred_clean, sample_shape, k_magnitude, k_bins)
    E_ref_batch = vmapped_spectrum_fn(ref_clean, sample_shape, k_magnitude, k_bins)

    E_pred = jnp.mean(E_pred_batch, axis=0)
    E_ref = jnp.mean(E_ref_batch, axis=0)

    if E_pred.shape[0] != E_ref.shape[0]:
        raise ValueError(
            f"Energy spectrum shapes do not match: E_pred={E_pred.shape}, E_ref={E_ref.shape}"
        )

    log_ratios = jnp.abs(jnp.log((E_pred + epsilon) / (E_ref + epsilon)))

    def weighted_calc():
        weights = E_ref / jnp.sum(E_ref)
        return jnp.sum(weights * log_ratios), jnp.std(weights * log_ratios)

    def unweighted_calc():
        return jnp.mean(log_ratios), jnp.std(log_ratios)

    return jax.lax.cond(weighted, weighted_calc, unweighted_calc)


def _single_dimension_calculate_wass1(
    predicted_samples_1d: jnp.ndarray,
    reference_samples_1d: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """1D Wasserstein-1 on a fixed histogram grid for one dimension."""
    integration_range = [
        -20.0,
        20.0,
    ]
    bins = jnp.linspace(integration_range[0], integration_range[1], num_bins + 1)

    pred_data = jnp.squeeze(predicted_samples_1d)
    ref_data = jnp.squeeze(reference_samples_1d)

    counts_pred, _ = jnp.histogram(pred_data, bins=bins, range=integration_range)
    counts_ref, _ = jnp.histogram(ref_data, bins=bins, range=integration_range)

    total_pred = jnp.sum(counts_pred)
    total_ref = jnp.sum(counts_ref)

    cdf_pred = jnp.cumsum(counts_pred) / (total_pred + 1e-10)
    cdf_ref = jnp.cumsum(counts_ref) / (total_ref + 1e-10)

    cdf_diff = jnp.abs(cdf_pred - cdf_ref)

    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    wass1_m = trapezoid(cdf_diff, x=bin_centers)

    return wass1_m


@partial(jax.jit, static_argnames="num_bins")
def _single_calculate_wass1(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """Average per-dimension 1-Wasserstein distance for a single pool."""
    if predicted_samples.shape[1] != reference_samples.shape[1]:
        raise ValueError(
            "Predicted and reference samples must have the same number of dimensions (columns)."
        )
    if predicted_samples.shape[2] != 1 or reference_samples.shape[2] != 1:
        raise ValueError(
            f"Expected trailing dimension of 1, but got {predicted_samples.shape} and {reference_samples.shape}"
        )

    wass1_vec = jax.vmap(
        _single_dimension_calculate_wass1, in_axes=(1, 1, None), out_axes=0
    )(predicted_samples, reference_samples, num_bins)

    mean_wass1 = jnp.mean(wass1_vec)

    return mean_wass1, jnp.std(wass1_vec)


@partial(jax.jit, static_argnames="num_bins")
def calculate_wass1_pooled(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """Wasserstein-1 pooled across samples and conditions (mean over dims)."""
    num_pooled_samples = predicted_samples.shape[0] * predicted_samples.shape[1]
    num_dimensions = predicted_samples.shape[2]
    pooled_predicted_samples = jnp.reshape(
        predicted_samples,
        (num_pooled_samples, num_dimensions, predicted_samples.shape[3]),
    )

    wass1 = _single_calculate_wass1(
        pooled_predicted_samples, reference_samples, num_bins
    )
    return wass1


def _flatten_channels(arr: np.ndarray) -> np.ndarray:
    """Flatten the channel dimension into the batch dimension.

    (B, n_x, d) -> (B*d, n_x, 1)

    Each spatial channel `d` becomes an independent sample, preserving the
    spatial structure along `n_x`.
    """
    B, n_x, d = arr.shape
    return arr.transpose(0, 2, 1).reshape(B * d, n_x, 1)


def evaluate_sample(
    samples_raw: jnp.ndarray,
    true_data_model,
    data_sett: Dict[str, Any],
    run_sett: Dict[str, Any],
    writer=None,
    key_suffix: str = "",
) -> None:
    """Full evaluation of generated samples against true test data.

    Mirrors the structure of ``evaluate_all_with_one_batch`` in utils_OT but
    operates on two distributions — x_gen vs x_true — instead of four.

    Args:
        samples_raw:      Generated samples, shape ``(N, C, n_x, d)``.
        true_data_model:  Data-model object with ``x_test`` and ``y_test``.
        data_sett:        Data-specific settings dict (e.g. ``run_sett["data_KS"]``).
        run_sett:         Full run-settings dict.
        writer:           Optional metric writer (e.g. WandbWriter).
        key_suffix:       String appended to all logged metric keys and file names.
    """
    from src.optimal_transport.utils_OT import (
        _w2_1d_sq,
        _ks_1d,
        _sliced_wasserstein_w2,
        _mmd2_rbf_rff,
        _median_heuristic_sigma,
        _adjacent_corr_from_trajs_np,
        plot_adjacent_corrs,
        _append_row_csv,
    )

    run_sett_global = run_sett["global"]
    run_sett_metrics = run_sett["metrics"]
    base_dir = run_sett.get("work_dir", os.getcwd())
    os.makedirs(base_dir, exist_ok=True)

    seed = int(run_sett_global["seed"])
    num_conditionings = int(data_sett["num_conditionings"])
    generation_type = str(run_sett_global["generation_type"])
    n_x = int(data_sett["n_x"])

    N, C, _, d = samples_raw.shape

    x_test_arr = np.asarray(true_data_model.x_test)

    if run_sett_global.get("debiased_conditioning", False):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        ot_settings_path = os.path.join(
            project_root, "src/optimal_transport/settings_OT.yaml"
        )
        with open(ot_settings_path, "r") as f:
            run_sett_ot = yaml.safe_load(f)
        seed_ot = int(run_sett_ot["global"]["seed"])
        ot_run_name = f"run_seed{seed_ot}"
        ot_dir = os.path.join(project_root, "main_OT", ot_run_name)
        _yp_path = os.path.join(ot_dir, "yp_trajs.h5")
        with h5py.File(_yp_path, "r") as f:
            y = np.asarray(f["yp_trajs"][()])
        print(f"Loaded yp_trajs from: {_yp_path}")
        y = y[:num_conditionings]
        if y.ndim == 4 and y.shape[-1] == 1:
            y = y[..., 0]  # (C, n_y, d, 1) -> (C, n_y, d)
    else:
        y = np.asarray(true_data_model.y_test)[:num_conditionings]

    samples_nc = np.asarray(samples_raw).reshape(N * C, n_x, d)
    samples_flat = _flatten_channels(samples_nc)
    samples_4d = samples_flat[:, np.newaxis, :, :]

    x_ref = _flatten_channels(x_test_arr)

    gen_flat = samples_flat[:, :, 0]
    ref_flat = x_ref[:, :, 0]

    constraint_rmse, constraint_rmse_sd = calculate_constraint_rmse(
        jnp.asarray(samples_raw),
        jnp.asarray(y),
        downsampling_type=str(data_sett.get("downsampling_type", "select")),
    )

    samples_for_var = jnp.asarray(samples_raw).reshape(N, C, n_x * d, 1)
    sample_variability, sample_variability_sd = calculate_sample_variability(
        samples_for_var
    )

    w2_list = [_w2_1d_sq(gen_flat[:, t], ref_flat[:, t]) for t in range(n_x)]
    ks_list = [_ks_1d(gen_flat[:, t], ref_flat[:, t]) for t in range(n_x)]
    w2_avg, w2_std = float(np.mean(w2_list)), float(np.std(w2_list))
    ks_avg, ks_std = float(np.mean(ks_list)), float(np.std(ks_list))

    swd_num_proj = int(run_sett_metrics.get("swd_num_proj", 256))
    swd, _, swd_std = _sliced_wasserstein_w2(
        gen_flat, ref_flat, num_proj=swd_num_proj, seed=seed
    )

    mmd_max_pairs = int(run_sett_metrics.get("mmd_max_pairs", 4096))
    mmd_rff_features = int(run_sett_metrics.get("mmd_rff_features", 256))
    sigma_mmd = _median_heuristic_sigma(
        gen_flat, ref_flat, seed=seed, max_pairs=mmd_max_pairs
    )
    mmd2_runs = [
        _mmd2_rbf_rff(
            gen_flat,
            ref_flat,
            sigma=sigma_mmd,
            num_features=mmd_rff_features,
            seed=seed + i,
        )
        for i in range(5)
    ]
    mmd2, mmd2_std = float(np.mean(mmd2_runs)), float(np.std(mmd2_runs))

    epsilon = float(run_sett_metrics.get("epsilon", 1e-5))
    jnp_samples_4d = jnp.asarray(samples_4d)
    jnp_x_ref = jnp.asarray(x_ref)
    melr_weighted, melr_weighted_sd = calculate_melr_pooled(
        jnp_samples_4d, jnp_x_ref, sample_shape=(n_x,), weighted=True, epsilon=epsilon
    )
    melr_unweighted, melr_unweighted_sd = calculate_melr_pooled(
        jnp_samples_4d, jnp_x_ref, sample_shape=(n_x,), weighted=False, epsilon=epsilon
    )

    wass1_num_bins = int(run_sett_metrics.get("wass1_num_bins", 1000))
    wass1, wass1_sd = calculate_wass1_pooled(
        jnp_samples_4d, jnp_x_ref, num_bins=wass1_num_bins
    )

    kld, kld_sd = calculate_kld_pooled(jnp_samples_4d, jnp_x_ref, epsilon=epsilon)

    print(
        f"cRMSE:           {float(constraint_rmse):.6f} (sd: {float(constraint_rmse_sd):.6f})"
    )
    print(f"W2_avg:          {w2_avg:.6f} (sd: {w2_std:.6f})")
    print(f"KS_avg:          {ks_avg:.6f} (sd: {ks_std:.6f})")
    print(f"SWD:             {swd:.6f} (sd: {swd_std:.6f})")
    print(f"MMD2:            {mmd2:.6f} (sd: {mmd2_std:.6f})")
    print(
        f"Variability:     {float(sample_variability):.6f} (sd: {float(sample_variability_sd):.6f})"
    )
    print(
        f"MELR_weighted:   {float(melr_weighted):.6f} (sd: {float(melr_weighted_sd):.6f})"
    )
    print(
        f"MELR_unweighted: {float(melr_unweighted):.6f} (sd: {float(melr_unweighted_sd):.6f})"
    )
    print(f"Wass1:           {float(wass1):.6f} (sd: {float(wass1_sd):.6f})")
    print(f"KLD:             {float(kld):.6f} (sd: {float(kld_sd):.6f})")

    csv_path = os.path.join(base_dir, f"eval_metrics_{generation_type}{key_suffix}.csv")
    _append_row_csv(
        csv_path,
        {
            "generation_type": generation_type,
            "cRMSE": float(constraint_rmse),
            "cRMSE_sd": float(constraint_rmse_sd),
            "W2_avg": w2_avg,
            "W2_avg_sd": w2_std,
            "KS_avg": ks_avg,
            "KS_avg_sd": ks_std,
            "SWD": swd,
            "SWD_sd": swd_std,
            "MMD2": mmd2,
            "MMD2_sd": mmd2_std,
            "Variability": float(sample_variability),
            "Variability_sd": float(sample_variability_sd),
            "MELR_weighted": float(melr_weighted),
            "MELR_weighted_sd": float(melr_weighted_sd),
            "MELR_unweighted": float(melr_unweighted),
            "MELR_unweighted_sd": float(melr_unweighted_sd),
            "Wass1": float(wass1),
            "Wass1_sd": float(wass1_sd),
            "KLD": float(kld),
            "KLD_sd": float(kld_sd),
        },
    )
    print(f"Metrics CSV saved to: {csv_path}")

    corr_gen = _adjacent_corr_from_trajs_np(samples_nc)
    corr_ref = _adjacent_corr_from_trajs_np(x_test_arr)
    plot_path = plot_adjacent_corrs(
        run_sett=run_sett,
        corr_flow=corr_gen,
        corr_true=corr_ref,
        writer=writer,
        first_k=n_x,
        key_suffix=key_suffix,
        out_name=f"adjcorr_{generation_type}",
        x_series=True,
    )
    print(f"Adjacent-corr plot saved to: {plot_path}")

    metrics_dict = {
        "cRMSE": float(constraint_rmse),
        "cRMSE_sd": float(constraint_rmse_sd),
        "W2_avg": w2_avg,
        "W2_avg_sd": w2_std,
        "KS_avg": ks_avg,
        "KS_avg_sd": ks_std,
        "SWD": swd,
        "SWD_sd": swd_std,
        "MMD2": mmd2,
        "MMD2_sd": mmd2_std,
        "Variability": float(sample_variability),
        "Variability_sd": float(sample_variability_sd),
        "MELR_weighted": float(melr_weighted),
        "MELR_weighted_sd": float(melr_weighted_sd),
        "MELR_unweighted": float(melr_unweighted),
        "MELR_unweighted_sd": float(melr_unweighted_sd),
        "Wass1": float(wass1),
        "Wass1_sd": float(wass1_sd),
        "KLD": float(kld),
        "KLD_sd": float(kld_sd),
    }

    if writer is not None and hasattr(writer, "write_scalars"):
        scalars = {f"metrics/{k}{key_suffix}": v for k, v in metrics_dict.items()}
        writer.write_scalars(step=0, scalars=scalars)

    return corr_gen, metrics_dict


def evaluate_all_samples(
    work_dir: str,
    true_data_model,
    data_sett: Dict[str, Any],
    run_sett: Dict[str, Any],
    writer=None,
    key_suffix: str = "",
) -> None:
    """Evaluate all generation types and produce a combined CSV and comparison plot.

    Expects h5 files in ``work_dir``:
      - ``samples_unconditional.h5``
      - ``samples_wan_conditional_biased.h5``
      - ``samples_wan_conditional_debiased.h5``

    Writes a combined CSV to ``work_dir/eval_all/eval_metrics_all.csv`` and a
    multi-series adjacent-correlation plot to ``work_dir/eval_all/adjcorr/``.
    """
    import copy

    from src.optimal_transport.utils_OT import (
        _adjacent_corr_from_trajs_np,
        plot_adjacent_corrs,
        _append_row_csv,
    )

    n_x = int(data_sett["n_x"])
    combined_dir = os.path.join(work_dir, "eval_all")
    os.makedirs(combined_dir, exist_ok=True)

    corr_flows = {}

    configs = [
        ("unconditional", False, None),
        ("wan_conditional", False, "biased"),
        ("wan_conditional", True, "debiased"),
    ]

    all_rows = []
    samples_dict = {}

    for gen_type, debiased, bias_tag in configs:
        if gen_type == "unconditional":
            h5_name = "samples_unconditional.h5"
            label = "unconditional"
            eval_subdir = os.path.join(work_dir, "unconditional")
        else:
            h5_name = f"samples_{gen_type}_{bias_tag}.h5"
            label = f"{gen_type}_{bias_tag}"
            eval_subdir = os.path.join(work_dir, label)

        h5_path = os.path.join(work_dir, h5_name)
        if not os.path.exists(h5_path):
            print(f"[WARN] h5 not found, skipping: {h5_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        with h5py.File(h5_path, "r") as f:
            samples_raw = jnp.asarray(f["samples"][()])

        patched = copy.deepcopy(run_sett)
        patched["global"]["generation_type"] = gen_type
        patched["global"]["debiased_conditioning"] = debiased
        os.makedirs(eval_subdir, exist_ok=True)
        patched["work_dir"] = eval_subdir

        corr_gen, metrics = evaluate_sample(
            samples_raw=samples_raw,
            true_data_model=true_data_model,
            data_sett=data_sett,
            run_sett=patched,
            writer=writer,
            key_suffix=key_suffix,
        )
        corr_flows[label] = corr_gen
        all_rows.append({"generation_type": label, **metrics})
        samples_dict[label] = samples_raw

    if not all_rows:
        print("[WARN] No sample files found; nothing to write.")
        return

    combined_csv = os.path.join(combined_dir, f"eval_metrics_all{key_suffix}.csv")
    for row in all_rows:
        _append_row_csv(combined_csv, row)
    print(f"\nCombined metrics CSV saved to: {combined_csv}")

    patched_plot = copy.deepcopy(run_sett)
    patched_plot["work_dir"] = combined_dir
    display_labels = ["uDfn", "cDfn", "OT+cDfn"]
    corr_test = _adjacent_corr_from_trajs_np(np.asarray(true_data_model.x_test))
    plot_path = plot_adjacent_corrs(
        run_sett=patched_plot,
        corr_flows=corr_flows,
        corr_test=corr_test,
        writer=writer,
        first_k=n_x,
        key_suffix=key_suffix,
        out_name="adjcorr_all",
        compare_all_x=True,
        labels=display_labels,
    )
    print(f"Combined adjacent-corr plot saved to: {plot_path}")

    density_paths = plot_marginal_densities(
        samples_dict=samples_dict,
        run_sett=patched_plot,
        data_sett=data_sett,
        x_test=np.asarray(true_data_model.x_test),
        writer=writer,
        key_suffix=key_suffix,
        out_name="marginal_densities",
        labels=display_labels,
    )
    for p in density_paths:
        print(f"Marginal density plot saved to: {p}")


def plot_marginal_densities(
    samples_dict: Dict[str, np.ndarray],
    run_sett: Dict[str, Any],
    data_sett: Dict[str, Any],
    x_test: np.ndarray = None,
    writer=None,
    key_suffix: str = "",
    out_name: str = "marginal_densities",
    positions: list = None,
    labels: list = None,
) -> list:
    """Marginal density plots at selected spatial positions vs. ground truth.

    For each channel d, produces a figure with one subplot per position.  Each
    subplot shows one histogram per generation type plus a ground-truth reference:

    - ``data_model == "ar"``: closed-form Gaussian AR(1) marginal.
    - Otherwise: empirical marginal of ``x_test`` (kernel-density estimate).

    The AR(1) marginal starting from Y_0 ~ N(mu_i, sigma^2):
      E[Y_n]   = mu_i * (1 - phi^{n+1}) / (1 - phi)
      Var[Y_n] = sigma^2 * (1 - phi^{2(n+1)}) / (1 - phi^2)
    with channel mean  mu_i = -2 + 4 * i / (d - 1)  (0-indexed channel i).

    Args:
        samples_dict: Mapping label -> array of shape ``(N, C, n_x, d)``.
        run_sett:     Full run-settings dict.
        data_sett:    Data-specific settings dict.
        x_test:       Test data array of shape ``(N, n_x, d)`` or ``(N, C, n_x, d)``;
                      used as empirical ground truth for non-AR models.
        writer:       Optional metric writer.
        key_suffix:   String appended to output file names.
        out_name:     Base name for output files.
        positions:    Spatial indices to plot; defaults to ``[0, 100, 200, n_x-1]``.

    Returns:
        List of saved file paths, one per channel.
    """
    import matplotlib.pyplot as plt

    data_model = str(run_sett["global"]["data_model"]).strip().lower()

    # Flatten each entry to (N*C, n_x, d)
    flat = {
        label: np.asarray(arr).reshape(-1, *np.asarray(arr).shape[2:])
        for label, arr in samples_dict.items()
    }

    first = next(iter(flat.values()))
    n_x, d = first.shape[1], first.shape[2]

    if positions is None:
        positions = [18, 36, 54] if data_model == "ks" else [100, 150, 200]
    positions = [min(p, n_x - 1) for p in positions]

    # AR-only: precompute closed-form marginal helpers
    if data_model == "ar":
        phi = float(run_sett["data_AR"]["phi"])
        sigma = 0.5  # N(0, 0.5^2) innovations
        channel_means = np.linspace(-2.0, 2.0, d)

        def _channel_mean(i):
            return float(channel_means[i])

        def _gt_mean_at_n(mu_i, n):
            if abs(phi - 1.0) < 1e-8:
                return mu_i * (n + 1)
            return mu_i * (1.0 - phi ** (n + 1)) / (1.0 - phi)

        def _gt_std_at_n(n):
            if abs(phi**2 - 1.0) < 1e-8:
                return sigma * np.sqrt(float(n + 1))
            return sigma * np.sqrt((1.0 - phi ** (2 * (n + 1))) / (1.0 - phi**2))

    # Non-AR: flatten x_test to (N_total, n_x, d)
    if data_model != "ar":
        if x_test is None:
            raise ValueError("x_test must be provided for non-AR data models")
        x_test_flat = np.asarray(x_test).reshape(-1, *np.asarray(x_test).shape[-2:])

    colors = ["steelblue", "darkorange", "crimson"]
    if labels is None:
        labels = list(flat.keys())

    base_dir = run_sett.get("work_dir", os.getcwd())
    out_dir = os.path.join(base_dir, "marginal_densities")
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []

    for dim in range(d):
        n_pos = len(positions)
        _, axes = plt.subplots(1, n_pos, figsize=(4 * n_pos, 4), sharey=False)
        if n_pos == 1:
            axes = [axes]

        for ax, p in zip(axes, positions):
            for (_, samples), color, label in zip(flat.items(), colors, labels):
                ax.hist(
                    samples[:, p, dim],
                    bins=60,
                    density=True,
                    color=color,
                    alpha=0.4,
                    label=label,
                )

            if data_model == "ar":
                mu_i = _channel_mean(dim)
                gt_mu = _gt_mean_at_n(mu_i, p)
                gt_sigma = _gt_std_at_n(p)
                x_range = np.linspace(
                    gt_mu - 4.5 * gt_sigma, gt_mu + 4.5 * gt_sigma, 300
                )
                gt_pdf = np.exp(-0.5 * ((x_range - gt_mu) / gt_sigma) ** 2) / (
                    gt_sigma * np.sqrt(2 * np.pi)
                )
                ax.plot(
                    x_range,
                    gt_pdf,
                    color="forestgreen",
                    linestyle="--",
                    linewidth=2,
                    label="Ground truth",
                )
            else:
                gt_vals = x_test_flat[:, p, dim]
                counts, bin_edges = np.histogram(gt_vals, bins=60, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax.plot(
                    bin_centers,
                    counts,
                    color="forestgreen",
                    linestyle="--",
                    linewidth=2,
                    label="Empirical",
                )

            ax.set_xlabel(f"$x_{{{p},{dim+1}}}$")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{out_name}_d{dim + 1}{key_suffix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Marginal density plot saved to: {out_path}")
        out_paths.append(out_path)

        if writer is not None and hasattr(writer, "write_images"):
            writer.write_images(images={f"{out_name}_d{dim + 1}{key_suffix}": out_path})

    return out_paths
