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


@jax.jit
def calculate_constraint_rmse(
    predicted_samples: jnp.ndarray,
    condition_reference_samples: jnp.ndarray,
) -> float:
    """Compute constraint RMSE pooled over conditions using direct strided indexing.

    Args:
        predicted_samples: Array `(N, C, n_x, d)` of HF predictions.
        condition_reference_samples: Array `(C, n_y, d)` of LR y' per condition.

    Returns:
        Scalar RMSE (mean, std) averaged over conditions.
    """
    n_x = predicted_samples.shape[2]
    n_y = condition_reference_samples.shape[1]
    step = n_x // n_y
    x_lr = predicted_samples[:, :, ::step, :]  # (N, C, n_y, d)
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


def evaluate_all(
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
    num_conditionings = int(run_sett_global["num_conditionings"])
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
            y = np.asarray(f["yp_trajs"][()])[..., 0]
        print(f"Loaded yp_trajs from: {_yp_path}")
        y = y[:num_conditionings]
    else:
        y = np.asarray(true_data_model.y_test)[:num_conditionings]

    samples_nc = np.asarray(samples_raw).reshape(N * C, n_x, d)
    samples_flat = _flatten_channels(samples_nc)
    samples_4d = samples_flat[:, np.newaxis, :, :]

    x_ref = _flatten_channels(x_test_arr)

    gen_flat = samples_flat[:, :, 0]
    ref_flat = x_ref[:, :, 0]

    constraint_rmse, constraint_rmse_sd = calculate_constraint_rmse(
        jnp.asarray(samples_raw), jnp.asarray(y)
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
        corr_gen,
        corr_ref,
        run_sett=run_sett,
        writer=writer,
        first_k=n_x,
        key_suffix=key_suffix,
        out_name=f"adjcorr_{generation_type}",
        x_series=True,
    )
    print(f"Adjacent-corr plot saved to: {plot_path}")

    if writer is not None and hasattr(writer, "write_scalars"):
        scalars = {
            f"metrics/cRMSE{key_suffix}": float(constraint_rmse),
            f"metrics/cRMSE_sd{key_suffix}": float(constraint_rmse_sd),
            f"metrics/W2_avg{key_suffix}": w2_avg,
            f"metrics/W2_avg_sd{key_suffix}": w2_std,
            f"metrics/KS_avg{key_suffix}": ks_avg,
            f"metrics/KS_avg_sd{key_suffix}": ks_std,
            f"metrics/SWD{key_suffix}": swd,
            f"metrics/SWD_sd{key_suffix}": swd_std,
            f"metrics/MMD2{key_suffix}": mmd2,
            f"metrics/MMD2_sd{key_suffix}": mmd2_std,
            f"metrics/Variability{key_suffix}": float(sample_variability),
            f"metrics/Variability_sd{key_suffix}": float(sample_variability_sd),
            f"metrics/MELR_weighted{key_suffix}": float(melr_weighted),
            f"metrics/MELR_weighted_sd{key_suffix}": float(melr_weighted_sd),
            f"metrics/MELR_unweighted{key_suffix}": float(melr_unweighted),
            f"metrics/MELR_unweighted_sd{key_suffix}": float(melr_unweighted_sd),
            f"metrics/Wass1{key_suffix}": float(wass1),
            f"metrics/Wass1_sd{key_suffix}": float(wass1_sd),
            f"metrics/KLD{key_suffix}": float(kld),
            f"metrics/KLD_sd{key_suffix}": float(kld_sd),
        }
        writer.write_scalars(step=0, scalars=scalars)
