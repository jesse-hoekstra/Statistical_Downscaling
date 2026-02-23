"""Metrics used to evaluate generated KS samples against reference data.

Provided metrics (pooled over samples/conditions where applicable):
- constraint RMSE: relative error of C x vs. y' per condition, averaged.
- sample variability: sqrt(mean pixel-wise variance) across generated samples.
- KLD (sum over dimensions) via KDE on a fixed grid.
- MELR: mismatch of log energy spectra (weighted or unweighted).
- 1-Wasserstein (per-dimension, histogram-based) averaged over dimensions.
"""

import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from jax.scipy.integrate import trapezoid
import jax
from functools import partial
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/generation/settings_GEN.yaml")
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)


def _single_calculate_constraint_rmse(
    predicted_samples: jnp.ndarray, condition_reference_samples: jnp.ndarray
) -> float:
    """Relative RMSE for one condition.

    Args:
        predicted_samples: Array `(d, 1)` or `(d,)` predicted at LR after C.
        condition_reference_samples: Array `(d, 1)` or `(d,)` reference y'.

    Returns:
        Scalar relative RMSE: ||p - r||_2 / ||p||_2 (0 if ||p||_2 == 0).
    """

    diff_norm = jnp.linalg.norm(predicted_samples - condition_reference_samples, axis=1)
    predicted_norm = jnp.linalg.norm(predicted_samples, axis=1)
    relative_errors = jnp.where(predicted_norm != 0, diff_norm / predicted_norm, 0.0)
    return jnp.mean(relative_errors)


@jax.jit
def calculate_constraint_rmse(
    predicted_samples: jnp.ndarray,
    condition_reference_samples: jnp.ndarray,
    C: jnp.ndarray,
) -> float:
    """Compute constraint RMSE pooled over conditions.

    Applies C to predicted HF samples and compares to provided LR conditions.

    Args:
        predicted_samples: Array `(N, C, d, 1)` of HF predictions.
        condition_reference_samples: Array `(C, d', 1)` of LR y' per condition.
        C: Observation matrix with shape `(d', d)`.

    Returns:
        Scalar RMSE averaged over conditions.
    """
    x = jnp.squeeze(predicted_samples, -1)
    C = C.astype(x.dtype)
    Cx = jnp.einsum("ncd,od->nco", x, C)
    predicted_samples_red_dim = Cx[..., None]
    vec_c = jax.vmap(_single_calculate_constraint_rmse, in_axes=(1, 0), out_axes=0)(
        predicted_samples_red_dim, condition_reference_samples
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
    predicted_samples: jnp.ndarray,  # Shape: (N, C, D, 1) e.g., (10, 2, 192, 1)
    reference_samples: jnp.ndarray,  # Shape: (M, D, 1) e.g., (163840, 192, 1)
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

    pred_clean = jnp.squeeze(pooled_predicted_samples, axis=-1)  # Shape: (N*C, D)
    ref_clean = jnp.squeeze(reference_samples, axis=-1)  # Shape: (M, D)

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

    # Vectorize the 1D calculation over all dimensions (axis=1)
    # in_axes=(1, 1, None) maps over dimension 'd' for both samples
    # and passes the static 'num_bins' argument
    wass1_vec = jax.vmap(
        _single_dimension_calculate_wass1, in_axes=(1, 1, None), out_axes=0
    )(predicted_samples, reference_samples, num_bins)

    # Average the Wass1 metric across all dimensions
    mean_wass1 = jnp.mean(wass1_vec)

    return mean_wass1, jnp.std(wass1_vec)


@partial(jax.jit, static_argnames="num_bins")
def calculate_wass1_pooled(
    predicted_samples: jnp.ndarray,
    reference_samples: jnp.ndarray,
    num_bins: int = 1000,
) -> float:
    """Wasserstein-1 pooled across samples and conditions (mean over dims)."""
    # Pool the N (batch) and C (condition) axes
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
