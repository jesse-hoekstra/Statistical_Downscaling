"""Preprocessing utilities for optimal transport experiments (JAX).

This module contains a small, stateful normalizer that estimates per-dimension
means and standard deviations from trajectories produced by a provided
true-data model. Estimation can be performed in two modes:
  - "time_varying": keep statistics per time step (do not pool over time)
  - "global": pool statistics over time into a single set, then broadcast

To keep memory bounded, statistics are accumulated in chunks by streaming over
randomly generated samples instead of materializing the entire sample set.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


class DataNormalizer:
    """Estimate and apply normalization statistics for (y, y') trajectories.

    Responsibilities:
      - Stream-sample trajectories from a given `true_data_model` and compute
        running sums of values and squared values for both y and y'.
      - Derive per-time (or global) mean, variance, standard deviation, and log determinant for y and y'.
      - Apply normalization (z = (x - mu) / std) with winsor clipping,
        and the inverse transformation.

    Expected `true_data_model` API:
      sample_true_trajectory(key) -> (y, yp)
        y, yp have shape (N+1, d, 1), matching this normalizer's configuration.
    """

    def __init__(self, run_sett: dict, true_data_model):
        """Initialize the normalizer from run settings and a data model.

        Args:
            run_sett: A dictionary with run settings.
            true_data_model: Object providing `sample_true_trajectory(key)`.
        """
        self.preprocessing_sett = run_sett["preprocessing"]
        self.evaluation_sett = run_sett["metrics"]["evaluation"]
        self.global_sett = run_sett["global"]
        self.seed = int(self.global_sett["seed"])
        self.N = int(self.global_sett["N"])
        self.use_data_normalization = bool(
            self.preprocessing_sett["use_data_normalization"]
        )
        self.winsor_clip_z = float(self.preprocessing_sett["winsor_clip_z"])
        self.num_samples = int(self.evaluation_sett["eval_B"])
        self.chunk_size = int(self.evaluation_sett["metrics_chunk_size"])
        self.mode = str(self.preprocessing_sett["mode"])
        self.eps = float(self.preprocessing_sett["eps"])
        self.RNG_NAMESPACE_NORM = int(self.preprocessing_sett["RNG_NAMESPACE_NORM"])
        self.true_data_model = true_data_model
        self.fitted = False

    def fit(self) -> None:
        """Estimate mean/variance/std statistics by streaming over samples.

        Sampling procedure:
          - Derive a namespaced key: fold_in(PRNGKey(seed), RNG_NAMESPACE_NORM)
          - Sample `num_samples` trajectories in chunks of size `chunk_size`
          - Accumulate sums and squared-sums for y and y'
          - If mode == "global": pool across time, elif mode == "time_varying": keep per-time stats

        Sets:
          - mu_y, mu_yp: means, shape (N+1, d, 1)
          - var_y, var_yp: variances with floor eps^2, shape (N+1, d, 1)
          - std_y, std_yp: standard deviations, shape (N+1, d, 1)
          - log_det: sum(log std_y) + sum(log std_yp); useful for change of variables formula due to normalization
        """
        self.fitted = True
        norm_key = jax.random.fold_in(
            jax.random.PRNGKey(self.seed), self.RNG_NAMESPACE_NORM
        )

        N_len = int(self.N + 1)
        mode = str(self.mode).lower()
        num_samples = int(self.num_samples)
        chunk_size = int(self.chunk_size)

        sum_y = sum_yp = sumsq_y = sumsq_yp = None
        total = 0
        cur_key = norm_key
        remaining = num_samples

        while remaining > 0:
            cur = min(remaining, chunk_size)
            cur_key, use_key = jax.random.split(cur_key)
            keys = jax.random.split(use_key, cur)
            y, yp = jax.vmap(self.true_data_model.sample_true_trajectory)(keys)

            if sum_y is None:
                d = y.shape[2]
                sum_y = jnp.zeros((N_len, d, 1), dtype=jnp.float32)
                sumsq_y = jnp.zeros((N_len, d, 1), dtype=jnp.float32)
                sum_yp = jnp.zeros((N_len, d, 1), dtype=jnp.float32)
                sumsq_yp = jnp.zeros((N_len, d, 1), dtype=jnp.float32)

            if mode == "global":
                y_flat = y.reshape((cur * N_len, d, 1))
                yp_flat = yp.reshape((cur * N_len, d, 1))
                sum_y += jnp.sum(y_flat, axis=0, keepdims=True).repeat(N_len, axis=0)
                sumsq_y += jnp.sum(y_flat * y_flat, axis=0, keepdims=True).repeat(
                    N_len, axis=0
                )
                sum_yp += jnp.sum(yp_flat, axis=0, keepdims=True).repeat(N_len, axis=0)
                sumsq_yp += jnp.sum(yp_flat * yp_flat, axis=0, keepdims=True).repeat(
                    N_len, axis=0
                )
                total += cur * N_len
            elif mode == "time_varying":
                sum_y += jnp.sum(y, axis=0)
                sumsq_y += jnp.sum(y * y, axis=0)
                sum_yp += jnp.sum(yp, axis=0)
                sumsq_yp += jnp.sum(yp * yp, axis=0)
                total += cur

            remaining -= cur

        total_f = jnp.array(max(total, 1), dtype=jnp.float32)
        self.mu_y = sum_y / total_f
        self.mu_yp = sum_yp / total_f
        self.var_y = jnp.maximum(sumsq_y / total_f - self.mu_y * self.mu_y, self.eps**2)
        self.var_yp = jnp.maximum(
            sumsq_yp / total_f - self.mu_yp * self.mu_yp, self.eps**2
        )
        self.std_y = jnp.sqrt(self.var_y)
        self.std_yp = jnp.sqrt(self.var_yp)
        self.log_det_y = jnp.sum(jnp.log(self.std_y))
        self.log_det_yp = jnp.sum(jnp.log(self.std_yp))
        self.log_det = self.log_det_y + self.log_det_yp

        return self

    def transform(
        self, mode, y: jnp.ndarray, yp: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Normalize or denormalize (y, y') using fitted statistics.

        Args:
            mode: Either "normalize" or "denormalize".
            y: Array shaped like (N+1, d, 1) or broadcastable to that shape.
            yp: Array shaped like (N+1, d, 1) or broadcastable to that shape.

        Returns:
            Tuple of arrays with the same shapes as inputs:
              - If "normalize": z-scores with optional winsor clipping.
              - If "denormalize": values restored to original scale.

        Notes:
            - If `use_data_normalization` is False, returns inputs unchanged.
            - Requires `fit()` to have been called before normalization.
        """
        if not self.use_data_normalization:
            return y, yp
        assert (
            self.fitted == True
        ), "DataNormalizer must be fitted before transforming data"
        if mode == "normalize":
            y_z = (y - self.mu_y) / self.std_y
            yp_z = (yp - self.mu_yp) / self.std_yp

            def _winsor(z: jnp.ndarray) -> jnp.ndarray:
                """Clip z-scores to [-winsor_clip_z, winsor_clip_z] if enabled."""
                if self.winsor_clip_z is not None and self.winsor_clip_z > 0.0:
                    return jnp.clip(z, -self.winsor_clip_z, self.winsor_clip_z)
                return z

            return _winsor(y_z), _winsor(yp_z)
        elif mode == "denormalize":
            y_denorm = y * self.std_y + self.mu_y
            yp_denorm = yp * self.std_yp + self.mu_yp
            return y_denorm, yp_denorm
        else:
            raise ValueError(
                f"Unknown mode '{mode}'; expected 'normalize' or 'denormalize'."
            )
