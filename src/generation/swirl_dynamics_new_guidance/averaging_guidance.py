"""Guidance transform for block-averaging downsampling operators.

Implements InfillFromBlockAverages, the analogue of InfillFromSlices for the
case where the downsampling operator C averages over contiguous blocks:

    (C x)[i] = mean(x[i*k : (i+1)*k])

instead of selecting every k-th point.
"""

from collections.abc import Callable, Mapping
from typing import Any

import chex
import flax
import jax
import jax.numpy as jnp

Array = jax.Array
PyTree = Any
ArrayMapping = Mapping[str, Array]
DenoiseFn = Callable[[Array, Array, ArrayMapping | None], Array]


@flax.struct.dataclass
class InfillFromBlockAverages:
    """Infilling guided by block-average constraints.

    For each block of `downsampling_factor` consecutive elements along axis 1,
    the block mean is constrained to equal the corresponding entry in
    `guidance_inputs['observed_averages']`.

    This mirrors the structure of InfillFromSlices exactly:
      1. Gradient step: penalise ||C*denoised - y||^2 where C is block-averaging.
      2. Projection:    shift each block uniformly so its mean equals y[i].

    Expected shapes (batch dimension is leading):
      denoised / x:        (batch, n_x, d)
      observed_averages:   (batch, n_y, d)   with n_y = n_x // downsampling_factor

    Example usage::

        guidance = InfillFromBlockAverages(downsampling_factor=60, guide_strength=0.5)
        guided_denoiser = guidance(denoiser, {"observed_averages": y_bar})
        denoised = guided_denoiser(noised, sigma=jnp.array(0.1), cond=None)

    Attributes:
        downsampling_factor: Number of fine-grid points per coarse-grid point (k).
        guide_strength: Guidance strength. Rescaled by cond_fraction = 1/k,
            consistent with the InfillFromSlices convention.
    """

    downsampling_factor: int
    guide_strength: chex.Numeric = 0.5

    def __call__(
        self, denoise_fn: DenoiseFn, guidance_inputs: ArrayMapping
    ) -> DenoiseFn:
        """Constructs a denoise function guided by block-average constraints."""
        k = self.downsampling_factor

        def _guided_denoise(
            x: Array, sigma: Array, cond: ArrayMapping | None = None
        ) -> Array:
            def constraint(xt: Array) -> tuple[Array, Array]:
                denoised = denoise_fn(xt, sigma, cond)
                batch, n_x, d = denoised.shape
                n_y = n_x // k
                block_means = denoised.reshape(batch, n_y, k, d).mean(axis=2)
                error = jnp.sum(
                    (block_means - guidance_inputs["observed_averages"]) ** 2
                )
                return error, denoised

            constraint_grad, denoised = jax.grad(constraint, has_aux=True)(x)
            cond_fraction = 1.0 / k
            guide_strength = self.guide_strength / cond_fraction
            denoised = denoised - guide_strength * constraint_grad

            # Projection choice (for reference from paper): shift each block uniformly so its mean equals observed_averages[i].
            batch, n_x, d = denoised.shape
            n_y = n_x // k
            blocks = denoised.reshape(batch, n_y, k, d)
            block_means = blocks.mean(axis=2, keepdims=True)
            correction = (
                guidance_inputs["observed_averages"][:, :, None, :] - block_means
            )
            return (blocks + correction).reshape(batch, n_x, d)

        return _guided_denoise
