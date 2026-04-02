"""Sampling helpers for unconditional and conditional generation.

This module provides thin wrappers around Swirl-Dynamics samplers to:
- draw unconditional samples,
- draw conditionally guided samples via a post-processed denoiser.
"""

import jax
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib

from src.generation.swirl_dynamics_new_guidance.averaging_guidance import (
    InfillFromBlockAverages,
)


def sample_unconditional(
    diffusion_scheme,
    denoise_fn,
    rng_key: jax.Array,
    num_samples: int,
    num_conditions: int,
    data_sett,
    run_sett,
):
    """Generate unconditional samples using an SDE sampler.

    Args:
        diffusion_scheme: Diffusion schedule object.
        denoise_fn: Callable denoiser inference function.
        rng_key: JAX PRNG key for sampling.
        num_samples: Number of independent samples to draw (number of PRNG splits).
        num_conditions: Number of samples generated per PRNG key; set equal to
            the number of conditioning observations used in conditional sampling
            so outputs are directly comparable.
        data_sett: Data settings dictionary (must contain 'n_x' and 'd').
        run_sett: Settings dictionary.

    Returns:
        Array of generated samples with shape `(num_samples, num_conditions, n_x, d)`.
    """
    sampler = dfn_lib.SdeSampler(
        input_shape=(data_sett["n_x"], data_sett["d"]),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=int(run_sett["exp_tspan"]["num_steps"]),
            end_sigma=float(run_sett["exp_tspan"]["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        apply_denoise_at_end=True,
        return_full_paths=False,
    )
    keys = jax.random.split(rng_key, int(num_samples))
    generate_one = jax.jit(
        lambda k: sampler.generate(rng=k, num_samples=num_conditions)
    )

    def loop_body(carry, key):
        samples = generate_one(key)
        return carry, samples

    _, samples_all = jax.lax.scan(loop_body, init=None, xs=keys)
    return samples_all


def sample_conditional(
    diffusion_scheme,
    denoise_fn,
    y_bar: jax.Array,
    rng_key: jax.Array,
    num_samples: int,
    data_sett,
    run_sett,
):
    """Generate conditionally guided samples.

    Wraps the denoiser with a guidance transform that enforces the low-resolution
    constraint during sampling. When `downsampling_type` is ``"average"``,
    `InfillFromBlockAverages` is used; otherwise `InfillFromSlices` is used.
    Guidance strength is read from `run_sett["train_denoiser"]["norm_guide_strength"]`.

    Args:
        diffusion_scheme: Diffusion schedule object.
        denoise_fn: Callable denoiser inference function.
        y_bar: Conditioning LR observations with shape `(num_conditions, n_y, d)`.
        rng_key: JAX PRNG key.
        num_samples: How many independent draws per condition.
        data_sett: Data settings dictionary (must contain 'n_x', 'n_y', 'd',
            'downsampling_type').
        run_sett: Settings dictionary.

    Returns:
        Array with shape `(num_samples, num_conditions, n_x, d)`.
    """
    downsampling_factor = int(data_sett["n_x"] // data_sett["n_y"])
    downsampling_type = str(data_sett["downsampling_type"])
    guide_strength = run_sett["train_denoiser"]["norm_guide_strength"]

    if downsampling_type == "average":
        guidance_transform = InfillFromBlockAverages(
            downsampling_factor=downsampling_factor,
            guide_strength=guide_strength,
        )
        guidance_inputs = {"observed_averages": y_bar}
    else:
        guidance_transform = dfn_lib.InfillFromSlices(
            slices=(slice(None), slice(None, None, downsampling_factor), slice(None)),
            guide_strength=guide_strength,
        )
        guidance_inputs = {"observed_slices": y_bar}

    sampler = dfn_lib.SdeSampler(
        input_shape=(data_sett["n_x"], data_sett["d"]),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=int(run_sett["exp_tspan"]["num_steps"]),
            end_sigma=float(run_sett["exp_tspan"]["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(guidance_transform,),
        apply_denoise_at_end=True,
        return_full_paths=False,
    )

    keys = jax.random.split(rng_key, num_samples)
    generate_one = jax.jit(
        lambda k: sampler.generate(
            rng=k, guidance_inputs=guidance_inputs, num_samples=int(y_bar.shape[0])
        )
    )

    def loop_body(carry, key):
        samples = generate_one(key)
        return carry, samples

    _, samples_all = jax.lax.scan(loop_body, init=None, xs=keys)
    return samples_all
