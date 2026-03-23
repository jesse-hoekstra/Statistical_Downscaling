"""Sampling helpers for unconditional and conditional generation.

This module provides thin wrappers around Swirl-Dynamics samplers to:
- build the linear observation matrix C' for downsampling,
- draw unconditional samples,
- draw WAN-style conditionally guided samples via a post-processed denoiser,
- draw PDE-guided samples using a learned log h guidance function (NewDriftSdeSampler).
"""

import jax
import jax.numpy as jnp
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib

from src.generation.swirl_dynamics_new_guidance.guidance import LinearConstraint
from src.generation.swirl_dynamics_new_sampler.samplers import NewDriftSdeSampler


def sample_unconditional(
    diffusion_scheme,
    denoise_fn,
    rng_key: jax.Array,
    num_samples: int,
    num_plots: int,
    run_sett,
):
    """Generate unconditional samples using an SDE sampler. num_plots is equal to the number of conditions in the conditional samplers.

    Args:
        diffusion_scheme: Diffusion schedule object.
        denoise_fn: Callable denoiser inference function.
        rng_key: JAX PRNG key for sampling.
        num_samples: Number of independent samples to draw.
        num_plots: Number of plots to generate, equal to the number of conditions in the conditional samplers.
        run_sett: Settings dictionary.
    Returns:
        Array of generated samples with shape `(num_samples, num_plots, d, 1)`.
    """
    sampler = dfn_lib.SdeSampler(
        input_shape=(run_sett["global"]["n_x"], run_sett["global"]["d"]),
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
    generate_one = jax.jit(lambda k: sampler.generate(rng=k, num_samples=num_plots))

    def loop_body(carry, key):
        samples = generate_one(key)
        return carry, samples

    _, samples_all = jax.lax.scan(loop_body, init=None, xs=keys)
    return samples_all


def sample_wan_guided(
    diffusion_scheme,
    denoise_fn,
    y_bar: jnp.ndarray,
    rng_key: jax.Array,
    num_samples: int,
    run_sett,
):
    """Generate WAN-style conditionally guided samples.

    Applies the LinearConstraint post-processing transform to the denoiser to
    enforce C' x ≈ y' during sampling. Guidance strength is read from
    `run_sett["train_denoiser"]["norm_guide_strength"]`.

    Args:
        diffusion_scheme: Diffusion schedule object.
        denoise_fn: Callable denoiser inference function.
        y_bar: Conditioning LR observations with shape `(num_conditions, d_prime, 1)`
          or `(num_conditions, d_prime)`.
        rng_key: JAX PRNG key.
        num_samples: How many independent draws per condition.
        run_sett: Settings dictionary.

    Returns:
        Array with shape `(num_samples, num_conditions, d, 1)`.
    """
    downsampling_factor = int(run_sett["global"]["n_x"] // run_sett["global"]["n_y"])

    if (
        False
    ):  # Use the LinearConstraint guidance transform, own code, delete in next update
        guidance_transform = LinearConstraint.create(
            C_prime=C_prime,
            y_bar=y_bar,
            norm_guide_strength=run_sett["train_denoiser"]["norm_guide_strength"],
        )
    else:  # Use the InfillFromSlices guidance transform, swirl_dynamics code
        guidance_transform = dfn_lib.InfillFromSlices(
            slices=(slice(None), slice(None, None, downsampling_factor), slice(None)),
            guide_strength=run_sett["train_denoiser"]["norm_guide_strength"],
        )

    sampler = dfn_lib.SdeSampler(
        input_shape=(run_sett["global"]["n_x"], run_sett["global"]["d"]),
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
    guidance_inputs = {"observed_slices": y_bar}
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


def sample_pde_guided(
    diffusion_scheme,
    denoise_fn,
    pde_solver,
    rng_key: jax.Array,
    samples_per_condition: int,
    y: jnp.ndarray,
):
    """Generate samples guided by a learned PDE-based guidance function.

    Uses `NewDriftSdeSampler` with `guidance_fn=pde_solver.grad_log_h_batched`,
    which supplies per-condition gradients of log h(t, x, y) to guide the SDE.

    Args:
        diffusion_scheme: Diffusion schedule object.
        denoise_fn: Callable denoiser inference function.
        pde_solver: Instance exposing `grad_log_h_batched` and run settings.
        rng_key: JAX PRNG key.
        samples_per_condition: Number of independent draws for each condition.
        y: Conditioning LR observations of shape `(num_conditions, d_prime[, 1])`.

    Returns:
        Array with shape `(samples_per_condition, num_conditions, d, 1)`.
    """
    num_conditionings = int(pde_solver.num_conditionings)
    if y.shape[0] != num_conditionings:
        raise ValueError(
            f"`y` must have leading size {num_conditionings}, but got {y.shape[0]}"
        )
    sampler = NewDriftSdeSampler(
        input_shape=(
            pde_solver.run_sett_global["n_x"],
            pde_solver.run_sett_global["d"],
        ),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=int(pde_solver.run_sett_exp_tspan["num_steps"]),
            end_sigma=float(pde_solver.run_sett_exp_tspan["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        guidance_fn=pde_solver.grad_log_h_batched,
        apply_denoise_at_end=True,
        return_full_paths=False,
    )

    keys = jax.random.split(rng_key, samples_per_condition)
    generate_one = jax.jit(
        lambda k: sampler.generate(
            rng=k, num_samples=num_conditionings, guidance_inputs={"y": y}
        )
    )

    def loop_body(carry, key):
        samples = generate_one(key)
        return carry, samples

    _, samples_all = jax.lax.scan(loop_body, init=None, xs=keys)
    return samples_all
