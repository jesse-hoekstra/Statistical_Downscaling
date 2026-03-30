import jax
import jax.numpy as jnp
import flax
from swirl_dynamics.lib.diffusion import diffusion
import swirl_dynamics.lib.diffusion as dfn_lib
from swirl_dynamics.lib.solvers import sde
from collections.abc import Mapping
from typing import Any, Callable, TypeAlias

Array: TypeAlias = jax.Array
ArrayMapping: TypeAlias = Mapping[str, Array]
Params: TypeAlias = Mapping[str, Any]


def dlog_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt log(f(t)) = ḟ(t)/f(t) given f(t)."""
    return jax.grad(lambda t: jnp.log(f(t)))


def dsquare_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
    """Returns d/dt (f(t))^2 = 2ḟ(t)f(t) given f(t)."""
    return jax.grad(lambda t: jnp.square(f(t)))


@flax.struct.dataclass
class NewDriftSdeSampler(dfn_lib.SdeSampler):
    """
    An SDE sampler with a custom drift term that reuses the original
    diffusion term from the parent SdeSampler class.
    """

    guidance_fn: Callable[[Array, Array, Array], Array] | None = flax.struct.field(
        default=None, pytree_node=False
    )
    T: float = 1.0

    def __post_init__(self):
        if self.guidance_fn is None:
            raise ValueError("guidance_fn is required for NewDriftSdeSampler")

    @property
    def dynamics(self) -> sde.SdeDynamics:
        f"""Drift and diffusion terms of the sampling SDE.

        In score function:

        dx = [ṡ(t)/s(t) x - 2 s(t)²σ̇(t)σ(t) ∇pₜ(x)- 2 s(t)²σ̇(t)σ(t) ∇h_T-t(x, y)] dt + s(t) √[2σ̇(t)σ(t)] dωₜ,

        obtained by substituting eq. 28, 34 of Karras et al.
        (https://arxiv.org/abs/2206.00364) into the reverse SDE formula - eq. 6 in
        Song et al. (https://arxiv.org/abs/2011.13456). Alternatively, it may be
        rewritten in terms of the denoise function (plugging in eq. 74 of
        Karras et al.) as:

        dx = [2 σ̇(t)/σ(t) + ṡ(t)/s(t)] x - [2 s(t)σ̇(t)/σ(t)] D(x/s(t), σ(t)) + s(t)**2 * 2σ̇(t)σ(t) * ∇h_T-t(x) dt
            + s(t) √[2σ̇(t)σ(t)] dωₜ

        where s(t), σ(t) are the scale and noise schedule of the diffusion scheme
        respectively.
        """
        # Get the original dynamics (drift and diffusion) from the parent class
        original_dynamics = super().dynamics

        def _new_drift(x: Array, t: Array, params: Params) -> Array:
            assert not t.ndim, "`t` must be a scalar."
            denoise_fn = self.get_guided_denoise_fn(
                guidance_inputs=params["guidance_inputs"]
            )  # technically not needed in our case
            s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
            x_hat = jnp.divide(x, s)
            dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
            dlog_s_dt = dlog_dt(self.scheme.scale)(t)
            drift = (2 * dlog_sigma_dt + dlog_s_dt) * x
            drift -= 2 * dlog_sigma_dt * s * denoise_fn(x_hat, sigma, params["cond"])

            if self.guidance_fn is not None:
                guidance_inputs = params["guidance_inputs"]
                if "y" not in guidance_inputs:
                    raise ValueError(
                        "Missing `y` in guidance_inputs; pass guidance_inputs={'y': <batch_y>} to `generate`."
                    )
                y = guidance_inputs["y"]

                x_for_guidance = x.squeeze(-1) if x.ndim == 3 else x
                y_for_guidance = y.squeeze(-1) if y.ndim == 3 else y
                grad_guidance = self.guidance_fn(
                    x_for_guidance,
                    y_for_guidance,
                    self.T - t,  # t runs from 1 to 0 so this is needed (T-s in notes)
                )
                dsquare_sigma_dt = dsquare_dt(self.scheme.sigma)(t)

                drift -= s**2 * dsquare_sigma_dt * grad_guidance[..., None]
            else:
                raise ValueError("Guidance function is not provided")

            return drift

        return sde.SdeDynamics(drift=_new_drift, diffusion=original_dynamics.diffusion)
