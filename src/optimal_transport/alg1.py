"""Core normalizing-flow and policy-gradient components for optimal transport.

This module implements:
- sinusoidal time embeddings for discrete time steps (as a separate function due to multiple call sites)
- a conditional spline coupling flow (per-dimension rational-quadratic splines)
- a correlation network producing time- and state-dependent correlation `rho`
- a normalizing-flow model that samples sequences and computes log-probabilities
- a policy-gradient trainer that fits the flow to minimize transport cost and
  match the true data distribution (via negative log-likelihood), optionally
  with EMA parameters for evaluation and control variates for variance reduction

Shapes and conventions
----------------------
- Dimension `d`: data dimensionality for each time step.
- Time length `N_len = N + 1`: we index time `n` in [0, N].
- A trajectory in normalized coordinates has shape (N+1, d, 1).
- Sampling typically works in normalized space and is converted to ORIGINAL
  space via the transform method of the `DataNormalizer`.

External interfaces used
------------------------
- `DataNormalizer` from `preprocessing_OT` providing:
    - `.fit()` to compute dataset statistics
    - `.transform(mode, y, yp)` where mode in {"normalize","denormalize"}
    - attribute `log_det` used to adjust log-probabilities if normalization is on
- `true_data_model` providing:
    - `.sample_true_trajectory(key) -> (y, y')` in ORIGINAL space
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import haiku as hk
import distrax
import optax
from typing import Optional, Tuple
from functools import partial
import os
import orbax.checkpoint as ocp

from preprocessing import DataNormalizer


def sinusoidal_time_embedding(n: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Return sinusoidal time embedding for step index `n`.

    This mirrors positional encodings: for frequencies geometrically spaced
    between [1, 1/10000] we apply sin/cos and concatenate.

    Parameters
    ----------
    n : jnp.ndarray
        Scalar or array of time indices.
    dim : int
        Embedding dimension.

    Returns
    -------
    jnp.ndarray
        Embedding with shape `n.shape + (dim,)`.
    """
    dim = int(dim)
    half = dim // 2
    n = n.astype(jnp.float32)
    freqs = jnp.exp(
        -jnp.log(10000.0)
        * jnp.arange(0, half, dtype=jnp.float32)
        / jnp.maximum(half, 1)
    )
    args = n[..., None] * freqs[None, ...]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, [(0, 0)] * (emb.ndim - 1) + [(0, 1)])
    return emb


class ConditionerMLP(hk.Module):
    """MLP producing spline parameters for a coupling flow conditioner.

    Given concatenated input `[x, context]`, outputs per-dimension parameters for
    a rational-quadratic spline: for each dimension we emit
    ``3 * num_bins + 1`` values: ``K`` bin widths, ``K`` bin heights, and ``K+1`` knot slopes
    (K-1 internal knots plus 2 boundary knots).
    """

    def __init__(self, name: str, d: int, num_bins: int, hidden_size: int):
        super().__init__(name=name)
        self.d = int(d)
        self.num_bins = int(num_bins)
        self.hidden_size = int(hidden_size)
        self.out_dim = self.d * (3 * self.num_bins + 1)

    def __call__(self, inp: jnp.ndarray) -> jnp.ndarray:
        """Return spline parameters shaped as (..., d, 3*num_bins + 1)."""
        h = hk.Linear(self.hidden_size)(inp)
        h = jax.nn.gelu(h)
        h = hk.Linear(self.hidden_size)(h)
        h = jax.nn.gelu(h)
        h = hk.Linear(
            self.out_dim,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
        )(h)
        return jnp.reshape(h, h.shape[:-1] + (self.d, 3 * self.num_bins + 1))


class ConditionalSplineCouplingFlow(hk.Module):
    """Masked coupling flow with per-dimension rational-quadratic splines.

    The conditioner is an MLP that consumes `[x, context]` and outputs spline
    parameters. Layers alternate masks over dimensions.
    """

    def __init__(
        self,
        run_sett: dict,
        name: str,
        context_dim: int,
    ):
        super().__init__(name=name)
        self.run_sett_marginal_flow = run_sett["marginal_flow"]
        self.run_sett_global = run_sett["global"]
        self.d = int(self.run_sett_global["d"])
        self.context_dim = int(context_dim)
        self.num_layers = int(self.run_sett_marginal_flow["num_layers"])
        self.hidden_size = int(self.run_sett_marginal_flow["hidden_size"])
        self.num_bins = int(self.run_sett_marginal_flow["num_bins"])
        self.range_min = float(self.run_sett_marginal_flow["range_min"])
        self.range_max = float(self.run_sett_marginal_flow["range_max"])

        self._conds = [
            ConditionerMLP(f"cond_{i}", self.d, self.num_bins, self.hidden_size)
            for i in range(self.num_layers)
        ]
        self._masks = [
            (jnp.arange(self.d) % 2 == (i % 2)) for i in range(self.num_layers)
        ]

        self._base = distrax.MultivariateNormalDiag(
            loc=jnp.zeros((self.d,), dtype=jnp.float32),
            scale_diag=jnp.ones((self.d,), dtype=jnp.float32),
        )

    def _make_dist(self, context: jnp.ndarray) -> distrax.Transformed:
        """Construct the transformed distribution for a given `context`."""
        layers = []
        for i in range(self.num_layers):
            mask = self._masks[i]
            cond_mlp = self._conds[i]

            def _broadcast_context(ctx: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
                return jnp.broadcast_to(ctx, x.shape[:-1] + ctx.shape)

            def _conditioner(x, ctx=context, mlp=cond_mlp):
                ctx_b = _broadcast_context(ctx, x)
                inp = jnp.concatenate([x, ctx_b], axis=-1)
                return mlp(inp)

            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,
                    conditioner=_conditioner,
                    bijector=lambda params: distrax.RationalQuadraticSpline(
                        params,
                        range_min=self.range_min,
                        range_max=self.range_max,
                        boundary_slopes="identity",
                        min_bin_size=1e-3,
                        min_knot_slope=1e-3,
                    ),
                )
            )
        return distrax.Transformed(self._base, distrax.Chain(layers))

    def forward_from_base(self, eps: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
        """Map base sample `eps` to data space given `context`."""
        dist = self._make_dist(context)
        return dist.bijector.forward(eps)

    def log_prob_and_base(
        self, x: jnp.ndarray, context: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return log-probability of `x` and corresponding base variable `u`."""
        dist = self._make_dist(context)
        u, ildj = dist.bijector.inverse_and_log_det(x)
        logp = self._base.log_prob(u) + ildj
        return logp, u


class RhoNet(hk.Module):
    """Network producing correlation vector `rho` in [-rho_max, rho_max]^d.

    Inputs are previous y, previous y' and a time embedding for step `n`. Output
    is clipped to [-rho_max, rho_max].
    """

    def __init__(self, run_sett: dict, name: str):
        super().__init__(name=name)
        self.run_sett_global = run_sett["global"]
        self.run_sett_correlation_flow = run_sett["correlation_flow"]
        self.d = int(self.run_sett_global["d"])
        self.time_emb_dim = int(self.run_sett_global["time_emb_dim"])
        self.hidden_size = int(self.run_sett_correlation_flow["hidden_size"])
        self.rho_max = float(self.run_sett_correlation_flow["rho_max"])

    def __call__(
        self, prev_y: jnp.ndarray, prev_yp: jnp.ndarray, n: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute per-dimension correlation `rho` for time step `n`."""
        te = sinusoidal_time_embedding(n, self.time_emb_dim).reshape((-1,))
        inp = jnp.concatenate([prev_y, prev_yp, te], axis=0)
        h = hk.Linear(self.hidden_size)(inp)
        h = jax.nn.gelu(h)
        h = hk.Linear(self.hidden_size)(h)
        h = jax.nn.gelu(h)
        out = hk.Linear(self.d)(h)
        rho = self.rho_max * jnp.tanh(out)
        return jnp.clip(rho, -self.rho_max, self.rho_max)


class NormalizingFlowModel:
    """Normalizing-flow model to sample sequences and compute log-probabilities.

    The model maintains two conditional flows (for y and y') and a correlation
    network that couples their base variables via a Gaussian copula with
    parameter `rho`. Sampling/log-likelihood computations are performed in the
    normalized space and converted if normalization is enabled.
    """

    def __init__(self, run_sett: dict, true_data_model):
        """Initialise the model, normalizer, and Haiku-transformed functions."""
        self.run_sett = run_sett
        self.run_sett_global = run_sett["global"]
        self.run_sett_marginal_flow = run_sett["marginal_flow"]
        self.run_sett_correlation_flow = run_sett["correlation_flow"]
        self.run_sett_preprocessing = run_sett["preprocessing"]

        self.d = int(self.run_sett_global["d"])
        self.N = int(self.run_sett_global["N"])
        self.N_len = int(self.N + 1)
        self.seed = int(self.run_sett_global["seed"])
        self.time_emb_dim = int(self.run_sett_global["time_emb_dim"])

        self.num_layers = int(self.run_sett_marginal_flow["num_layers"])
        self.hidden_size = int(self.run_sett_marginal_flow["hidden_size"])
        self.num_bins = int(self.run_sett_marginal_flow["num_bins"])

        self.rho_max = float(self.run_sett_correlation_flow["rho_max"])

        self.normalizer = DataNormalizer(run_sett, true_data_model).fit()
        self.use_data_normalization = bool(
            self.run_sett_preprocessing["use_data_normalization"]
        )

        self._hk_logprob_steps_y_norm = hk.without_apply_rng(
            hk.transform(self._logprob_steps_y_batch_norm_impl)
        )
        self._hk_logprob_total_y_norm = hk.without_apply_rng(
            hk.transform(self._logprob_total_y_batch_norm_impl)
        )
        self._hk_logprob_steps_yp_norm = hk.without_apply_rng(
            hk.transform(self._logprob_steps_yp_batch_norm_impl)
        )
        self._hk_logprob_total_yp_norm = hk.without_apply_rng(
            hk.transform(self._logprob_total_yp_batch_norm_impl)
        )
        self._hk_sample_norm = hk.without_apply_rng(
            hk.transform(self._sample_batch_norm_impl)
        )
        self._hk_logprob_steps_norm = hk.without_apply_rng(
            hk.transform(self._logprob_steps_batch_norm_impl)
        )
        self._hk_logprob_total_norm = hk.without_apply_rng(
            hk.transform(self._logprob_total_batch_norm_impl)
        )
        self._hk_transport = hk.without_apply_rng(hk.transform(self._transport_impl))
        init_key = jax.random.PRNGKey(self.seed)
        dummy_keys = jax.random.split(init_key, 2)
        self.params = self._hk_sample_norm.init(init_key, keys=dummy_keys)

    def _ctx(self, prev: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        """Concatenate previous state with sinusoidal time embedding."""
        te = sinusoidal_time_embedding(n, self.time_emb_dim).reshape((-1,))
        return jnp.concatenate([prev, te], axis=0)

    def _make_modules(self):
        """Create conditional flows for y and y' and the correlation network."""
        ctx_dim = self.d + self.time_emb_dim
        y_flow = ConditionalSplineCouplingFlow(
            run_sett=self.run_sett,
            name="y_flow",
            context_dim=ctx_dim,
        )
        yp_flow = ConditionalSplineCouplingFlow(
            run_sett=self.run_sett,
            name="yp_flow",
            context_dim=ctx_dim,
        )
        rho_net = RhoNet(
            run_sett=self.run_sett,
            name="rho_net",
        )
        return y_flow, yp_flow, rho_net

    def _sample_one_traj_norm(self, y_flow, yp_flow, rho_net, key: jax.Array):
        """Sample a single normalized trajectory (y, y') and return mean |rho|."""
        d, N_len = self.d, self.N_len

        def step(carry, n):
            prev_y, prev_yp, key_in, rho_acc = carry
            key_in, k1, k2 = jax.random.split(key_in, 3)

            rho = rho_net(prev_y, prev_yp, n)
            eps1 = jax.random.normal(k1, (d,))
            eps2 = jax.random.normal(k2, (d,))

            s = jnp.sqrt(jnp.clip(1.0 - rho * rho, 1e-6, 1.0))
            u = eps1
            v = rho * eps1 + s * eps2

            ctx_y = self._ctx(prev_y, n)
            ctx_yp = self._ctx(prev_yp, n)
            y_n = y_flow.forward_from_base(u, ctx_y)
            yp_n = yp_flow.forward_from_base(v, ctx_yp)

            rho_acc = rho_acc + jnp.mean(jnp.abs(rho))
            return (y_n, yp_n, key_in, rho_acc), (y_n, yp_n)

        prev0 = jnp.zeros((d,), dtype=jnp.float32)
        carry0 = (prev0, prev0, key, jnp.array(0.0, dtype=jnp.float32))
        ns = jnp.arange(N_len, dtype=jnp.int32)
        (_, _, _, rho_sum), (ys, yps) = hk.scan(step, carry0, ns)

        mean_abs_rho = rho_sum / N_len
        y_traj = ys.reshape((N_len, d, 1))
        yp_traj = yps.reshape((N_len, d, 1))
        return y_traj, yp_traj, mean_abs_rho

    def _logprob_steps_one_traj_norm(self, y_flow, yp_flow, rho_net, y_traj, yp_traj):
        """Compute per-step log-probabilities for a normalized `(y,y')` trajectory."""
        d, N_len = self.d, self.N_len
        y_vec = jnp.squeeze(y_traj, axis=-1)
        yp_vec = jnp.squeeze(yp_traj, axis=-1)

        def step(carry, n):
            prev_y, prev_yp = carry
            y_n = y_vec[n]
            yp_n = yp_vec[n]

            rho = rho_net(prev_y, prev_yp, n)
            ctx_y = self._ctx(prev_y, n)
            ctx_yp = self._ctx(prev_yp, n)

            logpy, u = y_flow.log_prob_and_base(y_n, ctx_y)
            logpyp, v = yp_flow.log_prob_and_base(yp_n, ctx_yp)

            def _gaussian_copula_correction(
                u: jnp.ndarray, v: jnp.ndarray, rho: jnp.ndarray
            ) -> jnp.ndarray:
                rho = jnp.clip(rho, -0.999, 0.999)
                one_minus = jnp.clip(1.0 - rho * rho, 1e-6, 1.0)
                quad = (u * u - 2.0 * rho * u * v + v * v) / one_minus
                corr = -0.5 * jnp.log(one_minus) - 0.5 * quad + 0.5 * (u * u + v * v)
                return jnp.sum(corr)

            corr = _gaussian_copula_correction(u, v, rho)

            logp_n = logpy + logpyp + corr
            return (y_n, yp_n), logp_n

        prev0 = jnp.zeros((d,), dtype=jnp.float32)
        ns = jnp.arange(N_len, dtype=jnp.int32)
        (_, _), logp_steps = hk.scan(step, (prev0, prev0), ns)
        return logp_steps

    def _logprob_steps_y_one_traj_norm(self, y_flow, y_traj):
        """Compute per-step marginal log-probabilities of ``y`` for a single normalized trajectory."""
        d, N_len = self.d, self.N_len
        y_vec = jnp.squeeze(y_traj, -1)

        def step(prev_y, t):
            y_t = y_vec[t]
            ctx_y = self._ctx(prev_y, t)
            logpy, _ = y_flow.log_prob_and_base(y_t, ctx_y)
            return y_t, logpy

        prev0 = jnp.zeros((d,))
        ts = jnp.arange(N_len)
        _, logp_steps = hk.scan(step, prev0, ts)
        return logp_steps

    def _logprob_steps_yp_one_traj_norm(self, yp_flow, yp_traj):
        """Compute per-step marginal log-probabilities of ``yp`` for a single normalized trajectory."""
        d, N_len = self.d, self.N_len
        yp_vec = jnp.squeeze(yp_traj, -1)

        def step(prev_yp, t):
            yp_t = yp_vec[t]
            ctx_yp = self._ctx(prev_yp, t)
            logpyp, _ = yp_flow.log_prob_and_base(yp_t, ctx_yp)
            return yp_t, logpyp

        prev0 = jnp.zeros((d,))
        ts = jnp.arange(N_len)
        _, logp_steps = hk.scan(step, prev0, ts)
        return logp_steps

    def _logprob_steps_y_batch_norm_impl(self, y_z):
        """Haiku-transformed: vmap per-step ``y`` log-probs over a batch of normalized trajectories."""
        y_flow, _, _ = self._make_modules()

        def one(y_one):
            return self._logprob_steps_y_one_traj_norm(y_flow, y_one)

        return hk.vmap(one, split_rng=False)(y_z)

    def _logprob_steps_yp_batch_norm_impl(self, yp_z):
        """Haiku-transformed: vmap per-step ``yp`` log-probs over a batch of normalized trajectories."""
        _, yp_flow, _ = self._make_modules()

        def one(yp_one):
            return self._logprob_steps_yp_one_traj_norm(yp_flow, yp_one)

        return hk.vmap(one, split_rng=False)(yp_z)

    def _logprob_total_y_batch_norm_impl(self, y_z):
        """Haiku-transformed: sum per-step ``y`` log-probs over time for each normalized trajectory."""
        return jnp.sum(self._logprob_steps_y_batch_norm_impl(y_z), axis=1)

    def _logprob_total_yp_batch_norm_impl(self, yp_z):
        """Haiku-transformed: sum per-step ``yp`` log-probs over time for each normalized trajectory."""
        return jnp.sum(self._logprob_steps_yp_batch_norm_impl(yp_z), axis=1)

    def _sample_batch_norm_impl(self, keys: jax.Array):
        """Haiku-transformed: vmap of `_sample_one_traj_norm` over `keys`."""
        y_flow, yp_flow, rho_net = self._make_modules()

        def one(k):
            y_z, yp_z, mean_abs_rho = self._sample_one_traj_norm(
                y_flow, yp_flow, rho_net, k
            )
            return y_z, yp_z, mean_abs_rho

        return hk.vmap(one, split_rng=False)(keys)

    def _logprob_steps_batch_norm_impl(self, y_z: jnp.ndarray, yp_z: jnp.ndarray):
        """Haiku-transformed: vmap per-step log-probs over a batch of trajectories."""
        y_flow, yp_flow, rho_net = self._make_modules()

        def one(a, b):
            return self._logprob_steps_one_traj_norm(y_flow, yp_flow, rho_net, a, b)

        return hk.vmap(one, split_rng=False)(y_z, yp_z)

    def _logprob_total_batch_norm_impl(self, y_z: jnp.ndarray, yp_z: jnp.ndarray):
        """Haiku-transformed: sum step log-probs over time for each trajectory."""
        steps = self._logprob_steps_batch_norm_impl(y_z, yp_z)
        return jnp.sum(steps, axis=1)

    def sample_batch_norm(self, params: hk.Params, keys: jax.Array):
        """Sample a batch of normalized trajectories `(y_z, yp_z, mean_abs_rho)`."""
        return self._hk_sample_norm.apply(params, keys=keys)

    def logprob_steps_y_batch_norm(self, params, y_z):
        """Compute per-step log-probabilities of ``y`` for a batch of normalized trajectories."""
        return self._hk_logprob_steps_y_norm.apply(params, y_z=y_z)

    def logprob_steps_yp_batch_norm(self, params, yp_z):
        """Compute per-step log-probabilities of ``yp`` for a batch of normalized trajectories."""
        return self._hk_logprob_steps_yp_norm.apply(params, yp_z=yp_z)

    def logprob_total_y_batch_norm(self, params, y_z):
        """Compute total log-probabilities of ``y`` (summed over time) for a batch of normalized trajectories."""
        return self._hk_logprob_total_y_norm.apply(params, y_z=y_z)

    def logprob_total_yp_batch_norm(self, params, yp_z):
        """Compute total log-probabilities of ``yp`` (summed over time) for a batch of normalized trajectories."""
        return self._hk_logprob_total_yp_norm.apply(params, yp_z=yp_z)

    def logprob_steps_batch_norm(
        self, params: hk.Params, y_z: jnp.ndarray, yp_z: jnp.ndarray
    ):
        """Compute per-step log-probabilities for a batch of normalized trajectories."""
        return self._hk_logprob_steps_norm.apply(params, y_z=y_z, yp_z=yp_z)

    def logprob_total_batch_norm(
        self, params: hk.Params, y_z: jnp.ndarray, yp_z: jnp.ndarray
    ):
        """Compute total log-probabilities for a batch of normalized trajectories."""
        return self._hk_logprob_total_norm.apply(params, y_z=y_z, yp_z=yp_z)

    def _log_det_y_per_t(self):
        """Per-step log-determinant of the ``y`` normalizer; shape ``(N+1,)``."""
        if (not self.use_data_normalization) or (self.normalizer is None):
            return jnp.zeros((self.N_len,))
        return jnp.sum(jnp.log(self.normalizer.std_y), axis=(1, 2))

    def _log_det_yp_per_t(self):
        """Per-step log-determinant of the ``yp`` normalizer; shape ``(N+1,)``."""
        if (not self.use_data_normalization) or (self.normalizer is None):
            return jnp.zeros((self.N_len,))
        return jnp.sum(jnp.log(self.normalizer.std_yp), axis=(1, 2))

    def logprob_steps_y_batch_original(self, params, y):
        """Per-step log-probabilities of ``y`` in ORIGINAL space, corrected for normalization."""
        if not self.use_data_normalization:
            y_z = y
        else:
            y_z, _ = self.normalizer.transform("normalize", y, jnp.zeros_like(y))
        logq_z = self.logprob_steps_y_batch_norm(params, y_z)
        if not self.use_data_normalization:
            return logq_z
        ld = self._log_det_y_per_t()
        return logq_z - ld[None, :]

    def logprob_steps_yp_batch_original(self, params, yp):
        """Per-step log-probabilities of ``yp`` in ORIGINAL space, corrected for normalization."""
        if not self.use_data_normalization:
            yp_z = yp
        else:
            _, yp_z = self.normalizer.transform("normalize", jnp.zeros_like(yp), yp)
        logq_z = self.logprob_steps_yp_batch_norm(params, yp_z)
        if not self.use_data_normalization:
            return logq_z
        ld = self._log_det_yp_per_t()
        return logq_z - ld[None, :]

    def sample_trajectory(self, key, params: hk.Params):
        """Sample a single trajectory ``(y, yp)`` in ORIGINAL space given explicit params."""
        if hasattr(key, "keys") and (not isinstance(params, dict)):
            key, params = params, key
        y_z, yp_z, _ = self.sample_batch_norm(params, jax.random.split(key, 1))
        y, yp = self.normalizer.transform("denormalize", y_z[0], yp_z[0])
        return y, yp

    def _transport_impl(self, y_z_in: jnp.ndarray, key: jax.Array):
        """Transport a batch of normalized trajectories to yp space.
        Steps (performed in normalized space):
            1) Compute base variables u_t by inverting y_flow given context from prev y.
            2) Sample v_t ~ N(rho_t * u_t, (1 - rho_t^2) * I) from the Gaussian copula.
            3) Map v_t forward through yp_flow to obtain yp_t.
        """
        y_flow, yp_flow, rho_net = self._make_modules()
        d, N_len = self.d, self.N_len

        def step(carry, n):
            prev_y_c, prev_yp_c, key_in = carry
            key_in, k_eps = jax.random.split(key_in)
            y_n = jnp.squeeze(y_z_in[n], axis=-1)
            rho = rho_net(prev_y_c, prev_yp_c, n)

            ctx_y = self._ctx(prev_y_c, n)
            _, u = y_flow.log_prob_and_base(y_n, ctx_y)

            eps = jax.random.normal(k_eps, (d,))
            s = jnp.sqrt(jnp.clip(1.0 - rho * rho, 1e-6, 1.0))
            v = rho * u + s * eps

            ctx_yp = self._ctx(prev_yp_c, n)
            yp_n = yp_flow.forward_from_base(v, ctx_yp)

            return (y_n, yp_n, key_in), yp_n

        prev0 = jnp.zeros((d,), dtype=jnp.float32)
        ns = jnp.arange(N_len, dtype=jnp.int32)
        (_, _, _), yp_seq = hk.scan(step, (prev0, prev0, key), ns)
        yp_seq = yp_seq.reshape((N_len, d, 1))
        return yp_seq

    def transport_y_to_yp(self, y_traj: jnp.ndarray, params: hk.Params, key: jax.Array):
        """Transport a single ORIGINAL-space trajectory y -> yp via learned flow.

        Parameters
        ----------
        y_traj : jax.Array
            Shape ``(N+1, d)`` or ``(N+1, d, 1)`` in ORIGINAL space.
        params : hk.Params
            Haiku params pytree for the model (matching init).
        key : jax.Array
            JAX PRNG key for sampling the copula noise.

        Returns
        -------
        jax.Array
            ``yp`` with shape matching ``y_traj`` in ORIGINAL space.
        """
        squeeze = y_traj.ndim == 2
        if squeeze:
            y_traj = y_traj[..., None]
        assert y_traj.shape == (
            self.N_len,
            self.d,
            1,
        ), f"Expected y_traj shape {(self.N_len, self.d, 1)}, got {y_traj.shape}"
        zeros_like = jnp.zeros_like(y_traj)
        y_traj_norm, _ = self.normalizer.transform("normalize", y_traj, zeros_like)

        yp_traj_norm = self._hk_transport.apply(params, y_traj_norm, key)
        _, yp_traj = self.normalizer.transform("denormalize", y_traj_norm, yp_traj_norm)

        if squeeze:
            yp_traj = yp_traj[..., 0]
        return yp_traj


class PolicyGradient:
    """Policy-gradient trainer for the normalizing-flow sequence model.

    Optimizes a mixture of:
    - score-function surrogate cost with advantage weighting and optional control variates
    - pathwise transport cost
    - negative log-likelihood on true data with weight beta

    Maintains an EMA of parameters for evaluation if configured.
    """

    def __init__(self, run_sett: dict, true_data_model, normalizing_flow_model=None):
        """Initialise the trainer, optimizer, schedules, and EMA state."""
        self.run_sett = run_sett
        self.run_sett_global = run_sett["global"]
        self.run_sett_beta = run_sett["beta_schedule"]
        self.run_sett_lr = run_sett["lr_schedule"]
        self.run_sett_policy_gradient = run_sett["policy_gradient"]
        self.run_sett_metrics = run_sett["metrics"]
        self.run_sett_evaluation = run_sett["metrics"]["evaluation"]
        self.run_sett_baseline_fitting = run_sett["baseline_fitting"]
        self.run_sett_ema = run_sett["ema"]

        self.d = int(self.run_sett_global["d"])
        self.N = int(self.run_sett_global["N"])
        self.N_len = int(self.N + 1)
        self.B = int(self.run_sett_global["B"])
        self.seed = int(self.run_sett_global["seed"])
        self.time_emb_dim = int(self.run_sett_global["time_emb_dim"])
        self.cv_ridge = float(self.run_sett_baseline_fitting["cv_ridge"])
        self.cv_split_ratio = float(self.run_sett_baseline_fitting["cv_split_ratio"])
        self._B_fit_static = int(
            max(0, min(self.B, round(self.cv_split_ratio * self.B)))
        )

        self.true_data_model = true_data_model
        self.model = normalizing_flow_model or NormalizingFlowModel(
            run_sett, true_data_model
        )

        self.ema_decay = float(self.run_sett_ema["ema_decay"])
        self.use_ema_eval = bool(self.run_sett_ema["use_ema_eval"])
        self.params = self.model.params
        self.ema_params = jax.tree_util.tree_map(lambda x: x, self.params)

        self.beta_schedule = self._build_beta_schedule()
        self._last_beta_value = float(self.beta_schedule(0))
        self.kl_warmup_steps = int(self.run_sett_beta["kl_only_warmup_steps"])

        self.lrate_schedule = self._build_lr_schedule()
        self.optimizer = optax.adam(learning_rate=self.lrate_schedule)
        self.opt_state = self.optimizer.init(self.params)

        self.mix_pathwise_alpha = float(
            self.run_sett_policy_gradient["mix_pathwise_alpha"]
        )
        self.use_control_variates = bool(
            self.run_sett_baseline_fitting["use_control_variates"]
        )
        self.use_advantage_standardization = bool(
            self.run_sett_baseline_fitting["use_advantage_standardization"]
        )

        self._step = 0

    def _build_beta_schedule(self):
        """Create a beta schedule for the NLL weight."""
        mode_type = str(self.run_sett_beta["type"]).lower()
        init_beta = float(self.run_sett_beta["init_boundary_value"])
        end_beta = float(self.run_sett_beta["end_boundary_value"])
        if mode_type == "constant":
            return optax.constant_schedule(end_beta)
        if mode_type == "linear":
            num_iter = int(self.run_sett_global["num_iterations"])
            warmup_ratio = float(self.run_sett_beta["warmup_end_ratio"])
            ramp_steps = max(1, int(num_iter * warmup_ratio))
            return optax.linear_schedule(init_beta, end_beta, ramp_steps)
        if mode_type == "schedule":
            num_iter = int(self.run_sett_global["num_iterations"])
            start_ramp = int(num_iter * 0.2)
            end_ramp = int(num_iter * 0.8)
            ramp = max(1, end_ramp - start_ramp)
            return optax.join_schedules(
                schedules=[
                    optax.constant_schedule(init_beta),
                    optax.linear_schedule(init_beta, end_beta, ramp),
                    optax.constant_schedule(end_beta),
                ],
                boundaries=[start_ramp, end_ramp],
            )
        return optax.constant_schedule(end_beta)

    def _build_lr_schedule(self):
        """Create learning-rate schedule (constant or cosine with warmup/decay)."""
        init_value = float(self.run_sett_lr["init_value"])
        peak_value = float(self.run_sett_lr["peak_value"])
        warmup_steps = int(self.run_sett_lr["warmup_steps"])
        decay_steps = int(self.run_sett_lr["decay_steps"])
        constant_value = float(self.run_sett_lr["constant_lr"])
        end_value = float(self.run_sett_lr["end_value"])
        step_decay_boundaries = self.run_sett_lr["step_decay_boundaries"]
        step_decay_factor = float(self.run_sett_lr["step_decay_factor"])
        mode_type = str(self.run_sett_lr["type"]).lower()
        warmup = optax.linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=max(warmup_steps, 1),
        )
        if mode_type == "constant":
            return optax.constant_schedule(constant_value)
        elif mode_type == "cosine":
            tail = optax.cosine_decay_schedule(
                init_value=peak_value,
                decay_steps=max(decay_steps, 1),
                alpha=end_value / max(peak_value, 1e-12),
            )
        elif mode_type == "step":
            boundaries_abs = [int(b) for b in step_decay_boundaries]
            warm = max(warmup_steps, 1)
            boundaries = [max(0, b - warm) for b in boundaries_abs]
            boundaries_and_scales = {}
            current_scale = 1.0
            for b in boundaries:
                current_scale *= step_decay_factor
                boundaries_and_scales[b] = current_scale
            tail = optax.piecewise_constant_schedule(
                init_value=peak_value, boundaries_and_scales=boundaries_and_scales
            )
        else:
            tail = optax.constant_schedule(peak_value)
        return optax.join_schedules(
            schedules=[warmup, tail], boundaries=[max(warmup_steps, 1)]
        )

    def get_eval_params_trees(self):
        """Return EMA params if enabled, else current params."""
        return self.ema_params if self.use_ema_eval else self.params

    def _phi(
        self, prev_y_z: jnp.ndarray, prev_yp_z: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Feature vector for baseline fitting at step `t` given previous states."""
        prev_y = jnp.squeeze(prev_y_z, -1)
        prev_yp = jnp.squeeze(prev_yp_z, -1)
        te = sinusoidal_time_embedding(t, self.time_emb_dim).reshape((-1,))
        return jnp.concatenate(
            [jnp.ones((1,), dtype=jnp.float32), prev_y, prev_yp, te], axis=0
        )

    def _fit_baseline_per_n(self, phi: jnp.ndarray, target: jnp.ndarray, ridge: float):
        """Fit a ridge-regression baseline per time step; ``phi`` ``(B,N+1,p)``, returns weights ``(N+1,p)``."""
        B, N_len, p = phi.shape

        def solve_one(n):
            X = phi[:, n, :]
            y = target[:, n]
            XtX = X.T @ X + ridge * jnp.eye(p, dtype=X.dtype)
            Xty = X.T @ y
            return jnp.linalg.solve(XtX, Xty)

        ns = jnp.arange(N_len, dtype=jnp.int32)
        return jax.vmap(solve_one)(ns)

    def _baseline_predict(self, phi: jnp.ndarray, w: jnp.ndarray):
        """Predict baseline values given features `phi` and weights `w`."""
        return jnp.einsum("bnp,np->bn", phi, w)

    @partial(jax.jit, static_argnums=0)
    def _train_step(self, params, opt_state, key: jax.Array, step_i: jnp.ndarray):
        """One training step: compute loss, update params and EMA, return metrics."""
        beta = jnp.asarray(self.beta_schedule(step_i), dtype=jnp.float32)
        warm = jnp.asarray(self.kl_warmup_steps, dtype=jnp.int32)
        beta = jnp.where(step_i < warm, jnp.array(0.0, dtype=jnp.float32), beta)

        key, k_model, k_true = jax.random.split(key, 3)
        keys_m = jax.random.split(k_model, self.B)
        keys_t = jax.random.split(k_true, self.B)

        y_true, yp_true = jax.vmap(self.true_data_model.sample_true_trajectory)(keys_t)
        y_true_z, yp_true_z = self.model.normalizer.transform(
            "normalize", y_true, yp_true
        )

        alpha = jnp.clip(
            jnp.asarray(self.mix_pathwise_alpha, dtype=jnp.float32), 0.0, 1.0
        )

        def loss_and_aux(p):
            y_z_s, yp_z_s, rho_abs = self.model.sample_batch_norm(p, keys_m)

            y_z_ng = lax.stop_gradient(y_z_s)
            yp_z_ng = lax.stop_gradient(yp_z_s)

            y_ng, yp_ng = self.model.normalizer.transform(
                "denormalize", y_z_ng, yp_z_ng
            )
            sq = jnp.sum((y_ng - yp_ng) ** 2, axis=(2, 3))
            V = jnp.flip(jnp.cumsum(jnp.flip(sq, axis=1), axis=1), axis=1)

            if self.use_control_variates:
                prev0 = jnp.zeros((self.d, 1), dtype=jnp.float32)

                def build_phi_one_traj(yz, ypz):
                    def step(carry, n):
                        prev_y, prev_yp = carry
                        cur_phi = self._phi(prev_y, prev_yp, n)
                        return (yz[n], ypz[n]), cur_phi

                    ns = jnp.arange(self.N_len, dtype=jnp.int32)
                    (_, _), phis = lax.scan(step, (prev0, prev0), ns)
                    return phis

                phi = jax.vmap(build_phi_one_traj)(y_z_ng, yp_z_ng)

                if self._B_fit_static <= 0 or self._B_fit_static >= self.B:
                    w = self._fit_baseline_per_n(phi, V, ridge=self.cv_ridge)
                    baseline = self._baseline_predict(phi, w)
                else:
                    phi_fit = phi[: self._B_fit_static]
                    V_fit = V[: self._B_fit_static]
                    w = self._fit_baseline_per_n(phi_fit, V_fit, ridge=self.cv_ridge)
                    baseline = self._baseline_predict(phi, w)
            else:
                baseline = jnp.zeros_like(V)

            Adv = V - baseline

            if self.use_advantage_standardization:
                mean_t = jnp.mean(Adv, axis=0, keepdims=True)
                std_t = jnp.std(Adv, axis=0, keepdims=True) + 1e-6
                Adv = (Adv - mean_t) / std_t

            Adv = lax.stop_gradient(Adv)

            # ---- score-function surrogate ----
            logp_steps = self.model.logprob_steps_batch_norm(p, y_z_ng, yp_z_ng)
            loss_cost_sf = jnp.mean(jnp.sum(Adv * logp_steps, axis=1))

            # ---- pathwise true cost ----
            y_s, yp_s = self.model.normalizer.transform("denormalize", y_z_s, yp_z_s)
            V0_pw = jnp.sum((y_s - yp_s) ** 2, axis=(1, 2, 3))
            cost_mean_pathwise = jnp.mean(V0_pw)

            loss_cost = (1.0 - alpha) * loss_cost_sf + alpha * cost_mean_pathwise

            logp_y_z = self.model.logprob_total_y_batch_norm(p, y_true_z)
            logp_yp_z = self.model.logprob_total_yp_batch_norm(p, yp_true_z)
            if self.model.use_data_normalization:
                assert self.model.normalizer is not None
                logp_y = logp_y_z - self.model.normalizer.log_det_y
                logp_yp = logp_yp_z - self.model.normalizer.log_det_yp
            else:
                logp_y = logp_y_z
                logp_yp = logp_yp_z
            nll_marg = -(jnp.mean(logp_y) + jnp.mean(logp_yp))

            loss = loss_cost + beta * nll_marg

            aux = {
                "loss": loss,
                "loss_cost_advantage": loss_cost_sf,
                "loss_cost_pathwise": cost_mean_pathwise,
                "nll_marg": nll_marg,
                "mean_abs_rho_model": jnp.mean(rho_abs),
            }
            return loss, aux

        (loss_val, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(params)
        clip_norm = float(self.run_sett_policy_gradient["grad_clip_norm"])
        if clip_norm and clip_norm > 0.0:
            g_norm = optax.global_norm(grads)
            scale = jnp.minimum(1.0, clip_norm / (g_norm + 1e-6))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, aux, beta

    def update_params(self, key: jax.Array):
        """Advance training by one step and return scalar metrics."""
        step_i = jnp.asarray(self._step, dtype=jnp.int32)
        self.params, self.opt_state, metrics, beta = self._train_step(
            self.params, self.opt_state, key, step_i
        )
        if self.use_ema_eval:
            self.ema_params = jax.tree_util.tree_map(
                lambda e, p: self.ema_decay * e + (1.0 - self.ema_decay) * p,
                self.ema_params,
                self.params,
            )
        self._last_beta_value = float(beta)
        self._step += 1
        return {k: float(v) for k, v in metrics.items()}

    def compute_logging_losses(self, key: jax.Array):
        """Compute logging metrics J_val, NLL_true, and combined J_beta."""
        rng_namespace = int(self.run_sett_policy_gradient["RNG_NAMESPACE_PG"])
        B = int(self.run_sett_evaluation["eval_B"])
        chunk = int(self.run_sett_evaluation["metrics_chunk_size"])
        params = self.get_eval_params_trees()

        remaining = B
        cur_key = jax.random.fold_in(key, rng_namespace)
        cost_sum = 0.0
        tot = 0
        while remaining > 0:
            cur = min(remaining, chunk)
            cur_key, use_key = jax.random.split(cur_key)
            keys = jax.random.split(use_key, cur)
            y_z, yp_z, _ = self.model.sample_batch_norm(params, keys)
            y, yp = self.model.normalizer.transform("denormalize", y_z, yp_z)
            diff = y - yp
            V0 = jnp.sum(diff * diff, axis=(1, 2, 3))
            cost_sum += float(jnp.sum(V0))
            tot += cur
            remaining -= cur
        J_val = cost_sum / max(tot, 1)

        remaining = B
        cur_key = jax.random.fold_in(key, rng_namespace + 111_111)
        nll_sum = 0.0
        tot = 0
        while remaining > 0:
            cur = min(remaining, chunk)
            cur_key, use_key = jax.random.split(cur_key)
            keys = jax.random.split(use_key, cur)
            y_t, yp_t = jax.vmap(self.true_data_model.sample_true_trajectory)(keys)
            y_z, yp_z = self.model.normalizer.transform("normalize", y_t, yp_t)
            logp_y_z = self.model.logprob_total_y_batch_norm(params, y_z)
            logp_yp_z = self.model.logprob_total_yp_batch_norm(params, yp_z)
            if self.model.use_data_normalization:
                assert self.model.normalizer is not None
                logp_y = logp_y_z - self.model.normalizer.log_det_y
                logp_yp = logp_yp_z - self.model.normalizer.log_det_yp
            else:
                logp_y = logp_y_z
                logp_yp = logp_yp_z
            nll_sum += float(jnp.sum(-(logp_y + logp_yp)))
            tot += cur
            remaining -= cur

        NLL_marg = nll_sum / max(tot, 1)
        beta = float(self._last_beta_value)
        J_beta = J_val + beta * NLL_marg
        return {
            "J_val": float(J_val),
            "NLL_marg": float(NLL_marg),
            "J_beta": float(J_beta),
            "beta": float(beta),
        }

    def sample_trajectory(self, key: jax.Array, params: Optional[hk.Params] = None):
        """Sample a single trajectory ``(y, yp)`` in ORIGINAL space; params default to EMA if enabled."""
        if params is None:
            params = self.get_eval_params_trees()
        y_z, yp_z, _ = self.model.sample_batch_norm(params, jax.random.split(key, 1))
        y, yp = self.model.normalizer.transform("denormalize", y_z[0], yp_z[0])
        return y, yp

    def sample_trajectories(
        self,
        key: jax.Array,
        num: int,
        params: Optional[hk.Params] = None,
    ):
        """Sample a batch of trajectories in ORIGINAL space.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.
        num : int
            Number of trajectories to sample.
        params : hk.Params, optional
            Model parameters; defaults to EMA params if enabled, else current params.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(y, yp)`` each of shape ``(num, N+1, d, 1)`` in ORIGINAL space.
        """
        if params is None:
            params = self.get_eval_params_trees()
        num = int(num)
        keys = jax.random.split(key, num)
        y_z, yp_z, _ = self.model.sample_batch_norm(params, keys)
        y, yp = self.model.normalizer.transform("denormalize", y_z, yp_z)
        return y, yp

    def joint_log_prob_batch(self, y, yp, params: Optional[hk.Params] = None):
        """Compute joint log-probabilities log p(y, yp) for a batch in ORIGINAL space.

        Parameters
        ----------
        y : jax.Array
            Batch of ``y`` trajectories, shape ``(B, N+1, d, 1)`` in ORIGINAL space.
        yp : jax.Array
            Batch of ``yp`` trajectories, shape ``(B, N+1, d, 1)`` in ORIGINAL space.
        params : hk.Params, optional
            Model parameters; defaults to EMA params if enabled, else current params.

        Returns
        -------
        jax.Array
            Log-probabilities of shape ``(B,)``.
        """
        if params is None:
            params = self.get_eval_params_trees()
        y_z, yp_z = self.model.normalizer.transform("normalize", y, yp)
        logp_z = self.model.logprob_total_batch_norm(params, y_z, yp_z)
        if self.model.use_data_normalization:
            return logp_z - self.model.normalizer.log_det
        return logp_z

    def save_params(self, ckpt_dir: str):
        """Save params, ema_params, and opt_state."""
        abs_dir = os.path.abspath(ckpt_dir)
        options = ocp.CheckpointManagerOptions(create=True, max_to_keep=10)
        item_handlers = {
            "params": ocp.PyTreeCheckpointHandler(),
            "ema_params": ocp.PyTreeCheckpointHandler(),
            "opt_state": ocp.PyTreeCheckpointHandler(),
        }
        manager = ocp.CheckpointManager(
            abs_dir, item_handlers=item_handlers, options=options
        )
        payload = {
            "params": self.params,
            "ema_params": self.ema_params,
            "opt_state": self.opt_state,
        }
        current_step = int(self._step)
        manager.save(step=current_step, items=payload)
        try:
            manager.wait_until_finished()
        except Exception as e:
            print(f"[PolicyGradient] Warning: checkpoint saving raised: {e}")
        print(
            f"[PolicyGradient] State saved to: {os.path.join(abs_dir, str(current_step))}"
        )

    def load_params(self, ckpt_dir: str) -> bool:
        """Load latest params, ema_params."""
        try:
            abs_dir = os.path.abspath(ckpt_dir)
            if not os.path.isdir(abs_dir):
                print(f"[PolicyGradient] Checkpoint dir does not exist: {abs_dir}")
                return False
            item_handlers = {
                "params": ocp.PyTreeCheckpointHandler(),
                "ema_params": ocp.PyTreeCheckpointHandler(),
                "opt_state": ocp.PyTreeCheckpointHandler(),
            }
            manager = ocp.CheckpointManager(
                abs_dir,
                item_handlers=item_handlers,
                options=ocp.CheckpointManagerOptions(create=False, max_to_keep=2),
            )
            latest_step = manager.latest_step()
            if latest_step is None:
                print(f"[PolicyGradient] No checkpoints found in: {abs_dir}")
                return False
            target_item = {
                "params": self.params,
                "ema_params": self.ema_params,
                "opt_state": self.opt_state,
            }
            restored = manager.restore(latest_step, items=target_item)
            self.params = restored["params"]
            self.ema_params = restored["ema_params"]
            self.opt_state = restored["opt_state"]
            self._step = int(latest_step)
            print(
                f"[PolicyGradient] State loaded from step {latest_step} at: {abs_dir}"
            )
            return True
        except Exception as e:
            print(f"[PolicyGradient] Failed to load state from '{ckpt_dir}': {e}")
            return False
