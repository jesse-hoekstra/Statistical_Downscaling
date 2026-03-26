"""Data-generation primitives for optimal transport experiments (JAX).

This module provides small stochastic state-transition models that generate
paired trajectories (y, y') for use in statistical downscaling experiments.
There are two variants:
  - TrueDataModelUnimodal: single-mode noise model
      Transition per dimension i:
        y_t[i] = base_means[i] + 0.5 * y_{t-1}[i] + eps_t[i],
        where eps_t[i] = [eps_y, eps_y'] with
          eps_y ~ Normal(0, 0.5^2),
          eps_y' = 4 * Beta(2, 5) - 1.5.
  - TrueDataModelBimodal: per-dimension fixed-mode (selector) mixture
      A Bernoulli selector s[i] = (s_y, s_y') is sampled once per dimension
      and held fixed across time. Then:
        y_t[i] = base_means[i] + 0.5 * y_{t-1}[i] + eps_t[i],
        with eps_t[i] = [eps_y, eps_y'] where:
          For y:
            if s_y:   eps_y ~ Normal(1.5, 0.5^2)
            else:     eps_y = 2.5 * Beta(2, 5) - 2.5
          For y':
            if s_y':  eps_y' = 2.5 * Beta(5, 2) + 0.5
            else:     eps_y' ~ Normal(-1.5, 0.5^2)

All sampling routines are JIT-compiled with JAX and operate on PRNG keys.
"""

import copy
import jax
import jax.numpy as jnp
from jax.scipy.special import betaln
from functools import partial
import h5py
from typing import Dict, Any

_LOG2PI = jnp.asarray(jnp.log(2.0 * jnp.pi), dtype=jnp.float32)


def _logpdf_normal(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """
    x, mean, std.
    return logpdf.
    """
    std = jnp.maximum(std, jnp.asarray(1e-12, dtype=jnp.float32))
    z = (x - mean) / std
    return -0.5 * _LOG2PI - jnp.log(std) - 0.5 * (z * z)


def _logpdf_beta_affine(
    x: jnp.ndarray, a: float, b: float, scale: float, shift: float
) -> jnp.ndarray:
    """
    x = scale * U + shift,  U ~ Beta(a,b)
    support: x in [shift, shift+scale] (assuming scale>0)
    return elementwise logpdf in x-space.
    """
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    scale = jnp.asarray(scale, dtype=jnp.float32)
    shift = jnp.asarray(shift, dtype=jnp.float32)

    u = (x - shift) / scale
    in_supp = (u > 0.0) & (u < 1.0)

    u_clip = jnp.clip(u, 1e-6, 1.0 - 1e-6)
    log_pdf_u = (
        (a - 1.0) * jnp.log(u_clip) + (b - 1.0) * jnp.log1p(-u_clip) - betaln(a, b)
    )
    log_pdf_x = log_pdf_u - jnp.log(jnp.abs(scale))

    return jnp.where(in_supp, log_pdf_x, -jnp.inf)


class TrueDataModelUnimodal:
    """Unimodal stochastic transition model for (y, y') trajectories.

    The latent state at each time step is a 2D vector per spatial dimension
    d, representing [y, y']. The next state is the sum of a linear carryover
    term and independent unimodal noise with fixed scale:
      - y-noise: Normal(0, 0.5^2)
      - y'-noise: 4 * Beta(2, 5) - 1.5
    """

    def __init__(self, run_sett: dict, phi: float = 0.5):
        """Initialize the model hyperparameters.

        Args:
            run_sett: Dictionary with required fields:
                - 'N' (int): Number of transition steps to sample.
                - 'd' (int): Number of spatial dimensions.
            phi: AR coefficient (default 0.5).
        """
        self.run_sett = run_sett
        run_sett_global = run_sett["global"]
        self.N = run_sett_global["N"]
        self.d = run_sett_global["d"]
        self.phi = float(phi)
        self.base_means = (
            jnp.linspace(-2, 2, self.d).reshape(self.d, 1).astype(jnp.float32)
        )

    def _step_dist(self, k, prev_val):
        """Sample a single transition given previous state.

        Args:
            k: JAX PRNG key for this transition.
            prev_val: Array with shape (d, 2), previous [y, y'] values.

        Returns:
            Array with shape (d, 2), the next [y, y'] state.

        Distributional details per dimension i:
          next_y[i]   = base_means[i] + phi * prev_y[i]   + Normal(0, 0.5^2)
          next_y'[i]  = base_means[i] + phi * prev_y'[i]  + (4*Beta(2,5)-1.5)
        """
        k_y, k_yp = jax.random.split(k)
        noise_y = jax.random.normal(k_y, shape=(self.d, 1)) * 0.5
        raw_beta = jax.random.beta(k_yp, a=2.0, b=5.0, shape=(self.d, 1))
        noise_yp = (raw_beta * 4.0) - 1.5
        noise = jnp.concatenate([noise_y, noise_yp], axis=1)
        return self.base_means + self.phi * prev_val + noise

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_true_trajectory(self, key, return_latents: bool = False):
        """Generate a full (y, y') trajectory.

        This returns the pair of sequences (y_t, y'_t) for t=0..N, where t=0
        is the initial step sampled from the same transition as the rest.

        Args:
            key: JAX PRNG key used to drive all randomness.

        Returns:
            Tuple (y, y_prime):
                - y: Array with shape (N+1, d, 1)
                - y_prime: Array with shape (N+1, d, 1)
        """
        key, k0 = jax.random.split(key)
        val0 = self._step_dist(k0, jnp.zeros((self.d, 2), dtype=jnp.float32))

        def step(carry, k):
            next_val = self._step_dist(k, carry)
            return next_val, next_val

        keys = jax.random.split(key, self.N)
        _, vals = jax.lax.scan(step, val0, keys)
        traj = jnp.concatenate([val0[None, ...], vals], axis=0)
        y = traj[..., 0:1]
        yp = traj[..., 1:2]
        if return_latents:
            lat = jnp.zeros((self.d, 2), dtype=jnp.bool_)
            return y, yp, lat
        return y, yp

    def log_prob_y_cond_steps(
        self, y: jnp.ndarray, latents=None, mode: str = "oracle"
    ) -> jnp.ndarray:
        # y: (B,N_len,d_prime,1) or (N_len,d_prime,1)
        if y.ndim == 3:
            y = y[None, ...]
        B, N_len, d_prime, _ = y.shape
        assert d_prime == self.d

        prev0 = jnp.zeros((B, 1, d_prime, 1), dtype=jnp.float32)
        prev = jnp.concatenate([prev0, y[:, :-1, :, :]], axis=1)

        noise = y - (
            self.base_means[None, None, :, :] + self.phi * prev
        )  # (B,N_len,d_prime,1)
        lp = _logpdf_normal(noise, mean=0.0, std=0.5)
        return jnp.sum(lp, axis=(2, 3))  # (B,N_len)

    def log_prob_yp_cond_steps(
        self, yp: jnp.ndarray, latents=None, mode: str = "oracle"
    ) -> jnp.ndarray:
        if yp.ndim == 3:
            yp = yp[None, ...]
        B, N_len, d_prime, _ = yp.shape
        assert d_prime == self.d

        prev0 = jnp.zeros((B, 1, d_prime, 1), dtype=jnp.float32)
        prev = jnp.concatenate([prev0, yp[:, :-1, :, :]], axis=1)

        noise = yp - (self.base_means[None, None, :, :] + self.phi * prev)
        lp = _logpdf_beta_affine(noise, a=2.0, b=5.0, scale=4.0, shift=-1.5)
        return jnp.sum(lp, axis=(2, 3))


class TrueDataModelBimodal:
    """Bimodal stochastic transition model for (y, y') trajectories.

    Similar to the unimodal variant but uses a fixed, per-dimension selector to
    switch between two noise modes for each of y and y'. This induces a
    mixture-like behavior across dimensions while keeping transitions simple.
    For each dimension i, we sample once s[i] = (s_y, s_y') ~ Bernoulli(0.5)
    and hold it fixed for all time steps:
      - If s_y is True:  y-noise ~ Normal(1.5, 0.5^2)
        else:            y-noise = 2.5 * Beta(2, 5) - 2.5
      - If s_y' is True: y'-noise = 2.5 * Beta(5, 2) + 0.5
        else:            y'-noise ~ Normal(-1.5, 0.5^2)
    """

    def __init__(self, run_sett: dict):
        """Initialize the model hyperparameters.

        Args:
            run_sett: Dictionary with required fields:
                - 'N' (int): Number of transition steps to sample.
                - 'd' (int): Number of spatial dimensions.
        """
        self.run_sett = run_sett
        run_sett_global = run_sett["global"]
        self.N = run_sett_global["N"]
        self.d = run_sett_global["d"]
        self.base_means = (
            jnp.linspace(-0.5, 0.0, self.d).reshape(self.d, 1).astype(jnp.float32)
        )

    def _step_dist(self, k, prev_val, selector):
        """Sample a single transition given previous state and mode selector.

        Args:
            k: JAX PRNG key for this transition.
            prev_val: Array with shape (d, 2), previous [y, y'] values.
            selector: Boolean array with shape (d, 2). For each dimension:
                - selector[:, 0] chooses left/right mode for y
                - selector[:, 1] chooses left/right mode for y'

        Returns:
            Array with shape (d, 2), the next [y, y'] state.

        Conditional noise per dimension i given selector s[i] = (s_y, s_y'):
          y:
            s_y=True  -> Normal(1.5, 0.5^2)
            s_y=False -> 2.5 * Beta(2, 5) - 2.5
          y':
            s_y'=True  -> 2.5 * Beta(5, 2) + 0.5
            s_y'=False -> Normal(-1.5, 0.5^2)
        """
        k_y_l, k_y_r, k_yp_l, k_yp_r = jax.random.split(k, 4)
        y_left = (jax.random.beta(k_y_l, 2.0, 5.0, shape=(self.d, 1)) * 2.5) - 2.5
        y_right = jax.random.normal(k_y_r, shape=(self.d, 1)) * 0.5 + 1.5
        yp_left = jax.random.normal(k_yp_l, shape=(self.d, 1)) * 0.5 - 1.5
        yp_right = (jax.random.beta(k_yp_r, 5.0, 2.0, shape=(self.d, 1)) * 2.5) + 0.5
        noise_y = jnp.where(selector[:, 0:1], y_right, y_left)
        noise_yp = jnp.where(selector[:, 1:2], yp_right, yp_left)
        noise = jnp.concatenate([noise_y, noise_yp], axis=1)
        return self.base_means + 0.5 * prev_val + noise

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_true_trajectory(self, key, return_latents: bool = False):
        """Generate a full (y, y') trajectory under a fixed per-dimension selector.

        A per-dimension, per-component Bernoulli selector is sampled once and
        held fixed across time, so each dimension consistently follows one of
        two noise modes for y and for y'.

        Args:
            key: JAX PRNG key used to drive all randomness.

        Returns:
            Tuple (y, y_prime):
                - y: Array with shape (N+1, d, 1)
                - y_prime: Array with shape (N+1, d, 1)
        """
        key, k_sel, k0 = jax.random.split(key, 3)
        selector = jax.random.bernoulli(k_sel, p=0.5, shape=(self.d, 2))
        val0 = self._step_dist(k0, jnp.zeros((self.d, 2), dtype=jnp.float32), selector)

        def step(carry, k):
            prev_val, sel = carry
            next_val = self._step_dist(k, prev_val, sel)
            return (next_val, sel), next_val

        keys = jax.random.split(key, self.N)
        _, vals = jax.lax.scan(step, (val0, selector), keys)
        traj = jnp.concatenate([val0[None, ...], vals], axis=0)
        y = traj[..., 0:1]
        yp = traj[..., 1:2]

        if return_latents:
            return y, yp, selector
        return y, yp

    def log_prob_y_cond_steps(
        self, y: jnp.ndarray, latents=None, mode: str = "oracle"
    ) -> jnp.ndarray:
        if y.ndim == 3:
            y = y[None, ...]
        B, N_len, d_prime, _ = y.shape
        assert d_prime == self.d

        mode = str(mode).lower()
        if mode != "oracle":
            raise ValueError("For bimodal you requested oracle mode only.")

        if latents is None:
            raise ValueError("Oracle bimodal KL needs latents/selector.")

        if latents.ndim == 2:
            latents = jnp.broadcast_to(latents[None, ...], (B, d_prime, 2))
        sel_y = latents[:, :, 0:1]

        prev0 = jnp.zeros((B, 1, d_prime, 1), dtype=jnp.float32)
        prev = jnp.concatenate([prev0, y[:, :-1, :, :]], axis=1)

        noise = y - (
            self.base_means[None, None, :, :] + 0.5 * prev
        )  # (B,N_len,d_prime,1)

        # right: Normal(1.5,0.5), left: 2.5*Beta(2,5)-2.5
        lp_right = _logpdf_normal(noise, mean=1.5, std=0.5)
        lp_left = _logpdf_beta_affine(noise, a=2.0, b=5.0, scale=2.5, shift=-2.5)

        sel_y_bt = sel_y[:, None, :, :]
        lp = jnp.where(sel_y_bt, lp_right, lp_left)
        return jnp.sum(lp, axis=(2, 3))  # (B,N_len)

    def log_prob_yp_cond_steps(
        self, yp: jnp.ndarray, latents=None, mode: str = "oracle"
    ) -> jnp.ndarray:
        if yp.ndim == 3:
            yp = yp[None, ...]
        B, N_len, d_prime, _ = yp.shape
        assert d_prime == self.d

        mode = str(mode).lower()
        if mode != "oracle":
            raise ValueError("For bimodal you requested oracle mode only.")

        if latents is None:
            raise ValueError("Oracle bimodal KL needs latents/selector.")

        if latents.ndim == 2:
            latents = jnp.broadcast_to(latents[None, ...], (B, d_prime, 2))
        sel_yp = latents[:, :, 1:2]  # (B,d_prime,1)

        prev0 = jnp.zeros((B, 1, d_prime, 1), dtype=jnp.float32)
        prev = jnp.concatenate([prev0, yp[:, :-1, :, :]], axis=1)

        noise = yp - (self.base_means[None, None, :, :] + 0.5 * prev)

        # right: 2.5*Beta(5,2)+0.5, left: Normal(-1.5,0.5)
        lp_right = _logpdf_beta_affine(noise, a=5.0, b=2.0, scale=2.5, shift=0.5)
        lp_left = _logpdf_normal(noise, mean=-1.5, std=0.5)

        sel_yp_bt = sel_yp[:, None, :, :]
        lp = jnp.where(sel_yp_bt, lp_right, lp_left)
        return jnp.sum(lp, axis=(2, 3))


class RobustHedgingModel:
    """
    Gaussian random-walk benchmark.
    """

    def __init__(self, run_sett: Dict[str, Any]):
        run_sett_global = run_sett["global"]
        run_sett_robust_hedging = run_sett["robust_hedging"]
        self.N = int(run_sett_global["N"])
        self.d = int(run_sett_global["d"])
        self.dt = float(run_sett_robust_hedging["dt"])
        self._sqrt_dt = jnp.sqrt(jnp.asarray(self.dt, dtype=jnp.float32))

        def _as_vec(x):
            a = jnp.asarray(x, dtype=jnp.float32)
            if a.ndim == 0:
                return jnp.full((self.d,), a, dtype=jnp.float32)
            if int(a.shape[0]) != self.d:
                raise ValueError(f"Expected length {self.d}, got shape {a.shape}.")
            return a.astype(jnp.float32)

        self.x0 = float(run_sett_robust_hedging["S0_y"])
        self.y0 = float(run_sett_robust_hedging["S0_yp"])
        self._x0_vec = _as_vec(self.x0)
        self._y0_vec = _as_vec(self.y0)

        self.sigma_x = _as_vec(run_sett_robust_hedging["sigma_y"])
        self.sigma_y = _as_vec(run_sett_robust_hedging["sigma_yp"])

    def _step_dist(self, k, prev_val: jnp.ndarray) -> jnp.ndarray:
        k1, k2 = jax.random.split(k, 2)
        eps1 = jax.random.normal(k1, shape=(self.d,), dtype=jnp.float32)
        eps2 = jax.random.normal(k2, shape=(self.d,), dtype=jnp.float32)

        dx = self.sigma_x * self._sqrt_dt * eps1
        dy = self.sigma_y * self._sqrt_dt * eps2

        x_next = prev_val[:, 0] + dx
        y_next = prev_val[:, 1] + dy
        return jnp.stack([x_next, y_next], axis=1)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_true_trajectory(self, key, return_latents: bool = False):
        val0 = jnp.stack([self._x0_vec, self._y0_vec], axis=1)

        def step(carry, k):
            nxt = self._step_dist(k, carry)
            return nxt, nxt

        keys = jax.random.split(key, self.N + 1)
        _, vals = jax.lax.scan(step, val0, keys)

        y = vals[..., 0:1]
        yp = vals[..., 1:2]

        if return_latents:
            lat = jnp.zeros((self.d, 2), dtype=jnp.bool_)
            return y, yp, lat
        return y, yp

    def log_prob_y_cond_steps(
        self, y: jnp.ndarray, latents=None, mode: str = "oracle"
    ) -> jnp.ndarray:
        if y.ndim == 3:
            y = y[None, ...]
        B, N_len, d_prime, _ = y.shape
        assert d_prime == self.d

        prev0 = jnp.broadcast_to(
            self._x0_vec.reshape(1, 1, d_prime, 1), (B, 1, d_prime, 1)
        )
        prev = jnp.concatenate([prev0, y[:, :-1, :, :]], axis=1)

        inc = y - prev  # dx
        std = (self.sigma_x * self._sqrt_dt).reshape(1, 1, d_prime, 1)
        lp = _logpdf_normal(inc, mean=0.0, std=std)
        return jnp.sum(lp, axis=(2, 3))

    def log_prob_yp_cond_steps(
        self, yp: jnp.ndarray, latents=None, mode: str = "oracle"
    ) -> jnp.ndarray:
        if yp.ndim == 3:
            yp = yp[None, ...]
        B, N_len, d_prime, _ = yp.shape
        assert d_prime == self.d

        prev0 = jnp.broadcast_to(
            self._y0_vec.reshape(1, 1, d_prime, 1), (B, 1, d_prime, 1)
        )
        prev = jnp.concatenate([prev0, yp[:, :-1, :, :]], axis=1)

        inc = yp - prev
        std = (self.sigma_y * self._sqrt_dt).reshape(1, 1, d_prime, 1)
        lp = _logpdf_normal(inc, mean=0.0, std=std)
        return jnp.sum(lp, axis=(2, 3))

    def optimal_cost_closed_form(self) -> float:
        d0 = self._x0_vec - self._y0_vec
        d0_sq = jnp.sum(d0 * d0)

        def _as_cov(sig_or_cov):
            a = jnp.asarray(sig_or_cov, dtype=jnp.float32)
            if a.ndim == 1:
                return self.dt * jnp.diag(a * a)
            if a.ndim == 2:
                return self.dt * a
            raise ValueError(
                f"sigma/cov must be 1D(std) or 2D(cov), got shape {a.shape}."
            )

        Sigma_x = _as_cov(self.sigma_x)
        Sigma_y = _as_cov(self.sigma_y)

        def _psd_sqrt(A):
            A = 0.5 * (A + A.T)
            w, V = jnp.linalg.eigh(A)
            w = jnp.clip(w, 0.0)
            return (V * jnp.sqrt(w)[None, :]) @ V.T

        Sx_sqrt = _psd_sqrt(Sigma_x)
        mid = Sx_sqrt @ Sigma_y @ Sx_sqrt
        mid_sqrt = _psd_sqrt(mid)

        tr_term = jnp.trace(Sigma_x) + jnp.trace(Sigma_y) - 2.0 * jnp.trace(mid_sqrt)
        T = jnp.asarray(self.N + 1, dtype=jnp.float32)
        total = T * d0_sq + (T * (T + 1.0) / 2.0) * tr_term
        return float(total)


class KSTrueDataModel:
    """True data model backed by Kuramoto–Sivashinsky simulation data.

    Paired trajectories (y, y') correspond to low-fidelity/low-resolution (LFLR)
    observations and a temporally-downsampled high-fidelity/high-resolution (HFHR)
    field, respectively.  All data are loaded from an HDF5 file at construction
    time so that `sample_true_trajectory` is a pure, jit/vmap-compatible lookup.
    """

    def __init__(self, run_sett: dict):
        self.run_sett = run_sett
        self.run_sett_data_KS = run_sett["data_KS"]
        self.d = int(run_sett["data_KS"]["d"])
        self.training_samples = self.run_sett_data_KS["training_samples"]
        self._load_data()

    def _load_data(self) -> None:
        """Load HDF5 file and prepare jnp train/test arrays."""
        with h5py.File(self.run_sett_data_KS["data_file_name"], "r") as f:
            u_LFLR = f["LFLR"][()]
            u_HFHR = f["HFHR"][()]

        u_LFLR = u_LFLR[:, :, ::2]

        x_train_eval, x_test, y_train_eval, y_test = self._get_train_test(
            u_HFHR, u_LFLR, split=48
        )

        self.x_train_eval = jnp.asarray(
            x_train_eval[: self.training_samples], dtype=jnp.float32
        )
        x_test_extra = jnp.asarray(
            x_train_eval[self.training_samples :], dtype=jnp.float32
        )
        n_generated = (
            self.run_sett["global"]["num_gen_samples"]
            * self.run_sett["global"]["num_conditionings"]
        )
        self.x_test = jnp.concatenate(
            [jnp.asarray(x_test, dtype=jnp.float32), x_test_extra], axis=0
        )[:n_generated]

        n_x, n_y = x_train_eval.shape[1], y_train_eval.shape[1]
        factor = int(n_x / n_y)
        yp_train_eval = x_train_eval[:, ::factor]  # (M, n_y, d)

        # x_train_eval / x_test: (M, n_x, d) — no trailing singleton.
        # y_test:                (M, n_y, d) — no trailing singleton.
        # y_train_eval / yp_train_eval: (M, n_y, d, 1) — trailing singleton kept
        # because sample_true_trajectory returns these to DataNormalizer, which
        # operates on (*, d, 1) shaped arrays throughout the OT training core.
        self.y_train_eval = jnp.asarray(y_train_eval[..., None], dtype=jnp.float32)
        self.yp_train_eval = jnp.asarray(yp_train_eval[..., None], dtype=jnp.float32)
        self.y_test = jnp.asarray(y_test, dtype=jnp.float32)
        self._n_train_y = int(self.y_train_eval.shape[0])
        self._n_train_yp = int(self.yp_train_eval.shape[0])

    def _get_train_test(self, u_HFHR, u_LFLR, split: int):
        """Reshape raw HDF5 arrays and split into train-eval / test sets.

        Each trajectory tensor has shape (n_traj, time, spatial_flat) where
        spatial_flat = n_inner * d.  After reshaping and transposing the spatial
        inner dimension becomes the sample dimension, yielding arrays of shape
        (n_traj * n_inner, time, d).

        Args:
            u_HFHR: Raw HFHR array, shape (n_traj, time_HF, n_inner_HF * d).
            u_LFLR: Raw LFLR array, shape (n_traj, time_LF, n_inner_LF * d).
            split:  Number of trajectories reserved for the test set.

        Returns:
            x_train, x_test: HFHR splits, shape (M_*, time_HF, d).
            y_train, y_test: LFLR splits, shape (M_*, n_y,     d).
        """
        d = self.d
        n_y = self.run_sett_data_KS["n_y"]
        n_x = self.run_sett_data_KS["n_x"]

        u_HFHR_train_eval, u_HFHR_test = u_HFHR[:-split], u_HFHR[-split:]
        u_LFLR_train_eval, u_LFLR_test = u_LFLR[:-split], u_LFLR[-split:]

        n_samples_train_eval = int(u_HFHR_train_eval.shape[0])
        n_samples_test = int(u_HFHR_test.shape[0])
        n_samples_inner_x = int(u_HFHR_train_eval.shape[2] / d)
        n_samples_inner_y = int(u_LFLR_train_eval.shape[2] / d)

        x_train_eval = (
            u_HFHR_train_eval[:, :n_x, :]
            .reshape(n_samples_train_eval, n_x, n_samples_inner_x, d)
            .transpose(0, 2, 1, 3)
            .reshape(n_samples_train_eval * n_samples_inner_x, n_x, d)
        )
        x_test = (
            u_HFHR_test[:, :n_x, :]
            .reshape(n_samples_test, n_x, n_samples_inner_x, d)
            .transpose(0, 2, 1, 3)
            .reshape(n_samples_test * n_samples_inner_x, n_x, d)
        )
        y_train_eval = (
            u_LFLR_train_eval[:, :n_y, :]
            .reshape(n_samples_train_eval, n_y, n_samples_inner_y, d)
            .transpose(0, 2, 1, 3)
            .reshape(n_samples_train_eval * n_samples_inner_y, n_y, d)
        )
        y_test = (
            u_LFLR_test[:, :n_y, :]
            .reshape(n_samples_test, n_y, n_samples_inner_y, d)
            .transpose(0, 2, 1, 3)
            .reshape(n_samples_test * n_samples_inner_y, n_y, d)
        )

        return (
            x_train_eval,
            x_test,
            y_train_eval,
            y_test,
        )

    def sample_true_trajectory(self, key):
        """Sample a single (y, y') pair uniformly from the training set.

        Compatible with ``jax.vmap`` and ``jax.jit``.

        Args:
            key: JAX PRNG key.

        Returns:
            y:  shape (N+1, d, 1) — LFLR observation.
            yp: shape (N+1, d, 1) — temporally-downsampled HFHR field.
        """
        key_y, key_yp = jax.random.split(key)
        idx_y = jax.random.randint(key_y, shape=(), minval=0, maxval=self._n_train_y)
        idx_yp = jax.random.randint(key_yp, shape=(), minval=0, maxval=self._n_train_yp)
        return (
            jnp.take(self.y_train_eval, idx_y, axis=0),
            jnp.take(self.yp_train_eval, idx_yp, axis=0),
        )

    def true_trajectory(self):
        """Return pre-loaded (y_train, yp_train, y_test) arrays.

        Returns:
            y_train:  (M_train, N+1, d, 1)
            yp_train: (M_train, N+1, d, 1)
            y_test:   (M_test,  N+1, d)
        """
        return self.y_train_eval, self.yp_train_eval, self.y_test


class ARTrueDataModel:

    def __init__(self, run_sett: dict):
        self.run_sett = run_sett
        self.seed = run_sett["global"]["seed"]
        self.d = int(run_sett["data_AR"]["d"])
        self.test_samples = int(run_sett["data_AR"]["test_samples"])
        self.y_nsamples = int(run_sett["data_AR"]["y_nsamples"])
        self.x_nsamples = int(run_sett["data_AR"]["x_nsamples"])
        self.n_x = int(run_sett["data_AR"]["n_x"])
        self.n_y = int(run_sett["data_AR"]["n_y"])
        self.phi = float(run_sett["data_AR"].get("phi", 0.5))
        self.y_factor = int(self.n_x / self.n_y)
        self._load_data()

    def _load_data(self) -> None:
        """Load HDF5 file and prepare jnp train/test arrays."""

        sett_ar = copy.deepcopy(self.run_sett)
        sett_ar["global"]["N"] = self.n_x - 1
        sett_ar["global"]["d"] = self.d
        model = TrueDataModelUnimodal(sett_ar, phi=self.phi)

        master_key = jax.random.PRNGKey(self.seed)
        train_key, test_key = jax.random.split(master_key)
        keys = jax.random.split(train_key, self.y_nsamples)
        keys_test = jax.random.split(test_key, self.test_samples)

        x_train, _ = jax.vmap(model.sample_true_trajectory)(keys)
        x_train = x_train.squeeze(-1)
        y_train_eval = x_train[:, :: self.y_factor]
        x_train_eval = x_train[: self.x_nsamples]
        yp_train_eval = x_train_eval.reshape(
            self.x_nsamples, self.n_x // self.y_factor, self.y_factor, self.d
        ).mean(axis=2)

        x_test, _ = jax.vmap(model.sample_true_trajectory)(keys_test)
        x_test = x_test.squeeze(-1)
        y_test = x_test[:, :: self.y_factor]

        self.x_train_eval = jnp.asarray(x_train_eval, dtype=jnp.float32)
        self.x_test = jnp.asarray(x_test, dtype=jnp.float32)
        self.y_train_eval = jnp.asarray(y_train_eval[..., None], dtype=jnp.float32)
        self.yp_train_eval = jnp.asarray(yp_train_eval[..., None], dtype=jnp.float32)
        self.y_test = jnp.asarray(y_test, dtype=jnp.float32)
        self._n_train_y = int(self.y_train_eval.shape[0])
        self._n_train_yp = int(self.yp_train_eval.shape[0])

    def sample_true_trajectory(self, key, return_latents: bool = False):
        """Sample a single (y, y') pair uniformly from the training set.

        Compatible with ``jax.vmap`` and ``jax.jit``.

        Args:
            key: JAX PRNG key.
            return_latents: If True, also return a dummy latent array.

        Returns:
            y:  shape (N+1, d, 1) — LFLR observation.
            yp: shape (N+1, d, 1) — temporally-downsampled HFHR field.
        """
        key_y, key_yp = jax.random.split(key)
        idx_y = jax.random.randint(key_y, shape=(), minval=0, maxval=self._n_train_y)
        idx_yp = jax.random.randint(key_yp, shape=(), minval=0, maxval=self._n_train_yp)
        y = jnp.take(self.y_train_eval, idx_y, axis=0)
        yp = jnp.take(self.yp_train_eval, idx_yp, axis=0)
        if return_latents:
            lat = jnp.zeros((self.d, 2), dtype=jnp.bool_)
            return y, yp, lat
        return y, yp

    def true_trajectory(self):
        """Return pre-loaded (y_train, yp_train, y_test) arrays.

        Returns:
            y_train:  (M_train, N+1, d, 1)
            yp_train: (M_train, N+1, d, 1)
            y_test:   (M_test,  N+1, d)
        """
        return self.y_train_eval, self.yp_train_eval, self.y_test
