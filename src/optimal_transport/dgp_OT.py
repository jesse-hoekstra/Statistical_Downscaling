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

import jax
import jax.numpy as jnp
from functools import partial
import h5py


class TrueDataModelUnimodal:
    """Unimodal stochastic transition model for (y, y') trajectories.

    The latent state at each time step is a 2D vector per spatial dimension
    d, representing [y, y']. The next state is the sum of a linear carryover
    term and independent unimodal noise with fixed scale:
      - y-noise: Normal(0, 0.5^2)
      - y'-noise: 4 * Beta(2, 5) - 1.5
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
        self.base_means = jnp.linspace(-2, 2, self.d).reshape(self.d, 1)

    def _step_dist(self, k, prev_val):
        """Sample a single transition given previous state.

        Args:
            k: JAX PRNG key for this transition.
            prev_val: Array with shape (d, 2), previous [y, y'] values.

        Returns:
            Array with shape (d, 2), the next [y, y'] state.

        Distributional details per dimension i:
          next_y[i]   = base_means[i] + 0.5 * prev_y[i]   + Normal(0, 0.5^2)
          next_y'[i]  = base_means[i] + 0.5 * prev_y'[i]  + (4*Beta(2,5)-1.5)
        """
        k_y, k_yp = jax.random.split(k)
        noise_y = jax.random.normal(k_y, shape=(self.d, 1)) * 0.5
        raw_beta = jax.random.beta(k_yp, a=2.0, b=5.0, shape=(self.d, 1))
        noise_yp = (raw_beta * 4.0) - 1.5
        noise = jnp.concatenate([noise_y, noise_yp], axis=1)
        return self.base_means + 0.5 * prev_val + noise

    @partial(jax.jit, static_argnums=0)
    def sample_true_trajectory(self, key):
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
        val0 = self._step_dist(k0, jnp.zeros((self.d, 2)))

        def step(carry, k):
            next_val = self._step_dist(k, carry)
            return next_val, next_val

        keys = jax.random.split(key, self.N)
        _, vals = jax.lax.scan(step, val0, keys)
        traj = jnp.concatenate([val0[None, ...], vals], axis=0)
        return traj[..., 0:1], traj[..., 1:2]


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
        self.base_means = jnp.linspace(-0.5, 0.0, self.d).reshape(self.d, 1)

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

    @partial(jax.jit, static_argnums=0)
    def sample_true_trajectory(self, key):
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
        val0 = self._step_dist(k0, jnp.zeros((self.d, 2)), selector)

        def step(carry, k):
            prev_val, sel = carry
            next_val = self._step_dist(k, prev_val, sel)
            return (next_val, sel), next_val

        keys = jax.random.split(key, self.N)
        _, vals = jax.lax.scan(step, (val0, selector), keys)
        traj = jnp.concatenate([val0[None, ...], vals], axis=0)
        return traj[..., 0:1], traj[..., 1:2]


class TrueDataModel:
    """True data model using fluid flows or other real data examples."""

    def __init__(self, run_sett: dict, mode: str):
        """Initialize the model hyperparameters.

        Args:
            run_sett: Dictionary with required fields:
        """
        self.run_sett = run_sett
        self.run_sett_data_KS = run_sett["data_KS"]
        self.d = self.run_sett["global"][
            "d"
        ]  # wrong notation, but copied from other datamodels
        self.downsampling_factor = (
            self.run_sett_data_KS["d"] // self.run_sett_data_KS["d_prime"]
        )
        self.N = int(self.run_sett["global"]["N"])
        self.mode = mode
        self.u_hflr_samples, self.u_lflr_samples = self.KS_true_trajectory()

    def sample_true_trajectory(self, key):
        if self.mode == "KS":
            num = self.u_hflr_samples.shape[0]
            block_len = int(self.N + 1)
            start = jax.random.randint(key, shape=(), minval=0, maxval=num - block_len)
            y = jax.lax.dynamic_slice(
                self.u_hflr_samples, (start, 0, 0), (block_len, self.d, 1)
            )
            y_prime = jax.lax.dynamic_slice(
                self.u_lflr_samples, (start, 0, 0), (block_len, self.d, 1)
            )

            return y, y_prime
        else:
            raise ValueError(f"Invalid data mode: {self.mode}")

    def KS_true_trajectory(self):
        """Load raw KS arrays and create a downsampled HF view.

        Args:
            file_name: Path to an HDF5 file with datasets 'LFLR', 'HFHR', 't', 'x'.

        Returns:
          - u_hflr_samples: High-fidelity, high-resolution array (512*320,24,1).
          - u_lflr_samples: Low-fidelity, low-resolution array (512*320,24,1).
        """
        with h5py.File(self.run_sett_data_KS["data_file_name"], "r+") as f1:
            u_LFLR = f1["LFLR"][()]
            u_HFHR = f1["HFHR"][()]

        u_HFLR = u_HFHR[:, :, :: self.downsampling_factor]
        u_LFLR = u_LFLR[:, :, ::2]

        u_hflr_samples = u_HFLR.reshape(-1, int(self.d), 1)
        u_lflr_samples = u_LFLR.reshape(-1, int(self.d), 1)

        return u_hflr_samples, u_lflr_samples
