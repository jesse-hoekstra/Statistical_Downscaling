"""Lightweight data-loading utilities for the KS experiments.

This module provides:
- get_raw_datasets: reads HF/LF/HR/LR arrays from an HDF5 file and
  constructs a stride-based low-resolution view of the HF data.
- get_ks_dataset: builds a deterministic, infinite tf.data pipeline that yields
  batched samples with a reproducible circular shift augmentation.
"""

import h5py
import jax.numpy as jnp
import tensorflow as tf
from typing import Optional


def get_raw_datasets(file_name, ds_x=4):
    """Load raw KS arrays and create a downsampled HF view.

    Args:
        file_name: Path to an HDF5 file with datasets 'LFLR', 'HFHR', 't', 'x'.
        ds_x: Spatial stride used to subsample `HFHR` into `HFLR` via `::ds_x`.

    Returns:
        A tuple `(u_HFHR, u_LFLR, u_HFLR, x, t)` where:
        - u_HFHR: High-fidelity, high-resolution array.
        - u_LFLR: Low-fidelity, low-resolution array (from file).
        - u_HFLR: Subsampled view of HFHR along the spatial axis: `HFHR[:, :, ::ds_x]`.
        - x: Spatial grid.
        - t: Temporal grid.
    """
    with h5py.File(file_name, "r+") as f1:
        u_LFLR = f1["LFLR"][()]
        u_HFHR = f1["HFHR"][()]
        t = f1["t"][()]
        x = f1["x"][()]

    u_HFLR = u_HFHR[:, :, ::ds_x]
    return u_HFHR, u_LFLR, u_HFLR, x, t


def get_dataset(
    u_samples: jnp.ndarray, split: str, batch_size: int, seed: Optional[int] = None
):
    """Create a deterministic, infinite NumPy iterator of batched KS samples.

    Each element is circularly shifted by a reproducible, per-index offset to
    encourage translation robustness during training.

    Args:
        u_samples: Array of shape (N, L, 1) containing KS fields.
        split: One of
            - "train": use full set;
            - "train[:p%]": take prefix p percent;
            - "train[p%:]": take suffix from p percent onward.
        batch_size: Number of samples per batch.
        seed: Base RNG seed for stateless, per-sample circular shifts.

    Returns:
        A NumPy iterator yielding dicts with key 'x' of shape (batch_size, L, 1).
    """
    ds = tf.data.Dataset.from_tensor_slices({"x": u_samples.astype(jnp.float32)})
    total_len = len(u_samples)

    if split == "train":
        pass
    elif split.startswith("train[:"):
        frac = float(split[len("train[:") : -2]) / 100
        ds = ds.take(int(frac * total_len))
    elif split.startswith("train["):
        frac = float(split[len("train[") : -3]) / 100
        ds = ds.skip(int(frac * total_len))
    else:
        raise ValueError(f"Unsupported split string: {split}")

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)
    ds = ds.repeat()

    global_seed = tf.cast(int(seed), tf.int32)
    ds = ds.enumerate()

    def _seeded_random_roll_map_fn(index, data_dict):
        sample = data_dict["x"]
        sample_len = tf.shape(sample)[0]
        idx_mod = tf.math.floormod(index, tf.cast(total_len, tf.int64))
        idx_mod_i32 = tf.cast(idx_mod, tf.int32)
        shift = tf.random.stateless_uniform(
            shape=[],
            minval=0,
            maxval=sample_len,
            dtype=tf.int32,
            seed=tf.stack([global_seed, idx_mod_i32]),
        )
        rolled_sample = tf.roll(sample, shift=shift, axis=0)
        return {"x": rolled_sample}

    ds = ds.map(_seeded_random_roll_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()
    return ds


def get_train_test(u_HFHR, u_LFLR, split, settings):
    u_HFHR_train_eval = u_HFHR[:-split]
    u_LFLR_train_eval = u_LFLR[:-split]
    u_HFHR_test = u_HFHR[-split:]
    u_LFLR_test = u_LFLR[-split:]

    d = int(settings["d"])
    n_x = int(settings["n_x"])
    n_y = int(settings["n_y"])
    n_samples_train_eval = int(u_HFHR_train_eval.shape[0])
    n_samples_test = int(u_HFHR_test.shape[0])
    n_samples_inner_x = int(u_HFHR_train_eval.shape[2] // d)
    n_samples_inner_y = int(u_LFLR_train_eval.shape[2] // d)

    x_samples_train_eval = (
        u_HFHR_train_eval[:, :n_x, :]
        .reshape(n_samples_train_eval, n_x, n_samples_inner_x, d)
        .transpose(0, 2, 1, 3)
        .reshape(n_samples_train_eval * n_samples_inner_x, n_x, d)
    )
    x_samples_test = (
        u_HFHR_test[:, :n_x, :]
        .reshape(n_samples_test, n_x, n_samples_inner_x, d)
        .transpose(0, 2, 1, 3)
        .reshape(n_samples_test * n_samples_inner_x, n_x, d)
    )
    y_samples_train_eval = (
        u_LFLR_train_eval[:, :n_y, :]
        .reshape(n_samples_train_eval, n_y, n_samples_inner_y, d)
        .transpose(0, 2, 1, 3)
        .reshape(n_samples_train_eval * n_samples_inner_y, n_y, d)
    )
    y_samples_test = (
        u_LFLR_test[:, :n_y, :]
        .reshape(n_samples_test, n_y, n_samples_inner_y, d)
        .transpose(0, 2, 1, 3)
        .reshape(n_samples_test * n_samples_inner_y, n_y, d)
    )

    return x_samples_train_eval, x_samples_test, y_samples_train_eval, y_samples_test
