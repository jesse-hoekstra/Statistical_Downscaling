from __future__ import annotations

import os
import json
import csv
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def _append_row_csv(path: str, row: Dict[str, Any]) -> None:
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    exists = os.path.exists(path)

    if exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        fieldnames = header if header is not None else list(row.keys())
    else:
        fieldnames = list(row.keys())

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def _get_eval_params(policy_gradient):
    """Fetch eval params (EMA if enabled) across old/new implementations."""
    if hasattr(policy_gradient, "get_eval_params_trees"):
        return policy_gradient.get_eval_params_trees()
    if hasattr(policy_gradient, "get_eval_params"):
        return policy_gradient.get_eval_params()

    use_ema = bool(getattr(policy_gradient, "use_ema_eval", True))
    if (
        use_ema
        and hasattr(policy_gradient, "ema_params_trees")
        and (policy_gradient.ema_params_trees is not None)
    ):
        return policy_gradient.ema_params_trees

    if hasattr(policy_gradient, "params_trees") and (
        policy_gradient.params_trees is not None
    ):
        return policy_gradient.params_trees

    if hasattr(policy_gradient, "normalizing_flow_model"):
        return policy_gradient.normalizing_flow_model.params_trees

    raise AttributeError("Cannot find eval params (no compatible API found).")


def _sample_flow_trajs(policy_gradient, key, num: int, params=None):
    """
    Sample flow trajectories in ORIGINAL space.
    Return: (y, yp) with shape (num, N_len, d_prime, 1) as JAX arrays.
    """
    num = int(num)
    params = _get_eval_params(policy_gradient) if params is None else params

    if hasattr(policy_gradient, "sample_trajectories"):
        try:
            return policy_gradient.sample_trajectories(key, num=num, params=params)
        except TypeError:
            return policy_gradient.sample_trajectories(key, num=num)

    keys = jax.random.split(key, num)
    y, yp = jax.vmap(
        policy_gradient.normalizing_flow_model.sample_trajectory, in_axes=(0, None)
    )(keys, params)
    return y, yp


@dataclass
class EvalBatch:
    true_y: np.ndarray
    true_yp: np.ndarray
    flow_y: np.ndarray
    flow_yp: np.ndarray
    meta: Dict[str, Any]
    true_latents: Optional[np.ndarray] = None


def save_eval_batch_npz(path: str, batch: EvalBatch, compress: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta_json = json.dumps(batch.meta, ensure_ascii=False)
    saver = np.savez_compressed if compress else np.savez
    payload = dict(
        true_y=batch.true_y.astype(np.float32),
        true_yp=batch.true_yp.astype(np.float32),
        flow_y=batch.flow_y.astype(np.float32),
        flow_yp=batch.flow_yp.astype(np.float32),
        meta=np.array([meta_json], dtype=object),
    )
    if batch.true_latents is not None:
        payload["true_latents"] = batch.true_latents.astype(np.int8)
    saver(path, **payload)


def load_eval_batch_npz(path: str) -> EvalBatch:
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"][0]))
    lat = data["true_latents"] if ("true_latents" in data.files) else None
    if lat is not None:
        lat = lat.astype(np.int8)
    return EvalBatch(
        true_y=data["true_y"],
        true_yp=data["true_yp"],
        flow_y=data["flow_y"],
        flow_yp=data["flow_yp"],
        meta=meta,
        true_latents=lat,
    )


def build_eval_batch(
    policy_gradient,
    true_data_model,
    run_sett: Dict[str, Any],
    step: Optional[int] = None,
    B: Optional[int] = None,
    chunk_size: Optional[int] = None,
    fold_in_tag: int = 999001,
) -> EvalBatch:
    """
    Sample ONE shared batch for all plots/metrics/tests.
    Deterministic on (seed, step, fold_in_tag).
    """
    run_sett_global = run_sett["global"]
    run_sett_metrics = run_sett["metrics"]
    seed0 = int(run_sett_global["seed"])
    B = int(B if B is not None else run_sett_metrics["evaluation"]["eval_B"])
    chunk_size = int(
        chunk_size
        if chunk_size is not None
        else run_sett_metrics["evaluation"]["metrics_chunk_size"]
    )

    key = jax.random.PRNGKey(seed0)
    key = jax.random.fold_in(key, int(fold_in_tag))
    if step is not None:
        key = jax.random.fold_in(key, int(step))
    key, key_true, key_flow = jax.random.split(key, 3)

    params = _get_eval_params(policy_gradient)

    def _sample_true(key_in):
        ys, yps, lats = [], [], []
        remaining, cur_key = B, key_in
        while remaining > 0:
            cur = min(remaining, chunk_size)
            cur_key, use_key = jax.random.split(cur_key)
            keys = jax.random.split(use_key, cur)

            def _one(k):
                return true_data_model.sample_true_trajectory(k, return_latents=True)

            ty, typ, lat = jax.vmap(_one)(keys)
            ys.append(np.array(jax.device_get(ty)))
            yps.append(np.array(jax.device_get(typ)))
            lats.append(
                np.array(jax.device_get(lat)).astype(np.int8)
            )  # (cur,d_prime,2)

            remaining -= cur

        return (
            np.concatenate(ys, axis=0),
            np.concatenate(yps, axis=0),
            np.concatenate(lats, axis=0),
        )

    def _sample_flow(key_in):
        ys, yps = [], []
        remaining, cur_key = B, key_in
        while remaining > 0:
            cur = min(remaining, chunk_size)
            cur_key, use_key = jax.random.split(cur_key)
            fy, fyp = _sample_flow_trajs(policy_gradient, use_key, cur, params=params)
            ys.append(np.array(jax.device_get(fy)))
            yps.append(np.array(jax.device_get(fyp)))
            remaining -= cur
        return np.concatenate(ys, axis=0), np.concatenate(yps, axis=0)

    true_y, true_yp, true_latents = _sample_true(key_true)
    flow_y, flow_yp = _sample_flow(key_flow)

    meta = dict(
        seed=seed0,
        step=None if step is None else int(step),
        B=B,
        chunk_size=chunk_size,
        fold_in_tag=int(fold_in_tag),
        N=int(run_sett_global["N"]),
        d_prime=int(run_sett_global["d_prime"]),
    )

    return EvalBatch(
        true_y=true_y,
        true_yp=true_yp,
        flow_y=flow_y,
        flow_yp=flow_yp,
        meta=meta,
        true_latents=true_latents,
    )


def _w2_1d_sq(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    m = min(len(x), len(y))
    if m == 0:
        return float("nan")
    xs = np.sort(x[:m])
    ys = np.sort(y[:m])
    return float(np.mean((xs - ys) ** 2))


def _ks_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return float("nan")
    xs = np.sort(x)
    ys = np.sort(y)
    allv = np.sort(np.concatenate([xs, ys]))
    cdf_x = np.searchsorted(xs, allv, side="right") / m
    cdf_y = np.searchsorted(ys, allv, side="right") / n
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _stack_traj_features(batch: EvalBatch, B: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build trajectory-level feature vectors:
      x_i = concat(vec(y_i), vec(y'_i)) in R^{N_len*2*d_prime}
    Returns:
      X_true: (B, P), X_flow: (B, P)
    """
    ty = batch.true_y[:B].astype(np.float32)  # (B,N_len,d_prime,1)
    typ = batch.true_yp[:B].astype(np.float32)
    fy = batch.flow_y[:B].astype(np.float32)
    fyp = batch.flow_yp[:B].astype(np.float32)

    # flatten each trajectory
    X_true = np.concatenate(
        [ty.reshape(B, -1), typ.reshape(B, -1)], axis=1
    )  # (B, 2*N_len*d_prime)
    X_flow = np.concatenate([fy.reshape(B, -1), fyp.reshape(B, -1)], axis=1)
    return X_true, X_flow


def _median_heuristic_sigma(
    X: np.ndarray, Y: np.ndarray, seed: int = 0, max_pairs: int = 20000
) -> float:
    """
    Median heuristic for RBF bandwidth.
    We estimate median of squared distances from random pairs to avoid O(B^2).
    Returns sigma (not sigma^2).
    """
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    Z = np.concatenate([X, Y], axis=0)
    n = Z.shape[0]
    if n < 2:
        return 1.0

    m = min(int(max_pairs), n * (n - 1) // 2)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    mask = i != j
    i, j = i[mask], j[mask]
    if len(i) == 0:
        return 1.0

    diff = Z[i] - Z[j]
    sq = np.sum(diff * diff, axis=1)
    med_sq = float(np.median(sq))
    # For k(x,y)=exp(-||x-y||^2/(2 sigma^2)), common heuristic: sigma^2 = 0.5 * median(||x-y||^2)
    sigma2 = max(0.5 * med_sq, 1e-12)
    return float(np.sqrt(sigma2))


def _mmd2_rbf_rff(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float,
    num_features: int = 1024,
    seed: int = 0,
) -> float:
    """
    Approximate MMD^2 with RBF kernel using Random Fourier Features:
      MMD^2 ≈ || mean(phi(X)) - mean(phi(Y)) ||^2
    Complexity: O(B * num_features * P)
    """
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    P = X.shape[1]
    R = int(num_features)

    sigma = float(max(sigma, 1e-6))
    # w ~ N(0, 1/sigma^2 I)
    W = rng.normal(size=(P, R)).astype(np.float32) / sigma
    b = rng.uniform(0.0, 2.0 * np.pi, size=(R,)).astype(np.float32)

    # phi(x) = sqrt(2/R) * cos(xW + b)
    scale = np.sqrt(2.0 / R).astype(np.float32)
    PhiX = scale * np.cos(X @ W + b)
    PhiY = scale * np.cos(Y @ W + b)

    muX = PhiX.mean(axis=0)
    muY = PhiY.mean(axis=0)
    mmd2 = float(np.sum((muX - muY) ** 2))
    return mmd2


def _sliced_wasserstein_w2(
    X: np.ndarray,
    Y: np.ndarray,
    num_proj: int = 128,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Sliced Wasserstein (W2) between empirical distributions in R^P.
    For each random unit direction theta:
      SWD2(theta) = mean( (sort(X@theta) - sort(Y@theta))^2 )
    Aggregate over projections:
      SWD2 = mean_theta SWD2(theta)
      SWD  = sqrt(SWD2)

    Returns (SWD, SWD2).
    """
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    Bx, P = X.shape
    By, _ = Y.shape
    m = min(Bx, By)
    if m <= 1:
        return float("nan"), float("nan")

    X = X[:m]
    Y = Y[:m]

    K = int(num_proj)
    Theta = rng.normal(size=(P, K)).astype(np.float32)
    Theta /= np.linalg.norm(Theta, axis=0, keepdims=True) + 1e-12

    projX = X @ Theta  # (m, K)
    projY = Y @ Theta

    projX.sort(axis=0)
    projY.sort(axis=0)

    swd2 = float(np.mean((projX - projY) ** 2))
    swd = float(np.sqrt(max(swd2, 0.0)))
    return swd, swd2


def compute_traj_dist_metrics_from_batch(
    batch: EvalBatch,
    run_sett: Dict[str, Any],
    writer=None,
    step: Optional[int] = None,
    csv_name: str = "traj_dist_metrics.csv",
    use_subset_B: Optional[int] = None,
) -> Dict[str, float]:
    """
    Trajectory-level metrics on x_i = concat(vec(y_i), vec(y'_i)):

      - MMD^2 (RBF) via RFF approximation (fast)
      - Sliced Wasserstein distance (W2), with SWD2 also logged

    Settings (optional in yaml):
      traj_metrics_samples: int (default: eval_B)
      mmd_rff_features: int (default: 1024)
      mmd_sigma: float or null (if null, median heuristic)
      mmd_seed_offset: int (default: 12345)
      swd_num_proj: int (default: 128)
      swd_seed_offset: int (default: 23456)
    """
    run_sett_global = run_sett["global"]
    run_sett_metrics = run_sett["metrics"]
    base_dir = run_sett.get("work_dir", os.getcwd())
    csv_path = os.path.join(base_dir, str(csv_name))

    B0 = int(batch.true_y.shape[0])
    B = int(
        min(
            (
                use_subset_B
                if use_subset_B is not None
                else run_sett_metrics["eval_traj_metrics"]["traj_metrics_samples"]
            ),
            B0,
        )
    )

    X_true, X_flow = _stack_traj_features(batch, B=B)

    seed0 = int(run_sett_global["seed"])
    mmd_seed = seed0 + int(run_sett_metrics["mmd"]["mmd_seed_offset"])
    swd_seed = seed0 + int(run_sett_metrics["swd"]["swd_seed_offset"])

    sigma_cfg = run_sett_metrics["mmd"]["mmd_sigma"]
    if sigma_cfg is None:
        sigma = _median_heuristic_sigma(
            X_true,
            X_flow,
            seed=mmd_seed,
            max_pairs=int(run_sett_metrics["mmd"]["mmd_median_pairs"]),
        )
    else:
        sigma = float(sigma_cfg)

    rff_features = int(run_sett_metrics["mmd"]["mmd_rff_features"])
    mmd2 = _mmd2_rbf_rff(
        X_true, X_flow, sigma=sigma, num_features=rff_features, seed=mmd_seed
    )

    num_proj = int(run_sett_metrics["swd"]["swd_num_proj"])
    swd, swd2 = _sliced_wasserstein_w2(X_true, X_flow, num_proj=num_proj, seed=swd_seed)

    out = {
        "eval/traj/B_used": float(B),
        "eval/traj/MMD2_rbf": float(mmd2),
        "eval/traj/MMD_sigma": float(sigma),
        "eval/traj/SWD": float(swd),
        "eval/traj/SWD2": float(swd2),
        "eval/traj/SWD_num_proj": float(num_proj),
        "eval/traj/MMD_rff_features": float(rff_features),
    }

    _append_row_csv(
        csv_path,
        {
            "step": -1 if step is None else int(step),
            "B_used": int(B),
            "mmd2_rbf_rff": float(mmd2),
            "mmd_sigma": float(sigma),
            "mmd_rff_features": int(rff_features),
            "swd": float(swd),
            "swd2": float(swd2),
            "swd_num_proj": int(num_proj),
        },
    )

    if writer is not None and hasattr(writer, "write_scalars") and step is not None:
        writer.write_scalars(step=int(step), scalars=out)

    return out


def compute_dist_metrics_all_times_from_batch(
    batch: EvalBatch,
    run_sett: Dict[str, Any],
    writer=None,
    step: Optional[int] = None,
    csv_name: str = "dist_metrics.csv",
    use_subset_B: Optional[int] = None,
) -> Dict[str, float]:
    """
    For each t: stack (y_t, y'_t) and flatten, compute W2^2 and KS.
    """
    run_sett_global = run_sett["global"]
    run_sett_metrics = run_sett["metrics"]
    N = int(run_sett_global["N"])
    base_dir = run_sett.get("work_dir", os.getcwd())
    csv_path = os.path.join(base_dir, csv_name)

    B0 = int(batch.true_y.shape[0])
    B = int(
        min(
            (
                use_subset_B
                if use_subset_B is not None
                else run_sett_metrics["distance_metrics"]["dist_metrics_samples"]
            ),
            B0,
        )
    )

    ty = batch.true_y[:B, :, :, 0]
    fy = batch.flow_y[:B, :, :, 0]
    typ = batch.true_yp[:B, :, :, 0]
    fyp = batch.flow_yp[:B, :, :, 0]

    out: Dict[str, float] = {}
    w2_list, ks_list = [], []

    for t in range(N + 1):
        true_flat = np.concatenate(
            [ty[:, t].reshape(-1), typ[:, t].reshape(-1)], axis=0
        )
        flow_flat = np.concatenate(
            [fy[:, t].reshape(-1), fyp[:, t].reshape(-1)], axis=0
        )
        w2 = _w2_1d_sq(true_flat, flow_flat)
        ks = _ks_1d(true_flat, flow_flat)

        out[f"dist/t{t}/W2"] = float(w2)
        out[f"dist/t{t}/KS"] = float(ks)
        w2_list.append(w2)
        ks_list.append(ks)

        _append_row_csv(
            csv_path,
            {
                "step": -1 if step is None else int(step),
                "t": int(t),
                "B_used": int(B),
                "W2": float(w2),
                "KS": float(ks),
            },
        )

    if writer is not None and hasattr(writer, "write_scalars") and step is not None:
        writer.write_scalars(
            step=int(step),
            scalars={
                "eval/dist/B_used": float(B),
                "eval/dist/Wasserstein_2": float(np.mean(w2_list)),
                "eval/dist/Kolmogorov-Smirnov": float(np.mean(ks_list)),
            },
        )

    return out


def compute_transition_marginal_kl_from_batch(
    batch: EvalBatch,
    policy_gradient,
    true_data_model,
    run_sett: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (kl_y_t, kl_yp_t, kl_total_t), each shape (T,)
    where T=N+1 and entry t corresponds to transition into time t given previous.
    We'll plot/use t=1..N.
    """
    run_sett_metrics = run_sett["metrics"]
    mode = str(run_sett_metrics["transition_kl_metrics"]["transition_kl_mode"]).lower()
    if mode != "oracle":
        raise ValueError("You requested oracle mode. Set transition_kl_mode='oracle'.")

    params = _get_eval_params(policy_gradient)
    model = getattr(policy_gradient, "model", None) or getattr(
        policy_gradient, "normalizing_flow_model", None
    )
    if model is None:
        raise ValueError("Cannot find model in policy_gradient.")

    y = jnp.asarray(batch.true_y)
    yp = jnp.asarray(batch.true_yp)
    lat = None
    if batch.true_latents is not None:
        lat = jnp.asarray(batch.true_latents).astype(jnp.bool_)

    logT_y = true_data_model.log_prob_y_cond_steps(y, latents=lat, mode="oracle")
    logT_yp = true_data_model.log_prob_yp_cond_steps(yp, latents=lat, mode="oracle")

    if (not hasattr(model, "logprob_steps_y_batch_original")) or (
        not hasattr(model, "logprob_steps_yp_batch_original")
    ):
        raise AttributeError(
            "NormalizingFlowModel missing logprob_steps_*_batch_original. Add the methods in section B."
        )

    logQ_y = model.logprob_steps_y_batch_original(params, y)
    logQ_yp = model.logprob_steps_yp_batch_original(params, yp)

    kl_y_t = jnp.mean(logT_y - logQ_y, axis=0)  # (N_len,)
    kl_yp_t = jnp.mean(logT_yp - logQ_yp, axis=0)
    kl_tot = kl_y_t + kl_yp_t

    return (
        np.array(jax.device_get(kl_y_t), dtype=np.float64),
        np.array(jax.device_get(kl_yp_t), dtype=np.float64),
        np.array(jax.device_get(kl_tot), dtype=np.float64),
    )


def save_transition_kl_to_csv(
    kl_y_t: np.ndarray,
    kl_yp_t: np.ndarray,
    kl_tot_t: np.ndarray,
    run_sett: Dict[str, Any],
    step: Optional[int],
    csv_name: str = "transition_kl_steps.csv",
) -> str:
    base_dir = run_sett.get("work_dir", os.getcwd())
    csv_path = os.path.join(base_dir, str(csv_name))
    T = len(kl_y_t)
    for t in range(T):
        _append_row_csv(
            csv_path,
            {
                "step": -1 if step is None else int(step),
                "t": int(t),
                "kl_y": float(kl_y_t[t]),
                "kl_yp": float(kl_yp_t[t]),
                "kl_total": float(kl_tot_t[t]),
            },
        )
    return csv_path


def plot_transition_kl(
    kl_y_t: np.ndarray,
    kl_yp_t: np.ndarray,
    kl_tot_t: np.ndarray,
    run_sett: Dict[str, Any],
    writer=None,
    step: Optional[int] = None,
    key_suffix: str = "",
    out_name: str = "transition_kl_steps",
) -> str:
    run_sett_global = run_sett["global"]
    N = int(run_sett_global["N"])
    N_len = N + 1
    assert len(kl_y_t) == N_len

    xs = np.arange(1, N_len)
    y1 = kl_y_t[1:N_len]
    y2 = kl_yp_t[1:N_len]
    y3 = kl_tot_t[1:N_len]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, y1, marker="o", linestyle="-", label="KL_y(t)")
    plt.plot(xs, y2, marker="s", linestyle="--", label="KL_y'(t)")
    plt.plot(xs, y3, marker="^", linestyle="-.", label="KL_total(t)")
    plt.xlabel("n")
    plt.ylabel("Estimated reverse KL")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    base_dir = run_sett.get("work_dir", os.getcwd())
    out_dir = os.path.join(base_dir, "kl")
    os.makedirs(out_dir, exist_ok=True)
    step_suffix = f"_step{int(step)}" if step is not None else ""
    out_path = os.path.join(out_dir, f"{out_name}{key_suffix}{step_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    if writer is not None and hasattr(writer, "write_images"):
        payload = {out_name + key_suffix: out_path}
        if step is not None:
            writer.write_images(images=payload, step=int(step))
        else:
            writer.write_images(images=payload)

    return out_path


def _adjacent_corr_from_trajs_np(trajs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    x = trajs[..., 0] if trajs.ndim == 4 else trajs  # (B,N_len,D)
    _, N_len, _ = x.shape
    out = []
    for t in range(1, N_len):
        x_prev = x[:, t - 1, :]
        x_curr = x[:, t, :]
        x_prev_c = x_prev - x_prev.mean(axis=0, keepdims=True)
        x_curr_c = x_curr - x_curr.mean(axis=0, keepdims=True)
        cov = (x_prev_c * x_curr_c).mean(axis=0)
        std_prev = np.sqrt((x_prev_c**2).mean(axis=0) + eps)
        std_curr = np.sqrt((x_curr_c**2).mean(axis=0) + eps)
        corr_dim = cov / (std_prev * std_curr + eps)
        out.append(float(corr_dim.mean()))
    return np.asarray(out, dtype=np.float64)


def compute_adjacent_corr_from_batch(
    batch: EvalBatch,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _adjacent_corr_from_trajs_np(batch.flow_y),
        _adjacent_corr_from_trajs_np(batch.flow_yp),
        _adjacent_corr_from_trajs_np(batch.true_y),
        _adjacent_corr_from_trajs_np(batch.true_yp),
    )


def plot_adjacent_corrs(
    corr_flow,
    corr_flow_prime,
    corr_true,
    corr_true_prime,
    run_sett: Dict[str, Any],
    writer=None,
    step: Optional[int] = None,
    first_k: int = 10,
    key_suffix: str = "",
    out_name: str = "adjcorr_comparison",
) -> str:
    max_k = min(int(first_k), len(corr_flow))
    xs = np.arange(1, max_k + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        xs, corr_flow[:max_k], marker="o", linestyle="-", label="Flow Corr(y_t,y_{t-1})"
    )
    plt.plot(
        xs,
        corr_flow_prime[:max_k],
        marker="s",
        linestyle="--",
        label="Flow Corr(y'_t,y'_{t-1})",
    )
    plt.plot(
        xs, corr_true[:max_k], marker="o", linestyle="-", label="True Corr(y_t,y_{t-1})"
    )
    plt.plot(
        xs,
        corr_true_prime[:max_k],
        marker="s",
        linestyle="--",
        label="True Corr(y'_t,y'_{t-1})",
    )
    plt.xlabel("t (corr between t and t-1)")
    plt.ylabel("Correlation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    base_dir = run_sett.get("work_dir", os.getcwd())
    out_dir = os.path.join(base_dir, "adjcorr")
    os.makedirs(out_dir, exist_ok=True)
    step_suffix = f"_step{int(step)}" if step is not None else ""
    out_path = os.path.join(out_dir, f"{out_name}{key_suffix}{step_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    if writer is not None and hasattr(writer, "write_images"):
        payload = {out_name + key_suffix: out_path}
        if step is not None:
            writer.write_images(images=payload, step=int(step))
        else:
            writer.write_images(images=payload)

    return out_path


def plot_hist_from_batch(
    batch: EvalBatch,
    run_sett: Dict[str, Any],
    t: int,
    writer=None,
    step: Optional[int] = None,
    key_suffix: str = "",
    num_bins: int = 50,
) -> str:
    run_sett_global = run_sett["global"]
    d_prime = int(run_sett_global["d_prime"])
    ty = batch.true_y[:, t, :, 0]
    fy = batch.flow_y[:, t, :, 0]
    typ = batch.true_yp[:, t, :, 0]
    fyp = batch.flow_yp[:, t, :, 0]

    fig, axes = plt.subplots(d_prime, 2, figsize=(12, 4 * d_prime))
    if d_prime == 1:
        axes = axes.reshape(1, 2)

    def _bins(a, b):
        data = np.concatenate([a, b], axis=0)
        lo, hi = float(np.min(data)), float(np.max(data))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
            lo, hi = -1.0, 1.0
        return np.linspace(lo, hi, int(num_bins))

    for d in range(d_prime):
        ax0 = axes[d, 0]
        bins = _bins(ty[:, d], fy[:, d])
        ax0.hist(ty[:, d], bins=bins, alpha=0.5, density=True, label="True")
        ax0.hist(fy[:, d], bins=bins, alpha=0.5, density=True, label="Flow")
        ax0.set_title(f"y[{d}] at t={t}")
        ax0.legend()

        ax1 = axes[d, 1]
        bins = _bins(typ[:, d], fyp[:, d])
        ax1.hist(typ[:, d], bins=bins, alpha=0.5, density=True, label="True")
        ax1.hist(fyp[:, d], bins=bins, alpha=0.5, density=True, label="Flow")
        ax1.set_title(f"y'[{d}] at t={t}")
        ax1.legend()

    fig.tight_layout()

    base_dir = run_sett.get("work_dir", os.getcwd())
    out_dir = os.path.join(base_dir, "hist")
    os.makedirs(out_dir, exist_ok=True)
    step_suffix = f"_step{int(step)}" if step is not None else ""
    out_path = os.path.join(
        out_dir, f"comparison_hist_t{t}_d_prime{d_prime}{key_suffix}{step_suffix}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if writer is not None and hasattr(writer, "write_images"):
        name = f"comparison_hist_t{t}" + key_suffix
        payload = {name: out_path}
        if step is not None:
            writer.write_images(images=payload, step=int(step))
        else:
            writer.write_images(images=payload)

    return out_path


def _perm_test_adjcorr_trajlevel(
    trajs_A: np.ndarray,
    trajs_B: np.ndarray,
    n_perm: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Permutation test for adjacent-corr difference between A and B.
    - Pointwise statistic: |corr_A(t) - corr_B(t)|
    - Global statistics: Tmax and T2
    """
    assert (
        trajs_A.shape == trajs_B.shape
    ), "Permutation test expects same shape for A/B."
    B = int(trajs_A.shape[0])
    N_len = int(trajs_A.shape[1])
    num_lags = N_len - 1

    corr_A = _adjacent_corr_from_trajs_np(trajs_A)
    corr_B = _adjacent_corr_from_trajs_np(trajs_B)
    delta_obs = corr_A - corr_B

    T_point = np.abs(delta_obs)
    Tmax_obs = float(T_point.max())
    T2_obs = float(np.sum(T_point**2))

    rng = np.random.default_rng(seed)
    combined = np.concatenate([trajs_A, trajs_B], axis=0)  # (2B,N_len,D,1)
    idx_all = np.arange(2 * B)

    cnt_point = np.zeros(num_lags, dtype=np.int64)
    cnt_Tmax, cnt_T2 = 0, 0

    for _ in range(int(n_perm)):
        perm = rng.permutation(idx_all)
        idxA, idxB = perm[:B], perm[B:]
        cA = _adjacent_corr_from_trajs_np(combined[idxA])
        cB = _adjacent_corr_from_trajs_np(combined[idxB])
        Tb = np.abs(cA - cB)
        cnt_point += (Tb >= T_point).astype(np.int64)
        cnt_Tmax += int(float(Tb.max()) >= Tmax_obs)
        cnt_T2 += int(float(np.sum(Tb**2)) >= T2_obs)

    denom = n_perm + 1
    p_point = (cnt_point + 1) / denom
    p_Tmax = (cnt_Tmax + 1) / denom
    p_T2 = (cnt_T2 + 1) / denom

    return dict(
        Tmax=Tmax_obs,
        p_Tmax=float(p_Tmax),
        T2=T2_obs,
        p_T2=float(p_T2),
        T_point=T_point,
        p_point=p_point,
        delta_obs=delta_obs,
        n_perm=int(n_perm),
        num_lags=int(num_lags),
        B=int(B),
    )


def run_adjacent_corr_permutation_tests_from_batch(
    batch: EvalBatch,
    run_sett: Dict[str, Any],
    writer=None,
    step: Optional[int] = None,
    key_suffix: str = "",
    csv_name: str = "adjcorr_tests.csv",
) -> Dict[str, Any]:
    """
    Permutation test using the SAME eval batch (stacked y,y').
    """
    base_dir = run_sett.get("work_dir", os.getcwd())
    csv_path = os.path.join(base_dir, csv_name)

    run_sett_global = run_sett["global"]
    run_sett_metrics = run_sett["metrics"]
    seed0 = int(run_sett_global["seed"])
    n_perm = int(run_sett_metrics["perm_test_metrics"]["adjcorr_perm_B"])

    B0 = int(batch.true_y.shape[0])
    B = int(min(run_sett_metrics["perm_test_metrics"]["adjcorr_test_samples"], B0))

    true_stacked = np.concatenate(
        [batch.true_y[:B], batch.true_yp[:B]], axis=2
    )  # (B,N_len,2d_prime,1)
    flow_stacked = np.concatenate([batch.flow_y[:B], batch.flow_yp[:B]], axis=2)

    res = _perm_test_adjcorr_trajlevel(
        flow_stacked, true_stacked, n_perm=n_perm, seed=seed0
    )

    if writer is not None and hasattr(writer, "write_scalars") and step is not None:
        writer.write_scalars(
            step=int(step),
            scalars={
                "eval/tests/adjcorr/Tmax": float(res["Tmax"]),
                "eval/tests/adjcorr/p_Tmax": float(res["p_Tmax"]),
                "eval/tests/adjcorr/T2": float(res["T2"]),
                "eval/tests/adjcorr/p_T2": float(res["p_T2"]),
            },
        )

    if step is not None:
        row = {
            "Tmax": float(res["Tmax"]),
            "p_Tmax": float(res["p_Tmax"]),
            "T2": float(res["T2"]),
            "p_T2": float(res["p_T2"]),
        }
        for n in range(1, int(res["num_lags"]) + 1):
            row[f"T_n{n}"] = float(res["T_point"][n - 1])
            row[f"p_n{n}"] = float(res["p_point"][n - 1])
            row[f"delta_n{n}"] = float(res["delta_obs"][n - 1])
        _append_row_csv(csv_path, row)

    return res


def _estimate_cost_from_batch(batch: EvalBatch) -> float:
    """
    Match training objective:
      V0 = sum_{t,d} (y - y')^2  (per path)
      return mean over paths.
    Uses FLOW samples in ORIGINAL space.
    """
    diff = batch.flow_y - batch.flow_yp  # (B,N_len,d_prime,1)
    v0 = np.sum(diff * diff, axis=(1, 2, 3))  # (B,)
    return float(np.mean(v0))


def evaluate_all_with_one_batch(
    policy_gradient,
    true_data_model,
    run_sett: Dict[str, Any],
    writer=None,
    step: Optional[int] = None,
    key_suffix: str = "",
) -> EvalBatch:
    """
    Sample once, optionally save, then:
      - dist metrics over t
      - adjacent corr plots
      - hist plots for first few t
      - optional permutation test
    """
    run_sett_global = run_sett["global"]
    run_sett_metrics = run_sett["metrics"]
    base_dir = run_sett.get("work_dir", os.getcwd())
    os.makedirs(base_dir, exist_ok=True)

    batch = build_eval_batch(
        policy_gradient=policy_gradient,
        true_data_model=true_data_model,
        run_sett=run_sett,
        step=step,
        B=int(run_sett_metrics["evaluation"]["eval_B"]),
        chunk_size=int(run_sett_metrics["evaluation"]["metrics_chunk_size"]),
        fold_in_tag=int(run_sett_metrics["evaluation"]["eval_fold_in_tag"]),
    )

    if bool(run_sett_metrics["evaluation"]["save_eval_samples"]):
        out_dir = os.path.join(
            base_dir, str(run_sett_metrics["evaluation"]["eval_samples_dir"])
        )
        fname = f"eval_step{int(step) if step is not None else -1}{key_suffix}.npz"
        save_eval_batch_npz(os.path.join(out_dir, fname), batch, compress=True)

    compute_dist_metrics_all_times_from_batch(
        batch=batch,
        run_sett=run_sett,
        writer=writer,
        step=step,
        csv_name=str(run_sett_metrics["distance_metrics"]["dist_csv_name"]),
        use_subset_B=int(run_sett_metrics["distance_metrics"]["dist_metrics_samples"]),
    )

    if bool(run_sett_metrics["eval_traj_metrics"]["run_traj_dist_metrics"]):
        compute_traj_dist_metrics_from_batch(
            batch=batch,
            run_sett=run_sett,
            writer=writer,
            step=step,
            csv_name=str(run_sett_metrics["eval_traj_metrics"]["traj_dist_csv_name"]),
            use_subset_B=int(
                run_sett_metrics["eval_traj_metrics"]["traj_metrics_samples"]
            ),
        )

    corr_flow, corr_flow_p, corr_true, corr_true_p = compute_adjacent_corr_from_batch(
        batch
    )
    plot_adjacent_corrs(
        corr_flow,
        corr_flow_p,
        corr_true,
        corr_true_p,
        run_sett=run_sett,
        writer=writer,
        step=step,
        first_k=int(run_sett_metrics["adj_corr_metrics"]["adjcorr_first_k"]),
        key_suffix=key_suffix,
        out_name=str(run_sett_metrics["adj_corr_metrics"]["adjcorr_plot_name"]),
    )

    if int(run_sett_metrics["adj_corr_metrics"]["log_hist_n"]) > 0:
        max_t = min(
            int(run_sett_global["N"]) + 1,
            int(run_sett_metrics["adj_corr_metrics"]["log_hist_n"]),
        )
        for t in range(max_t):
            plot_hist_from_batch(
                batch=batch,
                run_sett=run_sett,
                t=t,
                writer=writer,
                step=step,
                key_suffix=key_suffix,
                num_bins=int(run_sett_metrics["adj_corr_metrics"]["hist_num_bins"]),
            )

    if bool(run_sett_metrics["perm_test_metrics"]["run_adjcorr_perm_test"]):
        run_adjacent_corr_permutation_tests_from_batch(
            batch=batch,
            run_sett=run_sett,
            writer=writer,
            step=step,
            key_suffix=key_suffix,
            csv_name=str(
                run_sett_metrics["perm_test_metrics"]["adjcorr_test_csv_name"]
            ),
        )

    learned_cost = _estimate_cost_from_batch(batch)

    opt_cost = None
    if hasattr(true_data_model, "optimal_cost_closed_form") and callable(
        getattr(true_data_model, "optimal_cost_closed_form")
    ):
        try:
            opt_cost = float(true_data_model.optimal_cost_closed_form())
        except Exception:
            opt_cost = None

    scalars = {
        "eval/learned_cost_eval": float(learned_cost),
    }

    if opt_cost is not None and np.isfinite(opt_cost):
        gap = float(learned_cost) - float(opt_cost)
        rel_gap = float(gap) / (abs(float(opt_cost)) + 1e-12)
        scalars.update(
            {
                "eval/opt_cost_closed_form": float(opt_cost),
                "eval/learned_minus_opt": float(gap),
                "eval/learned_minus_opt_rel": float(rel_gap),
            }
        )

    if step is not None:
        base_dir = run_sett.get("work_dir", os.getcwd())
        csv_path = os.path.join(
            base_dir, str(run_sett_metrics["opt_gap_metrics"]["opt_gap_csv_name"])
        )
        row = {"step": int(step), **{k: float(v) for k, v in scalars.items()}}
        _append_row_csv(csv_path, row)

        if writer is not None and hasattr(writer, "write_scalars"):
            writer.write_scalars(step=int(step), scalars=scalars)

    if bool(run_sett_metrics["transition_kl_metrics"]["run_transition_kl"]):
        kl_y_t, kl_yp_t, kl_tot_t = compute_transition_marginal_kl_from_batch(
            batch=batch,
            policy_gradient=policy_gradient,
            true_data_model=true_data_model,
            run_sett=run_sett,
        )

        save_transition_kl_to_csv(
            kl_y_t,
            kl_yp_t,
            kl_tot_t,
            run_sett=run_sett,
            step=step,
            csv_name=str(
                run_sett_metrics["transition_kl_metrics"]["transition_kl_csv_name"]
            ),
        )

        plot_transition_kl(
            kl_y_t,
            kl_yp_t,
            kl_tot_t,
            run_sett=run_sett,
            writer=writer,
            step=step,
            key_suffix=key_suffix,
            out_name=str(
                run_sett_metrics["transition_kl_metrics"]["transition_kl_plot_name"]
            ),
        )

    return batch
