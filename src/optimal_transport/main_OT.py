import argparse
import yaml
import sys
import os
import jax
import jax.numpy as jnp
import h5py
from clu import metric_writers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from wandb_integration.wandb_adapter import WandbWriter
from src.optimal_transport.alg1_OT import PolicyGradient, NormalizingFlowModel
from src.optimal_transport.dgp_OT import TrueDataModelUnimodal, TrueDataModel
from src.optimal_transport.utils_OT import (
    calculate_adjacent_corr,
    plot_adjacent_corrs,
    plot_comparison,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="src/optimal_transport/settings_OT.yaml"
)
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)

use_wandb_cfg = bool(run_sett["wandb"]["use_wandb"])
env_disable = os.environ.get("WANDB_DISABLED", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
use_wandb = use_wandb_cfg and (not env_disable)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

env_run_name = os.environ.get("WANDB_NAME", "").strip()
if not env_run_name:
    env_run_name = f"run_seed{run_sett['global']['seed']}"

gpu_tag_env = os.environ.get("GPU_TAG", "").strip()
if not gpu_tag_env:
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_env:
        gpu_tag_env = f"cuda{cuda_env}"
if gpu_tag_env:
    env_run_name = f"{env_run_name}_{gpu_tag_env}"

root_work_dir = os.path.join(project_root, "main_OT")
work_dir = os.path.join(root_work_dir, env_run_name)
os.makedirs(work_dir, exist_ok=True)
run_sett["global"]["work_dir"] = work_dir

writer = None
key_suffix = ""

if use_wandb:
    base_writer = metric_writers.create_default_writer(work_dir, asynchronous=False)

    project = os.environ.get("WANDB_PROJECT", "optimal-transport")
    entity = os.environ.get("WANDB_ENTITY")  # optional
    run_name = os.environ.get("WANDB_NAME", env_run_name)
    if gpu_tag_env and gpu_tag_env not in run_name:
        run_name = f"{run_name}_{gpu_tag_env}"
    key_suffix = f"_{gpu_tag_env}" if gpu_tag_env else ""
    alpha = float(run_sett.get("mix_pathwise_alpha", 1.0))
    use_cv = bool(run_sett["policy_gradient"]["use_control_variates"])
    trainer_tag = (
        "guidedOT" if (abs(alpha - 1.0) < 1e-12 and (not use_cv)) else "guidedOT_CV"
    )

    writer = WandbWriter(
        base_writer,
        project=project,
        name=f"{run_name}_{trainer_tag}",
        entity=entity,
        config={"work_dir": work_dir, **run_sett},
        active=True,
    )
else:
    print(
        "[INFO] use_wandb=False -> disable ALL logging/plotting to avoid local memory pressure."
    )


def main():
    use_synthetic_data = bool(run_sett["global"]["use_synthetic_data"])
    if use_synthetic_data:
        true_data_model = TrueDataModelUnimodal(run_sett)
        normalizing_flow_model = NormalizingFlowModel(run_sett, true_data_model)
        policy_gradient = PolicyGradient(
            run_sett,
            true_data_model=true_data_model,
            normalizing_flow_model=normalizing_flow_model,
        )
    else:
        true_data_model = TrueDataModel(run_sett, mode="KS")
        normalizing_flow_model = NormalizingFlowModel(run_sett, true_data_model)
        policy_gradient = PolicyGradient(
            run_sett,
            true_data_model=true_data_model,
            normalizing_flow_model=normalizing_flow_model,
        )

    alpha = float(run_sett["policy_gradient"]["mix_pathwise_alpha"])
    use_cv = bool(run_sett["policy_gradient"]["use_control_variates"])
    use_advstd = bool(run_sett["policy_gradient"]["use_advantage_standardization"])
    cv_split_ratio = float(run_sett["policy_gradient"]["cv_split_ratio"])

    if abs(alpha - 1.0) < 1e-12 and (not use_cv):
        print("✓ Using Auto-grad Bicausal OT (pure pathwise; no control variates).")
    else:
        if abs(alpha - 1.0) < 1e-12:
            est = "pure pathwise (auto-grad)"
        elif abs(alpha - 0.0) < 1e-12:
            est = "pure score-function (REINFORCE)"
        else:
            est = f"unbiased mixture (alpha={alpha:g})"
        print(f"✓ Using trainer: {est}")
        print(
            f"  Control variates: {'ON' if use_cv else 'OFF'} | Adv std: {'ON' if use_advstd else 'OFF'} | cv_split_ratio={cv_split_ratio:g}"
        )

    if int(run_sett["beta_schedule"]["kl_only_warmup_steps"]) > 0:
        print(
            f"  KL weight warmup: beta=0 for first {int(run_sett['beta_schedule']['kl_only_warmup_steps'])} steps"
        )
    if bool(run_sett["ema"]["use_ema_eval"]):
        print("✓ EMA eval: ON")
    else:
        print("EMA eval: OFF")
    if bool(run_sett["preprocessing"]["use_data_normalization"]):
        print(f"✓ Data normalization: ON (mode={run_sett['preprocessing']['mode']})")
    else:
        print("Data normalization: OFF")
    if use_synthetic_data:
        print("✓ Using synthetic data.")
    else:
        print(f"✓ Using real data (mode={true_data_model.mode}).")

    N = int(run_sett["global"]["N"])
    d = int(run_sett["global"]["d"])
    RNG_NAMESPACE = int(run_sett["global"]["RNG_NAMESPACE"])
    key_master = jax.random.PRNGKey(int(run_sett["global"]["seed"]))
    num_iterations = int(run_sett["global"]["num_iterations"])
    kl_warmup = int(run_sett["beta_schedule"]["kl_only_warmup_steps"])
    log_every = int(run_sett["metrics"]["log_scalar_every"])
    log_corr_every = int(run_sett["metrics"]["log_adjcorr_every"])
    first_k = int(run_sett["metrics"]["adjcorr_first_k"])
    log_hist_every = int(run_sett["metrics"]["log_hist_every"])
    log_train_every = int(run_sett["metrics"]["log_train_every"])
    log_hist_n = int(run_sett["metrics"]["log_hist_n"])
    train_transform_mode = str(run_sett["global"]["train_tranform_mode"])

    last_metrics_key = None

    if train_transform_mode == "train":
        for it in range(num_iterations):
            key_step = jax.random.fold_in(key_master, RNG_NAMESPACE + it)
            metrics_key = jax.random.fold_in(key_master, RNG_NAMESPACE + it + 222_222)
            true_metrics_key, flow_metrics_key = jax.random.split(metrics_key)

            train_metrics = policy_gradient.update_params(key_step)
            global_step = int(policy_gradient._step)

            if use_wandb:
                scalars = {
                    "metrics/beta_value": float(policy_gradient._last_beta_value)
                }
                if (global_step % log_train_every) == 0:
                    scalars.update(
                        {f"train/{k}": float(v) for k, v in train_metrics.items()}
                    )
                writer.write_scalars(step=global_step, scalars=scalars)

            if global_step < kl_warmup:
                continue  # skip heavy logging during beta=0 warmup

            if use_wandb and (global_step % log_every == 0):
                diag = policy_gradient.compute_logging_losses(metrics_key)
                writer.write_scalars(
                    step=global_step,
                    scalars={
                        "metrics/J_val_log": float(diag["J_val"]),
                        "metrics/J_KL_proxy": float(diag["NLL_true"]),
                        "metrics/J_beta": float(diag["J_beta"]),
                        "metrics/beta_now": float(diag["beta_now"]),
                    },
                )

            if use_wandb and (global_step % log_corr_every == 0) and use_synthetic_data:
                try:
                    corr_flow, corr_flow_prime = calculate_adjacent_corr(
                        policy_gradient, "flow", true_metrics_key
                    )
                    corr_true, corr_true_prime = calculate_adjacent_corr(
                        true_data_model, "true", flow_metrics_key
                    )

                    corr_error = float(
                        jnp.mean(jnp.abs(jnp.array(corr_flow) - jnp.array(corr_true)))
                    )
                    corr_error_prime = float(
                        jnp.mean(
                            jnp.abs(
                                jnp.array(corr_flow_prime) - jnp.array(corr_true_prime)
                            )
                        )
                    )

                    writer.write_scalars(
                        step=global_step,
                        scalars={
                            "metrics/adjcorr_error": corr_error,
                            "metrics/adjcorr_error_prime": corr_error_prime,
                        },
                    )

                    try:
                        plot_adjacent_corrs(
                            corr_flow,
                            corr_flow_prime,
                            corr_true,
                            corr_true_prime,
                            run_sett,
                            writer,
                            first_k=first_k,
                            step=global_step,
                            key_suffix=key_suffix,
                        )
                    except Exception as e:
                        print(f"Warning: plot_adjacent_corrs failed: {e}")

                except Exception as e:
                    print(f"Warning: adjacent corr logging failed: {e}")

            if (
                use_wandb
                and log_hist_every > 0
                and (global_step % log_hist_every == 0)
                and use_synthetic_data
            ):
                try:
                    max_n = min(N + 1, log_hist_n)
                    for n in range(max_n):
                        plot_comparison(
                            n=n,
                            dims=d,
                            policy_gradient=policy_gradient,
                            true_data_model=true_data_model,
                            run_sett=run_sett,
                            writer=writer,
                            step=global_step,
                            key_suffix=key_suffix,
                        )
                except Exception as e:
                    print(f"Warning: periodic histogram logging failed: {e}")

        if use_wandb and use_synthetic_data:
            final_step = int(policy_gradient._step)
            if last_metrics_key is None:
                last_metrics_key = jax.random.fold_in(
                    key_master, RNG_NAMESPACE + 333_333
                )
                true_last_metrics_key, flow_last_metrics_key = jax.random.split(
                    last_metrics_key
                )

            for n in range(N + 1):
                plot_comparison(
                    n=n,
                    dims=d,
                    policy_gradient=policy_gradient,
                    true_data_model=true_data_model,
                    run_sett=run_sett,
                    writer=writer,
                    step=final_step,
                    key_suffix=key_suffix,
                )

            try:
                corr_flow, corr_flow_prime = calculate_adjacent_corr(
                    policy_gradient, "flow", flow_last_metrics_key
                )
                corr_true, corr_true_prime = calculate_adjacent_corr(
                    true_data_model, "true", true_last_metrics_key
                )

                corr_error = float(
                    jnp.mean(jnp.abs(jnp.array(corr_flow) - jnp.array(corr_true)))
                )
                corr_error_prime = float(
                    jnp.mean(
                        jnp.abs(jnp.array(corr_flow_prime) - jnp.array(corr_true_prime))
                    )
                )

                writer.write_scalars(
                    step=final_step,
                    scalars={
                        "metrics/adjcorr_error": corr_error,
                        "metrics/adjcorr_error_prime": corr_error_prime,
                    },
                )

                plot_adjacent_corrs(
                    corr_flow,
                    corr_flow_prime,
                    corr_true,
                    corr_true_prime,
                    run_sett,
                    writer,
                    first_k=first_k,
                    step=final_step,
                    key_suffix=key_suffix,
                )
            except Exception as e:
                print(f"Warning: final adjacent corr logging failed: {e}")

            try:
                writer.flush()
            except Exception:
                pass
            try:
                writer.close()
            except Exception:
                pass

        try:
            ckpt_dir = os.path.join(work_dir, "checkpoints_policy_gradient")
            policy_gradient.save_params(ckpt_dir)
        except Exception as e:
            print(f"Warning: saving PolicyGradient state failed: {e}")

    if train_transform_mode == "transform":
        _, u_lflr_samples = true_data_model.KS_true_trajectory()
        settings_gen = os.path.join(project_root, "src/generation/settings_GEN.yaml")
        with open(settings_gen, "r") as f:
            run_sett_gen = yaml.safe_load(f)
        num_conditionings = int(run_sett_gen["pde_solver"]["num_conditionings"])
        d = u_lflr_samples.shape[1]
        block_len = int(N + 1)
        num_blocks = int(num_conditionings / block_len)
        y_trajs = u_lflr_samples[:num_conditionings].reshape(
            num_blocks, block_len, d, 1
        )
        try:
            ckpt_dir = os.path.join(work_dir, "checkpoints_policy_gradient")
            ok = policy_gradient.load_params(ckpt_dir)
            if not ok:
                print(
                    "Warning: No PolicyGradient checkpoint found to load for transform mode."
                )
        except Exception as e:
            print(f"Warning: loading PolicyGradient state failed: {e}")
        try:
            yp_trajs = jax.vmap(
                normalizing_flow_model.transport_y_to_yp, in_axes=(0, None)
            )(y_trajs, policy_gradient.params)
            yp_trajs = yp_trajs.reshape(-1, d, 1)
            out_path = os.path.join(work_dir, "yp_trajs.h5")
            with h5py.File(out_path, "w") as f:
                f.create_dataset(
                    "yp_trajs", data=jax.device_get(yp_trajs), compression="gzip"
                )
            print(f"Saved yp_trajs to {out_path}")
        except Exception as e:
            print(f"Warning: final transport failed: {e}")


if __name__ == "__main__":
    main()
