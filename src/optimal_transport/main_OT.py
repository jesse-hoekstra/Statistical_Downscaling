import sys
import os
import argparse
import yaml

import jax
import numpy as np
from clu import metric_writers
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from wandb_integration.wandb_adapter import WandbWriter
except ImportError:
    print("WandbWriter not found, using dummy WandbWriter.")

    class WandbWriter:
        def __init__(self, base_writer, **kwargs):
            self.base_writer = base_writer
            self.active = kwargs.get("active", True)

        def write_scalars(self, step, scalars):
            if self.active and hasattr(self.base_writer, "write_scalars") and scalars:
                self.base_writer.write_scalars(step, scalars)

        def write_images(self, images, step=None):
            if self.active and hasattr(self.base_writer, "write_images"):
                if step is not None:
                    self.base_writer.write_images(step, images)
                else:
                    self.base_writer.write_images(images)

        def flush(self):
            if self.active and hasattr(self.base_writer, "flush"):
                self.base_writer.flush()

        def close(self):
            if self.active and hasattr(self.base_writer, "close"):
                self.base_writer.close()


from src.optimal_transport.utils_OT import evaluate_all_with_one_batch
from src.optimal_transport.alg1_OT import PolicyGradient

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="src/optimal_transport/settings_OT.yaml"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)

run_sett_global = run_sett["global"]
seed = int(run_sett_global["seed"])

USE_WANDB_DEFAULT = True
use_wandb_cfg = bool(run_sett.get("use_wandb", USE_WANDB_DEFAULT))
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
    env_run_name = f"run_seed{seed}"

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
run_sett["work_dir"] = work_dir

writer = None
key_suffix = f"_{gpu_tag_env}" if gpu_tag_env else ""

if use_wandb:
    base_writer = metric_writers.create_default_writer(work_dir, asynchronous=False)
    project = os.environ.get("WANDB_PROJECT", "optimal-transport")
    entity = os.environ.get("WANDB_ENTITY")
    run_name = os.environ.get("WANDB_NAME", env_run_name)

    alpha = float(run_sett.get("mix_pathwise_alpha", 1.0))
    use_cv = bool(run_sett.get("use_control_variates", False))
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
    print("WandB logging/plotting disabled.")


def _build_true_data_model(run_sett: dict):
    name = str(run_sett["global"]["true_data_model"]).strip().lower()
    from src.optimal_transport.dgp_OT import (
        TrueDataModelUnimodal,
        TrueDataModelBimodal,
        RobustHedgingModel,
        KSTrueDataModel,
    )

    registry = {
        "unimodal": TrueDataModelUnimodal,
        "bimodal": TrueDataModelBimodal,
        "robust_hedging": RobustHedgingModel,
        "ks": KSTrueDataModel,
    }
    if name not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown true_data_model='{name}'. Valid: {valid}")
    return registry[name](run_sett)


def main():
    true_data_model = _build_true_data_model(run_sett)
    policy_gradient = PolicyGradient(
        run_sett,
        true_data_model=true_data_model,
    )
    print(f"Using true_data_model: {run_sett['global']['true_data_model']}")

    run_sett_global = run_sett["global"]
    run_sett_logging = run_sett["logging"]
    run_sett_beta = run_sett["beta_schedule"]
    N = int(run_sett_global["N"])
    train_transform_mode = str(run_sett_global["train_tranform_mode"])
    num_iterations = int(run_sett_global["num_iterations"])
    RNG_NAMESPACE = int(run_sett_global["RNG_NAMESPACE"])
    key_master = jax.random.PRNGKey(seed)
    log_train_every = int(run_sett_logging["log_train_every"])
    log_eval_every = int(run_sett_logging["log_eval_every"])
    kl_warmup = int(run_sett_beta["kl_only_warmup_steps"])

    if train_transform_mode == "train":
        for it in range(num_iterations):
            key_step = jax.random.fold_in(key_master, RNG_NAMESPACE + it)
            metrics_key = jax.random.fold_in(key_master, RNG_NAMESPACE + it + 222_222)

            train_metrics = policy_gradient.update_params(key_step)
            global_step = int(policy_gradient._step)

            if use_wandb and (global_step % log_train_every) == 0:
                scalars = {f"train/{k}": float(v) for k, v in train_metrics.items()}
                if scalars:
                    writer.write_scalars(step=global_step, scalars=scalars)
            if global_step < kl_warmup:
                continue

            if (log_eval_every > 0) and ((global_step % log_eval_every) == 0):
                try:
                    diag = policy_gradient.compute_logging_losses(metrics_key)

                    if use_wandb:
                        nll_key = (
                            "NLL_marg"
                            if ("NLL_marg" in diag)
                            else ("NLL_true" if ("NLL_true" in diag) else None)
                        )
                        scalars = {
                            "eval/J_val": float(diag.get("J_val", np.nan)),
                            "eval/J_beta": float(diag.get("J_beta", np.nan)),
                            "eval/beta": float(diag.get("beta", np.nan)),
                        }
                        if nll_key is not None:
                            scalars["eval/NLL_marg"] = float(diag[nll_key])

                        writer.write_scalars(step=global_step, scalars=scalars)

                    if str(run_sett_global["true_data_model"]).strip().lower() not in {
                        "ks"
                    }:
                        evaluate_all_with_one_batch(
                            policy_gradient=policy_gradient,
                            true_data_model=true_data_model,
                            run_sett=run_sett,
                            writer=writer,
                            step=global_step,
                            key_suffix=key_suffix,
                        )
                except Exception as e:
                    print(f"evaluate_all_with_one_batch failed: {e}")

        if use_wandb:
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
        d_prime = u_lflr_samples.shape[1]
        N_len = int(N + 1)
        num_blocks = int(num_conditionings / N_len)
        y_trajs = u_lflr_samples[:num_conditionings].reshape(
            num_blocks, N_len, d_prime, 1
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
                policy_gradient.model.transport_y_to_yp, in_axes=(0, None)
            )(y_trajs, policy_gradient.params)
            yp_trajs = yp_trajs.reshape(-1, d_prime, 1)
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
