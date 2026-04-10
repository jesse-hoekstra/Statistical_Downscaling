"""
Entry point for the optimal-transport policy-gradient pipeline.

Supports two modes controlled by ``run_sett["global"]["train_transform_mode"]``:

- ``train``     – runs the policy-gradient training loop.
- ``transform`` – loads a checkpoint and applies the learned transport map.
"""

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


from src.optimal_transport.utils import evaluate_all_with_one_batch
from src.optimal_transport.alg1 import PolicyGradient

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="src/optimal_transport/settings_OT.yaml"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)

run_sett_global = run_sett["global"]
run_sett_logging = run_sett["logging"]
run_sett_beta = run_sett["beta_schedule"]
seed = int(run_sett_global["seed"])

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
    project = os.environ.get("WANDB_PROJECT", "OT_flow")
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
    """Instantiate the data-generating process specified by ``global.true_data_model``."""
    name = str(run_sett["global"]["true_data_model"]).strip().lower()
    from src.optimal_transport.dgp import (
        TrueDataModelUnimodal,
        TrueDataModelBimodal,
        RobustHedgingModel,
        KSTrueDataModel,
        ARTrueDataModel,
    )

    registry = {
        "unimodal": TrueDataModelUnimodal,
        "bimodal": TrueDataModelBimodal,
        "robust_hedging": RobustHedgingModel,
        "ks": KSTrueDataModel,
        "ar": ARTrueDataModel,
    }
    if name not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown true_data_model='{name}'. Valid: {valid}")
    return registry[name](run_sett)


def main():
    """Run the OT policy-gradient pipeline in the mode set by ``global.train_transform_mode``.

    Builds the true data model and dispatches to either the training loop or
    checkpoint-based transport application.
    """
    true_data_model = _build_true_data_model(run_sett)
    policy_gradient = PolicyGradient(
        run_sett,
        true_data_model=true_data_model,
    )

    data_model_name = str(run_sett_global["true_data_model"]).strip().lower()
    train_transform_mode = str(run_sett_global["train_transform_mode"])
    print(f"Using true_data_model: {data_model_name}")

    num_iterations = int(run_sett_global["num_iterations"])
    RNG_NAMESPACE = int(run_sett_global["RNG_NAMESPACE"])
    key_master = jax.random.PRNGKey(seed)
    log_train_every = int(run_sett_logging["log_train_every"])
    log_eval_every = int(run_sett_logging["log_eval_every"])
    kl_warmup = int(run_sett_beta["kl_only_warmup_steps"])
    save_every = int(run_sett_logging["save_every"])

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

                    if data_model_name not in {"ks"}:
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

            if (save_every > 0) and (global_step % save_every == 0):
                try:
                    ckpt_dir = os.path.join(work_dir, "checkpoints_policy_gradient")
                    policy_gradient.save_params(ckpt_dir)
                except Exception as e:
                    print(f"Warning: periodic save failed at step {global_step}: {e}")

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
        _, _, y_samples_test = true_data_model.true_trajectory()
        _data_sett_map = {"ks": "data_KS", "ar": "data_AR"}
        if data_model_name not in _data_sett_map:
            raise ValueError(
                f"transform mode is not supported for data model '{data_model_name}'. "
                f"Supported: {sorted(_data_sett_map.keys())}"
            )
        _data_sett_key = _data_sett_map[data_model_name]
        num_conditionings = int(run_sett[_data_sett_key]["num_conditionings"])
        y_trajs = y_samples_test[:num_conditionings]
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
            key_transform = jax.random.fold_in(key_master, RNG_NAMESPACE + 333_333)
            transport_keys = jax.random.split(key_transform, num_conditionings)
            yp_trajs = jax.vmap(
                policy_gradient.model.transport_y_to_yp, in_axes=(0, None, 0)
            )(y_trajs, policy_gradient.params, transport_keys)
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
