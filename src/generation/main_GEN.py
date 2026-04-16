"""
Entry point for the generative model pipeline.

Supports four modes controlled by ``run_sett["global"]["mode"]``:

- ``train``   – trains the score-based denoiser.
- ``sample``  – draws samples (unconditional or conditional) using a trained checkpoint.
- ``eval``    – evaluates a single set of saved samples.
- ``eval_all``– evaluates all generation types together in double precision.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from clu import metric_writers
import yaml
import argparse

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


from src.generation.denoiser_utils import (
    create_denoiser_model,
    create_diffusion_scheme,
    restore_denoise_fn,
    build_model,
    build_trainer,
    run_training,
)
from src.generation.metrics_utils import evaluate_sample, evaluate_all_samples
from src.generation.data_utils import get_dataset
from src.generation.sampler_utils import (
    sample_unconditional,
    sample_conditional,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/generation/settings_GEN.yaml")
parser.add_argument(
    "--config-ot", type=str, default="src/optimal_transport/settings_OT.yaml"
)
parser.add_argument(
    "--run-id",
    type=str,
    default="",
    help="Suffix appended to work_dir to distinguish parallel runs.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Override the seed in the config."
)
args = parser.parse_args()

with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)

if args.seed is not None:
    run_sett["global"]["seed"] = args.seed

run_sett_ot = None
if args.config_ot:
    with open(args.config_ot, "r") as f:
        run_sett_ot = yaml.safe_load(f)

run_sett_global = run_sett["global"]
run_sett_train_denoiser = run_sett["train_denoiser"]
run_sett_metrics = run_sett["metrics"]
run_sett_ema = run_sett["ema"]
run_sett_optimizer = run_sett["optimizer"]
run_sett_exp_tspan = run_sett["exp_tspan"]
seed = run_sett["global"]["seed"]

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

root_work_dir = os.path.join(project_root, "main_GEN")
run_id_suffix = f"_{args.run_id}" if args.run_id else ""
work_dir = os.path.join(root_work_dir, f"{env_run_name}{run_id_suffix}")
os.makedirs(work_dir, exist_ok=True)
run_sett["work_dir"] = work_dir

writer = None
key_suffix = ""

mode = str(run_sett_global["mode"])
data_model = str(run_sett_global["data_model"]).strip().lower()
data_sett = run_sett["data_KS" if data_model == "ks" else "data_AR"]
num_conditionings = int(data_sett["num_conditionings"])
use_ema_eval = bool(run_sett_ema["use_ema_eval"])
use_clip_gradient = bool(run_sett_optimizer["use_clip_gradient"])
clip_gradient = float(run_sett_optimizer["clip_gradient"])
seed = int(run_sett_global["seed"])
RNG_NAMESPACE = int(run_sett_global.get("RNG_NAMESPACE", 0))

key_master = jax.random.PRNGKey(seed)
BASE = jax.random.fold_in(key_master, int(RNG_NAMESPACE))
DENOISER_KEY_BASE = jax.random.fold_in(BASE, 0)
SAMPLE_KEY_BASE = jax.random.PRNGKey(888)

if use_wandb:
    base_writer = metric_writers.create_default_writer(work_dir, asynchronous=False)
    project = os.environ.get("WANDB_PROJECT", f"GEN_{mode}")
    entity = os.environ.get("WANDB_ENTITY")
    run_name = os.environ.get("WANDB_NAME", env_run_name)
    if gpu_tag_env and gpu_tag_env not in run_name:
        run_name = f"{run_name}_{gpu_tag_env}"
    key_suffix = f"_{gpu_tag_env}" if gpu_tag_env else ""

    writer = WandbWriter(
        base_writer,
        project=project,
        name=f"{run_name}_{mode}",
        entity=entity,
        config={"work_dir": work_dir, **run_sett},
        active=True,
    )
else:
    print("use_wandb=False: logging and plotting disabled.")


def _save_samples_h5(path, samples):
    """Save samples to an HDF5 file under the dataset key ``'samples'``."""
    arr = np.asarray(samples)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=arr)


def _load_samples_h5(path, *, as_jax=True):
    """Load samples from an HDF5 file and return a JAX array by default."""
    with h5py.File(path, "r") as f:
        samples_np = f["samples"][()]

    return jnp.asarray(samples_np) if as_jax else samples_np


def _build_true_data_model(run_sett: dict):
    """Instantiate the true data model (KS or AR) specified in the global config."""
    name = str(run_sett["global"]["data_model"]).strip().lower()
    from src.optimal_transport.dgp import KSTrueDataModel, ARTrueDataModel

    registry = {"ks": KSTrueDataModel, "ar": ARTrueDataModel}
    if name not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown true_data_model='{name}'. Valid: {valid}")
    return registry[name](run_sett)


def main():
    """Run the pipeline in the mode specified by ``run_sett["global"]["mode"]``.

    Builds the true data model, then dispatches to training, sampling, or
    evaluation logic.
    """
    true_data_model = _build_true_data_model(run_sett)

    DATA_STD = true_data_model.x_train_eval.std()

    if mode == "train":
        jax.config.update("jax_enable_x64", False)

        denoiser_model = create_denoiser_model(run_sett)
        diffusion_scheme = create_diffusion_scheme(DATA_STD, run_sett)

        batch_size = int(run_sett_train_denoiser["batch_size"])
        total_train_steps = int(run_sett_train_denoiser["total_train_steps"])
        metric_aggregation_steps = int(
            run_sett_train_denoiser["metric_aggregation_steps"]
        )
        eval_every_steps = int(run_sett_train_denoiser["eval_every_steps"])
        num_batches_per_eval = int(run_sett_train_denoiser["num_batches_per_eval"])
        save_interval_steps = int(run_sett_train_denoiser["save_interval_steps"])
        max_to_keep = int(run_sett_train_denoiser["max_to_keep"])

        model = build_model(
            denoiser_model, diffusion_scheme, DATA_STD, data_sett, run_sett
        )
        trainer = build_trainer(model, run_sett)

        denoiser_key_train = jax.random.fold_in(DENOISER_KEY_BASE, 0)
        denoiser_key_eval = jax.random.fold_in(DENOISER_KEY_BASE, 1)
        denoiser_seed_train = int(
            jax.random.randint(
                denoiser_key_train,
                shape=(),
                minval=0,
                maxval=2**31 - 1,
                dtype=jnp.int32,
            )
        )
        denoiser_seed_eval = int(
            jax.random.randint(
                denoiser_key_eval,
                shape=(),
                minval=0,
                maxval=2**31 - 1,
                dtype=jnp.int32,
            )
        )

        run_training(
            train_dataloader=get_dataset(
                true_data_model.x_train_eval,
                split="train[:75%]",
                batch_size=batch_size,
                seed=denoiser_seed_train,
            ),
            trainer=trainer,
            workdir=work_dir,
            total_train_steps=total_train_steps,
            metric_writer=writer,
            metric_aggregation_steps=metric_aggregation_steps,
            eval_dataloader=get_dataset(
                true_data_model.x_train_eval,
                split="train[75%:]",
                batch_size=batch_size,
                seed=denoiser_seed_eval,
            ),
            eval_every_steps=eval_every_steps,
            num_batches_per_eval=num_batches_per_eval,
            save_interval_steps=save_interval_steps,
            max_to_keep=max_to_keep,
        )
    elif mode == "sample":
        jax.config.update("jax_enable_x64", True)

        denoiser_model = create_denoiser_model(run_sett)
        diffusion_scheme = create_diffusion_scheme(DATA_STD, run_sett)

        generation_type = str(run_sett_global["generation_type"])
        num_gen_samples = int(data_sett["num_gen_samples"])

        if run_sett_global["debiased_conditioning"]:
            seed_ot = int(run_sett_ot["global"]["seed"])
            run_name = f"run_seed{seed_ot}"
            saved_dir = os.path.join(project_root, "main_OT", run_name)
            _yp_path = saved_dir + "/yp_trajs.h5"
            with h5py.File(_yp_path, "r") as f1:
                y = f1["yp_trajs"][()]
            print(f"Loaded yp_trajs from: {_yp_path}")
        else:
            y = true_data_model.y_test[:num_conditionings]
        denoise_fn = restore_denoise_fn(f"{work_dir}/checkpoints", denoiser_model)
        key_uncond = jax.random.fold_in(SAMPLE_KEY_BASE, 0)
        key_wan = jax.random.fold_in(SAMPLE_KEY_BASE, 1)
        key_dps = jax.random.fold_in(SAMPLE_KEY_BASE, 2)
        is_conditional = generation_type != "unconditional"
        if is_conditional:
            bias_tag = (
                "debiased" if run_sett_global["debiased_conditioning"] else "biased"
            )
            sample_file = os.path.join(
                work_dir, f"samples_{generation_type}_{bias_tag}{run_id_suffix}.h5"
            )
        else:
            sample_file = os.path.join(
                work_dir, f"samples_{generation_type}{run_id_suffix}.h5"
            )
        if generation_type == "unconditional":
            samples = sample_unconditional(
                diffusion_scheme,
                denoise_fn,
                key_uncond,
                num_samples=num_gen_samples,
                num_conditions=num_conditionings,
                data_sett=data_sett,
                run_sett=run_sett,
            )
            _save_samples_h5(sample_file, samples)
        elif generation_type == "conditional":
            samples = sample_conditional(
                diffusion_scheme,
                denoise_fn,
                y_bar=y,
                rng_key=key_wan,
                num_samples=num_gen_samples,
                data_sett=data_sett,
                run_sett=run_sett,
            )
            _save_samples_h5(sample_file, samples)
    elif mode == "eval":
        jax.config.update("jax_enable_x64", True)
        generation_type = run_sett_global["generation_type"]
        is_conditional = generation_type != "unconditional"
        if is_conditional:
            bias_tag = (
                "debiased" if run_sett_global["debiased_conditioning"] else "biased"
            )
            sample_file = os.path.join(
                work_dir, f"samples_{generation_type}_{bias_tag}{run_id_suffix}.h5"
            )
        else:
            sample_file = os.path.join(
                work_dir, f"samples_{generation_type}{run_id_suffix}.h5"
            )
        samples_raw = _load_samples_h5(sample_file, as_jax=True)

        eval_work_dir = os.path.join(
            work_dir,
            f"{generation_type}_{bias_tag}" if is_conditional else generation_type,
        )
        os.makedirs(eval_work_dir, exist_ok=True)
        run_sett["work_dir"] = eval_work_dir
        evaluate_sample(
            samples_raw=samples_raw,
            true_data_model=true_data_model,
            data_sett=data_sett,
            run_sett=run_sett,
            writer=writer,
            key_suffix=key_suffix,
        )
    elif mode == "eval_all":
        jax.config.update("jax_enable_x64", True)
        evaluate_all_samples(
            work_dir=work_dir,
            true_data_model=true_data_model,
            data_sett=data_sett,
            run_sett=run_sett,
            writer=writer,
            key_suffix=key_suffix,
            run_id_suffix=run_id_suffix,
        )

    try:
        writer.flush()
    except Exception:
        pass
    try:
        writer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
