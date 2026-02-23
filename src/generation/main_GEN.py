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
from wandb_integration.wandb_adapter import WandbWriter
from src.generation.Statistical_Downscaling_PDE_KS import (
    KSStatisticalDownscalingPDESolver,
)
from src.generation.denoiser_utils import (
    create_denoiser_model,
    create_diffusion_scheme,
    restore_denoise_fn,
    build_model,
    build_trainer,
    run_training,
)
from src.generation.utils_metrics import (
    calculate_constraint_rmse,
    calculate_sample_variability,
    calculate_melr_pooled,
    calculate_kld_pooled,
    calculate_wass1_pooled,
)
from src.generation.data_utils import get_raw_datasets, get_ks_dataset
from src.generation.sampler_utils import (
    sample_unconditional,
    sample_pde_guided,
    sample_wan_guided,
)
from src.generation.hyperparameter_utils import hyperparameter_step

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/generation/settings_GEN.yaml")
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

root_work_dir = os.path.join(project_root, "main_GEN")
work_dir = os.path.join(root_work_dir, env_run_name)
os.makedirs(work_dir, exist_ok=True)
run_sett["global"]["work_dir"] = work_dir

writer = None
key_suffix = ""

run_sett_train_denoiser = run_sett["train_denoiser"]
run_sett_pde_solver = run_sett["pde_solver"]
run_sett_global = run_sett["global"]
run_sett_metrics = run_sett["metrics"]
run_sett_ema = run_sett["ema"]
run_sett_optimizer = run_sett["optimizer"]
run_sett_exp_tspan = run_sett["exp_tspan"]

mode = str(run_sett_global["mode"])
use_ema_eval = bool(run_sett_ema["use_ema_eval"])
use_clip_gradient = bool(run_sett_optimizer["use_clip_gradient"])
clip_gradient = float(run_sett_optimizer["clip_gradient"])
adaptive_balancing_loss = bool(run_sett_pde_solver["adaptive_balancing_loss"])
normalize_data = bool(run_sett_pde_solver["normalize_data"])
num_conditionings = int(run_sett_pde_solver["num_conditionings"])
seed = int(run_sett_global["seed"])
RNG_NAMESPACE = int(run_sett_global.get("RNG_NAMESPACE", 0))

MASTER_KEY = jax.random.PRNGKey(seed)

BASE = jax.random.fold_in(MASTER_KEY, int(RNG_NAMESPACE))
DENOISER_KEY_BASE = jax.random.fold_in(BASE, 0)
SAMPLE_KEY_BASE = jax.random.fold_in(BASE, 1)
EVAL_KEY_BASE = jax.random.fold_in(BASE, 2)
DATA_KEY_BASE = jax.random.fold_in(BASE, 3)
PDE_KEY_BASE = jax.random.fold_in(BASE, 4)

if use_wandb:
    base_writer = metric_writers.create_default_writer(work_dir, asynchronous=False)

    project = os.environ.get("WANDB_PROJECT", "generation")
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
    print(
        "[INFO] use_wandb=False -> disable ALL logging/plotting to avoid local memory pressure."
    )


def _save_samples_h5(path, samples, *, y_bar=None, run_settings=None, rng_key=None):
    """Save only the samples to an HDF5 file as dataset 'samples'."""
    arr = np.asarray(samples)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=arr)


def _load_samples_h5(path, *, as_jax=True):
    """Load samples from an HDF5 file and return a JAX array by default."""
    with h5py.File(path, "r") as f:
        samples_np = f["samples"][()]

    return jnp.asarray(samples_np) if as_jax else samples_np


def _build_C_prime(d: int, d_prime: int) -> jax.Array:
    downsampling_factor = d // d_prime

    return jnp.array(
        [
            [1 if j == downsampling_factor * i else 0 for j in range(d)]
            for i in range(d_prime)
        ]
    )


def main():
    if use_ema_eval:
        print("✓ EMA eval: ON")
    else:
        print("EMA eval: OFF")
    if use_clip_gradient:
        print(f"✓ Gradient clipping: ON with clip value {clip_gradient}")
    else:
        print("Gradient clipping: OFF")
    if adaptive_balancing_loss:
        print("✓ Adaptive balancing loss: ON")
    else:
        print("Adaptive balancing loss: OFF")
    if normalize_data:
        print("✓ Normalize data: ON")
    else:
        print("Normalize data: OFF")
    if mode == "train":
        print("✓ Running in training mode.")
    elif mode == "sample":
        print("✓ Running in sampling mode.")
    elif mode == "eval":
        print("✓ Running in evaluation mode.")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    u_HFHR, u_LFLR, u_HFLR, x, t = get_raw_datasets(
        file_name=run_sett_global["data_file_name"]
    )
    u_LFLR = u_LFLR[:, :, ::2]

    u_hfhr_samples = u_HFHR.reshape(-1, int(run_sett_global["d"]), 1)
    u_lflr_samples = u_LFLR.reshape(-1, int(run_sett_global["d_prime"]), 1)
    DATA_STD = u_hfhr_samples.std()

    if mode == "train":
        # Training should be in single precision
        jax.config.update("jax_enable_x64", False)

        denoiser_model = create_denoiser_model()
        diffusion_scheme = create_diffusion_scheme(DATA_STD)

        if run_sett_global["train_denoiser"]:
            batch_size = int(run_sett_train_denoiser["batch_size"])
            total_train_steps = int(run_sett_train_denoiser["total_train_steps"])
            metric_aggregation_steps = int(
                run_sett_train_denoiser["metric_aggregation_steps"]
            )
            eval_every_steps = int(run_sett_train_denoiser["eval_every_steps"])
            num_batches_per_eval = int(run_sett_train_denoiser["num_batches_per_eval"])
            save_interval_steps = int(run_sett_train_denoiser["save_interval_steps"])
            max_to_keep = int(run_sett_train_denoiser["max_to_keep"])

            model = build_model(denoiser_model, diffusion_scheme, DATA_STD)
            trainer = build_trainer(model)

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
                train_dataloader=get_ks_dataset(
                    u_hfhr_samples,
                    split="train[:75%]",
                    batch_size=batch_size,
                    seed=denoiser_seed_train,
                ),
                trainer=trainer,
                workdir=work_dir,
                total_train_steps=total_train_steps,
                metric_writer=writer,
                metric_aggregation_steps=metric_aggregation_steps,
                eval_dataloader=get_ks_dataset(
                    u_hfhr_samples,
                    split="train[75%:]",
                    batch_size=batch_size,
                    seed=denoiser_seed_eval,
                ),
                eval_every_steps=eval_every_steps,
                num_batches_per_eval=num_batches_per_eval,
                save_interval_steps=save_interval_steps,
                max_to_keep=max_to_keep,
            )
        if run_sett_global["train_pde"]:
            log_train_metrics_every = int(run_sett_metrics["log_train_metrics_every"])
            log_ema_metrics_every = int(run_sett_metrics["log_ema_metrics_every"])
            denoise_fn = restore_denoise_fn(
                f"{work_dir}/checkpoints_denoise_model", denoiser_model
            )
            pde_solver = KSStatisticalDownscalingPDESolver(
                samples=u_hfhr_samples,
                settings=run_sett,
                denoise_fn=denoise_fn,
                scheme=diffusion_scheme,
            )
            for it in range(pde_solver.sampling_stages):
                key_step = jax.random.fold_in(PDE_KEY_BASE, it)
                train_metrics = pde_solver.update_params(key_step)
                global_step = int(pde_solver._step)
                if use_wandb:
                    scalars = {}
                    if (global_step % log_train_metrics_every) == 0:
                        scalars.update(
                            {f"train/{k}": float(v) for k, v in train_metrics.items()}
                        )
                    if use_ema_eval and (global_step % log_ema_metrics_every) == 0:
                        ema_metrics = pde_solver.compute_ema_metrics(key_step)
                        scalars.update(
                            {f"eval/{k}": float(v) for k, v in ema_metrics.items()}
                        )
                    if scalars:
                        writer.write_scalars(step=global_step, scalars=scalars)
            pde_params_dir = os.path.join(work_dir, "checkpoints_pde_solver")
            pde_solver.save_params(pde_params_dir)
    elif mode == "sample":
        # Sampling/generation should be in double precision
        jax.config.update("jax_enable_x64", True)

        denoiser_model = create_denoiser_model()
        diffusion_scheme = create_diffusion_scheme(DATA_STD)

        generation_type = str(run_sett_global["generation_type"])
        num_gen_samples = int(run_sett_pde_solver["num_gen_samples"])
        hyperparameter_tuning = bool(run_sett_global["hyperparameter_tuning"])

        if run_sett_global["debiased_conditioning"]:
            settings_ot = os.path.join(
                project_root, "src/optimal_transport/settings_OT.yaml"
            )
            with open(settings_ot, "r") as f:
                run_sett_ot = yaml.safe_load(f)
            seed = int(run_sett_ot["global"]["seed"])
            run_name = f"run_seed{seed}"
            saved_dir = os.path.join(project_root, "main_OT", run_name)
            with h5py.File(saved_dir + "/yp_trajs.h5", "r") as f1:
                y = f1["yp_trajs"][()]
        else:
            y = u_lflr_samples[:num_conditionings]
        denoise_fn = restore_denoise_fn(
            f"{work_dir}/checkpoints_denoise_model", denoiser_model
        )
        key_uncond = jax.random.fold_in(SAMPLE_KEY_BASE, 0)
        key_wan = jax.random.fold_in(SAMPLE_KEY_BASE, 1)
        key_cond = jax.random.fold_in(SAMPLE_KEY_BASE, 2)
        sample_file = os.path.join(work_dir, f"samples_{generation_type}.h5")
        if generation_type == "unconditional":
            samples = sample_unconditional(
                diffusion_scheme,
                denoise_fn,
                key_uncond,
                num_samples=num_gen_samples,
                num_plots=num_conditionings,
                run_sett=run_sett,
            )
            print(jnp.mean(samples))
            print(samples.std())
            _save_samples_h5(sample_file, samples)
        elif generation_type == "wan_conditional":
            if hyperparameter_tuning:
                alpha_tildes = [0.125, 0.25, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
                steps_N = [32, 64, 128, 256, 512, 1024]
                melr_results = jnp.zeros((len(alpha_tildes), len(steps_N)))
                variability_results = jnp.zeros((len(alpha_tildes), len(steps_N)))
                for i, a_tilde in enumerate(alpha_tildes):
                    for j, n_steps in enumerate(steps_N):
                        run_sett["train_denoiser"]["norm_guide_strength"] = float(
                            a_tilde
                        )
                        run_sett["exp_tspan"]["num_steps"] = int(n_steps)
                        key_ij = jax.random.fold_in(jax.random.fold_in(key_wan, i), j)
                        melr_unw_val, sam_var_val = hyperparameter_step(
                            run_sett,
                            diffusion_scheme,
                            denoise_fn,
                            y,
                            key_ij,
                            u_hfhr_samples,
                        )
                        melr_results = melr_results.at[i, j].set(melr_unw_val)
                        variability_results = variability_results.at[i, j].set(
                            sam_var_val
                        )
                        print(
                            f"Alpha tilde: {a_tilde}, Steps: {n_steps}, unweighted MELR: {melr_unw_val}, Variability: {sam_var_val}"
                        )
                if use_wandb and writer is not None:
                    series_labels = [
                        f"alpha_tilde={a_tilde}" for a_tilde in alpha_tildes
                    ]
                    writer.write_line_series(
                        "hp_tuning_MELR_unweighted/lines",
                        steps_N,
                        melr_results,
                        series_labels=series_labels,
                        title="MELR (unweighted) vs Steps",
                    )
                    writer.write_line_series(
                        "hp_tuning_Sample_Variability/lines",
                        steps_N,
                        variability_results,
                        series_labels=series_labels,
                        title="Sample Variability vs Steps",
                    )
            else:
                samples = sample_wan_guided(
                    diffusion_scheme,
                    denoise_fn,
                    y_bar=y,
                    rng_key=key_wan,
                    num_samples=num_gen_samples,
                    run_sett=run_sett,
                )
                print(samples.std())
                print(samples.shape)
                _save_samples_h5(sample_file, samples)
        elif generation_type == "conditional":
            pde_solver = KSStatisticalDownscalingPDESolver(
                samples=u_hfhr_samples,
                settings=run_sett,
                denoise_fn=denoise_fn,
                scheme=diffusion_scheme,
            )
            pde_params_dir = os.path.join(work_dir, "checkpoints_pde_solver")
            pde_solver.load_params(pde_params_dir)
            samples = sample_pde_guided(
                diffusion_scheme,
                denoise_fn,
                pde_solver,
                y=y,
                rng_key=key_cond,
                samples_per_condition=num_gen_samples,
            )
            print(samples.std())
            print(samples.shape)
            _save_samples_h5(sample_file, samples)
    elif mode == "eval":
        # Evaluation/metrics in double precision
        jax.config.update("jax_enable_x64", True)
        C_prime = _build_C_prime(run_sett_global["d"], run_sett_global["d_prime"])

        sample_file = os.path.join(
            work_dir, f"samples_{run_sett_global['generation_type']}.h5"
        )
        samples = _load_samples_h5(sample_file, as_jax=True)

        if run_sett_global["debiased_conditioning"]:
            settings_ot = os.path.join(
                project_root, "src/optimal_transport/settings_OT.yaml"
            )
            with open(settings_ot, "r") as f:
                run_sett_ot = yaml.safe_load(f)
            seed = int(run_sett_ot["global"]["seed"])
            run_name = f"run_seed{seed}"
            saved_dir = os.path.join(project_root, "main_OT", run_name)
            with h5py.File(saved_dir + "/yp_trajs.h5", "r") as f1:
                y = f1["yp_trajs"][()]
        else:
            y = u_lflr_samples[:num_conditionings]

        constraint_rmse, constraint_rmse_sd = calculate_constraint_rmse(
            samples,
            y,
            C_prime,
        )
        kld, kld_sd = calculate_kld_pooled(
            samples, u_hfhr_samples, epsilon=float(run_sett_metrics["epsilon"])
        )
        sample_variability, sample_variability_sd = calculate_sample_variability(
            samples
        )
        melr_weighted, melr_weighted_sd = calculate_melr_pooled(
            samples,
            u_hfhr_samples,
            sample_shape=(run_sett_global["d"],),
            weighted=True,
            epsilon=float(run_sett_metrics["epsilon"]),
        )

        melr_unweighted, melr_unweighted_sd = calculate_melr_pooled(
            samples,
            u_hfhr_samples,
            sample_shape=(run_sett_global["d"],),
            weighted=False,
            epsilon=float(run_sett_metrics["epsilon"]),
        )

        wass1, wass1_sd = calculate_wass1_pooled(
            samples,
            u_hfhr_samples,
            num_bins=1000,
        )

        print(
            "constraint_rmse: ",
            constraint_rmse,
            "(sd:",
            constraint_rmse_sd,
            ")",
            "sample_variability: ",
            sample_variability,
            "(sd:",
            sample_variability_sd,
            ")",
            "melr_unweighted: ",
            melr_unweighted,
            "(sd:",
            melr_unweighted_sd,
            ")",
            "melr_weighted: ",
            melr_weighted,
            "(sd:",
            melr_weighted_sd,
            ")",
            "kld: ",
            kld,
            "(sd:",
            kld_sd,
            ")",
            "wass1: ",
            wass1,
            "(sd:",
            wass1_sd,
            ")",
        )
        if use_wandb:
            writer.write_scalar("metrics/constraint_rmse", float(constraint_rmse))
            writer.write_scalar("metrics/constraint_rmse_sd", float(constraint_rmse_sd))
            writer.write_scalar("metrics/kld", float(kld))
            writer.write_scalar("metrics/kld_sd", float(kld_sd))
            writer.write_scalar("metrics/sample_variability", float(sample_variability))
            writer.write_scalar(
                "metrics/sample_variability_sd", float(sample_variability_sd)
            )
            writer.write_scalar("metrics/melr_weighted", float(melr_weighted))
            writer.write_scalar("metrics/melr_weighted_sd", float(melr_weighted_sd))
            writer.write_scalar("metrics/melr_unweighted", float(melr_unweighted))
            writer.write_scalar("metrics/melr_unweighted_sd", float(melr_unweighted_sd))
            writer.write_scalar("metrics/wass1", float(wass1))
            writer.write_scalar("metrics/wass1_sd", float(wass1_sd))

    # Flush/close the writer once
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
