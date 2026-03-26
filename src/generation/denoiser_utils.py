import jax.numpy as jnp
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
import optax
import jax
from swirl_dynamics import templates
import orbax.checkpoint as ocp
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/generation/settings_GEN.yaml")
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)

data_model = str(run_sett["global"]["data_model"]).strip().lower()
data_sett = run_sett["data_KS" if data_model == "ks" else "data_AR"]


def _as_tuple(value):
    """Accept tuple/list or comma-separated string and return a tuple of ints."""
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    # fallback for single scalar
    return (int(value),)


def vp_linear_beta_schedule(beta_min: float = None, beta_max: float = None):
    """Construct an invertible variance-preserving linear-β noise schedule.

    The schedule follows a linear β(t) in t and defines a mapping between
    time t ∈ [0, 1] and noise scale σ via
    forward(t) = sqrt(expm1(0.5 * (β_max-β_min) * t^2 + β_min * t)).

    If `beta_min`/`beta_max` are not provided, values are read from
    `run_sett['train_denoiser']['beta_min']` and `run_sett['train_denoiser']['beta_max']`.

    Args:
        beta_min: Optional lower bound for β. Defaults to config.
        beta_max: Optional upper bound for β. Defaults to config.

    Returns:
        dfn_lib.InvertibleSchedule with `forward(t)` and `inverse(σ)`.
    """
    beta_min = float(
        beta_min if beta_min is not None else run_sett["train_denoiser"]["beta_min"]
    )
    beta_max = float(
        beta_max if beta_max is not None else run_sett["train_denoiser"]["beta_max"]
    )
    bdiff = beta_max - beta_min
    forward = lambda t: jnp.sqrt(jnp.expm1(0.5 * bdiff * t * t + beta_min * t))

    def inverse(sig):
        L = jnp.log1p(jnp.square(sig))
        return (-beta_min + jnp.sqrt(beta_min**2 + 2.0 * bdiff * L)) / bdiff

    return dfn_lib.InvertibleSchedule(forward, inverse)


def create_denoiser_model():
    """Instantiate the UNet denoiser from configuration.

    Hyperparameters are read from `run_sett['UNET']` and passed to
    `dfn_lib.PreconditionedDenoiserUNet`.

    Returns:
        dfn_lib.PreconditionedDenoiserUNet ready to be wrapped in a model.
    """
    return dfn_lib.PreconditionedDenoiserUNet(
        out_channels=int(run_sett["UNET"]["out_channels"]),
        num_channels=_as_tuple(run_sett["UNET"]["num_channels"]),
        downsample_ratio=_as_tuple(run_sett["UNET"]["downsample_ratio"]),
        num_blocks=int(run_sett["UNET"]["num_blocks"]),
        noise_embed_dim=int(run_sett["UNET"]["noise_embed_dim"]),
        use_attention=bool(run_sett["UNET"]["use_attention"]),
        num_heads=int(run_sett["UNET"]["num_heads"]),
        use_position_encoding=bool(run_sett["UNET"]["use_position_encoding"]),
        dropout_rate=float(run_sett["UNET"]["dropout_rate"]),
    )


def create_diffusion_scheme(data_std: float):
    """Create a variance-preserving diffusion scheme.

    Uses the variance-preserving linear-β schedule from
    `vp_linear_beta_schedule()` and the provided `data_std`.

    Args:
        data_std: Standard deviation of the training data distribution.

    Returns:
        dfn_lib.Diffusion configured for variance preservation.
    """
    return dfn_lib.Diffusion.create_variance_preserving(
        sigma=vp_linear_beta_schedule(),
        data_std=data_std,
    )


def restore_denoise_fn(checkpoint_dir: str, denoiser_model):
    """Restore an EMA denoising inference function from a checkpoint.

    Args:
        checkpoint_dir: Directory containing Orbax checkpoints.
        denoiser_model: Model architecture to pair with restored parameters.

    Returns:
        A callable `denoise_fn` suitable for inference, using EMA parameters.
    """
    trained_state = dfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
        checkpoint_dir, step=None
    )
    print(
        f"Loaded denoiser checkpoint from: {checkpoint_dir} (step {int(trained_state.step)})"
    )
    return dfn.DenoisingTrainer.inference_fn_from_state_dict(
        trained_state, use_ema=True, denoiser=denoiser_model
    )


def build_model(denoiser_model, diffusion_scheme, data_std: float):
    """Wrap the denoiser into a `DenoisingModel` with sampling/weighting.

    The input shape is `(d, 1)` where `d` is read from `run_sett['global']['d']`.
    Time sampling uses `time_uniform_sampling` over the diffusion scheme and
    noise weighting uses EDM weighting parameterized by `data_std`.

    Args:
        denoiser_model: The UNet-like denoiser module.
        diffusion_scheme: The diffusion schedule object.
        data_std: Standard deviation of the data, used for weighting.

    Returns:
        dfn.DenoisingModel ready for training.
    """
    return dfn.DenoisingModel(
        input_shape=(int(data_sett["n_x"]), int(data_sett["d"])),
        denoiser=denoiser_model,
        noise_sampling=dfn_lib.time_uniform_sampling(
            diffusion_scheme,
            clip_min=float(run_sett["train_denoiser"]["clip_min"]),
            uniform_grid=True,
        ),
        noise_weighting=dfn_lib.edm_weighting(data_std=data_std),
    )


def build_trainer(model):
    """Create a `DenoisingTrainer` with optimizer, RNG, and EMA from config.

    The optimizer applies gradient clipping and Adam with a warmup cosine decay
    schedule. RNG seed and EMA decay are sourced from `run_sett`.

    Args:
        model: The `dfn.DenoisingModel` to train.

    Returns:
        dfn.DenoisingTrainer configured for training.
    """
    return dfn.DenoisingTrainer(
        model=model,
        rng=jax.random.PRNGKey(int(run_sett["global"]["seed"])),
        optimizer=optax.chain(
            optax.clip_by_global_norm(float(run_sett["optimizer"]["clip_norm"])),
            optax.adam(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=float(run_sett["optimizer"]["init_value"]),
                    peak_value=float(run_sett["optimizer"]["peak_value"]),
                    warmup_steps=int(run_sett["optimizer"]["warmup_steps"]),
                    decay_steps=int(run_sett["optimizer"]["decay_steps"]),
                    end_value=float(run_sett["optimizer"]["end_value"]),
                ),
            ),
        ),
        ema_decay=float(run_sett["train_denoiser"]["ema_decay"]),
    )


def run_training(
    *,
    train_dataloader,
    trainer,
    workdir: str,
    total_train_steps: int,
    metric_writer,
    metric_aggregation_steps: int,
    eval_dataloader,
    eval_every_steps: int,
    num_batches_per_eval: int,
    save_interval_steps: int,
    max_to_keep: int,
):
    """Train the model with standard progress and checkpoint callbacks.

    Thin wrapper around `templates.run_train` that runs training in batches of
    `metric_aggregation_steps`, aggregates metrics within each batch, and logs
    them via `metric_writer`. If `eval_dataloader` is provided, evaluation runs
    every `eval_every_steps` steps. Note that `eval_every_steps` must be an
    integer multiple of `metric_aggregation_steps` (otherwise a `ValueError` is
    raised by the underlying template). By default, when an evaluation loader is
    given, the template performs a single sanity evaluation batch before
    training starts.

    Args:
        train_dataloader: Infinite/long iterator of training batches.
        trainer: `dfn.DenoisingTrainer` instance.
        workdir: Directory for logs and checkpoints.
        total_train_steps: Total number of optimization steps.
        metric_writer: Writer for scalar metrics.
        metric_aggregation_steps: Number of steps over which to aggregate metrics.
        eval_dataloader: Iterator of evaluation batches (optional).
        eval_every_steps: Evaluation frequency in steps; must divide by
            `metric_aggregation_steps`.
        num_batches_per_eval: Number of batches per evaluation run.
        save_interval_steps: Checkpoint save frequency.
        max_to_keep: Maximum number of checkpoints to retain.

    Returns:
        None.

    Notes:
        Adds `TqdmProgressBar` and `TrainStateCheckpoint` callbacks.
    """
    return templates.run_train(
        train_dataloader=train_dataloader,
        trainer=trainer,
        workdir=workdir,
        total_train_steps=total_train_steps,
        metric_writer=metric_writer,
        metric_aggregation_steps=metric_aggregation_steps,
        eval_dataloader=eval_dataloader,
        eval_every_steps=eval_every_steps,
        num_batches_per_eval=num_batches_per_eval,
        callbacks=(
            templates.TqdmProgressBar(
                total_train_steps=total_train_steps, train_monitors=("train_loss",)
            ),
            templates.TrainStateCheckpoint(
                base_dir=workdir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=save_interval_steps, max_to_keep=max_to_keep
                ),
            ),
        ),
    )
