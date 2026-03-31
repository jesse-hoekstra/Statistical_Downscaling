# Statistical Downscaling for Time-Series

A JAX-based implementation of statistical downscaling for time-series data, combining **optimal transport** (OT) and **diffusion-based generation** (GEN) into a two-stage pipeline. Applied to autoregressive (AR) processes.

---

## Overview

The pipeline consists of two complementary components:

1. **Optimal Transport** (`src/optimal_transport/`) — Learns a transport policy that maps low-resolution conditional trajectories *y* to high-resolution counterparts *y′* via normalizing flows trained with a policy gradient objective.
2. **Generation** (`src/generation/`) — Trains a VP-diffusion denoiser (UNet) on high-resolution data, with optional guidance at sampling time: unconditional, constraint-aware (using OT output for debiased conditioning).

---

## Project Structure

```
Statistical_Downscaling/
├── src/
│   ├── generation/                     # Diffusion-based generation pipeline
│   │   ├── main_GEN.py                 # Entry point
│   │   ├── settings_GEN.yaml           # Configuration
│   │   ├── requirements_GEN.txt        # Dependencies
│   │   ├── data_utils.py               # HDF5 data loading & tf.data pipelines
│   │   ├── denoiser_utils.py           # UNet, VP diffusion scheme, training
│   │   ├── sampler_utils.py            # Unconditional 
│   │   ├── utils_metrics.py            # Evaluation metrics
│   │   └── swirl_dynamics_new_guidance/
│   └── optimal_transport/              # Optimal transport pipeline
│       ├── main_OT.py                  # Entry point
│       ├── settings_OT.yaml            # Configuration
│       ├── requirements_OT.txt         # Dependencies
│       ├── alg1_OT.py                  # Normalizing flow + policy gradient
│       ├── dgp_OT.py                   # Synthetic & real data generators
│       ├── preprocessing_OT.py         # Normalisation / winsorisation
│       └── utils_OT.py                 # Plotting & diagnostics
├── main_GEN/                           # Generation run outputs
├── main_OT/                            # OT run outputs


├── aggregate_metrics.py                # Aggregate metrics across parallel runs
└── scripts/
    └── pre_commit.sh                   # Code formatting hook
```

---

## Installation

The two pipelines have separate dependencies and virtual environments.

### Prerequisites

- Python 3.12.8
- GPU with CUDA (recommended)

### Optimal Transport

```bash
cd src/optimal_transport
python -m venv .venv_OT
source .venv_OT/bin/activate          # Windows: .venv_OT\Scripts\activate
pip install -r requirements_OT.txt
```

Key dependencies: JAX 0.8.0, Flax 0.12.0, Haiku 0.0.15, Distrax 0.1.7, Optax 0.2.6, TensorFlow 2.20.0, Orbax 0.11.28, W&B 0.23.0.

### Generation

```bash
cd src/generation
python -m venv .venv_GEN
source .venv_GEN/bin/activate          # Windows: .venv_GEN\Scripts\activate
pip install -r requirements_GEN.txt
```

---

## Usage

### Optimal Transport

Run the OT training and transport:

```bash
python src/optimal_transport/main_OT.py --config src/optimal_transport/settings_OT.yaml
```

This trains a `PolicyGradient` model for `num_iterations` steps and saves checkpoints under `main_OT/<run_name>/`. Optionally set `transform_mode: True` in the config to load a saved policy and generate transported samples *y → y′*.

### Generation

All generation workflows are driven by `main_GEN.py` and controlled via `global.mode` in `settings_GEN.yaml`.

#### Train the denoiser

```yaml
# settings_GEN.yaml
global:
  mode: train
  train_denoiser: True
```

```bash
python src/generation/main_GEN.py --config src/generation/settings_GEN.yaml
```


#### Sample / generate

```yaml
global:
  mode: sample
  generation_type: conditional   # unconditional | conditional 
```

```bash
python src/generation/main_GEN.py --config src/generation/settings_GEN.yaml
```

Samples are written to `main_GEN/<run_name>/samples_<generation_type>.h5`.

| `generation_type` | Description |
|---|---|
| `unconditional` | Solely dependent on the learned prior |
| `conditional` | Projecting the constraint space, correcting the unconstraint space. |

#### Evaluate

```yaml
global:
  mode: eval
  generation_type: conditional   # must match the saved sample file
```

```bash
python src/generation/main_GEN.py --config src/generation/settings_GEN.yaml
```

---

## Configuration

### `settings_OT.yaml`

| Section | Key parameters |
|---|---|
| `global` | `seed`, `N` (sequence length), `d` (dim), `batch_size`, `num_iterations` |
| `data_KS` / `data_AR` | Dataset paths, sample counts, spatial dimensions |
| `marginal_flow` | RQS flow: `num_bins`, `num_layers`, `hidden_size` |
| `correlation_flow` | Gaussian copula, `rho_max` |
| `policy_gradient` | Pathwise / score-function mixing weights |
| `preprocessing` | Normalisation, winsor clipping threshold |
| `ema` | `ema_decay`, `use_ema_eval` |
| `logging` | `log_train_every`, `log_eval_every`, `save_every` |

### `settings_GEN.yaml`

| Section | Key parameters |
|---|---|
| `global` | `seed`, `mode`, `generation_type`, `data_model` |
| `data_KS` / `data_AR` | Conditioning sample counts, downsampling strategy |
| `train_denoiser` | `batch_size`, `total_train_steps`, `beta_min/max` |
| `UNET` | Channel sizes, number of blocks, attention heads |
| `exp_tspan` | `num_steps` (diffusion steps), `end_sigma` |
| `ema` | `ema_decay`, `use_ema_eval` |

---

## Key Components

### Optimal Transport

| File | Description |
|---|---|
| `alg1_OT.py` | `ConditionalSplineCouplingFlow` (RQS masked coupling), `RhoNet` (time/state-dependent Gaussian copula), `NormalizingFlowModel`, `PolicyGradient` trainer with EMA, control variates, and configurable β/lr schedules |
| `dgp_OT.py` | `TrueDataModelUnimodal`, `TrueDataModelBimodal` (synthetic), `KSTrueDataModel`, `ARTrueDataModel` — all produce paired trajectories *(y, y′)* |
| `preprocessing_OT.py` | `DataNormalizer`: stream-estimated statistics, winsor-clipped normalisation, log-det tracking for change-of-variables |
| `utils_OT.py` | Adjacent correlation plots, trajectory comparisons, KS evaluator, metrics CSV I/O |

### Generation

| File | Description |
|---|---|
| `denoiser_utils.py` | UNet construction, VP diffusion scheme, Orbax-based checkpointing, EMA restore |
| `sampler_utils.py` | `sample_unconditional`, `sample_conditional` |
| `utils_metrics.py` | Constraint RMSE, KLD (KDE), MELR (spectral energy mismatch), 1-Wasserstein, sample variability |
| `data_utils.py` | HDF5 KS dataset loader, deterministic `tf.data` training/eval pipelines |

---

## Development

### Code formatting

A pre-commit hook is provided for automated code formatting:

```bash
# Install the hook
echo '#!/bin/sh\nsource scripts/pre_commit.sh' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```


