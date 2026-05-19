# Time Series Statistical Downscaling: [Bi-Causal Optimal Transport](https://arxiv.org/abs/2605.17271) and Diffusion Models
## Using Scalable Bi-causal Optimal Transport via KL Relaxation and Policy Gradients [Cao et al., 2026]



A JAX-based implementation of time series statistical downscaling, combining bi-causal **optimal transport** (OT) (https://arxiv.org/abs/2605.17271) and **diffusion-based generation** (GEN) into a two-stage pipeline. Applied to autoregressive (AR) processes. Supporting codebase to the paper: (Cao et al. 2026), including all other synthetic experiments. 

---

## Overview

The pipeline consists of two complementary components:

1. **Optimal Transport** (`src/optimal_transport/`) — Performs the bi-causal Optimal Transport (OT) methodology as per \cite{paper}, as an computationally efficient stochastic-optimization approach for computing bi-causal OT couplings with general, including continuous, marginals.
2. **Generation** (`src/generation/`) — Trains a VP denoiser (UNet) on high-resolution data, with optional guidance at sampling time: unconditional, constraint-aware (using OT output for debiased conditioning).

---

## Project Structure

```
statistical_downscaling/
├── src/
│   ├── generation/                       # Diffusion-based generation pipeline
│   │   ├── main_GEN.py                   # Entry point
│   │   ├── settings_GEN.yaml             # Configuration
│   │   ├── requirements_GEN.txt          # Dependencies
│   │   ├── data_utils.py                 # HDF5 data loading & tf.data pipelines
│   │   ├── denoiser_utils.py             # UNet, VP diffusion scheme, training
│   │   ├── sampler_utils.py              # Unconditional or conditional sampling
│   │   ├── utils_metrics.py              # Evaluation metrics
│   │   └── swirl_dynamics_new_guidance/  # Block-averaging guidance (extends swirl-dynamics)
│   │       └── averaging_guidance.py     # InfillFromBlockAverages guidance transform
│   └── optimal_transport/                # Optimal transport pipeline
│       ├── main_OT.py                    # Entry point
│       ├── settings_OT.yaml              # Configuration
│       ├── requirements_OT.txt           # Dependencies
│       ├── alg1.py                       # Normalizing flow + policy gradient
│       ├── dgp.py                        # Synthetic & real data generators
│       ├── preprocessing.py              # Normalisation / winsorisation
│       └── utils.py                      # Plotting & diagnostics
├── main_GEN/                             # Generation run outputs
├── main_OT/                              # OT run outputs
├── scripts/
│   └── pre_commit.sh                     # Code formatting hook
├── pyproject.toml                        # Package installation
├── LICENSE                               # MIT license
└── CITATION.cff                          # Citation metadata
```

---

## Installation

### Prerequisites

- Python 3.12.8
- GPU with CUDA (recommended)

Clone the repository and install from the root:

```bash
git clone https://github.com/jesse-hoekstra/statistical_downscaling.git
cd statistical_downscaling
```

Install the swirl-dynamics dependency (used by the generation pipeline):

```bash
git clone https://github.com/google-research/swirl-dynamics.git swirl_dynamics_main
pip install -e swirl_dynamics_main
```

The two pipelines have separate dependencies. Each command automatically installs all required packages — install only what you need:

```bash
# Optimal Transport pipeline
pip install -e ".[ot]"

# Generation pipeline
pip install -e ".[gen]"

# Both
pip install -e ".[all]"
```

For exact dependency versions used in the paper, see `src/optimal_transport/requirements_OT.txt` and `src/generation/requirements_GEN.txt`.

---

## Usage

### Optimal Transport

Run the OT training and transport:

```bash
python src/optimal_transport/main_OT.py --config src/optimal_transport/settings_OT.yaml
```

This trains a `OT map` for `num_iterations` steps and saves checkpoints under `main_OT/<run_name>/`. Optionally set `transform_mode: True` in the config to load a saved OT map and obtain debiased samples *y → y′* from biased inputs.

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

Samples are written to `main_GEN/<run_name>/samples_<generation_type>_<optional_input_type>.h5`.

| `generation_type` | Description |
|---|---|
| `unconditional` | Solely dependent on the learned prior |
| `conditional` | Projecting the constraint space, correcting the unconstraint space. |

#### Evaluate

```yaml
global:
  mode: eval
  generation_type: conditional   
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
| `alg1.py` | `ConditionalSplineCouplingFlow` (RQS masked coupling), `RhoNet` (time/state-dependent Gaussian copula), `NormalizingFlowModel`, `PolicyGradient` trainer with EMA, control variates, and configurable β/lr schedules |
| `dgp.py` | `TrueDataModelUnimodal`, `TrueDataModelBimodal` (synthetic), `KSTrueDataModel`, `ARTrueDataModel`  |
| `preprocessing.py` | `DataNormalizer`: stream-estimated statistics, winsor-clipped normalisation, log-det tracking for change-of-variables |
| `utils.py` | Adjacent correlation plots, trajectory comparisons, KS evaluator, metrics CSV I/O |

### Generation

| File | Description |
|---|---|
| `denoiser_utils.py` | UNet construction, VP diffusion scheme, Orbax-based checkpointing, EMA restore |
| `sampler_utils.py` | `sample_unconditional`, `sample_conditional` |
| `metrics_utils.py` | Constraint RMSE, KLD, 1-Wasserstein, sample variability, ... |
| `data_utils.py` | HDF5 KS dataset loader, deterministic `tf.data` training/eval pipelines |
| `swirl_dynamics_new_guidance/averaging_guidance.py` | `InfillFromBlockAverages`: guidance transform for block-averaging constraints; a gradient step corrects the unconstrained space by penalising deviation from observed block means, which are then exactly enforced in the constraint space by uniformly adjusting each block |

---

## Development

### Code formatting

A pre-commit hook is provided for automated code formatting:

```bash
# Install the hook
echo '#!/bin/sh\nsource scripts/pre_commit.sh' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

---

## Citation

If you use this code in your research, please cite our work. Click the **Cite this repository** button on the GitHub sidebar, or use the metadata in [CITATION.cff](CITATION.cff).


