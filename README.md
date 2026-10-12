# AI Photo Enhancer

A reinforcement learning agent that learns to automatically enhance photographs by predicting photo-editing parameters. Trained on the [MIT Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/) using **Soft Actor-Critic (SAC)**, the agent maps an input image to a set of slider values (exposure, temperature, saturation, etc.) that transform it to match a professionally retouched target.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Architecture](#architecture)
3. [Photo Editing Pipeline](#photo-editing-pipeline)
4. [Dataset Setup](#dataset-setup)
5. [Installation](#installation)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Interactive Demo](#interactive-demo)
9. [Experiment Structure](#experiment-structure)
10. [Configuration Reference](#configuration-reference)
11. [Pre-trained Models](#pre-trained-models)
12. [Backbone Options](#backbone-options)
13. [Project Structure](#project-structure)
14. [Acknowledgements](#acknowledgements)

---

## How It Works

The problem is framed as a **single-step Markov Decision Process**:

| Component   | Description                                                                                                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **State**   | 512-dim feature vector extracted from the input image by a ResNet18 backbone. Optional extensions: BERT+CLIP semantic embeddings, one-hot category labels, or CIE-Lab color histograms. |
| **Action**  | A vector of N continuous values ∈ `[-1, 1]`, one per enabled editing slider. Actions are discretized to a grid of resolution `discretize_step = 0.01` by default.    |
| **Reward**  | PSNR between the edited image and the expert-retouched target: `reward = 20 * log10(1 / RMSE) - 50`. A +10 bonus is given when PSNR exceeds `threshold_psnr`.        |
| **Episode** | Each episode processes a batch of images. With `max_episode_timesteps = 1`, the agent makes a single enhancement decision per image (single-shot MDP).                |

**Training** uses off-policy SAC with automatic entropy tuning and a 40,000-transition replay buffer backed by TorchRL.

---

## Architecture

```text
Input Image (B, H, W, 3)
        │
        ▼
┌──────────────────┐
│   ResNet18       │  ImageNet pretrained, classifier removed
│   + Normalize    │
└──────────────────┘
        │  512-dim features
        │
        ├──── [Optional: concat BERT/CLIP embeddings or histograms]
        │
        ▼
┌─────────────────────┐     ┌──────────────────────┐
│    Actor Network    │     │   Critic Networks     │
│  FC(512 → 256)      │     │  FC(512+N → 256)      │
│  FC(256 → 256)      │     │  FC(256 → 256)        │
│  mean(256 → N)      │     │  FC(256 → 1)          │
│  log_std(256 → N)   │     │  (x2: qf1, qf2)       │
└─────────────────────┘     └──────────────────────┘
        │
        ▼
  Squashed Gaussian → N editing parameters ∈ [-1, 1]
        │
        ▼
┌──────────────────────┐
│    Photo Editor      │  Batched, GPU-native 10-control pipeline
└──────────────────────┘
        │
        ▼
Enhanced Image (B, H, W, 3)
```

- **N** = number of enabled sliders (default: all 10 core controls)
- Actor and critics share the same backbone instance
- All linear layers use Xavier initialization

---

## Photo Editing Pipeline

The `PhotoEditor` class (`src/envs/new_edit_photo.py`) accepts display-encoded sRGB tensors in `(B, H, W, 3)` layout and normalized actions in `[-1, 1]`. It preserves the previous 10-slider action indexing:

`contrast, exposure, temp, tint, whites, blacks, highlights, shadows, vibrance, saturation`

The image is decoded to scene-linear Rec.2020, edited, gamut-mapped, and encoded back to sRGB. Exposure is a true light gain (±5 EV); temperature/tint use correlated color temperature plus Bradford adaptation; Highlights/Shadows use one fast edge-aware log-luminance pass; Contrast/Whites/Blacks use a luminance-preserving filmic tone map; and Vibrance/Saturation operate in perceptual OkLab with skin-tone protection. The implementation stays batched and on the input tensor's device—there are no per-image NumPy/OpenCV round trips.

Clarity and Dehaze are intentionally excluded. They are useful creative effects but require additional spatial filters, while the selected ten controls cover the main global correction workflow and keep RL rollouts fast. For rendered JPEG/PNG inputs, camera exposure metadata cannot reconstruct RAW sensor data or camera profiles. EXIF orientation and embedded ICC profiles should instead be resolved once while loading; the experiment notebook demonstrates this without adding work to each environment step.

Use `notebooks/photo_pipeline_experiment.ipynb` to load a sample, edit all ten actions, inspect individual slider sweeps, and benchmark preview/training batch latency.

> Existing actor checkpoints must be retrained. Their output dimensions and action indices still load, but their learned action values target the legacy image math and are not calibrated for this pipeline.

---

## Dataset Setup

The project uses the **MIT Adobe FiveK** dataset — 5,000 RAW photos, each retouched by 5 expert photographers. This project uses Expert C's JPEG retouches as ground truth.

### Download

```bash
conda activate kimeko
python dataset/download_dataset.py
```

### Expected Folder Structure

```text
dataset/
├── FiveK/
│   ├── train/
│   │   ├── input/        # source JPEGs (~4,500 images)
│   │   └── target/       # expert-retouched JPEGs
│   └── test/
│       ├── input/        # source JPEGs (~500 images)
│       └── target/       # expert-retouched JPEGs
├── processed_categories_2.txt   # per-image metadata: Location, Time, Light, Subject
└── categories.txt
```

The `processed_categories_2.txt` file maps each image name to 4 semantic category labels used by the `SemanticBackbone` and `SemanticBackboneOC` variants.

---

## Installation

```bash
git clone <repo_url>
cd ai-photo-enhancer-1

# Create and activate the conda environment
conda create -n kimeko python=3.10
conda activate kimeko

# Install PyTorch with CUDA 12.1 (adjust for your CUDA version)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** A GPU with at least 8 GB VRAM is recommended for training. CPU training is supported but slow.

---

## Training

```bash
conda activate kimeko
cd /path/to/ai-photo-enhancer-1

python src/train.py \
    <experiment_tag> \
    src/configs/hyperparameters.yaml \
    src/configs/config.yaml
```

**Arguments:**

| Argument            | Description                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| `experiment_tag`    | Short label appended to the run directory name (e.g. `"4_sliders_64px"`) |
| `sac_config`        | Path to SAC hyperparameters YAML                                          |
| `env_config`        | Path to environment config YAML                                           |
| `outdir` (optional) | Output directory for experiment results. Defaults to `experiments/runs/`  |
| `--logger_level`    | Logging verbosity. Default: `INFO`                                        |
| `--save_model`      | Save model weights at the end of training (flag, default: `True`)         |

**Key settings to adjust before training:**

- `sliders_to_use` in `config.yaml`: which editing sliders the agent controls
- `imsize` in `config.yaml`: image resolution (64 = fast, 224 = better quality)
- `total_timesteps` in `hyperparameters.yaml`: number of outer training iterations
- `device` in `hyperparameters.yaml`: `"cuda"` or `"cpu"`
- `train_batch_size` in `config.yaml`: parallel environments per step

**Monitor training with TensorBoard:**

```bash
tensorboard --logdir experiments/runs/
```

---

## Evaluation

```bash
conda activate kimeko
cd /path/to/ai-photo-enhancer-1

python src/test.py \
    experiments/runs/<run_name> \
    [--deterministic True] \
    [--device cuda:0] \
    [--plt_samples 3]
```

**Arguments:**

| Argument            | Default    | Description                                                                                                                    |
| ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `experiment_path`   | —          | Path to a saved run directory (must contain `models/` and `configs/`)                                                         |
| `--deterministic`   | `True`     | If `True`, use the actor mean action. If `False`, sample `n_actions_samples` actions and select the best by critic value.     |
| `--device`          | `cuda:0`   | Inference device                                                                                                               |
| `--plt_samples`     | `3`        | Number of sample images to visualize in output plot                                                                            |

**Output:** Prints mean PSNR and SSIM on the 500-image test set, and saves `samples_plot.svg` in the run directory.

**Metrics:**

- **PSNR** (Peak Signal-to-Noise Ratio): `20 * log10(1 / RMSE)`, higher is better
- **SSIM** (Structural Similarity Index): via `torchmetrics`, range [0, 1], higher is better

---

## Interactive Demo

```bash
conda activate kimeko
streamlit run demo.py
```

Opens a browser UI with:

- **Image upload** — upload any JPEG/PNG photograph
- **Auto Enhance** — deterministic enhancement using the actor's mean action
- **Auto Random Enhance** — stochastic enhancement (samples multiple actions, picks best)
- **Manual sliders** — adjust each editing parameter interactively (−100 to +100)
- **Before/After view** — side-by-side image comparison
- **Histogram display** — per-channel RGB histogram of the enhanced result

**To use a different pre-trained model**, edit the constants at the top of `demo.py`:

```python
MODEL_PATH = "experiments/runs/<your_run_name>"
SLIDERS_ORD = ["contrast", "exposure", "temp", ...]  # canonical PhotoEditor order
```

The included checkpoints target the legacy editor math. Train a new checkpoint before judging automatic enhancement with the new pipeline; the manual sliders work immediately.

---

## Experiment Structure

Every training run creates a directory under `experiments/runs/`:

```text
experiments/runs/<exp_name>__<tag>__<YYYY-MM-DD_HH-MM-SS>/
├── configs/
│   ├── sac_config.yaml        # copy of SAC hyperparameters used
│   └── env_config.yaml        # copy of environment config used
├── models/
│   ├── backbone.pth           # ResNet18 feature extractor weights
│   ├── actor_head.pth         # Policy head weights (fc1, fc2, fc_mean, fc_logstd)
│   ├── qf1_head.pth           # Critic 1 head weights (fc1, fc2, fc3)
│   └── qf2_head.pth           # Critic 2 head weights
├── events.out.tfevents.*      # TensorBoard event file
└── samples_plot.svg           # Generated by test.py after evaluation
```

**TensorBoard scalars logged during training:**

| Scalar                               | Description                                              |
| ------------------------------------ | -------------------------------------------------------- |
| `charts/mean_episodic_return`        | Rolling average reward on training batch                 |
| `charts/test_mean_episodic_return`   | Mean PSNR on the test set (logged every 200 steps)       |
| `charts/num_env_done`                | Number of images in batch that reached `threshold_psnr`  |
| `charts/SPS`                         | Training throughput (steps per second)                   |
| `losses/qf1_loss`, `losses/qf2_loss` | Critic losses                                            |
| `losses/actor_loss`                  | Policy gradient loss                                     |
| `losses/alpha`                       | Current entropy coefficient (if `autotune = True`)       |

---

## Configuration Reference

### `src/configs/config.yaml` — Environment Config

| Key                      | Default                                   | Description                                                                                                                   |
| ------------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `imsize`                 | `64`                                      | Image resolution in pixels (square crop). Use `224` for higher-quality training.                                             |
| `train_batch_size`       | `64`                                      | Number of parallel training environments (images processed per step).                                                        |
| `test_batch_size`        | `500`                                     | Batch size for test-set evaluation.                                                                                           |
| `threshold_psnr`         | `-25`                                     | PSNR threshold (shifted scale) to mark an episode done. Actual PSNR ≈ `threshold + 50`.                                     |
| `psnr_reward`            | `True`                                    | Use PSNR as the reward signal (vs. raw RMSE).                                                                                 |
| `sliders_to_use`         | All 10 core sliders                       | List of active editing sliders. Available: `temp`, `tint`, `exposure`, `contrast`, `highlights`, `shadows`, `whites`, `blacks`, `vibrance`, `saturation`. |
| `features_size`          | `512`                                     | Backbone output dimension (ResNet18 = 512).                                                                                   |
| `discretize`             | `True`                                    | Round continuous actions to a discrete grid.                                                                                  |
| `discretize_step`        | `0.01`                                    | Grid resolution for action discretization.                                                                                    |
| `use_txt_features`       | `False`                                   | Feature augmentation mode: `False`, `"one_hot"`, `"embedded"`, or `"histogram"`.                                             |
| `augment_data`           | `False`                                   | Enable random horizontal/vertical flip augmentation.                                                                          |
| `pre_encoding_device`    | `"cuda:0"`                                | Device used to pre-load and encode the dataset into memory.                                                                   |
| `pre_load_images`        | —                                         | If `True`, loads the full dataset into GPU memory before training starts (faster iterations).                                 |
| `preprocessor_agent_path`| `null`                                    | Path to a first-stage agent for two-stage pipeline experiments. Set to `null` to disable.                                    |
| `backbone_warmup`        | `0`                                       | Number of steps before the backbone's gradients are enabled (freeze backbone initially).                                      |

### `src/configs/hyperparameters.yaml` — SAC Config

| Key                       | Default          | Description                                                                          |
| ------------------------- | ---------------- | ------------------------------------------------------------------------------------ |
| `exp_name`                | `"ResNetEncoder"`| Name prefix for the run directory.                                                   |
| `seed`                    | `1`              | Global random seed for reproducibility.                                              |
| `device`                  | `"cuda"`         | Training device (`"cuda"` or `"cpu"`).                                               |
| `total_timesteps`         | `20`             | Number of outer training iterations (episodes over the full batch).                  |
| `buffer_size`             | `40000`          | Replay buffer capacity (transitions).                                                |
| `gamma`                   | `0`              | Discount factor. `0` = no discounting, appropriate for single-step episodes.         |
| `tau`                     | `0.005`          | Polyak averaging coefficient for target network updates.                             |
| `batch_size`              | `64`             | Minibatch size for gradient updates.                                                 |
| `learning_starts`         | `10`             | Steps of random exploration before learning begins.                                  |
| `policy_lr`               | `0.0003`         | Actor learning rate.                                                                 |
| `q_lr`                    | `0.0003`         | Critic learning rate.                                                                |
| `policy_frequency`        | `2`              | Actor update delay (TD3-style delayed policy updates).                               |
| `alpha`                   | `0.2`            | Entropy coefficient (ignored when `autotune = True`).                                |
| `autotune`                | `True`           | Automatically tune the entropy coefficient via Lagrange multiplier.                  |
| `max_episode_timesteps`   | `1`              | Steps per episode. `1` = single-shot enhancement (one action per image).             |

### `src/configs/inference_config.yaml` — Inference Config

| Key                 | Default    | Description                                                                          |
| ------------------- | ---------- | ------------------------------------------------------------------------------------ |
| `n_actions_samples` | `10`       | Number of stochastic action samples to draw in non-deterministic inference mode.     |
| `device`            | `"cuda:0"` | Inference device.                                                                    |

---

## Pre-trained Models

To use a pre-trained model:

1. Place the run directory under `experiments/runs/`
2. For **evaluation**: `python src/test.py experiments/runs/<run_name>`
3. For the **demo**: update `MODEL_PATH` at the top of `demo.py`

The pre-trained 10-slider ResNet18 model uses sliders in this action-vector order (must match training config):

```python
SLIDERS_ORD = [
    "contrast", "exposure", "temp", "tint",
    "whites", "blacks", "highlights", "shadows",
    "vibrance", "saturation",
]
```

> **Important:** `PhotoEditor` always uses the canonical order shown above, filtered to the enabled names. The order of names in `sliders_to_use` does not change action indexing. `SLIDERS_ORD` in `demo.py` must remain in that canonical order.

---

## Backbone Options

Select the backbone by setting `use_txt_features` in `config.yaml`:

| Class                | `use_txt_features` | Extra Input          | Description                                                                                                          |
| -------------------- | ------------------ | -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `ResNETBackbone`     | `False`            | —                    | ResNet18 only, 512-dim output. **Default.**                                                                          |
| `ResNETHistBackbone` | `"histogram"`      | CIE-Lab histogram    | ResNet18 + 192-dim color histogram (64 bins × 3 channels).                                                           |
| `SemanticBackboneOC` | `"one_hot"`        | One-hot labels       | ResNet18 + 16-dim one-hot encoded category labels.                                                                   |
| `SemanticBackbone`   | `"embedded"`       | BERT + CLIP features | ResNet18 + cross-modal attention over BERT (768-dim) and CLIP (512-dim) text embeddings. Downloads ~1.5 GB on first use. |

The correct backbone class is selected automatically based on the `use_txt_features` setting.

---

## Project Structure

```text
ai-photo-enhancer-1/
├── src/
│   ├── train.py                     # Training entry point
│   ├── test.py                      # Evaluation entry point
│   ├── configs/
│   │   ├── config.yaml              # Environment configuration
│   │   ├── hyperparameters.yaml     # SAC algorithm hyperparameters
│   │   └── inference_config.yaml    # Inference settings
│   ├── sac/
│   │   ├── sac_algorithm.py         # SAC training loop (collect → store → update)
│   │   ├── sac_networks.py          # Actor, Critic, and Backbone architectures
│   │   ├── sac_inference.py         # InferenceAgent wrapper for deployment
│   │   └── utils.py                 # Model save/load utilities
│   └── envs/
│       ├── photo_env.py             # Gymnasium-compatible photo enhancement environment
│       ├── new_edit_photo.py        # Fast scene-linear 10-control editing pipeline
│       ├── image_dataset.py         # FiveKDataset with optional semantic feature loading
│       ├── env_dataloader.py        # PyTorch DataLoader factory
│       └── dehaze/                  # Dark channel prior dehaze (not yet integrated)
├── dataset/
│   ├── download_dataset.py          # Dataset download script (gdown)
│   └── processed_categories_2.txt  # Image category metadata
├── demo.py                          # Streamlit interactive demo
├── experiments/
│   └── runs/                        # Saved training runs (auto-created)
├── pyproject.toml                   # Ruff linter/formatter configuration
└── requirements.txt
```

---

## Acknowledgements

- **MIT Adobe FiveK Dataset:** Vladimir Bychkovsky, Sylvain Paris, Eric Chan, Frédo Durand. *"Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs."* CVPR 2011.
- **Soft Actor-Critic:** Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine. *"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor."* ICML 2018.
- Built with [PyTorch](https://pytorch.org/), [TorchRL](https://pytorch.org/rl/), [Gymnasium](https://gymnasium.farama.org/), and [Streamlit](https://streamlit.io/).
