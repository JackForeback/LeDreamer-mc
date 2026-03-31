<img src="./dreamer4-fig2.png" width="500px"></img>

# LeDreamer-mc

**Dream-based reinforcement learning for Minecraft, combining Dreamer4 with LeWM (JEPA-style) world model dynamics.**

LeDreamer-mc trains agents to play Minecraft entirely from imagination. It learns a world model from [OpenAI VPT](https://github.com/openai/Video-Pre-Training) demonstration recordings, then trains a policy inside the model's "dreams" without any real environment interaction. The project provides two dynamics model variants:

- **Dreamer4** (default) — Flow-matching dynamics with shortcut consistency training, following [Hafner et al. (2025)](https://arxiv.org/abs/2509.24527v1)
- **LeWM** — A JEPA-style single-step next-embedding predictor with SIGReg regularization, inspired by [Maes et al. (2026)](https://arxiv.org/abs/2603.19312) and [Balestriero & LeCun (2025)](https://arxiv.org/abs/2511.08544)

Both variants share the same transformer backbone, action space, training infrastructure, and evaluation pipeline, allowing direct comparison of flow-matching vs. joint-embedding dynamics.

Forked from [lucidrains/dreamer4](https://github.com/lucidrains/dreamer4). Evaluation uses [openai/Video-Pre-Training](https://github.com/openai/Video-Pre-Training) as a git subtree.

**Disclaimer: This is a preliminary implementation. All criticism and suggestions for improvement are welcome and encouraged.**


## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [Phase 1: Video Tokenizer](#phase-1-video-tokenizer)
  - [Phase 2: Dynamics World Model](#phase-2-dynamics-world-model)
  - [Phase 3: Agent Training in Dreams](#phase-3-agent-training-in-dreams)
- [Evaluation](#evaluation)
- [LeWM: JEPA-Style Dynamics](#lewm-jepa-style-dynamics)
- [Architecture](#architecture)
- [Testing](#testing)
- [HPC / Slurm](#hpc--slurm)
- [Moving MNIST Quick Start](#moving-mnist-quick-start)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation

LeDreamer-mc requires **two separate Python environments** due to MineRL's dependency constraints:

### Training Environment (Python 3.10+)

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install torch torchvision einops accelerate adam-atan2-pytorch \
    x-mlps-pytorch hyper-connections vit-pytorch assoc-scan \
    discrete-continuous-embed-readout torch-einops-utils ema-pytorch \
    einx opencv-python
```

### Evaluation Environment (Python 3.9)

MineRL requires Python 3.9 and older versions of several packages. The setup script creates a separate virtualenv and patches incompatible packages automatically:

```bash
bash scripts/setup_eval_venv.sh
```

This creates `eval-env/` with MineRL, a compatible PyTorch build, and monkey-patched versions of `beartype`, `x_mlps_pytorch`, and `discrete_continuous_embed_readout` for Python 3.9 compatibility.

## Data Preparation

LeDreamer-mc trains on [OpenAI VPT](https://github.com/openai/Video-Pre-Training) contractor demonstration recordings. These consist of paired files:

- `.mp4` — 640x360 Minecraft gameplay video at 20 FPS
- `.jsonl` — One JSON object per frame containing keyboard, mouse, and camera actions

Place your `.mp4`/`.jsonl` pairs in a directory (e.g., `./download/data/`). The dataset loader (`minecraft_vpt_dataset.py`) handles all preprocessing automatically:

- Zero-pads video frames from 640x360 to 640x384 (matching the paper's preprocessing)
- Converts VPT's action format to Dreamer4's discrete action space
- Filters null actions (matching the VPT paper's preprocessing)
- Applies mu-law camera quantization (11x11 = 121 bins)
- Lazy-loads frames from MP4 on each training step (no pre-extraction required)

**Action space mapping:** VPT recordings use 20 binary keyboard/mouse buttons + 2D continuous camera. LeDreamer-mc discretizes everything into a tuple of 21 discrete action types: `(2, 2, ..., 2, 121)` — 20 binary buttons plus 121 mu-law camera bins.

## Training

All three training phases use a single entry point with `--phase 1|2|3`. Each phase saves a checkpoint that the next phase loads.

### Phase 1: Video Tokenizer

Trains a **VideoTokenizer** to compress 384x640 Minecraft frames (zero-padded from native 360x640) into compact latent representations using:

- MAE-style patch masking for regularization
- LPIPS perceptual loss for visual quality
- Axial space-time attention (separate spatial and temporal attention blocks)
- Temporal and spatial decorrelation losses

```bash
python train_dreamer4_minecraft.py --phase 1 \
    --data_dir ./download/data/ \
    --output_dir ./checkpoints \
    --num_steps 50000
```

**Output:** `checkpoints/tokenizer.pt`

### Phase 2: Dynamics World Model

Trains the **DynamicsWorldModel** to predict future latent states given past observations and actions. The default variant uses flow matching with shortcut consistency training:

1. Encodes video into latents using the frozen Phase 1 tokenizer
2. Adds noise at random signal levels (flow matching)
3. Predicts clean latents from noised versions
4. Shortcut consistency training allows larger denoising jumps at inference time

```bash
python train_dreamer4_minecraft.py --phase 2 \
    --data_dir ./download/data/ \
    --output_dir ./checkpoints \
    --tokenizer_ckpt ./checkpoints/tokenizer.pt \
    --num_steps 100000
```

**Output:** `checkpoints/dynamics.pt`

For the LeWM variant, add `--use_lewm` (see [LeWM section](#lewm-jepa-style-dynamics)).

### Phase 3: Agent Training in Dreams

Trains policy and value heads using **imagined rollouts** inside the learned world model. No real Minecraft environment is needed:

1. Generates synthetic trajectories using the frozen dynamics model
2. The agent token embedding at each step provides policy distributions and value estimates
3. Updates policy/value heads with PPO/PMPO while the world model weights remain frozen

```bash
python train_dreamer4_minecraft.py --phase 3 \
    --output_dir ./checkpoints \
    --dynamics_ckpt ./checkpoints/dynamics.pt \
    --num_steps 50000
```

**Output:** `checkpoints/dreamer4_minecraft.pt`

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tokenizer_batch_size` | 2 | Keep small — attention layers are memory-intensive |
| `--dynamics_batch_size` | 4 | Batch size for Phase 2 |
| `--dream_batch_size` | 16 | Batch size for Phase 3 imagined rollouts |
| `--tokenizer_seq_len` | 16 | Frames per clip for tokenizer training |
| `--dynamics_seq_len` | 16 | Frames per clip for dynamics training |
| `--dream_generate_timesteps` | 16 | Length of imagined trajectories |
| `--use_tensorboard` | off | Enable TensorBoard logging |
| `--log_video` | off | Log video reconstructions (Phase 1) |
| `--max_trajectories` | all | Limit number of VPT recordings (for debugging) |

## Evaluation

Evaluation runs the trained agent in MineRL's `HumanSurvival` environment and tracks Minecraft diamond tech tree progression.

**Requires the evaluation environment** (`eval-env/`):

```bash
source eval-env/bin/activate
```

### Single Agent with Rendering

```bash
python Video-Pre-Training/dreamer4_minecraft_agent.py \
    --checkpoint ./checkpoints/dreamer4_minecraft.pt
```

Rendering uses OpenCV instead of MineRL's built-in LWJGL renderer for cross-platform compatibility (including WSL2). Press `q` to quit.

### Parallel Headless Evaluation

```bash
python Video-Pre-Training/evaluate_dreamer4_minecraft.py \
    --checkpoint ./checkpoints/dreamer4_minecraft.pt \
    --n_episodes 100 \
    --n_workers 1 \
    --output_json eval_results.json
```

### Evaluation Notes

- **Default `--n_workers 1`** to prevent out-of-memory kills. Each worker spawns a Minecraft JVM (~2 GB heap) plus a copy of the model.
- The JVM heap is monkey-patched from MineRL's 4 GB default to **2 GB** per instance.
- If a Minecraft process crashes mid-episode, the environment is **recreated from scratch** (works around a MineRL bug where `self.instance` does not exist on `_SingleAgentEnv`).
- Workers are staggered by 10 seconds to avoid JVM launch resource contention.

## LeWM: JEPA-Style Dynamics

LeWM (Le World Model) provides an alternative dynamics objective inspired by Joint-Embedding Predictive Architectures. Instead of multi-step flow-matching denoising, LeWM uses **single-step next-embedding prediction** with explicit SIGReg regularization to prevent representation collapse.

| Aspect | Dreamer4 (Flow Matching) | LeWM (JEPA) |
|--------|--------------------------|-------------|
| Training signal | Denoise noisy latents | Predict next clean embedding |
| Noise injection | Yes (random signal levels) | No (clean latents only) |
| Regularization | Implicit (noise) | Explicit (SIGReg) |
| Generation speed | Multi-step denoising per frame | Single forward pass per frame |
| Shortcut training | Yes | N/A |

### Training with LeWM

Phase 1 (tokenizer) is identical. Add `--use_lewm` to Phase 2 and Phase 3:

```bash
# Phase 2: LeWM Dynamics
python train_dreamer4_minecraft.py --phase 2 --use_lewm \
    --data_dir ./download/data/ \
    --output_dir ./checkpoints \
    --tokenizer_ckpt ./checkpoints/tokenizer.pt \
    --num_steps 100000

# Phase 3: Agent in LeWM Dreams
python train_dreamer4_minecraft.py --phase 3 --use_lewm \
    --output_dir ./checkpoints \
    --dynamics_ckpt ./checkpoints/lewm_dynamics.pt \
    --num_steps 50000
```

**Checkpoints:** `lewm_dynamics.pt` (Phase 2) and `lewm_minecraft.pt` (Phase 3). Phase 3 auto-detects the model variant from the checkpoint metadata.

### Evaluating a LeWM Agent

No special flags needed — just point to the LeWM checkpoint:

```bash
python Video-Pre-Training/dreamer4_minecraft_agent.py \
    --checkpoint ./checkpoints/lewm_minecraft.pt
```

### LeWM Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_lewm_dynamics` | `False` | Enable LeWM mode |
| `lewm_loss_weight` | `1.0` | Weight for next-embedding prediction loss |
| `lewm_sigreg_loss_weight` | `0.05` | Weight for SIGReg regularization |
| `lewm_layer` | `-1` | Transformer layer for AR prediction (-1 = last) |
| `lewm_action_conditioned` | `True` | Condition AR predictor on actions |

For a detailed explanation of the LeWM implementation, see [`LEWM.md`](LEWM.md).

## Architecture

### Overview

```
Phase 1: VideoTokenizer       Encode 384x640 video to latents (MAE + LPIPS)
             |                          checkpoints/tokenizer.pt
             v
Phase 2: DynamicsWorldModel    Learn latent dynamics from VPT data + actions
             |                          checkpoints/dynamics.pt  (or lewm_dynamics.pt)
             v
Phase 3: DreamTrainer          Train policy/value heads in imagined rollouts
             |                          checkpoints/dreamer4_minecraft.pt  (or lewm_minecraft.pt)
             v
Evaluation: MineRL Agent       Play Minecraft using learned policy
```

### Core Components (`dreamer4/dreamer4.py`)

- **VideoTokenizer** — MAE encoder/decoder with `AxialSpaceTimeTransformer`. Encodes video frames to `(batch, frames, num_latent_tokens, dim_latent)`. Losses: reconstruction MSE + LPIPS perceptual + time/space decorrelation.

- **DynamicsWorldModel** — Transformer that predicts next latent states. Supports two dynamics modes: flow matching (continuous-time ODE with shortcut consistency) and LeWM (JEPA single-step prediction). Token packing at each timestep combines spatial tokens, action tokens, reward tokens, register tokens, and an agent token for policy/value heads.

- **LatentAutoregressiveLoss** — AR predictor MLP for next-step spatial token prediction with SIGReg loss. Acts as an auxiliary loss in flow matching mode; becomes the primary dynamics loss in LeWM mode.

### Trainers (`dreamer4/trainers.py`)

- **VideoTokenizerTrainer** — Accelerate-based training with EMA and TensorBoard logging
- **BehaviorCloneTrainer** — Supervised learning from VPT demonstrations (used in Phase 2)
- **DreamTrainer** — Imagined rollout generation + PPO/PMPO policy optimization (Phase 3)

### Dataset (`minecraft_vpt_dataset.py`)

Lazy-loading PyTorch Dataset that decodes VPT `.mp4`/`.jsonl` pairs on-the-fly. Supports `decord` for fast random-access seeking (falls back to OpenCV). Memory-efficient: stores only metadata (~46 bytes/frame vs. ~196 KB/frame for pre-loaded tensors).

### Evaluation Agent (`Video-Pre-Training/dreamer4_minecraft_agent.py`)

Wraps a trained Dreamer4 model as a MineRL-compatible agent. Inference pipeline: observation → tokenize → dynamics forward → policy sample → MineRL action dict. Uses clean-latent inference (`signal_level = max_steps - 1`), which is the same regime used by LeWM.

## Testing

The test suite is heavily parametrized over 16 feature flags (shortcut training, GQA, latent AR, PoPE, variable length, spatial tokens, LeWM, etc.):

```bash
# Full parametrized suite (~2^16 combinations)
pytest tests/test_dreamer.py -v

# Just the main end-to-end test
pytest tests/test_dreamer.py -v -k "test_e2e"

# LeWM-specific tests
pytest tests/test_dreamer.py -v -k "use_lewm_dynamics"

# Shard for CI (split across 4 runners)
pytest tests/test_dreamer.py --shard-id=0 --num-shards=4
```

Each combination creates a small tokenizer + dynamics model, runs a forward pass, and validates output shapes and losses.

## HPC / Slurm

Pre-configured SBATCH scripts run all three phases sequentially, then evaluation:

```bash
# Dreamer4 (flow matching)
sbatch scripts/dreamer4-full-train.sh

# LeWM (JEPA dynamics)
sbatch scripts/lewm-full-train.sh
```

Both scripts request a single V100 GPU with 170 GB system RAM and a 24-hour time limit. Adjust the `#SBATCH` headers and data paths for your cluster.

## Moving MNIST Quick Start

For a fast sanity check without Minecraft data, train on synthetic Moving MNIST:

```bash
# Tokenizer (~5 min)
uv run train_moving_mnist_tokenizer.py --num_train_steps 5000

# Dynamics with action conditioning (~5 min)
uv run train_moving_mnist_dynamics.py --num_train_steps 5000 --condition_on_actions True
```

The baseline synthesizes digits floating in random directions. Passing `--condition_on_actions True` lets you command digit trajectories with velocity actions.

## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@misc{hafner2025trainingagentsinsidescalable,
    title   = {Training Agents Inside of Scalable World Models},
    author  = {Danijar Hafner and Wilson Yan and Timothy Lillicrap},
    year    = {2025},
    eprint  = {2509.24527},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url     = {https://arxiv.org/abs/2509.24527},
}
```

```bibtex
@misc{maes2026leworldmodelstableendtoendjointembedding,
    title   = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
    author  = {Lucas Maes and Quentin Le Lidec and Damien Scieur and Yann LeCun and Randall Balestriero},
    year    = {2026},
    eprint  = {2603.19312},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2603.19312},
}
```

```bibtex
@misc{balestriero2025lejepa,
    title   = {LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
    author  = {Randall Balestriero and Yann LeCun},
    year    = {2025},
    eprint  = {2511.08544},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2511.08544},
}
```

<details>
<summary>Additional citations (click to expand)</summary>

```bibtex
@misc{fang2026racrectifiedflowauto,
    title   = {RAC: Rectified Flow Auto Coder},
    author  = {Sen Fang and Yalin Feng and Yanxin Zhang and Dimitris N. Metaxas},
    year    = {2026},
    eprint  = {2603.05925},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2603.05925},
}
```

```bibtex
@misc{chefer2026self,
    title   = {Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis},
    author  = {Hila Chefer and Patrick Esser and Dominik Lorenz and Dustin Podell and Vikash Raja and Vinh Tong and Antonio Torralba and Robin Rombach},
    year    = {2026},
    url     = {https://bfl.ai/research/self-flow},
    note    = {Preprint}
}
```

```bibtex
@misc{li2025basicsletdenoisinggenerative,
    title   = {Back to Basics: Let Denoising Generative Models Denoise},
    author  = {Tianhong Li and Kaiming He},
    year    = {2025},
    eprint  = {2511.13720},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2511.13720},
}
```

```bibtex
@misc{kimiteam2026attentionresiduals,
    title   = {Attention Residuals},
    author  = {Kimi Team and Guangyu Chen and Yu Zhang and Jianlin Su and Weixin Xu and Siyuan Pan and Yaoyu Wang and Yucheng Wang and Guanduo Chen and Bohong Yin and Yutian Chen and Junjie Yan and Ming Wei and Y. Zhang and Fanqing Meng and Chao Hong and Xiaotong Xie and Shaowei Liu and Enzhe Lu and Yunpeng Tai and Yanru Chen and Xin Men and Haiqing Guo and Y. Charles and Haoyu Lu and Lin Sui and Jinguo Zhu and Zaida Zhou and Weiran He and Weixiao Huang and Xinran Xu and Yuzhi Wang and Guokun Lai and Yulun Du and Yuxin Wu and Zhilin Yang and Xinyu Zhou},
    year    = {2026},
    eprint  = {2603.15031},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2603.15031},
}
```

```bibtex
@misc{zhang2026beliefformer,
    title   = {BeliefFormer: Belief Attention in Transformer},
    author  = {Guoqiang Zhang},
    year    = {2026},
    url     = {https://openreview.net/forum?id=Ard2QzPAUK}
}
```

```bibtex
@misc{osband2026delightfulpolicygradient,
    title   = {Delightful Policy Gradient},
    author  = {Ian Osband},
    year    = {2026},
    eprint  = {2603.14608},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2603.14608},
}
```

```bibtex
@misc{gopalakrishnan2025decouplingwhatwherepolar,
    title   = {Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings},
    author  = {Anand Gopalakrishnan and Robert Csordas and Jürgen Schmidhuber and Michael C. Mozer},
    year    = {2025},
    eprint  = {2509.10534},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2509.10534},
}
```

</details>

## Acknowledgements

- [Phil Wang (lucidrains)](https://github.com/lucidrains) for the original [Dreamer4](https://github.com/lucidrains/dreamer4) implementation
- [@dirkmcpherson](https://github.com/dirkmcpherson) for typo fixes and bug corrections
- [@witherhoard99](https://github.com/witherhoard99) and [Vish](https://github.com/humboldt123) for [improvements](https://github.com/lucidrains/dreamer4/pull/10) to video tokenizer convergence, proprioception handling, discrete action bug fixes, and TensorBoard logging
- [OpenAI](https://github.com/openai/Video-Pre-Training) for the VPT demonstration data and evaluation infrastructure

## License

MIT
