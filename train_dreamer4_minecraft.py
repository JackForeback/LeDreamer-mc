"""
Training script for Dreamer4 on Minecraft using VPT data.

This script implements the full 3-phase Dreamer4 training pipeline:

  Phase 1: Video Tokenizer Training
    Train the VideoTokenizer to compress 384x640 Minecraft frames into
    compact latent representations. Uses MAE-style masking, LPIPS perceptual
    loss, and axial space-time attention.

  Phase 2: World Model (Dynamics) Training
    Train the DynamicsWorldModel on tokenized video + actions from VPT data.
    Uses flow matching with shortcut consistency training. The model learns
    to predict future latent states given past observations and actions.

  Phase 3: Agent Training (Dream-based RL)
    Train the policy/value heads inside the learned world model using
    imagined rollouts (DreamTrainer). No real environment needed — the
    world model generates synthetic experience for PPO/PMPO training.

Each phase saves checkpoints that the next phase loads.

Usage:
    # Phase 1: Train tokenizer
    python train_dreamer4_minecraft.py --phase 1 \
        --data_dir ./data/vpt-recordings \
        --output_dir ./checkpoints \
        --num_steps 50000

    # Phase 2: Train world model
    python train_dreamer4_minecraft.py --phase 2 \
        --data_dir ./data/vpt-recordings \
        --output_dir ./checkpoints \
        --tokenizer_ckpt ./checkpoints/tokenizer.pt \
        --num_steps 100000

    # Phase 3: Train agent in dreams
    python train_dreamer4_minecraft.py --phase 3 \
        --output_dir ./checkpoints \
        --dynamics_ckpt ./checkpoints/dynamics.pt \
        --num_steps 50000
"""

import os
import argparse
import sys
from pathlib import Path

import torch

# cuDNN autotuner picks the fastest conv/attn algorithm for the (fixed) input
# shapes used by this script. One-line win, safe because input shapes are
# static (384x640 video, constant seq_len per phase).
torch.backends.cudnn.benchmark = True

from dreamer4 import VideoTokenizer, DynamicsWorldModel
from dreamer4.trainers import (
    VideoTokenizerTrainer,
    BehaviorCloneTrainer,
    DreamTrainer,
)

from minecraft_vpt_dataset import (
    MinecraftVPTDataset,
    DREAMER4_NUM_DISCRETE_ACTIONS,
)


# ─── CUDA / resume helpers ───────────────────────────────────────────

def _enforce_cuda_or_exit(args, phase_name: str) -> bool:
    """Return True if the phase should run on CPU, False for GPU.

    If CUDA is unavailable and --allow_cpu was not passed, print a loud
    warning and exit(1). This prevents silently burning HPC walltime
    on a CPU fallback caused by e.g. a CUDA driver version mismatch.
    """
    if torch.cuda.is_available():
        print(f"[{phase_name}] CUDA OK — device={torch.cuda.get_device_name(0)} "
              f"| torch={torch.__version__} | cuda={torch.version.cuda}")
        return False

    msg = (
        "\n" + "=" * 70 + "\n"
        f"[{phase_name}] ERROR: torch.cuda.is_available() is False.\n"
        "This is almost always a CUDA driver / PyTorch mismatch.\n"
        f"  torch version : {torch.__version__}\n"
        f"  torch.cuda    : {torch.version.cuda}\n"
        "Training on CPU for this model size is ~100x slower than a V100S.\n"
        "To fix: install a PyTorch build matching your cluster driver, e.g.\n"
        "pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "or ask your admin which CUDA version the driver supports.\n"
        "\n"
        "Pass --allow_cpu to override this check (e.g. for a laptop smoke test).\n"
        + "=" * 70 + "\n"
    )
    if not getattr(args, 'allow_cpu', False):
        print(msg, file=sys.stderr)
        sys.exit(1)
    print(msg, file=sys.stderr)
    print(f"[{phase_name}] --allow_cpu set, proceeding on CPU (will be slow).",
          file=sys.stderr)
    return True


def _resolve_resume_from(raw: str | None, output_dir: str) -> str | None:
    """Resolve --resume_from into a concrete state-<N> directory path.

    Acceptable inputs:
      None                   → no resume
      'latest'               → look in output_dir/latest_state.txt
      <path to state-N dir>  → use as-is
      <path to dir containing state-N dirs>
                             → look for latest_state.txt inside it
    """
    if raw is None:
        return None

    if raw == 'latest':
        base = Path(output_dir)
    else:
        base = Path(raw)

    if base.is_dir():
        # Accept either a state-<N> dir directly, or a parent with a pointer.
        pointer = base / 'latest_state.txt'
        if pointer.exists():
            target = base / pointer.read_text().strip()
            if target.exists():
                print(f"[resume] using {target} from {pointer}")
                return str(target)
            print(f"[resume] pointer {pointer} references {target} but it does not exist")
            return None
        # Might already be a state-<N> dir
        if (base / 'random_states_0.pkl').exists() or any(base.glob('*.safetensors')):
            print(f"[resume] using state dir {base}")
            return str(base)
        print(f"[resume] {base} is not a state dir and has no latest_state.txt; starting fresh")
        return None

    print(f"[resume] resume_from path {base} does not exist; starting fresh")
    return None


# ─── Default Hyperparameters ────────────────────────────────────────

# Paper-matched defaults for Minecraft at 384x640 resolution (360x640 zero-padded).
# See data.txt: 16x16 patches → 960 tokens, bottleneck (N_b=512)x(D_b=16).

TOKENIZER_DEFAULTS = dict(
    dim=256,                    # Hidden dimension of transformer
    dim_latent=16,              # Latent bottleneck dimension (D_b=16)
    patch_size=16,              # 384/16=24, 640/16=40 → 960 patch tokens
    image_height=384,           # VPT 360x640 zero-padded to 384x640
    image_width=640,
    num_latent_tokens=512,      # Bottleneck token count (N_b=512)
    encoder_depth=4,            # Transformer depth
    decoder_depth=4,
    time_block_every=4,         # Temporal attention every 4th block
    attn_heads=4,
    attn_dim_head=64,
    lpips_loss_weight=0.2,      # Perceptual loss weight
    per_image_patch_mask_prob=(0.0, 0.9),  # MAE masking range
    use_loss_normalization=True,
)

DYNAMICS_DEFAULTS = dict(
    dim=256,                    # Hidden dimension
    dim_latent=16,              # Must match tokenizer (D_b=16)
    max_steps=64,               # K_max for flow matching (power of 2)
    num_register_tokens=8,      # Register tokens for temporal consistency
    num_spatial_tokens=256,     # Paper: N_z=256 spatial tokens
    num_latent_tokens=512,      # Must match tokenizer (N_b=512)
    depth=8,                    # Transformer depth
    time_block_every=4,
    attn_heads=4,
    attn_dim_head=64,
    use_time_rnn=True,          # GRU on temporal blocks
    # Action space: 20 binary buttons + 1 camera (121 choices)
    num_discrete_actions=DREAMER4_NUM_DISCRETE_ACTIONS,
    num_continuous_actions=0,   # All actions are discrete
    multi_token_pred_len=8,     # Multi-token prediction horizon
    pred_orig_latent=True,      # x-space prediction (better than v-space)
    # RL hyperparameters (used in Phase 3)
    gae_discount_factor=0.997,
    gae_lambda=0.95,
    ppo_eps_clip=0.2,
    policy_entropy_weight=0.01,
)

# LeWM dynamics: replaces flow matching with JEPA-style next-embedding prediction
LEWM_DYNAMICS_DEFAULTS = dict(
    **DYNAMICS_DEFAULTS,
    use_lewm_dynamics=True,         # Enable LeWM mode
    lewm_loss_weight=1.0,           # Next-embedding prediction loss weight
    lewm_sigreg_loss_weight=0.05,   # SIGReg regularization weight
    lewm_layer=-1,                  # Use last transformer layer for prediction
    lewm_action_conditioned=True,   # Condition predictions on actions
)

TRAINING_DEFAULTS = dict(
    # Phase 1
    tokenizer_batch_size=2,             # Keep small — AttentionResidual layers are memory-intensive
    tokenizer_lr=3e-4,
    tokenizer_num_steps=50000,
    tokenizer_max_grad_norm=1.0,
    tokenizer_seq_len=16,
    # Phase 2
    dynamics_batch_size=4,
    dynamics_lr=3e-4,
    dynamics_num_steps=100000,
    dynamics_max_grad_norm=1.0,
    dynamics_seq_len=16,
    # Phase 3
    dream_batch_size=16,
    dream_lr=3e-4,
    dream_num_steps=50000,
    dream_max_grad_norm=1.0,
    dream_generate_timesteps=16,
)


# ─── Phase 1: Train Video Tokenizer ────────────────────────────────

def train_tokenizer(args):
    """Train the VideoTokenizer on Minecraft video data.

    The tokenizer learns to compress 384x640 RGB frames into compact
    latent representations using:
      - Patch embedding (16x16 patches → 24x40 spatial grid)
      - Axial space-time transformer encoder
      - Latent bottleneck with Tanh activation
      - MAE-style patch masking for regularization
      - LPIPS perceptual loss for visual quality
      - Temporal/spatial decorrelation losses

    The encoder output shape per frame: (num_latent_tokens, dim_latent) = (512, 16)
    """
    print("=" * 60)
    print("PHASE 1: Training Video Tokenizer")
    print("=" * 60)

    use_cpu = _enforce_cuda_or_exit(args, "Phase 1")
    resume_from = _resolve_resume_from(args.resume_from, args.output_dir)

    # Load dataset — tokenizer only needs video, no actions
    dataset = MinecraftVPTDataset(
        data_dir=args.data_dir,
        seq_len=args.tokenizer_seq_len,
        stride=args.tokenizer_seq_len // 2,
        image_height=384,
        image_width=640,
        max_trajectories=args.max_trajectories,
    )

    # The VideoTokenizerTrainer expects a dataset that yields video tensors.
    # Our dataset yields dicts, so we wrap it to extract just the video.
    class VideoOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base = base_dataset
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            return self.base[idx]['video']  # (3, T, H, W)

    video_dataset = VideoOnlyDataset(dataset)

    if len(video_dataset) == 0:
        raise RuntimeError(
            f"No training clips were created from data in '{args.data_dir}'. "
            f"Check that the --data_dir path is correct and contains .mp4/.jsonl pairs. "
            f"(Note: the default folder is 'data/vpt-recordings' with hyphens, not underscores.)"
        )

    # Create tokenizer
    tokenizer = VideoTokenizer(**TOKENIZER_DEFAULTS)
    print(f"VideoTokenizer parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")

    # Train
    trainer = VideoTokenizerTrainer(
        model=tokenizer,
        dataset=video_dataset,
        batch_size=args.tokenizer_batch_size,
        learning_rate=args.tokenizer_lr,
        max_grad_norm=args.tokenizer_max_grad_norm,
        num_train_steps=args.num_steps or args.tokenizer_num_steps,
        cpu=use_cpu,
        mixed_precision=args.mixed_precision,
        use_tensorboard_logger=args.use_tensorboard,
        log_dir=args.output_dir,
        log_video=args.log_video,
        video_fps=20,
        log_video_every=args.log_video_every,
        checkpoint_folder=args.output_dir,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=not args.no_pin_memory,
        dataloader_prefetch_factor=args.prefetch_factor,
        resume_from=resume_from,
    )

    trainer()

    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "tokenizer.pt")
    torch.save({
        'model': tokenizer.state_dict(),
        'config': TOKENIZER_DEFAULTS,
    }, ckpt_path)
    print(f"Tokenizer saved to {ckpt_path}")


# ─── Phase 2: Train Dynamics World Model ───────────────────────────

def train_dynamics(args):
    """Train the DynamicsWorldModel on tokenized video + actions.

    The dynamics model learns to predict future latent states using
    flow matching with shortcut consistency training:

    1. Takes clean latents from the tokenizer
    2. Adds noise at random signal levels (flow matching)
    3. Predicts the clean latents from noised versions
    4. Also predicts rewards and actions (multi-token prediction)

    Shortcut training (from Frans et al.):
      - Sometimes trains with step_size > 1, allowing the model to
        make larger jumps in the denoising process
      - Consistency loss ensures half-step predictions compose correctly
      - This makes generation faster at inference time

    The world model processes sequences of:
      - Spatial tokens (from latents)
      - Action tokens (embedded discrete actions)
      - Reward tokens (SymExp two-hot encoded)
      - Register tokens (for temporal consistency)
      - Agent tokens (for policy/value heads)
    """
    print("=" * 60)
    print("PHASE 2: Training Dynamics World Model")
    print("=" * 60)

    use_cpu = _enforce_cuda_or_exit(args, "Phase 2")
    resume_from = _resolve_resume_from(args.resume_from, args.output_dir)

    # Load tokenizer from Phase 1 checkpoint
    assert args.tokenizer_ckpt is not None, "Must provide --tokenizer_ckpt for Phase 2"
    tok_ckpt = torch.load(args.tokenizer_ckpt, map_location='cpu', weights_only=False)
    tok_config = tok_ckpt.get('config', TOKENIZER_DEFAULTS)

    tokenizer = VideoTokenizer(**tok_config)
    tokenizer.load_state_dict(tok_ckpt['model'])
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    print("Loaded frozen tokenizer")

    # Load dataset with actions
    dataset = MinecraftVPTDataset(
        data_dir=args.data_dir,
        seq_len=args.dynamics_seq_len,
        stride=args.dynamics_seq_len // 2,
        image_height=384,
        image_width=640,
        max_trajectories=args.max_trajectories,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No training clips were created from data in '{args.data_dir}'. "
            f"Check that the --data_dir path is correct and contains .mp4/.jsonl pairs. "
            f"(Note: the default folder is 'data/vpt-recordings' with hyphens, not underscores.)"
        )

    # Create dynamics model with the tokenizer
    base_defaults = LEWM_DYNAMICS_DEFAULTS if args.use_lewm else DYNAMICS_DEFAULTS
    dynamics_config = base_defaults.copy()
    dynamics_config['num_latent_tokens'] = tok_config.get('num_latent_tokens', 16)
    dynamics_config['dim_latent'] = tok_config.get('dim_latent', 32)

    variant = "LeWM" if args.use_lewm else "Dreamer4"
    dynamics = DynamicsWorldModel(
        video_tokenizer=tokenizer,
        **dynamics_config,
    )
    print(f"DynamicsWorldModel ({variant}) parameters: {sum(p.numel() for p in dynamics.parameters()):,}")

    # Train using BehaviorCloneTrainer
    # The trainer accepts dict batches and calls dynamics(**batch_data)
    trainer = BehaviorCloneTrainer(
        model=dynamics,
        dataset=dataset,
        batch_size=args.dynamics_batch_size,
        learning_rate=args.dynamics_lr,
        max_grad_norm=args.dynamics_max_grad_norm,
        num_train_steps=args.num_steps or args.dynamics_num_steps,
        cpu=use_cpu,
        mixed_precision=args.mixed_precision,
        use_tensorboard_logger=args.use_tensorboard,
        log_dir=args.output_dir,
        checkpoint_folder=args.output_dir,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=not args.no_pin_memory,
        dataloader_prefetch_factor=args.prefetch_factor,
        resume_from=resume_from,
    )

    trainer()

    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = "lewm_dynamics.pt" if args.use_lewm else "dynamics.pt"
    ckpt_path = os.path.join(args.output_dir, ckpt_name)
    torch.save({
        'model': dynamics.state_dict(),
        'config': dynamics_config,
        'tokenizer_config': tok_config,
        'use_lewm': args.use_lewm,
    }, ckpt_path)
    print(f"Dynamics model saved to {ckpt_path}")


# ─── Phase 3: Dream-based Agent Training ───────────────────────────

def train_agent(args):
    """Train the policy/value heads using imagined rollouts.

    DreamTrainer generates experience entirely inside the world model:

    1. Start with random noise latents
    2. Iteratively denoise using the dynamics model (generate())
    3. At each step, the agent token embedding provides:
       - Policy distribution (via policy head MLP)
       - Value estimate (via value head MLP)
       - Predicted reward (via reward prediction heads)
    4. After generating a trajectory, compute GAE returns
    5. Update policy head with PPO/PMPO loss
    6. Update value head with clipped value loss

    Only the policy and value head parameters are updated.
    The world model transformer weights remain frozen.

    This is the key advantage of Dreamer-style methods:
    the agent can train on unlimited imagined experience
    without needing access to the real Minecraft environment.
    """
    print("=" * 60)
    print("PHASE 3: Training Agent in Dreams")
    print("=" * 60)

    use_cpu = _enforce_cuda_or_exit(args, "Phase 3")
    resume_from = _resolve_resume_from(args.resume_from, args.output_dir)

    # Load dynamics model from Phase 2
    assert args.dynamics_ckpt is not None, "Must provide --dynamics_ckpt for Phase 3"
    dyn_ckpt = torch.load(args.dynamics_ckpt, map_location='cpu', weights_only=False)
    is_lewm = dyn_ckpt.get('use_lewm', False) or args.use_lewm
    default_config = LEWM_DYNAMICS_DEFAULTS if is_lewm else DYNAMICS_DEFAULTS
    dyn_config = dyn_ckpt.get('config', default_config)
    tok_config = dyn_ckpt.get('tokenizer_config', TOKENIZER_DEFAULTS)
    if is_lewm:
        print("Detected LeWM dynamics checkpoint")

    # Rebuild tokenizer (frozen)
    tokenizer = VideoTokenizer(**tok_config)
    tokenizer.eval()

    # Rebuild dynamics model
    dynamics = DynamicsWorldModel(
        video_tokenizer=tokenizer,
        **dyn_config,
    )
    dynamics.load_state_dict(dyn_ckpt['model'])
    print("Loaded dynamics model")

    # Freeze everything except policy and value heads
    for p in dynamics.parameters():
        p.requires_grad_(False)
    for p in dynamics.policy_head_parameters():
        p.requires_grad_(True)
    for p in dynamics.value_head_parameters():
        p.requires_grad_(True)

    print(f"Trainable parameters: {sum(p.numel() for p in dynamics.parameters() if p.requires_grad):,}")

    # Train using DreamTrainer
    trainer = DreamTrainer(
        model=dynamics,
        batch_size=args.dream_batch_size,
        generate_timesteps=args.dream_generate_timesteps,
        learning_rate=args.dream_lr,
        max_grad_norm=args.dream_max_grad_norm,
        num_train_steps=args.num_steps or args.dream_num_steps,
        cpu=use_cpu,
        mixed_precision=args.mixed_precision,
        use_tensorboard_logger=args.use_tensorboard,
        log_dir=args.output_dir,
        checkpoint_every=args.dream_checkpoint_every,
        checkpoint_folder=args.output_dir,
        resume_from=resume_from,
    )

    trainer()

    # Save final checkpoint with everything
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = "lewm_minecraft.pt" if is_lewm else "dreamer4_minecraft.pt"
    ckpt_path = os.path.join(args.output_dir, ckpt_name)
    torch.save({
        'model': dynamics.state_dict(),
        'config': dyn_config,
        'tokenizer_config': tok_config,
        'use_lewm': is_lewm,
    }, ckpt_path)
    print(f"Trained agent saved to {ckpt_path}")


# ─── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Dreamer4 on Minecraft using VPT data"
    )

    # Required arguments
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="Training phase: 1=tokenizer, 2=dynamics, 3=agent")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory for saving checkpoints and logs")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing VPT .mp4/.jsonl pairs")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Limit number of trajectories (for debugging)")

    # Checkpoint arguments
    parser.add_argument("--tokenizer_ckpt", type=str, default=None,
                        help="Path to tokenizer checkpoint (Phase 2)")
    parser.add_argument("--dynamics_ckpt", type=str, default=None,
                        help="Path to dynamics checkpoint (Phase 3)")

    # Training arguments
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Override number of training steps")

    # Tokenizer hyperparameters
    parser.add_argument("--tokenizer_batch_size", type=int,
                        default=TRAINING_DEFAULTS['tokenizer_batch_size'])
    parser.add_argument("--tokenizer_lr", type=float,
                        default=TRAINING_DEFAULTS['tokenizer_lr'])
    parser.add_argument("--tokenizer_max_grad_norm", type=float,
                        default=TRAINING_DEFAULTS['tokenizer_max_grad_norm'])
    parser.add_argument("--tokenizer_num_steps", type=int,
                        default=TRAINING_DEFAULTS['tokenizer_num_steps'])
    parser.add_argument("--tokenizer_seq_len", type=int,
                        default=TRAINING_DEFAULTS['tokenizer_seq_len'])

    # Dynamics hyperparameters
    parser.add_argument("--dynamics_batch_size", type=int,
                        default=TRAINING_DEFAULTS['dynamics_batch_size'])
    parser.add_argument("--dynamics_lr", type=float,
                        default=TRAINING_DEFAULTS['dynamics_lr'])
    parser.add_argument("--dynamics_max_grad_norm", type=float,
                        default=TRAINING_DEFAULTS['dynamics_max_grad_norm'])
    parser.add_argument("--dynamics_num_steps", type=int,
                        default=TRAINING_DEFAULTS['dynamics_num_steps'])
    parser.add_argument("--dynamics_seq_len", type=int,
                        default=TRAINING_DEFAULTS['dynamics_seq_len'])

    # Dream training hyperparameters
    parser.add_argument("--dream_batch_size", type=int,
                        default=TRAINING_DEFAULTS['dream_batch_size'])
    parser.add_argument("--dream_lr", type=float,
                        default=TRAINING_DEFAULTS['dream_lr'])
    parser.add_argument("--dream_max_grad_norm", type=float,
                        default=TRAINING_DEFAULTS['dream_max_grad_norm'])
    parser.add_argument("--dream_num_steps", type=int,
                        default=TRAINING_DEFAULTS['dream_num_steps'])
    parser.add_argument("--dream_generate_timesteps", type=int,
                        default=TRAINING_DEFAULTS['dream_generate_timesteps'])

    # Model variant
    parser.add_argument("--use_lewm", action="store_true",
                        help="Use LeWM dynamics (JEPA-style next-embedding prediction) "
                             "instead of flow matching")

    # Logging
    parser.add_argument("--use_tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--log_video", action="store_true",
                        help="Log video reconstructions (Phase 1 only)")
    parser.add_argument("--log_video_every", type=int, default=1000,
                        help="Log video every N steps")

    # Resume / checkpointing
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from an accelerator.save_state() dump. "
                             "Accepts either a specific state-<N>/ directory, the "
                             "parent dir containing latest_state.txt, or the literal "
                             "'latest' to read the pointer inside --output_dir.")
    parser.add_argument("--dream_checkpoint_every", type=int, default=500,
                        help="Phase 3: save a DreamTrainer checkpoint every N steps "
                             "(0 to disable). Phases 1/2 use the trainer defaults.")

    # Performance / hardware
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers. Parallel frame decoding so "
                             "data prep does not bottleneck GPU training.")
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Disable DataLoader pin_memory (default: enabled).")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch_factor (per worker). Bigger "
                             "value keeps the GPU fed while workers decode "
                             "the next batches of video.")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode passed to Accelerate. "
                             "Use 'fp16' on V100S (Volta), 'bf16' on A100/H100.")
    parser.add_argument("--allow_cpu", action="store_true",
                        help="Allow running on CPU when torch.cuda.is_available() "
                             "is False. Without this flag, a missing GPU is a hard "
                             "error — preventing a silent 24h CPU fallback.")

    args = parser.parse_args()

    # Validate arguments
    if args.phase in [1, 2]:
        assert args.data_dir is not None, f"Phase {args.phase} requires --data_dir"

    # Run the appropriate phase
    if args.phase == 1:
        train_tokenizer(args)
    elif args.phase == 2:
        train_dynamics(args)
    elif args.phase == 3:
        train_agent(args)


if __name__ == "__main__":
    main()
