"""
Dreamer4 Minecraft Agent — drop-in replacement for VPT's MineRLAgent.

This module wraps a trained Dreamer4 world model (VideoTokenizer +
DynamicsWorldModel with trained policy/value heads) to create an agent
that can play Minecraft through the VPT evaluation infrastructure.

It replaces the broken examples/dreamer4_agent.py which imported
non-existent classes (Encoder, Decoder, Tokenizer, Dynamics) from a
non-existent 'model' module. This version uses the actual Dreamer4 API:
  - VideoTokenizer (from dreamer4.dreamer4)
  - DynamicsWorldModel (from dreamer4.dreamer4)

Inference pipeline at each step:
  1. Receive MineRL observation dict with "pov" key (H, W, 3) uint8
  2. Resize to 128x128, convert to (1, 3, 1, 128, 128) float tensor
  3. VideoTokenizer.tokenize() → latent (1, 1, num_latents, dim_latent)
  4. Concatenate with history latents, add view dim for DynamicsWorldModel
  5. DynamicsWorldModel.forward() with signal_level=max_steps-1 (clean)
     → agent embedding h_t
  6. Policy head MLP → action distribution
  7. ActionEmbedder.sample() → discrete actions (buttons + camera bin)
  8. Convert discrete actions to MineRL env action dict

Key differences from examples/dreamer4_agent.py:
  - Uses actual VideoTokenizer/DynamicsWorldModel classes
  - Uses DynamicsWorldModel's built-in policy_head and action_embedder
  - No separate PolicyHead class needed
  - Correct action space mapping matching training
  - Proper signal_level/step_size conditioning
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
import cv2

# Add paths for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))       # .../Video-Pre-Training
_project_root = os.path.dirname(_this_dir)                    # .../dreamer4 (repo root)
sys.path.insert(0, _this_dir)                                 # VPT local imports (agent.py, lib/)
sys.path.insert(0, _project_root)                             # dreamer4 package + minecraft_vpt_dataset

from dreamer4 import VideoTokenizer, DynamicsWorldModel

from agent import AGENT_RESOLUTION, ENV_KWARGS, validate_env, resize_image
from lib.actions import ActionTransformer, Buttons

from minecraft_vpt_dataset import (
    BUTTONS_ALL,
    N_BUTTONS,
    N_CAMERA_BINS,
    CAMERA_MAXVAL,
    CAMERA_BINSIZE,
    CAMERA_MU,
    DREAMER4_NUM_DISCRETE_ACTIONS,
)


# VPT action transformer for converting discrete bins back to env actions
ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=CAMERA_BINSIZE,
    camera_maxval=CAMERA_MAXVAL,
    camera_mu=CAMERA_MU,
    camera_quantization_scheme="mu_law",
)


def dreamer4_actions_to_minerl(
    discrete_actions: torch.Tensor,
    action_transformer: ActionTransformer,
) -> dict:
    """Convert Dreamer4 discrete action tensor to MineRL env action dict.

    Args:
        discrete_actions: (21,) long tensor
            [0:20] = button states (0 or 1)
            [20]   = camera joint index (0 to 120)
        action_transformer: VPT ActionTransformer for camera undiscretization

    Returns:
        MineRL-compatible action dict with all required keys
    """
    actions_np = discrete_actions.cpu().numpy()

    env_action = {}

    # Buttons: map discrete values to button names
    for i, button_name in enumerate(BUTTONS_ALL):
        env_action[button_name] = int(actions_np[i])

    # Keys not in our 20-button set default to 0
    env_action["ESC"] = 0
    env_action["pickItem"] = 0
    env_action["swapHands"] = 0

    # Camera: decompose joint index to pitch/yaw bins, then undiscretize
    camera_joint = int(actions_np[N_BUTTONS])
    pitch_bin = camera_joint // N_CAMERA_BINS
    yaw_bin = camera_joint % N_CAMERA_BINS

    # Use VPT's undiscretize to convert bins back to degrees
    camera_bins = np.array([[pitch_bin, yaw_bin]])
    camera_degrees = action_transformer.undiscretize_camera(camera_bins)
    env_action["camera"] = camera_degrees[0]  # (2,) array [pitch, yaw]

    return env_action


class Dreamer4MinecraftAgent:
    """Dreamer4-based Minecraft agent compatible with VPT evaluation scripts.

    This class provides the same interface as VPT's MineRLAgent:
      - __init__(env, ...) validates the environment
      - reset() resets internal state
      - get_action(minerl_obs) returns MineRL-compatible action dict

    Architecture:
      The agent uses the DynamicsWorldModel in a special "inference" mode:
      - Signal level = max_steps - 1 (fully denoised / clean observation)
      - Step size = max_steps (single step, since observation is already clean)
      - The model processes the observation history and outputs agent embeddings
      - Policy head maps embeddings to action distributions

      This is equivalent to what interact_with_env() does internally,
      but we handle the environment stepping ourselves to match VPT's interface.
    """

    def __init__(
        self,
        env,
        *,
        checkpoint_path: str,
        device: Optional[str] = None,
        stochastic: bool = True,
        max_context_len: int = 32,
    ):
        """
        Args:
            env: MineRL environment (validated against VPT settings)
            checkpoint_path: Path to dreamer4_minecraft.pt from training
            device: "cuda" or "cpu"
            stochastic: Sample from policy (True) or take argmax (False)
            max_context_len: Maximum history length for the dynamics model
        """
        validate_env(env)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.stochastic = stochastic
        self.max_context_len = max_context_len

        # VPT action transformer for converting bins back to degrees
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        tok_config = ckpt.get('tokenizer_config', {})
        dyn_config = ckpt.get('config', {})

        # Rebuild models
        self.tokenizer = VideoTokenizer(**tok_config)
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        self.dynamics = DynamicsWorldModel(
            video_tokenizer=self.tokenizer,
            **dyn_config,
        )
        self.dynamics.load_state_dict(ckpt['model'])
        self.dynamics.eval()
        for p in self.dynamics.parameters():
            p.requires_grad_(False)

        self.dynamics.to(self.device)

        print(f"Loaded Dreamer4 agent ({sum(p.numel() for p in self.dynamics.parameters()):,} params)")

        # Internal state
        self._latent_history = []     # List of (1, 1, num_views, num_latents, dim_latent) tensors
        self._action_history = []     # List of (1, 1, 21) discrete action tensors
        self._reward_history = []     # List of (1, 1) reward tensors
        self._step_count = 0

    def load_weights(self, path: str):
        """Compatibility stub — weights are loaded in __init__."""
        pass

    def reset(self):
        """Reset agent state for a new episode."""
        self._latent_history = []
        self._action_history = []
        self._reward_history = []
        self._step_count = 0

    @torch.no_grad()
    def _encode_observation(self, minerl_obs: dict) -> torch.Tensor:
        """Encode a MineRL observation to latent tokens.

        Args:
            minerl_obs: MineRL observation dict with "pov" key

        Returns:
            latents: (1, 1, num_latent_tokens, dim_latent) tensor
        """
        # Extract and resize image
        frame = minerl_obs["pov"]  # (H, W, 3) uint8
        frame = resize_image(frame, AGENT_RESOLUTION)  # (128, 128, 3) uint8

        # Convert to (1, 3, 1, 128, 128) float tensor
        frame_t = torch.from_numpy(frame).float() / 255.0
        frame_t = frame_t.permute(2, 0, 1)  # (3, 128, 128)
        video = frame_t.unsqueeze(0).unsqueeze(2).to(self.device)  # (1, 3, 1, 128, 128)

        # Tokenize: returns (1, 1, num_latents, dim_latent)
        latents = self.tokenizer.tokenize(video)
        return latents

    @torch.no_grad()
    def get_action(self, minerl_obs: dict) -> dict:
        """Get action for a MineRL observation.

        Main inference entry point. Matches MineRLAgent.get_action() interface.

        Flow:
        1. Encode observation to latent
        2. Build history context (latents, actions, rewards)
        3. Run dynamics model forward pass with fully-denoised signal
        4. Extract agent embedding → policy head → sample action
        5. Convert to MineRL format

        Args:
            minerl_obs: MineRL observation dict with "pov" key

        Returns:
            MineRL action dict compatible with env.step()
        """
        # 1. Encode current observation
        latents = self._encode_observation(minerl_obs)  # (1, 1, n, d)
        self._latent_history.append(latents)

        # 2. Build context window
        # Trim to max context length
        start = max(0, len(self._latent_history) - self.max_context_len)
        ctx_latents = self._latent_history[start:]
        ctx_actions = self._action_history[start:]
        ctx_rewards = self._reward_history[start:]

        # Stack latents: (1, T, num_latents, dim_latent)
        latents_seq = torch.cat(ctx_latents, dim=1)

        # Stack actions: (1, T-1, 21) or None
        # Actions are shifted: action[t] is the action taken AFTER observing frame[t]
        # So for T frames we have T-1 past actions (none before first frame)
        discrete_actions = None
        if len(ctx_actions) > 0:
            discrete_actions = torch.cat(ctx_actions, dim=1)

        # Stack rewards: (1, T-1) or None
        rewards = None
        if len(ctx_rewards) > 0:
            rewards = torch.cat(ctx_rewards, dim=1)

        # 3. Forward pass through dynamics model
        # Signal level = max_steps - 1 means "fully denoised / clean observation"
        # Step size = max_steps // num_steps where num_steps determines denoising granularity
        # For inference on real observations, we use a single step
        max_steps = self.dynamics.max_steps
        num_steps = 4  # Number of denoising steps (must divide max_steps)
        assert max_steps % num_steps == 0, \
            f"max_steps ({max_steps}) must be divisible by num_steps ({num_steps})"
        step_size = max_steps // num_steps

        _, (embeds, _) = self.dynamics(
            latents=latents_seq,
            signal_levels=max_steps - 1,    # Clean signal (fully denoised)
            step_sizes=step_size,
            rewards=rewards,
            discrete_actions=discrete_actions,
            latent_is_noised=True,           # Skip noise injection — latents are clean from tokenizer
            return_pred_only=True,
            return_intermediates=True,
        )

        # 4. Extract agent embedding and sample action
        agent_embed = embeds.agent  # (B, T, num_agents, dim)
        assert agent_embed.ndim == 4, \
            f"Expected 4D agent_embed, got shape {agent_embed.shape}"
        # Take last timestep, first agent
        one_agent_embed = agent_embed[:, -1:, 0:1, :]  # (1, 1, 1, dim)

        # Policy head → action distribution
        policy_embed = self.dynamics.policy_head(one_agent_embed)

        # Sample actions from the action embedder
        if self.stochastic:
            sampled_discrete, _ = self.dynamics.action_embedder.sample(
                policy_embed, pred_head_index=0, squeeze=True
            )
        else:
            # For deterministic: sample with temperature 0 (argmax)
            sampled_discrete, _ = self.dynamics.action_embedder.sample(
                policy_embed, pred_head_index=0,
                discrete_temperature=0.01, squeeze=True
            )

        # sampled_discrete: (1, 1, 21) → squeeze to (21,) for action conversion
        action_tensor = sampled_discrete.squeeze()  # (21,)

        # 5. Store action and dummy reward in history
        self._action_history.append(sampled_discrete)
        # Use zero reward (we don't know the real reward until env.step)
        dummy_reward = torch.zeros(1, 1, device=self.device)
        self._reward_history.append(dummy_reward)

        # 6. Convert to MineRL action format
        minerl_action = dreamer4_actions_to_minerl(action_tensor, self.action_transformer)

        self._step_count += 1
        return minerl_action


# ─── CLI for quick testing ──────────────────────────────────────────

if __name__ == "__main__":
    from argparse import ArgumentParser
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

    parser = ArgumentParser("Run Dreamer4 agent on MineRL")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to dreamer4_minecraft.pt checkpoint")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max_steps", type=int, default=36000)
    args = parser.parse_args()

    env = HumanSurvival(**ENV_KWARGS).make()
    agent = Dreamer4MinecraftAgent(
        env,
        checkpoint_path=args.checkpoint,
        stochastic=not args.deterministic,
    )

    obs = env.reset()
    agent.reset()
    total_reward = 0.0

    for step in range(args.max_steps):
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if step % 100 == 0:
            print(f"Step {step}, reward: {total_reward:.2f}")
        if args.render:
            env.render()
        if done:
            print(f"Episode done at step {step}, reward: {total_reward:.2f}")
            obs = env.reset()
            agent.reset()
            total_reward = 0.0
