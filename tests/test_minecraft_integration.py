"""Integration tests spanning dataset → tokenizer → dynamics.

Exercises the contract that ties the three Minecraft-specific modules
together: batches produced by MinecraftVPTDataset + collate_minecraft_batch
must be digestible by DynamicsWorldModel configured with
DREAMER4_NUM_DISCRETE_ACTIONS.

We build everything at tiny sizes so the whole flow runs in seconds on CPU.
"""
import pytest
import torch
from torch.utils.data import DataLoader

from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer
from minecraft_vpt_dataset import (
    DREAMER4_NUM_DISCRETE_ACTIONS,
    MinecraftVPTDataset,
    collate_minecraft_batch,
)


pytestmark = pytest.mark.integration


def _tiny_tokenizer(image_h=64, image_w=64):
    return VideoTokenizer(
        dim=16,
        dim_latent=8,
        patch_size=32,
        image_height=image_h,
        image_width=image_w,
        encoder_depth=1,
        decoder_depth=1,
        time_block_every=1,
        attn_heads=2,
        attn_dim_head=8,
        num_latent_tokens=4,
    )


def _tiny_dynamics(tokenizer):
    return DynamicsWorldModel(
        dim=16,
        dim_latent=tokenizer.dim_latent,
        num_latent_tokens=tokenizer.num_latent_tokens,
        num_spatial_tokens=1,
        max_steps=8,
        depth=1,
        time_block_every=1,
        attn_heads=2,
        attn_dim_head=8,
        video_tokenizer=tokenizer,
        num_discrete_actions=DREAMER4_NUM_DISCRETE_ACTIONS,
        num_continuous_actions=0,
        pred_orig_latent=True,
        prob_shortcut_train=0.9,
    )


class TestDatasetToDynamics:
    """End-to-end smoke test: synthetic VPT recordings → training step."""

    def test_dataloader_collate_shapes(self, vpt_data_dir):
        ds = MinecraftVPTDataset(
            data_dir=str(vpt_data_dir),
            seq_len=4, stride=4,
            image_height=64, image_width=64,
        )
        dl = DataLoader(
            ds, batch_size=2, shuffle=False,
            collate_fn=collate_minecraft_batch, num_workers=0,
        )
        batch = next(iter(dl))
        assert batch["video"].shape == (2, 3, 4, 64, 64)
        assert batch["discrete_actions"].shape == (2, 4, 21)
        assert batch["rewards"].shape == (2, 4)

    def test_forward_pass_loss_is_scalar(self, vpt_data_dir):
        ds = MinecraftVPTDataset(
            data_dir=str(vpt_data_dir),
            seq_len=4, stride=4,
            image_height=64, image_width=64,
        )
        dl = DataLoader(
            ds, batch_size=1, shuffle=False,
            collate_fn=collate_minecraft_batch, num_workers=0,
        )
        batch = next(iter(dl))

        tokenizer = _tiny_tokenizer()
        dynamics = _tiny_dynamics(tokenizer)

        loss = dynamics(
            video=batch["video"],
            discrete_actions=batch["discrete_actions"],
            rewards=batch["rewards"],
        )
        assert loss.numel() == 1
        assert torch.isfinite(loss)

    def test_gradients_flow(self, vpt_data_dir):
        ds = MinecraftVPTDataset(
            data_dir=str(vpt_data_dir),
            seq_len=4, stride=4,
            image_height=64, image_width=64,
        )
        dl = DataLoader(
            ds, batch_size=1, shuffle=False,
            collate_fn=collate_minecraft_batch, num_workers=0,
        )
        batch = next(iter(dl))

        tokenizer = _tiny_tokenizer()
        dynamics = _tiny_dynamics(tokenizer)

        loss = dynamics(
            video=batch["video"],
            discrete_actions=batch["discrete_actions"],
            rewards=batch["rewards"],
        )
        loss.backward()
        n_with_grad = sum(
            1 for p in dynamics.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert n_with_grad > 0, "no parameters received gradient"

    def test_generate_emits_valid_action_shape(self):
        """Generating from the dynamics model returns actions compatible
        with our 21-slot discrete action layout."""
        tokenizer = _tiny_tokenizer()
        dynamics = _tiny_dynamics(tokenizer)

        gen = dynamics.generate(
            time_steps=3,
            batch_size=1,
            image_height=64,
            image_width=64,
            return_agent_actions=True,
            return_decoded_video=True,
        )
        discrete_actions, _ = gen.actions
        assert discrete_actions.shape == (1, 3, 21)
        # Button values ∈ {0,1}, camera bin ∈ [0, 120]
        assert discrete_actions[..., :20].max() <= 1
        assert discrete_actions[..., :20].min() >= 0
        assert discrete_actions[..., 20].max() <= 120
        assert discrete_actions[..., 20].min() >= 0
