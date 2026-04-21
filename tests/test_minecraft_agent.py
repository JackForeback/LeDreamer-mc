"""Tests for Video-Pre-Training/dreamer4_minecraft_agent.py.

The agent pulls in VPT's ActionTransformer, which imports MineRL at module
load. That's only available in the eval-env (Python 3.9 — see CLAUDE.md §5.1).
When MineRL is missing we fall back to a pure-python duplicate of the
camera-decode math to still cover the correctness claim:

    encode(decode(bin)) == bin, for every bin in [0, 120]

In eval-env (or any env where MineRL is present) the full suite runs and
also exercises the top-level dreamer4_actions_to_minerl function end-to-end.
"""
import os
import sys

import numpy as np
import pytest

# These helper math functions don't need MineRL and give us a deterministic
# reference to compare the agent decode against.
from minecraft_vpt_dataset import (
    BUTTONS_ALL,
    CAMERA_BINSIZE,
    CAMERA_MAXVAL,
    CAMERA_MU,
    N_BUTTONS,
    N_CAMERA_BINS,
    discretize_camera,
)


# ─── Pure-python reference for camera undiscretize ─────────────────
# Mirrors VPT's CameraQuantizer.undiscretize() with scheme='mu_law'.
# Kept out-of-line so we can test roundtrip even without MineRL installed.

def _reference_undiscretize(bins: np.ndarray) -> np.ndarray:
    xy = bins * CAMERA_BINSIZE - CAMERA_MAXVAL
    xy = xy / CAMERA_MAXVAL
    xy = (
        np.sign(xy) * (1.0 / CAMERA_MU)
        * ((1.0 + CAMERA_MU) ** np.abs(xy) - 1.0)
    )
    return xy * CAMERA_MAXVAL


class TestCameraRoundtripPurePython:
    """Verify the encode/decode pair is (approximately) an inverse.

    This is the correctness invariant the agent relies on: a camera bin
    predicted by the policy must decode back to roughly the angle the
    dataset would have produced for that same motion.
    """

    def test_each_bin_roundtrips(self):
        for bin_idx in range(N_CAMERA_BINS):
            # Undiscretize this bin → re-discretize → should map back to itself.
            one_bin = np.array([bin_idx, bin_idx])
            degrees = _reference_undiscretize(one_bin)
            re_bin = discretize_camera(degrees)
            assert tuple(re_bin.tolist()) == (bin_idx, bin_idx)

    def test_center_bin_decodes_to_zero(self):
        center = N_CAMERA_BINS // 2
        result = _reference_undiscretize(np.array([center, center]))
        assert np.allclose(result, 0.0, atol=1e-6)


# ─── Tests that require MineRL / VPT ───────────────────────────────

def _try_import_agent():
    try:
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "..", "Video-Pre-Training")
        )
        import dreamer4_minecraft_agent  # noqa: F401
        return dreamer4_minecraft_agent
    except Exception:
        return None


_ag = _try_import_agent()
_needs_minerl = pytest.mark.skipif(
    _ag is None,
    reason="Agent tests need MineRL (eval-env / Python 3.9)",
)


@_needs_minerl
@pytest.mark.needs_minerl
class TestDreamer4ActionsToMinerl:
    def test_button_positions_roundtrip(self):
        import torch
        t = _ag.ActionTransformer(**_ag.ACTION_TRANSFORMER_KWARGS)
        disc = torch.zeros(N_BUTTONS + 1, dtype=torch.long)
        disc[BUTTONS_ALL.index("forward")] = 1
        disc[BUTTONS_ALL.index("jump")] = 1
        disc[N_BUTTONS] = 60  # center bin
        env_action = _ag.dreamer4_actions_to_minerl(disc, t)
        assert env_action["forward"] == 1
        assert env_action["jump"] == 1
        assert env_action["attack"] == 0
        assert env_action["camera"].shape == (2,)
        # Center bin decodes to ~zero pitch/yaw
        assert abs(env_action["camera"][0]) < 1e-6
        assert abs(env_action["camera"][1]) < 1e-6

    def test_extremes_decode_within_maxval(self):
        import torch
        t = _ag.ActionTransformer(**_ag.ACTION_TRANSFORMER_KWARGS)
        disc = torch.zeros(N_BUTTONS + 1, dtype=torch.long)
        disc[N_BUTTONS] = N_CAMERA_BINS ** 2 - 1  # corner bin
        env_action = _ag.dreamer4_actions_to_minerl(disc, t)
        # mu-law undiscretize at the corner maps back to ±CAMERA_MAXVAL
        assert abs(env_action["camera"][0]) <= CAMERA_MAXVAL + 1e-6
        assert abs(env_action["camera"][1]) <= CAMERA_MAXVAL + 1e-6
