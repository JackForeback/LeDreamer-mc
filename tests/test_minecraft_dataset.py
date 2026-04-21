"""Unit + I/O tests for minecraft_vpt_dataset.

Covers:
  - Camera mu-law encoding / bin discretization
  - Camera bin joint-index math
  - JSONL action parsing (null detection, attack-stuck bug, button mapping,
    camera clamp, hotbar diff)
  - env_action → Dreamer4 tensor conversion
  - _zero_pad_frame (frame padding vs. resize fallback)
  - _composite_cursor overlay alpha blending
  - prescan_trajectory (metadata-only path)
  - load_trajectory (full decode path)
  - MinecraftVPTDataset __len__/__getitem__, skip too-short trajectories,
    retry on corrupt clip
  - collate_minecraft_batch stacking

Action-constant sanity:
  DREAMER4_NUM_DISCRETE_ACTIONS must be a tuple of 21 ints matching
  (2,)*20 + (121,) — this is the single source of truth used by both the
  dataset and the trained DynamicsWorldModel, so a regression here would
  silently break the whole training pipeline.
"""
import json

import numpy as np
import pytest
import torch

from minecraft_vpt_dataset import (
    BUTTONS_ALL,
    CAMERA_BINSIZE,
    CAMERA_MAXVAL,
    CAMERA_MU,
    DREAMER4_NUM_DISCRETE_ACTIONS,
    MinecraftVPTDataset,
    N_BUTTONS,
    N_CAMERA_BINS,
    _composite_cursor,
    _zero_pad_frame,
    camera_bins_to_joint_index,
    collate_minecraft_batch,
    discretize_camera,
    env_action_to_dreamer4,
    load_trajectory,
    mu_law_encode,
    parse_jsonl_action,
    prescan_trajectory,
)


# ─── Action-space constants ─────────────────────────────────────────

class TestActionSpaceConstants:
    def test_button_count(self):
        assert N_BUTTONS == 20
        assert len(BUTTONS_ALL) == 20

    def test_button_order_matches_vpt(self):
        # Must match lib/actions.py Buttons.ALL ordering — verified below by
        # a spot-check of key positions used by other tests/agent code.
        assert BUTTONS_ALL[0] == "attack"
        assert BUTTONS_ALL[2] == "forward"
        assert BUTTONS_ALL[3] == "jump"
        assert BUTTONS_ALL[10] == "inventory"
        assert BUTTONS_ALL[11] == "hotbar.1"
        assert BUTTONS_ALL[19] == "hotbar.9"

    def test_camera_bin_constants(self):
        assert N_CAMERA_BINS == 11
        assert CAMERA_BINSIZE == 2
        assert CAMERA_MAXVAL == 10
        assert CAMERA_MU == 10

    def test_dreamer4_action_shape(self):
        # 20 × Discrete(2) + 1 × Discrete(121)
        assert len(DREAMER4_NUM_DISCRETE_ACTIONS) == N_BUTTONS + 1
        assert list(DREAMER4_NUM_DISCRETE_ACTIONS[:N_BUTTONS]) == [2] * N_BUTTONS
        assert DREAMER4_NUM_DISCRETE_ACTIONS[N_BUTTONS] == N_CAMERA_BINS ** 2


# ─── Camera encoding ────────────────────────────────────────────────

class TestMuLawEncode:
    def test_zero_maps_to_zero(self):
        assert mu_law_encode(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_clips_to_maxval(self):
        # Values beyond maxval get clipped before encoding.
        assert mu_law_encode(np.array([50.0]))[0] == pytest.approx(CAMERA_MAXVAL)
        assert mu_law_encode(np.array([-50.0]))[0] == pytest.approx(-CAMERA_MAXVAL)

    def test_sign_preserved(self):
        v = mu_law_encode(np.array([-5.0, 5.0]))
        assert v[0] < 0 and v[1] > 0
        assert abs(v[0]) == pytest.approx(abs(v[1]))

    def test_monotonic(self):
        xs = np.linspace(-CAMERA_MAXVAL, CAMERA_MAXVAL, 50)
        ys = mu_law_encode(xs)
        assert np.all(np.diff(ys) >= -1e-8), "mu-law encoding must be non-decreasing"


class TestDiscretizeCamera:
    def test_output_in_valid_range(self):
        pts = np.array([[0.0, 0.0], [CAMERA_MAXVAL, -CAMERA_MAXVAL], [5.0, -5.0]])
        bins = discretize_camera(pts)
        assert bins.shape == pts.shape
        assert bins.dtype == np.int64
        assert bins.min() >= 0
        assert bins.max() <= N_CAMERA_BINS - 1

    def test_center_maps_to_center_bin(self):
        # Mu-law of 0 = 0; (0 + 10) / 2 = 5 → center of 11-bin grid
        assert list(discretize_camera(np.array([0.0, 0.0]))) == [5, 5]

    def test_extremes_map_to_endpoints(self):
        assert list(discretize_camera(np.array([CAMERA_MAXVAL, -CAMERA_MAXVAL]))) == [
            N_CAMERA_BINS - 1, 0
        ]


class TestCameraBinsToJointIndex:
    def test_range(self):
        idx = camera_bins_to_joint_index(0, 0)
        assert idx == 0
        idx = camera_bins_to_joint_index(N_CAMERA_BINS - 1, N_CAMERA_BINS - 1)
        assert idx == N_CAMERA_BINS ** 2 - 1

    def test_row_major(self):
        # pitch_bin * 11 + yaw_bin
        assert camera_bins_to_joint_index(3, 7) == 3 * N_CAMERA_BINS + 7


# ─── JSONL parsing ──────────────────────────────────────────────────

def _make_step(keys=None, mouse_buttons=None, dx=0, dy=0, hotbar=0, gui=False):
    return {
        "keyboard": {"keys": keys or [], "newKeys": [], "chars": ""},
        "mouse": {
            "x": 0, "y": 0, "dx": dx, "dy": dy,
            "buttons": mouse_buttons or [], "newButtons": [],
        },
        "hotbar": hotbar, "tick": 0, "isGuiOpen": gui,
    }


class TestParseJsonlAction:
    def test_null_action(self):
        env, is_null = parse_jsonl_action(_make_step())
        assert is_null is True
        # all buttons zero
        for b in BUTTONS_ALL:
            assert env[b] == 0
        assert env["camera"].tolist() == [0.0, 0.0]

    def test_forward_key_maps_to_forward(self):
        env, is_null = parse_jsonl_action(_make_step(keys=["key.keyboard.w"]))
        assert env["forward"] == 1
        assert is_null is False

    def test_esc_key_not_counted_as_button(self):
        # ESC maps to the "ESC" slot but is not in BUTTONS_ALL, so a lone
        # ESC should NOT flip is_null to False.
        env, is_null = parse_jsonl_action(_make_step(keys=["key.keyboard.escape"]))
        assert env["ESC"] == 1
        assert is_null is True

    def test_mouse_button_zero_is_attack(self):
        env, is_null = parse_jsonl_action(_make_step(mouse_buttons=[0]))
        assert env["attack"] == 1
        assert is_null is False

    def test_mouse_button_one_is_use(self):
        env, _ = parse_jsonl_action(_make_step(mouse_buttons=[1]))
        assert env["use"] == 1

    def test_mouse_button_two_is_pickitem(self):
        env, _ = parse_jsonl_action(_make_step(mouse_buttons=[2]))
        assert env["pickItem"] == 1

    def test_camera_dx_dy_scaled(self):
        env, is_null = parse_jsonl_action(_make_step(dx=100, dy=50))
        assert is_null is False
        # CAMERA_SCALER = 360/2400
        assert env["camera"][0] == pytest.approx(50 * 360.0 / 2400.0)   # pitch (dy)
        assert env["camera"][1] == pytest.approx(100 * 360.0 / 2400.0)  # yaw (dx)

    def test_attack_stuck_drops_attack(self):
        # When attack_is_stuck, the persistent mouse button 0 gets stripped.
        env, _ = parse_jsonl_action(
            _make_step(mouse_buttons=[0]), attack_is_stuck=True
        )
        assert env["attack"] == 0


class TestEnvActionToDreamer4:
    def test_shape_and_dtype(self):
        env_action = {b: 0 for b in BUTTONS_ALL}
        env_action["camera"] = np.array([0.0, 0.0])
        arr = env_action_to_dreamer4(env_action)
        assert arr.shape == (N_BUTTONS + 1,)
        assert arr.dtype == np.int64

    def test_button_positions(self):
        env_action = {b: 0 for b in BUTTONS_ALL}
        env_action["camera"] = np.array([0.0, 0.0])
        env_action["forward"] = 1
        env_action["hotbar.3"] = 1
        arr = env_action_to_dreamer4(env_action)
        assert arr[BUTTONS_ALL.index("forward")] == 1
        assert arr[BUTTONS_ALL.index("hotbar.3")] == 1
        # Camera bin 5,5 = 60 (center)
        assert arr[N_BUTTONS] == 60

    def test_camera_encoding_in_range(self):
        env_action = {b: 0 for b in BUTTONS_ALL}
        env_action["camera"] = np.array([CAMERA_MAXVAL, -CAMERA_MAXVAL])
        arr = env_action_to_dreamer4(env_action)
        assert 0 <= arr[N_BUTTONS] < N_CAMERA_BINS ** 2


# ─── Low-level frame utilities ──────────────────────────────────────

class TestZeroPadFrame:
    def test_pads_height_when_width_matches(self):
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 17
        out = _zero_pad_frame(frame, 384, 640)
        assert out.shape == (384, 640, 3)
        # Top/bottom pad bands should be zero, the inside preserved
        assert out[0, 0, 0] == 0
        assert out[-1, 0, 0] == 0
        assert out[12, 0, 0] == 17  # pad_top = 12

    def test_resizes_when_dimensions_mismatch(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        out = _zero_pad_frame(frame, 384, 640)
        assert out.shape == (384, 640, 3)

    def test_passthrough_when_already_target_size(self):
        frame = np.ones((384, 640, 3), dtype=np.uint8) * 5
        out = _zero_pad_frame(frame, 384, 640)
        # Either the same buffer or an identical copy is acceptable.
        assert out.shape == (384, 640, 3)
        assert (out == 5).all()


class TestCompositeCursor:
    def test_overlay_changes_pixels_in_place(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cursor = np.full((16, 16, 3), 255, dtype=np.uint8)
        alpha = np.ones((16, 16, 1), dtype=np.float32)  # fully opaque
        _composite_cursor(image, cursor, alpha, 10, 10)
        assert (image[10:26, 10:26, :] == 255).all()
        # outside region untouched
        assert (image[0:10, 0:10, :] == 0).all()

    def test_clip_out_of_bounds(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        cursor = np.full((16, 16, 3), 255, dtype=np.uint8)
        alpha = np.ones((16, 16, 1), dtype=np.float32)
        _composite_cursor(image, cursor, alpha, 18, 18)  # only 2x2 visible
        assert (image[18:20, 18:20, :] == 255).all()

    def test_fully_off_screen_noop(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        cursor = np.full((16, 16, 3), 255, dtype=np.uint8)
        alpha = np.ones((16, 16, 1), dtype=np.float32)
        _composite_cursor(image, cursor, alpha, 40, 40)  # outside image
        assert (image == 0).all()


# ─── I/O: trajectory loading ────────────────────────────────────────

class TestPrescanTrajectory:
    def test_returns_valid_indices_and_actions(self, vpt_recording):
        mp4, jsonl = vpt_recording
        valid_indices, actions = prescan_trajectory(str(mp4), str(jsonl))
        assert valid_indices.dtype == np.int32
        assert actions.dtype == np.int16
        # Synthetic fixture: half of 32 frames are non-null → 16
        assert len(valid_indices) == 16
        assert actions.shape == (16, N_BUTTONS + 1)

    def test_no_skip_keeps_all_frames(self, vpt_recording):
        mp4, jsonl = vpt_recording
        valid_indices, _ = prescan_trajectory(
            str(mp4), str(jsonl), skip_null_actions=False
        )
        assert len(valid_indices) == 32

    def test_null_only_returns_empty(self, null_only_recording):
        mp4, jsonl = null_only_recording
        valid_indices, actions = prescan_trajectory(str(mp4), str(jsonl))
        assert len(valid_indices) == 0
        assert actions.shape == (0, N_BUTTONS + 1)


class TestLoadTrajectory:
    def test_shapes_and_dtypes(self, vpt_recording):
        mp4, jsonl = vpt_recording
        frames, actions, rewards = load_trajectory(str(mp4), str(jsonl))
        assert frames.dtype == np.uint8
        assert actions.dtype == np.int64
        assert rewards.dtype == np.float32
        assert frames.ndim == 4 and frames.shape[1:] == (384, 640, 3)
        assert actions.shape[1] == N_BUTTONS + 1
        assert len(frames) == len(actions) == len(rewards)

    def test_no_skip_has_more_frames(self, vpt_recording):
        mp4, jsonl = vpt_recording
        _, actions_skip, _ = load_trajectory(str(mp4), str(jsonl))
        _, actions_all, _ = load_trajectory(
            str(mp4), str(jsonl), skip_null_actions=False
        )
        assert len(actions_all) > len(actions_skip)

    def test_raises_for_missing_video(self, tmp_path):
        bogus = tmp_path / "nope.mp4"
        bogus.touch()
        jsonl = tmp_path / "nope.jsonl"
        jsonl.write_text("")
        with pytest.raises(RuntimeError):
            load_trajectory(str(bogus), str(jsonl))


# ─── PyTorch Dataset + collate ─────────────────────────────────────

class TestMinecraftVPTDataset:
    def test_lengths_and_item_shapes(self, vpt_data_dir):
        ds = MinecraftVPTDataset(
            data_dir=str(vpt_data_dir),
            seq_len=4,
            stride=4,
            image_height=384,
            image_width=640,
        )
        assert len(ds) > 0
        sample = ds[0]
        assert set(sample.keys()) == {"video", "discrete_actions", "rewards"}
        assert sample["video"].shape == (3, 4, 384, 640)
        assert sample["video"].dtype == torch.float32
        assert sample["video"].min() >= 0 and sample["video"].max() <= 1.0
        assert sample["discrete_actions"].shape == (4, N_BUTTONS + 1)
        assert sample["discrete_actions"].dtype == torch.int64
        assert sample["rewards"].shape == (4,)

    def test_short_trajectory_excluded(self, vpt_short_recording, tmp_path):
        # vpt_short_recording lives in tmp_path — we reuse that same dir
        ds = MinecraftVPTDataset(data_dir=str(tmp_path), seq_len=16, stride=16)
        assert len(ds) == 0

    def test_max_trajectories_limit(self, vpt_data_dir):
        ds = MinecraftVPTDataset(
            data_dir=str(vpt_data_dir), seq_len=4, stride=4, max_trajectories=1
        )
        # With max_trajectories=1 we should have clips from exactly one traj
        assert len(ds.trajectories) == 1

    def test_getitem_retries_on_error(self, vpt_data_dir, monkeypatch):
        ds = MinecraftVPTDataset(
            data_dir=str(vpt_data_dir), seq_len=4, stride=4
        )
        # Inject a one-shot failure into the decode function so we see
        # the retry branch flip to a different clip successfully.
        import minecraft_vpt_dataset as mvd

        orig = mvd._decode_frames
        calls = {"n": 0}

        def flaky(mp4_path, frame_indices, h, w):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic decode failure")
            return orig(mp4_path, frame_indices, h, w)

        monkeypatch.setattr(mvd, "_decode_frames", flaky)
        sample = ds[0]
        assert sample["video"].shape[0] == 3  # recovered after retry


class TestCollateMinecraftBatch:
    def test_stack_shapes(self):
        batch = [
            {
                "video": torch.zeros(3, 4, 32, 32),
                "discrete_actions": torch.zeros(4, 21, dtype=torch.long),
                "rewards": torch.zeros(4),
            }
            for _ in range(2)
        ]
        out = collate_minecraft_batch(batch)
        assert out["video"].shape == (2, 3, 4, 32, 32)
        assert out["discrete_actions"].shape == (2, 4, 21)
        assert out["rewards"].shape == (2, 4)
