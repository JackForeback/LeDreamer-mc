"""Shared pytest fixtures for LeDreamer-mc Minecraft tests.

Provides synthetic VPT-style MP4/JSONL recording pairs so tests don't need
real VPT data on disk. Generated files live in tmp_path_factory scoped
directories and are cleaned up automatically.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make the project root and Video-Pre-Training importable in every test.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "Video-Pre-Training"))


def _write_vpt_recording(
    dir_path: Path,
    stem: str,
    n_frames: int,
    width: int = 640,
    height: int = 360,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Write a small mp4+jsonl pair mimicking a VPT recording.

    The actions alternate between null (all-zero) and non-null (attack=1,
    camera motion), so null-action filtering halves the usable frame count.
    """
    rng = np.random.default_rng(seed)
    mp4_path = dir_path / f"{stem}.mp4"
    jsonl_path = dir_path / f"{stem}.jsonl"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, 20.0, (width, height))
    try:
        for i in range(n_frames):
            frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    records = []
    for i in range(n_frames):
        is_non_null = (i % 2 == 1)
        rec = {
            "mouse": {
                "x": 320.0,
                "y": 180.0,
                "dx": 5.0 if is_non_null else 0.0,
                "dy": -3.0 if is_non_null else 0.0,
                "buttons": [0] if is_non_null else [],
                "newButtons": [],
            },
            "keyboard": {
                "keys": ["key.keyboard.w"] if is_non_null else [],
                "newKeys": [],
                "chars": "",
            },
            "hotbar": 0,
            "tick": i,
            "isGuiOpen": False,
        }
        records.append(rec)

    with jsonl_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return mp4_path, jsonl_path


@pytest.fixture
def vpt_recording(tmp_path):
    """Single synthetic VPT recording: returns (mp4_path, jsonl_path)."""
    return _write_vpt_recording(tmp_path, "test-rec-0", n_frames=32)


@pytest.fixture
def vpt_data_dir(tmp_path):
    """Directory with multiple synthetic VPT recordings."""
    for i in range(3):
        _write_vpt_recording(tmp_path, f"rec-{i}", n_frames=32, seed=i)
    return tmp_path


@pytest.fixture
def vpt_short_recording(tmp_path):
    """A recording too short for the default seq_len=16 — used to test filtering."""
    return _write_vpt_recording(tmp_path, "short", n_frames=6)


@pytest.fixture
def vpt_gui_recording(tmp_path):
    """A recording with isGuiOpen=True on some frames to exercise cursor overlay."""
    mp4_path = tmp_path / "gui.mp4"
    jsonl_path = tmp_path / "gui.jsonl"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, 20.0, (640, 360))
    for _ in range(10):
        writer.write(np.full((360, 640, 3), 128, dtype=np.uint8))
    writer.release()

    records = []
    for i in range(10):
        records.append({
            "mouse": {"x": 100.0 + i * 5, "y": 50.0, "dx": 1.0, "dy": 1.0,
                      "buttons": [], "newButtons": []},
            "keyboard": {"keys": ["key.keyboard.w"], "newKeys": [], "chars": ""},
            "hotbar": 0,
            "tick": i,
            "isGuiOpen": i >= 5,
        })
    with jsonl_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return mp4_path, jsonl_path


@pytest.fixture
def null_only_recording(tmp_path):
    """All-null-action recording — exercises the null-skip branch."""
    mp4_path = tmp_path / "null.mp4"
    jsonl_path = tmp_path / "null.jsonl"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, 20.0, (640, 360))
    for _ in range(8):
        writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
    writer.release()

    with jsonl_path.open("w") as f:
        for i in range(8):
            rec = {
                "mouse": {"x": 0, "y": 0, "dx": 0, "dy": 0,
                          "buttons": [], "newButtons": []},
                "keyboard": {"keys": [], "newKeys": [], "chars": ""},
                "hotbar": 0, "tick": i, "isGuiOpen": False,
            }
            f.write(json.dumps(rec) + "\n")
    return mp4_path, jsonl_path
