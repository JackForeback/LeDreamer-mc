"""Tests for train_dreamer4_minecraft.py (CLI + helpers).

Focus on the two helpers that have non-trivial behaviour and are the
common source of HPC bugs:

  _enforce_cuda_or_exit — silent CPU fallback is explicitly forbidden
  (CLAUDE.md §2). This function either returns, exits, or prints based
  on (cuda_available, --allow_cpu). We test all three branches.

  _resolve_resume_from — resume-from-latest was historically fragile
  (see documentation/CHECKPOINT_FIX.md). We test pointer, direct
  state-dir, and the two "start fresh" paths.

Also smoke-tests the CLI argument parser so the --phase / --use_lewm /
--allow_cpu surface doesn't silently regress.
"""
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_dreamer4_minecraft as tdm


# ─── _enforce_cuda_or_exit ──────────────────────────────────────────

class TestEnforceCuda:
    def test_cuda_available_returns_false(self, monkeypatch):
        monkeypatch.setattr(tdm.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(tdm.torch.cuda, "get_device_name", lambda i: "FakeGPU")
        args = SimpleNamespace(allow_cpu=False)
        assert tdm._enforce_cuda_or_exit(args, "unit") is False

    def test_allow_cpu_returns_true_on_no_cuda(self, monkeypatch):
        monkeypatch.setattr(tdm.torch.cuda, "is_available", lambda: False)
        args = SimpleNamespace(allow_cpu=True)
        assert tdm._enforce_cuda_or_exit(args, "unit") is True

    def test_no_cuda_no_flag_exits(self, monkeypatch):
        monkeypatch.setattr(tdm.torch.cuda, "is_available", lambda: False)
        args = SimpleNamespace(allow_cpu=False)
        with pytest.raises(SystemExit) as exc_info:
            tdm._enforce_cuda_or_exit(args, "unit")
        assert exc_info.value.code == 1


# ─── _resolve_resume_from ───────────────────────────────────────────

class TestResolveResumeFrom:
    def test_none_returns_none(self, tmp_path):
        assert tdm._resolve_resume_from(None, str(tmp_path)) is None

    def test_latest_with_pointer(self, tmp_path):
        state_dir = tmp_path / "state-42"
        state_dir.mkdir()
        (state_dir / "random_states_0.pkl").touch()
        (tmp_path / "latest_state.txt").write_text("state-42")

        resolved = tdm._resolve_resume_from("latest", str(tmp_path))
        assert resolved is not None
        assert Path(resolved).name == "state-42"

    def test_latest_pointer_references_missing_dir(self, tmp_path):
        (tmp_path / "latest_state.txt").write_text("state-ghost")
        assert tdm._resolve_resume_from("latest", str(tmp_path)) is None

    def test_direct_state_dir(self, tmp_path):
        state = tmp_path / "state-7"
        state.mkdir()
        (state / "random_states_0.pkl").touch()
        resolved = tdm._resolve_resume_from(str(state), str(tmp_path))
        assert resolved == str(state)

    def test_state_dir_via_safetensors(self, tmp_path):
        state = tmp_path / "state-7"
        state.mkdir()
        (state / "model.safetensors").touch()
        assert tdm._resolve_resume_from(str(state), str(tmp_path)) == str(state)

    def test_nonexistent_path_returns_none(self, tmp_path):
        assert tdm._resolve_resume_from(
            str(tmp_path / "does-not-exist"), str(tmp_path)
        ) is None

    def test_empty_dir_returns_none(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert tdm._resolve_resume_from(str(empty), str(tmp_path)) is None


# ─── CLI surface ────────────────────────────────────────────────────

class TestCLI:
    """Parse the CLI without executing any phase.

    We patch the three phase functions and call main() for each mode.
    This ensures argparse stays in sync with the registered flags the
    training scripts (scripts/*.sh) rely on.
    """

    def _patch_and_run(self, monkeypatch, argv):
        called = {}
        monkeypatch.setattr(tdm, "train_tokenizer", lambda a: called.setdefault("phase", 1))
        monkeypatch.setattr(tdm, "train_dynamics", lambda a: called.setdefault("phase", 2))
        monkeypatch.setattr(tdm, "train_agent",    lambda a: called.setdefault("phase", 3))
        monkeypatch.setattr("sys.argv", ["train_dreamer4_minecraft.py", *argv])
        tdm.main()
        return called

    def test_phase1_requires_data_dir(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog", "--phase", "1"])
        with pytest.raises(AssertionError):
            tdm.main()

    def test_phase1_dispatches(self, monkeypatch, tmp_path):
        called = self._patch_and_run(monkeypatch, ["--phase", "1", "--data_dir", str(tmp_path)])
        assert called == {"phase": 1}

    def test_phase2_dispatches(self, monkeypatch, tmp_path):
        called = self._patch_and_run(monkeypatch, ["--phase", "2", "--data_dir", str(tmp_path)])
        assert called == {"phase": 2}

    def test_phase3_skips_data_dir_requirement(self, monkeypatch):
        called = self._patch_and_run(monkeypatch, ["--phase", "3"])
        assert called == {"phase": 3}

    def test_use_lewm_flag_parses(self, monkeypatch, tmp_path):
        captured = {}
        def fake_train(args):
            captured["use_lewm"] = args.use_lewm
        monkeypatch.setattr(tdm, "train_dynamics", fake_train)
        monkeypatch.setattr("sys.argv", [
            "prog", "--phase", "2", "--data_dir", str(tmp_path), "--use_lewm",
        ])
        tdm.main()
        assert captured["use_lewm"] is True

    def test_mixed_precision_choices(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.argv", [
            "prog", "--phase", "1", "--data_dir", str(tmp_path),
            "--mixed_precision", "invalid",
        ])
        with pytest.raises(SystemExit):
            tdm.main()


# ─── Default hyperparams are paper-aligned ─────────────────────────

class TestHyperparameterDefaults:
    """Defaults encode paper choices — catch accidental drift."""

    def test_tokenizer_defaults(self):
        d = tdm.TOKENIZER_DEFAULTS
        assert d["patch_size"] == 16
        assert d["image_height"] == 384 and d["image_width"] == 640
        assert d["num_latent_tokens"] == 512
        assert d["dim_latent"] == 16

    def test_dynamics_defaults(self):
        d = tdm.DYNAMICS_DEFAULTS
        # Power of two: critical for shortcut consistency training
        assert d["max_steps"] & (d["max_steps"] - 1) == 0
        assert d["dim_latent"] == 16
        assert d["num_latent_tokens"] == 512
        assert d["num_discrete_actions"] == tdm.DREAMER4_NUM_DISCRETE_ACTIONS

    def test_lewm_extends_dynamics(self):
        lewm = tdm.LEWM_DYNAMICS_DEFAULTS
        base = tdm.DYNAMICS_DEFAULTS
        assert lewm["use_lewm_dynamics"] is True
        # Every non-LeWM key matches the base, so switching variants doesn't
        # silently change unrelated hyperparameters.
        for k, v in base.items():
            assert lewm[k] == v, f"LeWM drift for {k!r}"
