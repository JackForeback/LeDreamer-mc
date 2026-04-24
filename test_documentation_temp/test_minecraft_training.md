# `tests/test_minecraft_training.py` — Trainer CLI & helpers

> **Requirements verified:** Req1, Req4, Req5, Req6, Req7. See
> `TRACEABILITY_MATRIX.md` §1 for the per-test breakdown.

19 tests, runtime ~4 s. Focuses on the thin "driver" layer in
`train_dreamer4_minecraft.py`: the CLI parser, two load-bearing helpers,
and the default-hyperparameter constants.

## Scope — what we test and what we don't

The `train_dreamer4_minecraft.py` module is ~500 lines of code, most of
which is three functions called `train_tokenizer`, `train_dynamics`, and
`train_agent`. Each constructs a big trainer object and runs
`trainer.train()`. Testing those function bodies would require a real
VPT dataset, real GPUs, and hours of wall time per test.

**Instead, we test:**

1. **`_enforce_cuda_or_exit`** — the strict-CUDA guard. Three branches.
2. **`_resolve_resume_from`** — resume-from-latest pointer logic. Seven cases.
3. **`main()` / argparse** — phase dispatch and flag parsing, with the
   three phase functions monkeypatched out so nothing real runs.
4. **Default-hyperparameter dicts** — paper-aligned constants that
   shouldn't drift.

The existing `tests/test_dreamer.py` already exercises the underlying
`VideoTokenizerTrainer`, `BehaviorCloneTrainer`, and `DreamTrainer`
classes with CPU-sized mocks, so duplicating that here would add runtime
without catching new bugs.

## Imports

```python
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_dreamer4_minecraft as tdm
```

The `SimpleNamespace` import is used to fake argparse results — we can
build an object with `.allow_cpu=True` (and nothing else) without
actually running argparse.

---

## `TestEnforceCuda` (3 tests)

`_enforce_cuda_or_exit(args, phase_name)` is the safety valve that
ensures training doesn't silently run on CPU on an HPC node where CUDA
was supposed to be available. Per `CLAUDE.md §2`:

> This codebase trains ML models on HPC — silent failures waste days of
> compute. Respect existing guards like `--allow_cpu` (which exists
> specifically to prevent silent CUDA→CPU fallback).

The function has three distinct return behaviours:

| `torch.cuda.is_available()` | `args.allow_cpu` | Behaviour | Return |
|---|---|---|---|
| True | (either) | OK, using GPU | `False` (i.e. don't force CPU) |
| False | True | OK, user opted in | `True` (force CPU) |
| False | False | **FAIL LOUD** | `sys.exit(1)` |

### `test_cuda_available_returns_false`
`monkeypatch.setattr(tdm.torch.cuda, "is_available", lambda: True)` —
note that we patch the attribute **on the tdm module's bound `torch`**,
not on the global `torch`. This avoids accidentally changing cuda state
for every other test file in the run.

Also stubs `get_device_name` because `_enforce_cuda_or_exit` prints
`"Using GPU: {name}"` and would raise on a real CPU host.

With CUDA available and `allow_cpu=False`, the function returns `False`
(no CPU forcing).

### `test_allow_cpu_returns_true_on_no_cuda`
`is_available()` → False, `allow_cpu=True` → returns `True`. The user
asked for CPU, the function says "OK."

### `test_no_cuda_no_flag_exits`
`is_available()` → False, `allow_cpu=False` → `SystemExit(1)`.

Captured via `pytest.raises(SystemExit)` and the exit code is explicitly
asserted:

```python
with pytest.raises(SystemExit) as exc_info:
    tdm._enforce_cuda_or_exit(args, "unit")
assert exc_info.value.code == 1
```

Code 1 (not 0) is what matters — an `exit(0)` would look like success to
a Slurm wrapper and that's exactly the silent-failure mode the guard is
meant to prevent.

---

## `TestResolveResumeFrom` (7 tests)

`_resolve_resume_from(value, output_dir)` accepts whatever the user
passed to `--resume_from` and returns either `None` (start fresh) or an
absolute path to a valid `state-<N>/` directory.

Historical context: the resume logic was rewritten once already — see
`documentation/CHECKPOINT_FIX.md`. Several of these tests are
regression guards for that fix.

Valid inputs the function must handle:

1. `None` — no resume requested.
2. `"latest"` — read `<output_dir>/latest_state.txt`, go there.
3. An explicit `state-<N>/` path — use it directly.
4. A parent dir containing a `latest_state.txt` — same behaviour as
   (2), for convenience.
5. Anything else → return `None` + warn.

A "valid" state dir is one that contains at least one of:
`random_states_0.pkl` (accelerate's RNG dump) **or** `model.safetensors`
(the model weights). Either marker proves it was produced by
`accelerator.save_state()` rather than a stray empty directory.

### Per-test breakdown

| Test | Scenario | Expected return |
|---|---|---|
| `test_none_returns_none` | `value=None` | `None` |
| `test_latest_with_pointer` | `"latest"` + `latest_state.txt` points to existing `state-42/` with `random_states_0.pkl` | path ending in `state-42` |
| `test_latest_pointer_references_missing_dir` | `latest_state.txt` points to `state-ghost/` which doesn't exist | `None` (soft-fail, don't crash) |
| `test_direct_state_dir` | `value=/.../state-7` (exists, has `random_states_0.pkl`) | that exact path |
| `test_state_dir_via_safetensors` | `value=/.../state-7` with `model.safetensors` (no pkl) | that exact path |
| `test_nonexistent_path_returns_none` | `value=/.../does-not-exist` | `None` |
| `test_empty_dir_returns_none` | Directory exists but has neither marker file | `None` |

The "soft-fail on missing state-ghost" case is subtle: if the pointer is
stale (e.g. someone manually deleted a checkpoint) we prefer
"start fresh + warn" over "crash" because a multi-day job should not die
at restart over a stale symlink.

### Pattern used

All tests use the `tmp_path` pytest fixture, which pytest provides
automatically — no `conftest.py` setup needed. For each test we build
the required directory layout with `.mkdir()` and `.touch()`, then call
the function:

```python
def test_latest_with_pointer(self, tmp_path):
    state_dir = tmp_path / "state-42"
    state_dir.mkdir()
    (state_dir / "random_states_0.pkl").touch()
    (tmp_path / "latest_state.txt").write_text("state-42")

    resolved = tdm._resolve_resume_from("latest", str(tmp_path))
    assert resolved is not None
    assert Path(resolved).name == "state-42"
```

---

## `TestCLI` (6 tests) — the argparse surface

The goal is to ensure the CLI flags the shell scripts (`scripts/*.sh`)
depend on keep parsing correctly. We don't want to actually run a
training phase — so we **monkeypatch the three phase functions** to
trivial no-ops that just record which phase got called.

### The `_patch_and_run` helper

```python
def _patch_and_run(self, monkeypatch, argv):
    called = {}
    monkeypatch.setattr(tdm, "train_tokenizer", lambda a: called.setdefault("phase", 1))
    monkeypatch.setattr(tdm, "train_dynamics", lambda a: called.setdefault("phase", 2))
    monkeypatch.setattr(tdm, "train_agent",    lambda a: called.setdefault("phase", 3))
    monkeypatch.setattr("sys.argv", ["train_dreamer4_minecraft.py", *argv])
    tdm.main()
    return called
```

- Replaces each phase function with a lambda that records which phase
  fired in a shared `called` dict.
- Sets `sys.argv` to a synthetic command line (argparse reads from
  `sys.argv` by default).
- Calls `tdm.main()` — which does full CLI parsing, dispatches to the
  (now-patched) phase function, and returns.
- Test body inspects `called` to see which phase was hit.

### Per-test breakdown

| Test | argv | Assertion |
|---|---|---|
| `test_phase1_requires_data_dir` | `["--phase", "1"]` — no `--data_dir` | `AssertionError` at `main()` |
| `test_phase1_dispatches` | `["--phase", "1", "--data_dir", <tmp>]` | `called == {"phase": 1}` |
| `test_phase2_dispatches` | `["--phase", "2", "--data_dir", <tmp>]` | `called == {"phase": 2}` |
| `test_phase3_skips_data_dir_requirement` | `["--phase", "3"]` — note, no `--data_dir` | `called == {"phase": 3}`, no error |
| `test_use_lewm_flag_parses` | `["--phase", "2", "--data_dir", ..., "--use_lewm"]` | `args.use_lewm is True` |
| `test_mixed_precision_choices` | `["--phase", "1", ..., "--mixed_precision", "invalid"]` | `SystemExit` (argparse reject) |

The Phase 3 case is architecturally important: Phase 3 ("agent" — PPO in
dreams) does not need real data, only a trained dynamics checkpoint. The
CLI must not require `--data_dir` for phase 3 or else the LeWM training
script would fail to launch.

The `test_use_lewm_flag_parses` test captures `args.use_lewm` through a
fake `train_dynamics` that writes the value to an outer dict — a
slightly different pattern from `_patch_and_run`, but the idea is the
same: replace the real phase function with something that spies on the
parsed args.

---

## `TestHyperparameterDefaults` (3 tests)

Three frozen dicts inside `train_dreamer4_minecraft.py` —
`TOKENIZER_DEFAULTS`, `DYNAMICS_DEFAULTS`, `LEWM_DYNAMICS_DEFAULTS` —
are passed to the trainer constructors when the user doesn't override
them. They encode "paper-aligned" choices. Drift catches drift.

### `test_tokenizer_defaults`
- `patch_size == 16` — per `DreamerPaper.md §D.1`, the ViT patch is 16×16.
- `image_height == 384, image_width == 640` — Minecraft native-ish.
- `num_latent_tokens == 512` — total latent-token count for the MAE.
- `dim_latent == 16` — per-token latent dim.

### `test_dynamics_defaults`
- `max_steps` is a power of 2. This is **critical** — flow-matching
  shortcut consistency training indexes into a log-spaced step schedule
  that breaks if `max_steps` isn't a power of two. The bitwise check
  `d["max_steps"] & (d["max_steps"] - 1) == 0` captures "is 0 or a power
  of two"; we separately assume max_steps > 0.
- `dim_latent == 16` and `num_latent_tokens == 512` — must match the
  tokenizer's output, else the two modules can't be chained.
- `num_discrete_actions == DREAMER4_NUM_DISCRETE_ACTIONS` — the action
  contract (see `test_minecraft_dataset.md`).

### `test_lewm_extends_dynamics`
The LeWM variant reuses the flow-matching defaults as a base, then flips
`use_lewm_dynamics=True`. The test iterates every key in the base dict
and asserts LeWM's value matches — guarding against a common refactor
bug where someone updates one dict but not the other. If drift happens
the failure message is `f"LeWM drift for {k!r}"`, which pinpoints the
culprit.

---

## Running this file

```bash
# full file
env/bin/pytest tests/test_minecraft_training.py -v

# just helpers
env/bin/pytest tests/test_minecraft_training.py::TestEnforceCuda \
                tests/test_minecraft_training.py::TestResolveResumeFrom -v

# with coverage of the training module
env/bin/pytest tests/test_minecraft_training.py \
    --cov=train_dreamer4_minecraft --cov-report=term-missing
```

Current coverage of `train_dreamer4_minecraft.py`: **53 %**. Unreached
lines are mostly inside the three phase function bodies —
intentionally skipped per the scoping note at the top.
