# `tests/test_minecraft_agent.py` — Agent & camera-roundtrip tests

> **Requirements verified:** Req2, Req3, Req8, Req9. See
> `TRACEABILITY_MATRIX.md` §1 for the per-test breakdown.

2 tests always run + 2 tests that require MineRL. Runtime <1 s in
training env; more in eval-env.

## The dual-environment problem this file solves

`Video-Pre-Training/dreamer4_minecraft_agent.py` imports VPT's
`ActionTransformer`, which transitively imports MineRL. MineRL requires
**Python 3.9** and a JVM, and is only installed in the eval-env
(`scripts/setup_eval_venv.sh`). In the training env (Python 3.12), the
module can't even be imported.

But there's a **mathematical correctness claim** that the agent relies
on which does not depend on MineRL:

> For every camera bin `b ∈ [0, 120]`:
> `discretize(undiscretize(b)) == b`

If that roundtrip ever breaks, the agent will emit camera bins that
decode to angles that round to a *different* bin — i.e. the action it
thought it was taking isn't the action the env receives. We want this
claim tested in **every** environment.

**Solution:** a dual-track structure.
1. Pure-Python tests for the mu-law math that always run (use a local
   copy of VPT's `undiscretize`).
2. MineRL-gated tests for the `dreamer4_actions_to_minerl` end-to-end
   function, skipped if `dreamer4_minecraft_agent` can't be imported.

An earlier version of this file used a module-level
`pytest.importorskip("minerl")`, which caused the *entire module*
(including the pure-Python tests) to be skipped in training env. That
was fixed — see `TASK_SUMMARY_minecraft-tests.md` §3 for the rationale.

## Imports

```python
import os
import sys
import numpy as np
import pytest

from minecraft_vpt_dataset import (
    BUTTONS_ALL, CAMERA_BINSIZE, CAMERA_MAXVAL, CAMERA_MU,
    N_BUTTONS, N_CAMERA_BINS,
    discretize_camera,
)
```

`discretize_camera` is imported from the dataset module (not the agent
module) because the dataset module does not depend on MineRL.

---

## The pure-Python reference: `_reference_undiscretize`

```python
def _reference_undiscretize(bins: np.ndarray) -> np.ndarray:
    xy = bins * CAMERA_BINSIZE - CAMERA_MAXVAL
    xy = xy / CAMERA_MAXVAL
    xy = (
        np.sign(xy) * (1.0 / CAMERA_MU)
        * ((1.0 + CAMERA_MU) ** np.abs(xy) - 1.0)
    )
    return xy * CAMERA_MAXVAL
```

This is a line-for-line re-implementation of
VPT's `CameraQuantizer.undiscretize(scheme="mu_law")`. Takes bin
indices → returns degrees in `[-CAMERA_MAXVAL, CAMERA_MAXVAL]`.

Breakdown of the formula:
1. `xy = bin * BINSIZE - MAXVAL` — map bin ∈ [0, 10] to
   linearly-spaced value in `[-MAXVAL, MAXVAL]`.
2. `xy / MAXVAL` — normalize to `[-1, 1]`.
3. The mu-law **inverse**: `sign(u) * ((1+mu)^|u| - 1) / mu`. This
   expands the compressed signal back out.
4. `* MAXVAL` — rescale back to degrees.

Keeping this reference **out of `minecraft_vpt_dataset.py`** is
intentional: the dataset module only ever needs `discretize`, not
`undiscretize`. The agent module is where `undiscretize` lives — and
that module is MineRL-gated. So we duplicate the ~6 lines of math here
to cover the math even when we can't import the agent.

---

## `TestCameraRoundtripPurePython` (2 tests, always run)

### `test_each_bin_roundtrips`
```python
for bin_idx in range(N_CAMERA_BINS):
    one_bin = np.array([bin_idx, bin_idx])
    degrees = _reference_undiscretize(one_bin)
    re_bin = discretize_camera(degrees)
    assert tuple(re_bin.tolist()) == (bin_idx, bin_idx)
```

Iterates every bin in `[0, 11)` (both pitch and yaw, same bin), decodes
to degrees via the reference, re-encodes with the dataset's
`discretize_camera`, and asserts we land on the same bin. This is the
full roundtrip correctness claim: 11 × 1 = 11 checks.

Note: we only check the diagonal (`[bin, bin]`), not all 11² pairs. The
encode and decode functions are coordinate-wise independent so the
diagonal is sufficient — if any individual axis were broken, the
diagonal would catch it.

### `test_center_bin_decodes_to_zero`
```python
center = N_CAMERA_BINS // 2  # == 5
result = _reference_undiscretize(np.array([center, center]))
assert np.allclose(result, 0.0, atol=1e-6)
```

The center bin must correspond to exactly zero camera motion. Not
"approximately zero"; **exactly zero** up to FP noise. If this drifted,
"no camera motion" and "tiny camera motion" would get confused, which
is common in held controllers and causes the agent to wiggle on idle.

---

## MineRL-gated tests

### The import guard

```python
def _try_import_agent():
    try:
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "..", "Video-Pre-Training")
        )
        import dreamer4_minecraft_agent
        return dreamer4_minecraft_agent
    except Exception:
        return None

_ag = _try_import_agent()
_needs_minerl = pytest.mark.skipif(
    _ag is None,
    reason="Agent tests need MineRL (eval-env / Python 3.9)",
)
```

Three important properties:

1. **`except Exception` (not `ImportError`)** — MineRL's import chain
   raises various oddities: `RuntimeError` from the JVM if Java is
   missing, `AttributeError` from `beartype` version mismatches, etc.
   A broad catch keeps the test file loadable regardless.
2. **Function-level, not module-level** — the `skipif` is applied per
   class via `@_needs_minerl`, not via `pytest.importorskip` at top of
   file. This is what keeps the pure-Python tests running.
3. **Imports only performed once per pytest session** — `_ag` is
   module-level, computed when the test file is first loaded.

### `TestDreamer4ActionsToMinerl` (2 tests, skipped if `_ag is None`)

Decorated with **both** `@_needs_minerl` (the skip) and
`@pytest.mark.needs_minerl` (the custom marker registered in
`pyproject.toml`). The marker lets you filter:

```bash
pytest -m needs_minerl        # run only these
pytest -m "not needs_minerl"  # skip them
```

#### `test_button_positions_roundtrip`

```python
t = _ag.ActionTransformer(**_ag.ACTION_TRANSFORMER_KWARGS)
disc = torch.zeros(N_BUTTONS + 1, dtype=torch.long)
disc[BUTTONS_ALL.index("forward")] = 1
disc[BUTTONS_ALL.index("jump")] = 1
disc[N_BUTTONS] = 60  # center bin
env_action = _ag.dreamer4_actions_to_minerl(disc, t)
```

Builds a Dreamer4-format discrete-action tensor with:
- "forward" bit flipped (index 2)
- "jump" bit flipped (index 3)
- camera bin = 60 (center: `5*11 + 5`)

Then calls the agent's conversion function and asserts the resulting
MineRL env-action dict has:
- `forward == 1`, `jump == 1`
- `attack == 0` (not flipped)
- `camera.shape == (2,)`
- `abs(camera[0]) < 1e-6` and `abs(camera[1]) < 1e-6` — center bin
  decodes to zero.

This is the end-to-end correctness claim: the Dreamer4 model outputs
discrete action tensors in a specific layout, and
`dreamer4_actions_to_minerl` must translate them correctly into
MineRL's dict format. Getting this wrong = agent plays the game
"blindfolded."

#### `test_extremes_decode_within_maxval`

```python
disc[N_BUTTONS] = N_CAMERA_BINS ** 2 - 1  # corner bin == 120
env_action = _ag.dreamer4_actions_to_minerl(disc, t)
assert abs(env_action["camera"][0]) <= CAMERA_MAXVAL + 1e-6
assert abs(env_action["camera"][1]) <= CAMERA_MAXVAL + 1e-6
```

Sets the camera bin to 120 (the "corner" of the 11×11 grid, i.e.
extreme pitch + extreme yaw) and asserts the decoded degrees stay
within `±CAMERA_MAXVAL`. Off-by-one errors in the mu-law inverse
would overshoot. The `+ 1e-6` tolerance accommodates FP noise.

---

## Running this file

In **training env** (MineRL absent):
```bash
env/bin/pytest tests/test_minecraft_agent.py -v
# → 2 passed, 2 skipped
```

In **eval-env** (MineRL present):
```bash
bash scripts/setup_eval_venv.sh   # one-time setup
eval-env/bin/pytest tests/test_minecraft_agent.py -v
# → 4 passed
```

To list only MineRL-gated tests:
```bash
env/bin/pytest tests/test_minecraft_agent.py -m needs_minerl
```

## Why we don't unit-test `Dreamer4MinecraftAgent.get_action()`

`get_action` is the inference path: load checkpoint → preprocess obs →
run tokenizer + dynamics + policy → decode action. It needs:

1. MineRL (for `validate_env` + observation shape helpers).
2. A real trained checkpoint (not just any random-init model — the
   policy head has specific output expectations).

Testing it would add a ~30 MB fake checkpoint and a MineRL-gated test
scaffold, which the summary flags as a follow-up. The math the
inference path depends on (camera roundtrip, button positions) is
already exercised by the tests in this file, so we're not blind to
regressions — we just can't currently assert the full end-to-end call.
