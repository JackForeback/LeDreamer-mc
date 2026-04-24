# `tests/test_minecraft_dataset.py` — Dataset & action-space tests

> **Requirements verified:** Req5, Req6, Req9, Req10. See
> `TRACEABILITY_MATRIX.md` §1 for the per-test breakdown.

41 tests, runtime ~4 s. This is the biggest single file in the suite and
covers the entire `minecraft_vpt_dataset.py` module: from low-level math
(mu-law encoding) up through the `torch.utils.data.Dataset` surface.

## What this module is protecting

`minecraft_vpt_dataset.py` is the bridge between **VPT's on-disk format**
(raw `.mp4` + per-frame `.jsonl`) and **Dreamer4's tensor contract**
(video `[B, 3, T, H, W]`, discrete actions `[B, T, 21]`, rewards `[B, T]`).
Every field that crosses this bridge is tested here. If any of these
tests fail, training will either crash or silently mis-encode actions —
the latter being much worse.

The single most important invariant in this file is:

```
DREAMER4_NUM_DISCRETE_ACTIONS = (2,)*20 + (121,)
```

which encodes "20 binary buttons followed by one 121-bin camera slot."
The dataset produces tensors shaped by this; the `DynamicsWorldModel`
is instantiated expecting it; the agent encodes from it. Pinned in
`TestActionSpaceConstants::test_dreamer4_action_shape`.

---

## Imports

```python
from minecraft_vpt_dataset import (
    BUTTONS_ALL, CAMERA_BINSIZE, CAMERA_MAXVAL, CAMERA_MU,
    DREAMER4_NUM_DISCRETE_ACTIONS, MinecraftVPTDataset,
    N_BUTTONS, N_CAMERA_BINS,
    _composite_cursor, _zero_pad_frame,
    camera_bins_to_joint_index, collate_minecraft_batch,
    discretize_camera, env_action_to_dreamer4,
    load_trajectory, mu_law_encode,
    parse_jsonl_action, prescan_trajectory,
)
```

The underscore-prefixed imports (`_composite_cursor`, `_zero_pad_frame`,
`_decode_frames`) are "private" helpers — normally we don't test private
functions, but these are load-bearing enough (frame padding and cursor
overlay shape up what the model actually sees) that we verify their
behaviour directly rather than infer it through the dataset class.

---

## `TestActionSpaceConstants` (4 tests) — the action-space contract

### `test_button_count`
Asserts `N_BUTTONS == 20` and `len(BUTTONS_ALL) == 20`. These are
**constants** but the test still pins them so a refactor (adding a new
button, say) can't silently widen the action space without the paired
changes in `DREAMER4_NUM_DISCRETE_ACTIONS` and the dynamics model.

### `test_button_order_matches_vpt`
Spot-checks positions 0, 2, 3, 10, 11, 19:

| Index | Button |
|---|---|
| 0 | `attack` |
| 2 | `forward` |
| 3 | `jump` |
| 10 | `inventory` |
| 11 | `hotbar.1` |
| 19 | `hotbar.9` |

The order **must** match `Video-Pre-Training/lib/actions.py :: Buttons.ALL`
because the agent code (`dreamer4_actions_to_minerl`) indexes by position
to translate Dreamer4's output back into MineRL's dict-form env action.
If these positions drift, the agent will press the wrong keys.

### `test_camera_bin_constants`
Asserts all four mu-law constants simultaneously:

- `N_CAMERA_BINS == 11` — 11 bins per axis ⇒ 11² = 121 joint bins.
- `CAMERA_BINSIZE == 2` — spacing in mu-law'd coordinates.
- `CAMERA_MAXVAL == 10` — clamp bound in degrees.
- `CAMERA_MU == 10` — mu-law compression strength.

Changing any of these without updating the others silently breaks the
bin grid.

### `test_dreamer4_action_shape`
The lynchpin check — asserts that `DREAMER4_NUM_DISCRETE_ACTIONS` is
literally the tuple `(2, 2, ..., 2, 121)` with 20 twos. This is the
contract every other test file also pins via different angles
(integration test asserts the shape of `generate`'s output; training
test asserts `DYNAMICS_DEFAULTS` matches it). If any one fires, you've
broken the contract; the overlap is deliberate.

---

## `TestMuLawEncode` (4 tests) — the camera compression curve

`mu_law_encode(x)` is the scalar mu-law compression used by VPT's
`CameraQuantizer`:

```
y = sign(x) * log(1 + mu*|x|/maxval) / log(1 + mu) * maxval
```

This "spreads out" small motions (small yaw/pitch changes get more bins)
and "compresses" large motions, which matches how humans actually move
the mouse in Minecraft.

### `test_zero_maps_to_zero`
`mu_law_encode(0) == 0`. The identity at the origin — if this breaks,
the center bin is no longer "no motion."

### `test_clips_to_maxval`
Inputs ±50 get clipped to ±`CAMERA_MAXVAL=10` before the log. Verifies
the `np.clip` pre-step runs.

### `test_sign_preserved`
`mu_law_encode(-5)` negative, `mu_law_encode(+5)` positive, and
`abs(encode(-5)) ≈ abs(encode(+5))`. Confirms the function is odd
(symmetric about zero).

### `test_monotonic`
Sample 50 points across `[-maxval, maxval]` and assert `diff(ys) >= -1e-8`
(tolerant of FP noise). A non-monotonic encoding would put neighbouring
inputs in non-neighbouring bins — catastrophic for discretization.

---

## `TestDiscretizeCamera` (3 tests)

`discretize_camera(points)` takes a `(..., 2)` array of `(pitch, yaw)`
degrees and returns int bin indices in `[0, 11)`. This is what
`env_action_to_dreamer4` calls on the per-step camera vector.

### `test_output_in_valid_range`
Run the function on a batch of points including both center and extremes,
then assert `bins.min() >= 0`, `bins.max() <= N_CAMERA_BINS-1`, and the
dtype is `np.int64` with preserved shape. A regression where an
out-of-bounds bin leaks through would cause a CUDA assertion later when
the world model tries to look up that bin's embedding.

### `test_center_maps_to_center_bin`
`discretize_camera([0, 0]) == [5, 5]`. The mu-law of zero is zero;
`(0 + 10) / 2 = 5` — the middle bin of an 11-bin grid. Any drift in
`CAMERA_BINSIZE` or `CAMERA_MAXVAL` would move this.

### `test_extremes_map_to_endpoints`
`discretize_camera([10, -10]) == [10, 0]`. Confirms that when input hits
the clamp, it lands exactly on the edge bin — not one-past-the-end.

---

## `TestCameraBinsToJointIndex` (2 tests)

Combines the two 1D bins into a single `[0, 121)` index used by the
`Discrete(121)` action slot.

### `test_range`
`(0,0) → 0`, `(10,10) → 120`. End-to-end bounds.

### `test_row_major`
`(3, 7) → 3*11 + 7 = 40`. Explicitly checks **pitch-major** ordering.
If this ever got flipped to yaw-major, the agent would roll instead of
look up/down.

---

## `TestParseJsonlAction` (7 tests)

`parse_jsonl_action(step_dict)` is the single most complex function in
`minecraft_vpt_dataset.py`. Takes a parsed JSONL line, returns
`(env_action_dict, is_null)`. All the following subtleties need to be
exercised because they each cost time to debug when wrong.

### The `_make_step` helper

```python
def _make_step(keys=None, mouse_buttons=None, dx=0, dy=0, hotbar=0, gui=False):
```

Builds a minimal valid JSONL record. Each test calls this with just the
field it wants to vary.

### Individual tests

| Test | What's being checked |
|---|---|
| `test_null_action` | Empty input → all buttons zero, camera `[0,0]`, `is_null=True`. |
| `test_forward_key_maps_to_forward` | `"key.keyboard.w"` sets `env["forward"] = 1`, `is_null=False`. |
| `test_esc_key_not_counted_as_button` | **Important edge case.** ESC appears in the keymap but is not in `BUTTONS_ALL`, so pressing *only* ESC should leave `is_null=True`. Without this check, the null-filter in the dataset would accidentally keep every pause-menu frame. |
| `test_mouse_button_zero_is_attack` | Mouse button 0 → `env["attack"] = 1`. |
| `test_mouse_button_one_is_use` | Mouse button 1 → `env["use"] = 1`. |
| `test_mouse_button_two_is_pickitem` | Mouse button 2 → `env["pickItem"] = 1`. |
| `test_camera_dx_dy_scaled` | Mouse `(dx=100, dy=50)` → `env["camera"] ≈ (50·360/2400, 100·360/2400)`. The `360/2400` is VPT's `CAMERA_SCALER`. Note the **dy→pitch, dx→yaw** ordering, which trips people up. |
| `test_attack_stuck_drops_attack` | When called with `attack_is_stuck=True`, a mouse-button-0 press should *not* flip `env["attack"]`. This models VPT's known "stuck attack" bug where some recordings have attack-on through an entire clip; if we left those in, the dataset distribution would skew to all-attack. |

---

## `TestEnvActionToDreamer4` (3 tests)

`env_action_to_dreamer4(env_action)` packs the dict-form action into a
flat `int64[N_BUTTONS + 1]` array that the world model ingests.

### `test_shape_and_dtype`
Output is `(21,) int64`. Wrong dtype would break `F.one_hot` or embedding
lookups downstream.

### `test_button_positions`
Set `env["forward"] = env["hotbar.3"] = 1`, camera at origin; assert:
- `arr[BUTTONS_ALL.index("forward")] == 1`
- `arr[BUTTONS_ALL.index("hotbar.3")] == 1`
- `arr[20] == 60` (center-bin joint index = `5*11 + 5`)

### `test_camera_encoding_in_range`
Camera at `[10, -10]` (extremes) produces a bin in `[0, 121)`. This is a
soft correctness check — we don't assert the exact value (that's
`TestCameraBinsToJointIndex`'s job), just that it stays in bounds.

---

## `TestZeroPadFrame` (3 tests)

`_zero_pad_frame(frame, target_h, target_w)` handles the resolution
mismatch between VPT's 640×360 and Dreamer4's 640×384 (which has to be
divisible by `patch_size=16`). It pads the shorter dimension with zeros;
if the aspect is wrong it falls back to `cv2.resize`.

### `test_pads_height_when_width_matches`
Input `(360, 640, 3)` filled with 17 → output `(384, 640, 3)` where:
- `out[0, 0, 0] == 0` (top pad band)
- `out[-1, 0, 0] == 0` (bottom pad band)
- `out[12, 0, 0] == 17` — verifies `pad_top == 12` (since `384-360=24`
  split into 12 top + 12 bottom).

### `test_resizes_when_dimensions_mismatch`
Input `(100, 200, 3)` → output `(384, 640, 3)`. Both dims off → resize
fallback; output shape only, content unchecked.

### `test_passthrough_when_already_target_size`
Input `(384, 640, 3)` filled with 5 → output still `(384, 640, 3)` with
all 5s. Neither branch should mutate the content.

---

## `TestCompositeCursor` (3 tests)

`_composite_cursor(image, cursor, alpha, x, y)` draws a cursor sprite
onto an image **in place** with alpha blending, used when
`isGuiOpen=True`. It must handle partial-overlap and off-screen cases
because the recorded mouse position can be anywhere including off-canvas.

### `test_overlay_changes_pixels_in_place`
Cursor is 16×16 white, alpha fully opaque, placed at `(10, 10)` on a
100×100 black image → the 16×16 block at `[10:26, 10:26]` is 255, outside
is 0.

### `test_clip_out_of_bounds`
Same cursor placed at `(18, 18)` on a 20×20 image → only the bottom-right
2×2 is visible. Clipping must not index-error.

### `test_fully_off_screen_noop`
Cursor at `(40, 40)` on a 20×20 image → nothing changes, no crash.

---

## `TestPrescanTrajectory` (3 tests)

`prescan_trajectory(mp4, jsonl, skip_null_actions=True)` is the
"metadata-only" pass: parse the entire JSONL, produce arrays of
`(valid_frame_indices, packed_actions)`, without decoding any video.
This runs on dataset `__init__` for every trajectory so the scan has to
be fast and correct.

### `test_returns_valid_indices_and_actions`
Using `vpt_recording` (32 frames, odd-indexed are non-null): asserts
`len(valid_indices) == 16`, `actions.shape == (16, 21)`, dtypes are
`int32` (indices) and `int16` (actions). Small integer dtypes matter
because in-memory storage for tens of thousands of trajectories adds up.

### `test_no_skip_keeps_all_frames`
Same fixture, `skip_null_actions=False` → all 32 frames kept. Confirms
the bypass flag works — useful for debugging distributions.

### `test_null_only_returns_empty`
`null_only_recording` (8 frames, all null) → `len(valid_indices) == 0`
and `actions.shape == (0, 21)`. The empty-but-correctly-shaped actions
array matters because downstream concat operations assume a 2D shape.

---

## `TestLoadTrajectory` (3 tests)

`load_trajectory(mp4, jsonl)` is the full decode path — prescans, then
reads the valid frames out of the MP4, pads/resizes, and returns
`(frames, actions, rewards)`.

### `test_shapes_and_dtypes`
From `vpt_recording`:
- `frames.dtype == np.uint8`, shape `(16, 384, 640, 3)` — 16 because of
  the null-skip, 384 because of the pad, 3 because RGB.
- `actions.dtype == np.int64`, shape `(16, 21)`.
- `rewards.dtype == np.float32`, shape `(16,)`.
- `len(frames) == len(actions) == len(rewards)` — the "per-frame
  alignment" invariant; if this ever drifts by one, the agent learns to
  predict next-step instead of same-step actions.

### `test_no_skip_has_more_frames`
Compares `skip_null_actions=True` to `False` and asserts the latter
returns strictly more frames. A sanity check that `skip_null_actions` is
actually wired through.

### `test_raises_for_missing_video`
Write empty `.mp4` + empty `.jsonl`, expect `RuntimeError`. A **fail-loud**
requirement: a corrupt or empty video file must not silently return an
empty tensor, because training on "nothing" would waste hours before
anyone noticed the loss plateaued.

---

## `TestMinecraftVPTDataset` (4 tests)

The user-facing `torch.utils.data.Dataset` class — this is what the
trainer actually sees.

### `test_lengths_and_item_shapes`
Instantiate with `seq_len=4, stride=4` on `vpt_data_dir` (3 recordings).
Assert `len(ds) > 0`; take `ds[0]` and verify the sample dict:

| Key | Shape | Dtype | Range |
|---|---|---|---|
| `video` | `(3, 4, 384, 640)` | `float32` | `[0, 1]` (normalized) |
| `discrete_actions` | `(4, 21)` | `int64` | — |
| `rewards` | `(4,)` | — | — |

The `video.min()/max()` range check is important — if normalization was
skipped, frames would be in `[0, 255]` and the tokenizer's MAE loss
would explode in the first step.

### `test_short_trajectory_excluded`
With `vpt_short_recording` (6 frames, 3 valid) and `seq_len=16`, the
dataset has `len(ds) == 0`. If a trajectory can't yield even one clip, it
should be dropped rather than cause `__getitem__` to crash at training
time.

### `test_max_trajectories_limit`
`max_trajectories=1` on a dir with 3 recordings → exactly one trajectory
loaded. This flag is commonly used with `--max_trajectories 10` during
debugging to boot fast.

### `test_getitem_retries_on_error`
Uses `monkeypatch.setattr(mvd, "_decode_frames", flaky)` to inject a
one-shot `RuntimeError` into the MP4 decode call. Then asserts
`ds[0]` still returns a valid sample — the dataset must catch the
failure, log, and retry on a different clip index. This models real
failures: corrupt MP4s, decord/cv2 hiccups, disk read errors. If the
retry path was broken, a single bad file could kill a multi-hour
training run.

The `monkeypatch` pattern:
```python
monkeypatch.setattr(mvd, "_decode_frames", flaky)
sample = ds[0]  # original flaky call raises; retry path runs success path
assert sample["video"].shape[0] == 3
```

---

## `TestCollateMinecraftBatch` (1 test)

`collate_minecraft_batch(list_of_samples)` is passed as
`collate_fn=` to the DataLoader. Stacks a list of `B` sample dicts into
a single batched dict.

### `test_stack_shapes`
Build two identical samples with zero tensors; after collate:

| Field | Shape |
|---|---|
| `video` | `(2, 3, 4, 32, 32)` |
| `discrete_actions` | `(2, 4, 21)` |
| `rewards` | `(2, 4)` |

The leading `B=2` axis is what matters; the inner shapes are just
whatever we put in. If this function hit a typed-dict key mismatch you'd
get a `KeyError` at the first DataLoader fetch.

---

## Running this file

```bash
# everything in this file
env/bin/pytest tests/test_minecraft_dataset.py -v

# just one class
env/bin/pytest tests/test_minecraft_dataset.py::TestMuLawEncode -v

# one test
env/bin/pytest tests/test_minecraft_dataset.py::TestParseJsonlAction::test_esc_key_not_counted_as_button -v

# with coverage for the module under test
env/bin/pytest tests/test_minecraft_dataset.py \
    --cov=minecraft_vpt_dataset --cov-report=term-missing
```

Current coverage of `minecraft_vpt_dataset.py`: **82 %**. The unreached
lines are the `decord` backend branch (only used when decord is
installed; we run on cv2) and a handful of rarely-hit cv2 error paths.
