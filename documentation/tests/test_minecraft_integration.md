# `tests/test_minecraft_integration.py` — Cross-module integration tests

> **Requirements verified:** Req3, Req6, Req9. See
> `TRACEABILITY_MATRIX.md` §1 for the per-test breakdown.

4 tests, runtime ~10 s. The only file in the suite that wires the
**dataset**, **tokenizer**, and **dynamics world model** together and
runs a real forward + backward pass.

## Why this file exists

All the other test files unit-test their module in isolation. But the
thing that actually matters is: **can a batch produced by
`MinecraftVPTDataset` + `collate_minecraft_batch` be fed into
`DynamicsWorldModel.forward()` without error?**

That composition is what phase 2 training does every step. If any shape
or dtype drifts at the seam — the dataset outputs `int64` actions but
the dynamics model expects `int32`, or the image size doesn't match the
tokenizer's patch grid — nothing in the unit tests catches it. The
integration tests do.

They also serve as the canonical "how to use these classes together"
example: a reader unfamiliar with the codebase can look at
`test_forward_pass_loss_is_scalar` and see the minimum required
arguments.

## Module-level marker

```python
pytestmark = pytest.mark.integration
```

Tags every test in this file with `@pytest.mark.integration`, so you can
filter with:

```bash
pytest -m integration       # run only these
pytest -m "not integration"  # skip them
```

The marker is registered in `pyproject.toml`'s
`[tool.pytest.ini_options].markers` list so pytest doesn't warn about an
unknown marker.

## Imports

```python
import pytest
import torch
from torch.utils.data import DataLoader

from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer
from minecraft_vpt_dataset import (
    DREAMER4_NUM_DISCRETE_ACTIONS,
    MinecraftVPTDataset,
    collate_minecraft_batch,
)
```

Note: **no MineRL**. The dataset / tokenizer / dynamics triple lives
entirely in training env.

---

## Tiny-model factories

To keep these tests fast on CPU, we instantiate the tokenizer and
dynamics model at the smallest configurations that still exercise the
code paths.

### `_tiny_tokenizer(image_h=64, image_w=64)`

```python
VideoTokenizer(
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
```

Design choices:
- `dim=16` — width of transformer features. Real model uses 768; we use
  16 so every matmul is tiny.
- `patch_size=32` on a 64×64 image → 2×2 spatial grid = 4 spatial
  patches per frame. The minimum that exercises "more than one patch."
- `encoder_depth=1, decoder_depth=1` — one transformer block each.
- `time_block_every=1` — a time-attention block on every layer (so the
  temporal dimension of the input matters).
- `attn_heads=2, attn_dim_head=8` — `heads × dim_head = 16` matches
  `dim`, as the code expects.
- `num_latent_tokens=4` — tiny latent budget, enough to see the MAE
  structure.

### `_tiny_dynamics(tokenizer)`

```python
DynamicsWorldModel(
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
```

The **important** argument: `num_discrete_actions=DREAMER4_NUM_DISCRETE_ACTIONS`.
This is the contract with the dataset — if either side's shape drifts,
the forward pass will either crash or silently encode garbage.

Other choices:
- `max_steps=8` (power of 2, required for shortcut consistency).
- `num_continuous_actions=0` — Minecraft has no continuous actions in
  the Dreamer4-VPT mapping; camera was discretized upstream.
- `pred_orig_latent=True` — matches the training config.
- `prob_shortcut_train=0.9` — high probability so the shortcut branch
  runs most calls (testing the common path).

---

## The four tests

### `test_dataloader_collate_shapes`

The **cheapest** integration check — just run the dataset through a
DataLoader and inspect batch shapes.

```python
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
```

`num_workers=0` is important — multi-worker DataLoaders fork the
process, which doesn't play well with pytest's test isolation on CI.

`image_height=image_width=64` overrides the dataset default (384×640).
This is *only* used for fast CPU-size tests — production training uses
the default 384×640.

### `test_forward_pass_loss_is_scalar`

The flagship test. End-to-end: build dataset → build tiny tokenizer and
dynamics → forward pass → assert loss is a finite scalar.

```python
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
```

What this proves:
1. The batched tensors from `collate_minecraft_batch` have exactly the
   keys `DynamicsWorldModel.forward()` expects.
2. The shapes match the declared `num_discrete_actions`.
3. The internal tokenization + flow-matching forward doesn't produce
   NaNs at init.

If either contract broke, you'd get a shape mismatch (crash) or a
non-finite loss (silent — which is why we explicitly check
`torch.isfinite`).

### `test_gradients_flow`

```python
loss.backward()
n_with_grad = sum(
    1 for p in dynamics.parameters()
    if p.grad is not None and p.grad.abs().sum() > 0
)
assert n_with_grad > 0, "no parameters received gradient"
```

Runs a backward pass through the same tiny-model setup and counts
parameters with nonzero gradients. The bar is deliberately low —
"at least one parameter has a gradient" — because a fully-broken graph
(e.g. the loss is computed from a detached tensor) would result in
**zero** parameters having gradients.

A stricter check would be "ratio > X" but with random init at tiny size
some legit paths (e.g. infrequently-hit modality embeddings) might not
receive gradient on a single batch. Keeping the threshold at "> 0" is
the right balance between "useful signal" and "not flaky."

### `test_generate_emits_valid_action_shape`

```python
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
assert discrete_actions[..., :20].max() <= 1
assert discrete_actions[..., :20].min() >= 0
assert discrete_actions[..., 20].max() <= 120
assert discrete_actions[..., 20].min() >= 0
```

Closes the loop on the **output side** of the action contract: when the
dynamics model *generates* (rolls forward in dream), the actions it
emits must be shape-compatible with what the dataset produced.

- Shape `(B=1, T=3, 21)` — same layout as `batch["discrete_actions"]`.
- The first 20 slots are button bits, each ∈ {0, 1}.
- Slot 20 is the camera joint index, ∈ [0, 120].

If generation ever emitted e.g. a `(B, T, 2)` tensor (just pitch/yaw),
the agent would crash at inference time when trying to index into
`BUTTONS_ALL`. Catching it here is an order of magnitude faster than at
eval time.

No fixtures needed — generation starts from a learned prior, not from
real data, so this test works in isolation.

---

## Why only four tests?

Integration tests have a cost (runtime) and a benefit
(cross-module bug detection). Beyond these four, additional integration
tests would mostly re-prove things the unit tests already cover:

- Gradient values — already checked in `tests/test_dreamer.py`.
- Per-module shapes — already unit-tested.
- Action-space constants — already pinned in multiple files.

The four here are the **minimal spanning set** over the critical
contracts:
1. Dataset → DataLoader → batch (test 1).
2. Batch → Dynamics.forward (test 2).
3. Backward graph integrity (test 3).
4. Generation output shape (test 4).

Any breakage in the dataset↔model coupling fires at least one of them.

## Running this file

```bash
env/bin/pytest tests/test_minecraft_integration.py -v
# 4 passed in ~10s
```

Just the integration marker:
```bash
env/bin/pytest tests/ -m integration -v
```

Or skip them during a quick unit-only check:
```bash
env/bin/pytest tests/ -m "not integration" -v
```

## Gotchas / things that could make this flaky

1. **CPU determinism.** We don't set `torch.manual_seed(...)` at the
   top of each test. If generation ever produces NaN for certain
   random inits, these tests would start flaking. If that happens, add
   a seed in each test body — the goal isn't determinism of values, just
   determinism of "does this pass."
2. **PyTorch version.** `DynamicsWorldModel.generate()` uses some
   newer attention kernels; if the pinned PyTorch version is ever
   downgraded, generation could fail. Pin-check in CI would help.
3. **Tiny model sizes ≠ production.** Shapes in test_dreamer.py's
   parametrized suite cover more combos; this file only checks one
   tiny config. A bug that only manifests at large sizes wouldn't be
   caught here.
