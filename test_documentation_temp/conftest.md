# `tests/conftest.py` — Shared fixtures

> **Requirements verified by tests that use these fixtures:** Req5,
> Req6, Req9, Req10. See `TRACEABILITY_MATRIX.md` for the exact
> requirement-to-test mapping.

## Purpose

`conftest.py` is automatically loaded by pytest before any test file in the
`tests/` directory. It has two jobs for this suite:

1. **Make imports work.** Insert the repo root and `Video-Pre-Training/` into
   `sys.path` so every test file can `import minecraft_vpt_dataset`,
   `import train_dreamer4_minecraft`, and
   `import evaluate_dreamer4_minecraft` without any per-file path hacks.
2. **Generate synthetic VPT data on demand.** Provide fixtures that write
   tiny `.mp4` + `.jsonl` pairs that mimic the schema of real VPT recordings.
   This lets I/O and integration tests run without the ~200 GB VPT dataset
   on disk; everything is written into `tmp_path` and cleaned up by pytest
   automatically.

> The real VPT recordings are 640×360 @ 20 fps with a side-car `.jsonl` log
> listing per-frame mouse/keyboard/hotbar/GUI state. The synthetic fixtures
> here produce the **same schema** at tiny sizes (≤ 32 frames), so code
> paths that parse JSONL, decode MP4, or pair them up all execute against
> realistic-shaped inputs.

---

## Path setup (module top)

```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "Video-Pre-Training"))
```

- `_REPO_ROOT` resolves to `/home/jackf/LeDreamer-mc/` no matter where
  pytest is invoked from.
- Inserting at index 0 puts both directories at the front of `sys.path` so
  they win over any system-installed packages with clashing names.
- The `Video-Pre-Training/` entry is what allows `import evaluate_dreamer4_minecraft`
  and `import dreamer4_minecraft_agent` to work — neither module is part
  of a formal Python package; they're bare scripts living inside the VPT
  subtree.

This duplicates what `pyproject.toml`'s `[tool.pytest.ini_options].pythonpath`
does, but belt-and-braces: if someone runs `pytest` without using the
project's config (e.g. from inside an IDE's test runner), the sys.path
munging still kicks in.

---

## The helper: `_write_vpt_recording(...)`

```python
def _write_vpt_recording(
    dir_path: Path,
    stem: str,
    n_frames: int,
    width: int = 640,
    height: int = 360,
    seed: int = 0,
) -> tuple[Path, Path]:
```

Generates a single recording pair at `{dir_path}/{stem}.mp4` +
`{dir_path}/{stem}.jsonl` and returns the two paths.

### Video half

- Uses OpenCV's `cv2.VideoWriter` with the `mp4v` FourCC at 20 fps and the
  default VPT resolution 640×360.
- Each frame is random noise from a seeded `np.random.default_rng(seed)`,
  so recordings with different seeds don't look identical (important for
  sanity — if two recordings collapsed to the same bytes, decoding tests
  could pass trivially).
- The `try / finally writer.release()` ensures the MP4 trailer gets written
  even if frame generation raises — without it, you can end up with
  truncated files that aren't decodable.

### Actions half (JSONL)

One record per frame; `is_non_null = (i % 2 == 1)` — so **odd-indexed frames
get real actions, even-indexed frames are null**. The null-skip filter in
`minecraft_vpt_dataset.prescan_trajectory` throws away the null ones, so
you can do the arithmetic: a 32-frame fixture yields exactly 16 "valid"
frames once the filter runs. Several dataset tests rely on this 16.

Per-frame record fields match real VPT JSONL output:

| Field | Non-null value | Null value | Purpose |
|---|---|---|---|
| `mouse.dx` | `5.0` | `0.0` | Yaw delta in pixels (scaled by 360/2400 downstream) |
| `mouse.dy` | `-3.0` | `0.0` | Pitch delta in pixels |
| `mouse.buttons` | `[0]` (attack) | `[]` | 0=attack, 1=use, 2=pickItem |
| `keyboard.keys` | `["key.keyboard.w"]` | `[]` | "w" maps to the `forward` button |
| `hotbar` | `0` | `0` | Currently-selected hotbar slot 0-8 |
| `tick` | `i` | `i` | Monotonic tick index |
| `isGuiOpen` | `False` | `False` | Inventory/crafting screen open? |

The values are small but non-trivial so the parser has something to map.
With `dx=5, dy=-3`, the pitch/yaw camera vector becomes
`(dy * 360/2400, dx * 360/2400) = (-0.45°, 0.75°)`. Those are well inside
`CAMERA_MAXVAL=10` so they don't get clipped — they discretize to a
near-center bin.

---

## Fixtures

All fixtures are function-scoped (default), so each test gets a fresh
`tmp_path` and a fresh set of generated files.

### `vpt_recording` — the common case

```python
@pytest.fixture
def vpt_recording(tmp_path):
    return _write_vpt_recording(tmp_path, "test-rec-0", n_frames=32)
```

One 32-frame recording. 16 valid frames after null-filter. Returns
`(mp4_path, jsonl_path)` as a tuple — tests unpack with
`mp4, jsonl = vpt_recording`.

**Used by:** most `TestPrescanTrajectory`, `TestLoadTrajectory` tests in
`test_minecraft_dataset.py`.

### `vpt_data_dir` — a directory of recordings

```python
@pytest.fixture
def vpt_data_dir(tmp_path):
    for i in range(3):
        _write_vpt_recording(tmp_path, f"rec-{i}", n_frames=32, seed=i)
    return tmp_path
```

Three recordings with different seeds. Returns the containing directory,
ready to pass as `data_dir=...` to `MinecraftVPTDataset`. The dataset
globs for `*.mp4` + matching `*.jsonl`, so three pairs → three
trajectories.

**Used by:** `TestMinecraftVPTDataset` in `test_minecraft_dataset.py` and
every test in `test_minecraft_integration.py`.

### `vpt_short_recording` — deliberately too short

```python
@pytest.fixture
def vpt_short_recording(tmp_path):
    return _write_vpt_recording(tmp_path, "short", n_frames=6)
```

6 frames. After null-filter you get 3 valid frames — fewer than any
reasonable `seq_len`. Used to verify that the dataset correctly **drops**
trajectories it can't carve a single clip out of.

**Used by:** `test_short_trajectory_excluded` in
`test_minecraft_dataset.py`.

### `vpt_gui_recording` — exercises cursor overlay

```python
@pytest.fixture
def vpt_gui_recording(tmp_path):
```

10 frames with `isGuiOpen=True` starting at frame 5. When the GUI is
open the dataset code overlays a synthetic cursor at the mouse `(x,y)`
position (`_composite_cursor`). This fixture sweeps the cursor `x` from
100 to 145 so a test reading these frames can verify the overlay
activated on the right half of the recording.

**Note:** no test currently calls this fixture directly — it's provided
for future cursor-overlay tests. Leaving it here means a future test
author doesn't have to re-derive the schema.

### `null_only_recording` — all frames null

```python
@pytest.fixture
def null_only_recording(tmp_path):
```

8 frames, every record empty (`keys=[]`, `buttons=[]`, `dx=dy=0`). Used
to prove that `prescan_trajectory` correctly returns
**zero valid indices** and an empty actions array of the correct shape
(`(0, N_BUTTONS+1)` rather than failing outright or returning `(0,)`).

**Used by:** `test_null_only_returns_empty` in
`test_minecraft_dataset.py`.

---

## Why synthetic data rather than "use a checked-in tiny real clip"?

1. **Reproducibility.** RNG-seeded, text-only fixture definition — no
   binary blobs in git.
2. **Flexibility.** New edge-case fixtures (all-null, GUI-open, short) are
   one function call each.
3. **Speed.** Writing + decoding 32 frames at 640×360 with cv2 is <100 ms
   per fixture on any machine.
4. **Avoids licensing questions** about shipping subsets of the VPT
   dataset.

The trade-off: synthetic frames are random noise, so tokenizer/dynamics
tests can't meaningfully assert on reconstruction quality — but that's
not what those tests are trying to verify anyway. They check shapes,
gradient flow, and scalar-ness of the loss, which are all content-agnostic.

---

## How to add a new fixture

1. Write a helper (or extend `_write_vpt_recording`) that produces the
   specific edge case you need.
2. Add a `@pytest.fixture` that calls it with `tmp_path`.
3. In your test, take `your_fixture` as a parameter — pytest injects it.
4. If you need the fixture in multiple modules, keep it here; if only one
   file needs it, it's fine to define it inline in that file.

## How to add a new data-layer test that uses fixtures

Request one of these fixtures by name in the test signature:

```python
def test_my_new_thing(vpt_data_dir):
    ds = MinecraftVPTDataset(data_dir=str(vpt_data_dir), ...)
    ...
```

Pytest resolves the name, runs the fixture, and passes the return value.
No manual setup/teardown needed.
