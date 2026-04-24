# `tests/test_minecraft_evaluate.py` — Evaluation helpers

> **Requirements verified:** Req8, Req10, Req11, Req12, Req14. See
> `TRACEABILITY_MATRIX.md` §1 for the per-test breakdown.

14 tests, runtime ~0.4 s. Tests the **data-layer helpers** inside
`Video-Pre-Training/evaluate_dreamer4_minecraft.py` — the pure-Python
functions that don't require MineRL or a live game.

## What the module does (and what we test)

`evaluate_dreamer4_minecraft.py` is the end-to-end evaluation driver:
spawn worker subprocesses, each running MineRL's HumanSurvival env with a
trained Dreamer4 agent, collect per-episode results, aggregate tech-tree
success rates, write out a JSON report.

The game-driving parts — `run_worker()`, `main()` — are multi-process
MineRL orchestration. They spawn Xvfb, start a MineRL instance manager,
load a PyTorch checkpoint, step the env thousands of times, all
inside an isolated process per worker. Testing that requires the
**eval-env** (Python 3.9 + MineRL + Java).

The data-layer pieces are pure dict/list arithmetic:

- `TECH_TREE_TASKS` — a list of 11 `(task_name, inventory_key)` pairs
  that defines which Minecraft items constitute "progress" (log →
  planks → crafting_table → … → diamond).
- `check_inventory(info, key)` — returns `True` iff the MineRL `info`
  dict reports at least one of `key` in inventory.
- `EpisodeResult` — a dataclass holding per-episode stats.
- `aggregate_results(results)` — turns a list of `EpisodeResult` into
  summary metrics (mean reward, success rate per task, mean steps-to-success).

None of those touch MineRL, so we test them here in the normal training
env.

## Imports

```python
import evaluate_dreamer4_minecraft as ev
```

Works because `conftest.py` put `Video-Pre-Training/` on `sys.path`.
The MineRL import inside `evaluate_dreamer4_minecraft.py` sits inside
`run_worker()` (i.e. inside a function body, not at module load), so
simply importing the module doesn't require MineRL.

---

## `TestTechTree` (3 tests)

`TECH_TREE_TASKS` defines the diamond-progression ladder reported in the
Dreamer4 paper. Wrong entries here would make evaluation silently
inflate or deflate success metrics.

### `test_eleven_tasks`
`len(TECH_TREE_TASKS) == 11`. Exactly 11 is what the paper reports; a
drift would fail the regression.

### `test_progression_order`
Check that later items appear strictly after their prerequisites in the
list:

- `log` before `planks` (you chop a tree before you craft planks)
- `planks` before `crafting_table`
- `wooden_pickaxe` before `stone_pickaxe`
- `iron_ingot` before `iron_pickaxe`
- `diamond` is last (the terminal goal)

This order matters for the report — some aggregation code walks the
list assuming topological order of prerequisites. If someone reshuffles
the list alphabetically, downstream analysis breaks subtly.

### `test_each_task_has_inventory_key`
Every entry is `(str, str)` with a non-empty inventory key. Catches
copy-paste mistakes like `("diamond", "")` or `("log", None)`.

---

## `TestCheckInventory` (5 tests)

`check_inventory(info, key)` is called after every MineRL step to see if
a tech-tree item was obtained. It has to be defensive because different
MineRL versions return `info["inventory"]` as either a dict or a list,
and sometimes the key is absent entirely.

### `test_dict_with_item`
Standard case: `{"inventory": {"log": 3}}, key="log"` → `True`.

### `test_dict_zero_count`
Edge case: `{"inventory": {"log": 0}}, key="log"` → `False`. Presence
alone is not enough; count must be > 0.

### `test_dict_missing_item`
`{"inventory": {"log": 1}}, key="diamond"` → `False`. Don't match on
substring, don't crash on missing keys.

### `test_missing_inventory`
`{}, key="log"` → `False`. Happens on the very first reset before any
step has been taken.

### `test_non_dict_inventory`
`{"inventory": [1, 2, 3]}, key="log"` → `False`. Some MineRL versions
return inventory as a list of item counts indexed by some enum; we don't
try to translate, just soft-fail to `False`. This keeps evaluation
robust across MineRL versions.

---

## `TestEpisodeResult` (2 tests)

`EpisodeResult` is a `@dataclass` holding one episode's stats:

```python
@dataclass
class EpisodeResult:
    episode_id: int
    worker_id: int
    total_steps: int
    total_reward: float
    wall_time_seconds: float
    tech_tree_steps: dict = field(default_factory=dict)
    tech_tree_achieved: dict = field(default_factory=dict)
```

### `test_construction_defaults`
Construct without the tech-tree dicts; assert they default to empty
`{}` — not `None`, not shared references (which `field(default_factory=dict)`
guarantees). Without `default_factory`, every instance would share a
single dict and one episode's achievements would leak into the next.

### `test_construction_with_trees`
Pass explicit `tech_tree_steps` and `tech_tree_achieved`; assert the
values round-trip through the dataclass unchanged.

---

## `TestAggregateResults` (4 tests)

`aggregate_results(list_of_EpisodeResult)` reduces many episodes to a
single summary dict. This is the "outer loop" of the eval script's
reporting step.

### The `_make_result` helper

```python
def _make_result(ep, reward, steps, achievements):
    steps_map = {name: (ep + 1) if achievements.get(name) else -1
                 for name, _ in ev.TECH_TREE_TASKS}
    achieved = {name: achievements.get(name, False)
                for name, _ in ev.TECH_TREE_TASKS}
    return ev.EpisodeResult(
        episode_id=ep,
        worker_id=0,
        total_steps=steps,
        total_reward=reward,
        wall_time_seconds=1.0,
        tech_tree_steps=steps_map,
        tech_tree_achieved=achieved,
    )
```

Produces a synthetic `EpisodeResult` from a compact spec:
- `ep` — episode index (used as fake steps-to-success value).
- `reward`, `steps` — scalar episode stats.
- `achievements` — a dict of just the tasks to mark achieved; everything
  else defaults to "not achieved."

The per-task `steps_map` uses `-1` for "not achieved" (the convention
the aggregator uses to skip those episodes when computing
`mean_steps_to_success`).

### Per-test breakdown

### `test_empty_input_returns_empty_dict`
`aggregate_results([]) == {}` — not a crash, not `None`, just an empty
dict. Important for when all eval workers time out.

### `test_counts_and_means`
Three episodes, rewards 10/20/0, steps 100/200/50:
- `n_episodes == 3`
- `mean_reward == 10.0` (pytest.approx)
- `mean_steps == 116.666...` (pytest.approx, `abs=1e-2`)

Uses `pytest.approx` because floating-point division won't be exact.

### `test_success_rate`
Four episodes, two with `log=True`:
- `stats["tasks"]["log"]["success_rate"] == 0.5`
- `stats["tasks"]["log"]["n_successes"] == 2`

The two stats encode redundant information (rate = successes / total),
but both are in the output for convenience — the test pins both so a
refactor can't silently drop one.

### `test_mean_steps_infinity_when_no_success`
Three episodes, none achieve diamond:
- `stats["tasks"]["diamond"]["mean_steps_to_success"]` is `math.inf`

Choosing `inf` over `None`/`-1`/`nan` is deliberate: JSON serialization
renders it as `Infinity`, and downstream plotting code can filter on
`>= 0 and < inf` to exclude unachieved tasks. Each of the alternatives
(`None` breaks `json.dumps` in strict mode; `-1` would mean
"unachieved-as-faster-than-any-real-value"; `nan` silently propagates
through averages) is worse.

---

## Why we can't test `run_worker()` / `main()` here

1. They `import minerl` inside the function body — MineRL requires
   Python 3.9 (training env is 3.12).
2. They spawn real subprocesses via `multiprocessing` with
   `spawn_method='spawn'`, which requires a clean re-import chain that
   our conftest's sys.path insertion doesn't fully cover.
3. They need a real PyTorch checkpoint — without one, the worker would
   exit immediately and not exercise any interesting code.

`TASK_SUMMARY_minecraft-tests.md` flags this explicitly in the "Known
limitations" section. If someone wants to add coverage there, the
approach is to run the existing tests from inside the eval-env
(`bash scripts/setup_eval_venv.sh`) and add a fixture that produces a
tiny eval-only checkpoint.

## Running this file

```bash
env/bin/pytest tests/test_minecraft_evaluate.py -v

# with coverage
env/bin/pytest tests/test_minecraft_evaluate.py \
    --cov=evaluate_dreamer4_minecraft --cov-report=term-missing
```

Coverage of `evaluate_dreamer4_minecraft.py`: **21 %**. The unreached
79 % is the worker/main MineRL driver code noted above.
