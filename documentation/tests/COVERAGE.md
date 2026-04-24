# Code Coverage

This file covers the "code coverage" leg of verification
(`instructions.md` lines 7–18). The "requirements coverage" leg is
`TRACEABILITY_MATRIX.md`.

## 1. Required coverage modes

Per `instructions.md`, we track three code-coverage numbers
separately so a regression in any one of them is visible:

1. **Automated tests only** — `pytest tests/ --cov=<module>`.
2. **Manual / integration tests only** — run manual tests from
   `MANUAL_TESTS.md` under `coverage run`, then `coverage report`.
3. **Combined** — union of the two coverage datasets (`coverage
   combine`).

## 2. Tooling

We use the [`coverage`](https://coverage.readthedocs.io/en/latest/cmd.html)
package, driven through `pytest-cov` for automated runs and raw
`coverage run` for manual ones.

Install:

```bash
pip install coverage pytest-cov
```

The config lives in `pyproject.toml` under `[tool.coverage.*]`. Adjust
`source =` there if you add a new top-level module that needs tracking.

## 3. Automated-only coverage

```bash
# Single-shot, terminal report
env/bin/pytest tests/ \
    --cov=minecraft_vpt_dataset \
    --cov=train_dreamer4_minecraft \
    --cov=evaluate_dreamer4_minecraft \
    --cov=dreamer4_minecraft_agent \
    --cov-report=term-missing

# HTML report — open htmlcov/index.html
env/bin/pytest tests/ \
    --cov=minecraft_vpt_dataset \
    --cov=train_dreamer4_minecraft \
    --cov=evaluate_dreamer4_minecraft \
    --cov-report=html
```

Current numbers (as of 2026-04-23, training env, CPU):

| Module | Auto coverage | Notes |
|---|---:|---|
| `minecraft_vpt_dataset.py` | 82 % | Decord backend branch unexecuted (cv2 preferred). |
| `train_dreamer4_minecraft.py` | 53 % | Phase function bodies skipped intentionally (need GPU + real data). |
| `evaluate_dreamer4_minecraft.py` | 21 % | Worker / main MineRL driver excluded — requires eval-env. |
| `dreamer4_minecraft_agent.py` | n/a in training env | Import gated behind MineRL; see eval-env runs. |

## 4. Manual-only coverage

Manual tests (`MT-1`..`MT-14`) exercise code paths that pytest cannot
touch without GPU or MineRL. To record their coverage:

```bash
# Each manual test is a shell invocation — prefix its python command
# with `coverage run --append --source=<modules> -m`, e.g. for MT-6:
coverage run --append \
    --source=minecraft_vpt_dataset,train_dreamer4_minecraft \
    train_dreamer4_minecraft.py \
    --phase 2 --data_dir ./download/data \
    --output_dir /tmp/ledreamer-mt6 \
    --tokenizer_ckpt ./checkpoints/tokenizer.pt \
    --num_steps 5 --batch_size 1 --allow_cpu

# After running every MT test that applies (not all of them run
# python scripts directly — see below), produce the report:
coverage report -m
coverage html -d htmlcov-manual
```

For MT tests that run **eval-env** code, switch the virtualenv first:

```bash
source eval-env/bin/activate
coverage run --append \
    --source=dreamer4_minecraft_agent,evaluate_dreamer4_minecraft \
    Video-Pre-Training/evaluate_dreamer4_minecraft.py \
    --checkpoint ./checkpoints/dreamer4_minecraft.pt \
    --n_episodes 1 --n_workers 1
```

The `--append` flag is crucial: without it each `coverage run` clobbers
the `.coverage` data file and only the last MT test's coverage survives.

### Which MT tests produce Python-side coverage?

| MT test | Runs Python? | Coverage target module |
|---|---|---|
| MT-1  | Yes | `train_dreamer4_minecraft` |
| MT-2  | Yes | `train_dreamer4_minecraft` |
| MT-3  | Yes (eval-env) | `dreamer4_minecraft_agent`, `minecraft_vpt_dataset` |
| MT-4  | Yes (eval-env) | `evaluate_dreamer4_minecraft` |
| MT-5  | Yes | `train_dreamer4_minecraft` |
| MT-6  | Yes | `train_dreamer4_minecraft`, `minecraft_vpt_dataset`, `dreamer4.dreamer4` |
| MT-7  | Yes (same run as MT-6) | `train_dreamer4_minecraft` |
| MT-8  | Yes (eval-env) | `evaluate_dreamer4_minecraft` |
| MT-9  | Yes (eval-env) | `dreamer4_minecraft_agent` |
| MT-10 | Yes (eval-env) | `evaluate_dreamer4_minecraft` |
| MT-11 | No (JSON inspection only) | — |
| MT-12 | No (stdout inspection) | — |
| MT-13 | Yes (when implemented) | `train_dreamer4_minecraft` |
| MT-14 | Yes (when implemented) | `evaluate_dreamer4_minecraft` |

## 5. Combined coverage

```bash
# 1. Run pytest under coverage into a uniquely-named datafile
COVERAGE_FILE=.coverage.auto \
    env/bin/pytest tests/ --cov=minecraft_vpt_dataset \
                          --cov=train_dreamer4_minecraft \
                          --cov=evaluate_dreamer4_minecraft

# 2. Run each manual test under coverage, each into its own datafile
COVERAGE_FILE=.coverage.manual \
    coverage run --append --source=minecraft_vpt_dataset,... <MT command>
# ...repeat for each MT test you want to include...

# 3. Combine
coverage combine .coverage.auto .coverage.manual
coverage report -m
coverage html -d htmlcov-combined
```

In `pyproject.toml`'s `[tool.coverage.paths]` the `source` mapping
should list the absolute repo root so `combine` can unify paths across
the two venvs.

## 6. Storing the reports

Keep the three `htmlcov-*` directories as release artifacts:

- `release2_artifacts/coverage/auto/index.html`
- `release2_artifacts/coverage/manual/index.html`
- `release2_artifacts/coverage/combined/index.html`

These are the reviewable artifacts for a release; record the headline
percentages in `release2_artifacts/release-notes.md`.

## 7. Targets and interpretation

- **Automated coverage ≥ 80 %** for `minecraft_vpt_dataset.py` — this
  is the module with the richest unit-test surface; a drop below 80 %
  usually means a newly added branch has no test.
- **No target** for `train_dreamer4_minecraft.py` / `evaluate_dreamer4_minecraft.py`
  **in automated mode** — the phase-function bodies and the worker loop
  are intentionally unreachable without GPU / MineRL.
- **Combined coverage ≥ 80 %** is the release-review gate. If a line is
  not touched by either automated or manual tests it is either dead
  code (remove it) or needs a test (add one).

## 8. Known exclusions

The following are intentionally excluded from coverage accounting:

- `dreamer4/` fork internals (upstream from `lucidrains/dreamer4`) —
  tested by `tests/test_dreamer.py`'s parametrized suite separately.
- `Video-Pre-Training/lib/` (VPT subtree) — upstream code, not owned.
- Any `if TYPE_CHECKING:` block.
- `except ImportError: pass` fallbacks used for optional dependencies
  (marked with `# pragma: no cover`).

Add new exclusions as `# pragma: no cover` comments rather than editing
`.coveragerc` whenever possible — they are easier to audit at
review time.
