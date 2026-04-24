# Manual / Integration Tests

Per `instructions.md` lines 30–54, each manual test contains a series of
**repeatable steps** and **expected results**. These tests complement
the automated suite documented in `TRACEABILITY_MATRIX.md` — they
exercise paths that require a GPU, a real MineRL runtime, multi-hour
wall time, or human observation of on-screen output. Each test id
(MT-1…MT-14) is referenced from the traceability matrix.

## Preconditions (common to all)

Before running any manual test, set up the environment once:

```bash
# From repo root
cd /home/jackf/LeDreamer-mc

# Training env (Python 3.10+; used by MT-1, MT-2, MT-4..MT-7, MT-10..MT-14)
python -m venv env
source env/bin/activate
pip install -e ".[test]"

# Eval env (Python 3.9 + MineRL; used by MT-3, MT-8, MT-9)
bash scripts/setup_eval_venv.sh
```

Record the tester, date, and outcome for each run in a testing
worksheet (`release2_artifacts/manual-test-log.md` or similar).

---

## MT-1 — Load a saved model by path (Req1)

**Purpose:** verify that `train_dreamer4_minecraft.py --resume_from
<path>` picks up an explicit checkpoint directory.

**Steps:**
1. Create a state-dir marker:
   ```bash
   mkdir -p /tmp/ledreamer-mt1/state-5
   touch   /tmp/ledreamer-mt1/state-5/random_states_0.pkl
   ```
2. Invoke with `--resume_from` pointing at that directory:
   ```bash
   env/bin/python train_dreamer4_minecraft.py \
       --phase 2 \
       --data_dir ./download/data \
       --output_dir /tmp/ledreamer-mt1 \
       --resume_from /tmp/ledreamer-mt1/state-5 \
       --allow_cpu --num_steps 1
   ```
3. Read the log line printed at startup that contains `resume_from`.

**Expected results:**
- The program resolves `resume_from` to
  `/tmp/ledreamer-mt1/state-5` and prints a message indicating it is
  loading that checkpoint.
- The process does **not** fail with "checkpoint not found."
- Automated equivalents: `TestResolveResumeFrom::test_direct_state_dir`.

---

## MT-2 — Resume training from `latest` (Req1)

**Purpose:** verify the `--resume_from latest` convenience path.

**Steps:**
1. Stage a state dir and a pointer file:
   ```bash
   mkdir -p /tmp/ledreamer-mt2/state-9
   touch   /tmp/ledreamer-mt2/state-9/random_states_0.pkl
   echo "state-9" > /tmp/ledreamer-mt2/latest_state.txt
   ```
2. Invoke:
   ```bash
   env/bin/python train_dreamer4_minecraft.py \
       --phase 2 --data_dir ./download/data \
       --output_dir /tmp/ledreamer-mt2 \
       --resume_from latest --allow_cpu --num_steps 1
   ```

**Expected results:**
- Startup log indicates resume from `.../state-9/`.
- Automated equivalent: `TestResolveResumeFrom::test_latest_with_pointer`.

---

## MT-3 — Launch agent against HumanSurvival env (Req2, Req3)

**Purpose:** prove a trained checkpoint can drive a live MineRL env.
Requires eval-env and a trained checkpoint (`./checkpoints/dreamer4_minecraft.pt`).

**Steps:**
1. `source eval-env/bin/activate`
2. Run:
   ```bash
   python Video-Pre-Training/dreamer4_minecraft_agent.py \
       --checkpoint ./checkpoints/dreamer4_minecraft.pt
   ```
3. Observe the first ~30 s of stdout and the rendered frame window (if
   DISPLAY is set).

**Expected results:**
- MineRL JVM starts (Xvfb/LWJGL logs visible).
- "Environment initialized" message appears.
- At least one action dict is printed per tick (forward/back/jump bits
  plus a `camera` entry in `[-10, 10]°`).
- No `MalformedActionError` or `ValueError: discrete_actions shape`.
- Automated equivalent (action-conversion only):
  `TestDreamer4ActionsToMinerl::test_button_positions_roundtrip`.

---

## MT-4 — Bounded episode termination (Req4)

**Purpose:** confirm the user-supplied run duration is honoured.

**Steps:**
1. From eval-env, run a short evaluation:
   ```bash
   python Video-Pre-Training/evaluate_dreamer4_minecraft.py \
       --checkpoint ./checkpoints/dreamer4_minecraft.pt \
       --n_episodes 2 --n_workers 1
   ```
2. Record wall-clock start and end times.

**Expected results:**
- Exactly 2 episodes run (observable in progress log).
- Process terminates cleanly with exit code `0`
  (`echo $?` after completion).
- No orphaned MineRL JVM processes left behind (`pgrep -f minerl` returns
  nothing).
- Automated equivalent: `TestCLI::test_phase3_skips_data_dir_requirement`
  covers the argparse surface but not the termination behaviour.

---

## MT-5 — User-authored CLI config reaches trainer (Req5)

**Purpose:** verify CLI-provided hyperparameters take effect.

**Steps:**
1. Launch with non-default flags on a tiny run:
   ```bash
   env/bin/python train_dreamer4_minecraft.py \
       --phase 1 --data_dir ./download/data \
       --output_dir /tmp/ledreamer-mt5 \
       --batch_size 1 --num_steps 2 \
       --mixed_precision fp16 \
       --allow_cpu
   ```
2. Inspect the first few lines of stdout for the hyperparameter echo
   that the trainer prints.

**Expected results:**
- `batch_size=1`, `num_steps=2`, `mixed_precision=fp16` are all echoed.
- Passing an invalid value (e.g. `--mixed_precision invalid`) terminates
  with a non-zero exit and an argparse error.
- Automated equivalents: `TestCLI::test_mixed_precision_choices`,
  `TestHyperparameterDefaults::*`.

---

## MT-6 — End-to-end tiny phase-2 run (Req6)

**Purpose:** smoke-check that the Dreamer V4 dynamics training loop
actually executes end-to-end, not just at tensor-shape level.

**Steps:**
1. Verify synthetic data or a small slice of VPT exists under
   `./download/data/` (at minimum one `.mp4` + `.jsonl` pair).
2. Run:
   ```bash
   env/bin/python train_dreamer4_minecraft.py \
       --phase 2 --data_dir ./download/data \
       --output_dir /tmp/ledreamer-mt6 \
       --tokenizer_ckpt ./checkpoints/tokenizer.pt \
       --num_steps 5 --batch_size 1 --allow_cpu
   ```
3. Tail the output.

**Expected results:**
- A non-NaN loss value is printed for every one of the 5 steps.
- The loss decreases or oscillates — it does not diverge to inf.
- A `state-<N>/` directory appears under `/tmp/ledreamer-mt6/` at the
  end of the run.
- Automated equivalents:
  `TestDatasetToDynamics::test_forward_pass_loss_is_scalar`,
  `TestDatasetToDynamics::test_gradients_flow`.

---

## MT-7 — Checkpoint artifacts written at end of run (Req7)

**Purpose:** confirm model weights are persisted after training.

**Steps:**
1. Run the command from MT-6 (or any complete training phase).
2. After completion:
   ```bash
   ls /tmp/ledreamer-mt6/
   ls /tmp/ledreamer-mt6/state-*
   cat /tmp/ledreamer-mt6/latest_state.txt
   ```

**Expected results:**
- At least one `state-<N>/` directory exists.
- Inside it, **both** of:
  - `model.safetensors` (weights), OR
  - `random_states_0.pkl` (RNG state) — accelerate writes at least one.
- `latest_state.txt` contains the name of the most-recent state dir
  (e.g. `state-5`).
- Automated equivalents:
  `TestResolveResumeFrom::test_state_dir_via_safetensors`,
  `TestResolveResumeFrom::test_latest_with_pointer`.

---

## MT-8 — Evaluation opens Minecraft environment (Req8)

**Purpose:** confirm `evaluate_dreamer4_minecraft.py` can initialize
MineRL at the start of an evaluation run.

**Steps:**
1. `source eval-env/bin/activate`
2. Run with one episode:
   ```bash
   python Video-Pre-Training/evaluate_dreamer4_minecraft.py \
       --checkpoint ./checkpoints/dreamer4_minecraft.pt \
       --n_episodes 1 --n_workers 1
   ```
3. Observe startup logs.

**Expected results:**
- Log line includes `Starting MineRL` or equivalent (JVM + env boot).
- No `JVMException` or `EOFError` before the first step.
- Automated equivalent:
  `TestCheckInventory::test_missing_inventory` (covers empty-info edge
  case on first reset).

---

## MT-9 — Episode-long agent drive (Req9)

**Purpose:** verify the agent keeps taking valid actions for the full
duration of an episode (not just the first few steps).

**Steps:**
1. From eval-env:
   ```bash
   python Video-Pre-Training/evaluate_dreamer4_minecraft.py \
       --checkpoint ./checkpoints/dreamer4_minecraft.pt \
       --n_episodes 1 --n_workers 1 --max_steps 2000
   ```
2. Leave the run unattended for ~10 minutes.

**Expected results:**
- Exactly 2000 steps recorded in the episode log, or the env
  terminates earlier (death/quit) — either is valid.
- No `MalformedActionError` raised anywhere during the run.
- No silent hang (`htop` shows the Python process remains active).
- Automated equivalents: the full
  `TestActionSpaceConstants`, `TestMuLawEncode`,
  `TestCameraRoundtripPurePython`, `TestDreamer4ActionsToMinerl`
  groups all pin pieces of the action contract this test exercises
  live.

---

## MT-10 — JSON eval report written to `--output` (Req10)

**Purpose:** verify the evaluation driver records per-episode data in
the user-specified location.

**Steps:**
1. ```bash
   python Video-Pre-Training/evaluate_dreamer4_minecraft.py \
       --checkpoint ./checkpoints/dreamer4_minecraft.pt \
       --n_episodes 3 --n_workers 1 \
       --output /tmp/ledreamer-mt10/results.json
   ```
2. After completion:
   ```bash
   python -c "import json; d=json.load(open('/tmp/ledreamer-mt10/results.json')); print(d.keys())"
   ```

**Expected results:**
- File exists at the specified path.
- Parses as valid JSON.
- Contains at least `n_episodes`, `mean_reward`, `tasks` keys.
- Automated equivalents:
  `TestEpisodeResult::*`, `TestMinecraftVPTDataset::test_lengths_and_item_shapes`.

---

## MT-11 — Final metrics JSON content inspection (Req11)

**Purpose:** verify the numeric content of the final metrics report.

**Steps:**
1. Use the JSON produced in MT-10.
2. ```bash
   python -c "
   import json, math
   d = json.load(open('/tmp/ledreamer-mt10/results.json'))
   assert d['n_episodes'] == 3
   assert isinstance(d['mean_reward'], (int, float))
   for task, stats in d['tasks'].items():
       assert 'success_rate' in stats
       assert 'n_successes' in stats
       assert 'mean_steps_to_success' in stats
   print('OK')
   "
   ```

**Expected results:**
- Script prints `OK`.
- `mean_steps_to_success` is either a non-negative float or
  `Infinity` (for tasks never achieved).
- Automated equivalents: the full `TestAggregateResults` and
  `TestTechTree` suites.

---

## MT-12 — Display metrics summary to stdout (Req12)

**Purpose:** confirm a human-readable summary prints after evaluation.

**Steps:**
1. From MT-10's command, capture stdout:
   ```bash
   ... > /tmp/ledreamer-mt12/stdout.log 2>&1
   less /tmp/ledreamer-mt12/stdout.log
   ```
2. Scroll to the end of the log.

**Expected results:**
- A table or formatted block summarizing at minimum `mean_reward`,
  `n_episodes`, and per-task `success_rate` appears near EOF.
- No raw Python `repr(dict)` dumps — output is intended for human
  reading.
- Automated equivalent (data-layer robustness):
  `TestCheckInventory::*`.

---

## MT-13 — Live metrics stream during training (Req13, optional)

**Purpose:** confirm the optional live-metrics path, when enabled,
produces periodic output during a long run.

**Steps:**
1. If the codebase grows a live-metrics flag (e.g.
   `--log_interval 10`), enable it:
   ```bash
   env/bin/python train_dreamer4_minecraft.py \
       --phase 2 --data_dir ./download/data \
       --output_dir /tmp/ledreamer-mt13 \
       --num_steps 50 --log_interval 10 --allow_cpu
   ```
2. Watch stdout during the run.

**Expected results:**
- A metrics line is printed every ~10 steps — not only at the end.
- When the flag is *not* provided, no per-step metrics appear (only
  summary at end). This confirms the feature is opt-in.
- Automated equivalent (flag-parse pattern):
  `TestCLI::test_use_lewm_flag_parses` is the closest analogue.

*Note:* if the live-metrics flag does not yet exist, mark this test
**N/A — feature pending**; Req13 is explicitly optional in
`instructions.md` line 69.

---

## MT-14 — User selects metric subset (Req14, optional)

**Purpose:** confirm the user can filter which metrics are reported.

**Steps:**
1. Invoke evaluation restricting metrics (if flag exists):
   ```bash
   python Video-Pre-Training/evaluate_dreamer4_minecraft.py \
       --checkpoint ./checkpoints/dreamer4_minecraft.pt \
       --n_episodes 1 --n_workers 1 \
       --metrics log,planks,diamond \
       --output /tmp/ledreamer-mt14/results.json
   ```
2. Inspect the output JSON:
   ```bash
   python -c "import json; d=json.load(open('/tmp/ledreamer-mt14/results.json')); print(sorted(d['tasks'].keys()))"
   ```

**Expected results:**
- Only the three requested tasks appear under `tasks`, not all 11.
- If the flag is absent, the output contains every
  `TECH_TREE_TASKS` entry.
- Automated equivalent:
  `TestTechTree::test_each_task_has_inventory_key` confirms the
  metric registry is well-formed.

*Note:* if the filter flag does not yet exist, mark this test
**N/A — feature pending**; Req14 is explicitly optional in
`instructions.md` line 70.

---

## Test log template

For each manual run, record:

```
Test id:           MT-<n>
Requirement:       Req<n>
Tester:            <name>
Date:              YYYY-MM-DD
Env:               training | eval
Checkpoint used:   <path or N/A>
Outcome:           PASS | FAIL | N/A — feature pending
Notes:             <anything surprising — include stderr excerpts>
```

Keep the log under version control (`release2_artifacts/` is a good
home) so release review has a traceable history.
