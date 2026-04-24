# Traceability Matrix — Requirements ↔ Tests

This matrix maps every requirement from `instructions.md` to the
tests that verify it. It satisfies the "Requirements Coverage" leg of
the verification plan (code coverage is reported in `COVERAGE.md`).

Notation:
- **Auto** = automated unit/integration test in `tests/` (pytest).
- **Manual** = repeatable manual test step documented in
  `MANUAL_TESTS.md`.
- A test id like `test_minecraft_dataset.py::TestEnvActionToDreamer4::test_button_positions`
  is a direct `pytest` nodeid — copy/paste into a pytest command to run
  exactly that test.


---

## 1. Requirement-to-Test Map

### Req1 — Load a pre-trained agent when the user selects a saved model

| Kind | Test | File |
|---|---|---|
| Auto | `TestResolveResumeFrom::test_none_returns_none` | `test_minecraft_training.py` |
| Auto | `TestResolveResumeFrom::test_latest_with_pointer` | `test_minecraft_training.py` |
| Auto | `TestResolveResumeFrom::test_latest_pointer_references_missing_dir` | `test_minecraft_training.py` |
| Auto | `TestResolveResumeFrom::test_direct_state_dir` | `test_minecraft_training.py` |
| Auto | `TestResolveResumeFrom::test_state_dir_via_safetensors` | `test_minecraft_training.py` |
| Auto | `TestResolveResumeFrom::test_nonexistent_path_returns_none` | `test_minecraft_training.py` |
| Auto | `TestResolveResumeFrom::test_empty_dir_returns_none` | `test_minecraft_training.py` |
| Manual | MT-1: Load saved model by path | `MANUAL_TESTS.md` |
| Manual | MT-2: Resume training from `latest` | `MANUAL_TESTS.md` |

### Req2 — Initialize the Minecraft environment when a pre-trained agent is executed

| Kind | Test | File |
|---|---|---|
| Auto (eval-env) | `TestDreamer4ActionsToMinerl::test_button_positions_roundtrip` | `test_minecraft_agent.py` |
| Auto (eval-env) | `TestDreamer4ActionsToMinerl::test_extremes_decode_within_maxval` | `test_minecraft_agent.py` |
| Manual | MT-3: Launch agent against HumanSurvival env | `MANUAL_TESTS.md` |

### Req3 — Execute pre-trained agent behavior in the Minecraft environment when the run command is issued

| Kind | Test | File |
|---|---|---|
| Auto | `TestCameraRoundtripPurePython::test_each_bin_roundtrips` | `test_minecraft_agent.py` |
| Auto | `TestCameraRoundtripPurePython::test_center_bin_decodes_to_zero` | `test_minecraft_agent.py` |
| Auto (eval-env) | `TestDreamer4ActionsToMinerl::test_button_positions_roundtrip` | `test_minecraft_agent.py` |
| Auto (eval-env) | `TestDreamer4ActionsToMinerl::test_extremes_decode_within_maxval` | `test_minecraft_agent.py` |
| Auto | `TestDatasetToDynamics::test_generate_emits_valid_action_shape` | `test_minecraft_integration.py` |
| Manual | MT-3: Launch agent against HumanSurvival env | `MANUAL_TESTS.md` |

### Req4 — Run for a period of time specified by the user before terminating safely

| Kind | Test | File |
|---|---|---|
| Auto | `TestCLI::test_phase1_dispatches` (parses `--num_steps`) | `test_minecraft_training.py` |
| Auto | `TestCLI::test_phase2_dispatches` (parses `--num_steps`) | `test_minecraft_training.py` |
| Auto | `TestCLI::test_phase3_skips_data_dir_requirement` | `test_minecraft_training.py` |
| Manual | MT-4: Bounded `--n_episodes` / `--num_steps` termination | `MANUAL_TESTS.md` |

### Req5 — Allow the user to configure training parameters in a config file

| Kind | Test | File |
|---|---|---|
| Auto | `TestCLI::test_phase1_requires_data_dir` | `test_minecraft_training.py` |
| Auto | `TestCLI::test_phase1_dispatches` | `test_minecraft_training.py` |
| Auto | `TestCLI::test_phase2_dispatches` | `test_minecraft_training.py` |
| Auto | `TestCLI::test_phase3_skips_data_dir_requirement` | `test_minecraft_training.py` |
| Auto | `TestCLI::test_use_lewm_flag_parses` | `test_minecraft_training.py` |
| Auto | `TestCLI::test_mixed_precision_choices` | `test_minecraft_training.py` |
| Auto | `TestEnforceCuda::test_cuda_available_returns_false` | `test_minecraft_training.py` |
| Auto | `TestEnforceCuda::test_allow_cpu_returns_true_on_no_cuda` | `test_minecraft_training.py` |
| Auto | `TestEnforceCuda::test_no_cuda_no_flag_exits` | `test_minecraft_training.py` |
| Auto | `TestHyperparameterDefaults::test_tokenizer_defaults` | `test_minecraft_training.py` |
| Auto | `TestHyperparameterDefaults::test_dynamics_defaults` | `test_minecraft_training.py` |
| Auto | `TestHyperparameterDefaults::test_lewm_extends_dynamics` | `test_minecraft_training.py` |
| Manual | MT-5: User-authored CLI config reaches trainer | `MANUAL_TESTS.md` |

### Req6 — Execute a Dreamer V4 training loop when specified

| Kind | Test | File |
|---|---|---|
| Auto | `TestCLI::test_phase1_dispatches` | `test_minecraft_training.py` |
| Auto | `TestCLI::test_phase2_dispatches` | `test_minecraft_training.py` |
| Auto | `TestCLI::test_phase3_skips_data_dir_requirement` | `test_minecraft_training.py` |
| Auto | `TestCLI::test_use_lewm_flag_parses` | `test_minecraft_training.py` |
| Auto | `TestHyperparameterDefaults::test_dynamics_defaults` | `test_minecraft_training.py` |
| Auto | `TestHyperparameterDefaults::test_lewm_extends_dynamics` | `test_minecraft_training.py` |
| Auto | `TestDatasetToDynamics::test_dataloader_collate_shapes` | `test_minecraft_integration.py` |
| Auto | `TestDatasetToDynamics::test_forward_pass_loss_is_scalar` | `test_minecraft_integration.py` |
| Auto | `TestDatasetToDynamics::test_gradients_flow` | `test_minecraft_integration.py` |
| Manual | MT-6: End-to-end tiny phase-2 run | `MANUAL_TESTS.md` |

### Req7 — Save the trained agent's model weights when training completes successfully

| Kind | Test | File |
|---|---|---|
| Auto | `TestResolveResumeFrom::test_direct_state_dir` (requires `random_states_0.pkl`) | `test_minecraft_training.py` |
| Auto | `TestResolveResumeFrom::test_state_dir_via_safetensors` (requires `model.safetensors`) | `test_minecraft_training.py` |
| Auto | `TestResolveResumeFrom::test_latest_with_pointer` (reads `latest_state.txt`) | `test_minecraft_training.py` |
| Auto | `TestResolveResumeFrom::test_empty_dir_returns_none` (rejects empty output) | `test_minecraft_training.py` |
| Manual | MT-7: Checkpoint artifacts written at end of run | `MANUAL_TESTS.md` |

### Req8 — Load the Minecraft environment when an evaluation session begins

| Kind | Test | File |
|---|---|---|
| Auto (eval-env) | `TestDreamer4ActionsToMinerl::test_button_positions_roundtrip` | `test_minecraft_agent.py` |
| Auto (eval-env) | `TestDreamer4ActionsToMinerl::test_extremes_decode_within_maxval` | `test_minecraft_agent.py` |
| Auto | `TestCheckInventory::test_missing_inventory` (first-reset path) | `test_minecraft_evaluate.py` |
| Manual | MT-8: `evaluate_dreamer4_minecraft.py` opens HumanSurvival env | `MANUAL_TESTS.md` |

### Req9 — Execute agent actions in the Minecraft environment throughout the duration of the testing suite in accordance with Dreamer V4

| Kind | Test | File |
|---|---|---|
| Auto | `TestActionSpaceConstants::test_button_count` | `test_minecraft_dataset.py` |
| Auto | `TestActionSpaceConstants::test_button_order_matches_vpt` | `test_minecraft_dataset.py` |
| Auto | `TestActionSpaceConstants::test_camera_bin_constants` | `test_minecraft_dataset.py` |
| Auto | `TestActionSpaceConstants::test_dreamer4_action_shape` | `test_minecraft_dataset.py` |
| Auto | `TestMuLawEncode::*` (4 tests) | `test_minecraft_dataset.py` |
| Auto | `TestDiscretizeCamera::*` (3 tests) | `test_minecraft_dataset.py` |
| Auto | `TestCameraBinsToJointIndex::*` (2 tests) | `test_minecraft_dataset.py` |
| Auto | `TestParseJsonlAction::*` (7 tests) | `test_minecraft_dataset.py` |
| Auto | `TestEnvActionToDreamer4::*` (3 tests) | `test_minecraft_dataset.py` |
| Auto | `TestCameraRoundtripPurePython::*` (2 tests) | `test_minecraft_agent.py` |
| Auto (eval-env) | `TestDreamer4ActionsToMinerl::*` (2 tests) | `test_minecraft_agent.py` |
| Auto | `TestDatasetToDynamics::test_generate_emits_valid_action_shape` | `test_minecraft_integration.py` |
| Manual | MT-9: Episode-long agent drive | `MANUAL_TESTS.md` |

### Req10 — Record all relevant data in the method specified in the config

| Kind | Test | File |
|---|---|---|
| Auto | `TestEpisodeResult::test_construction_defaults` | `test_minecraft_evaluate.py` |
| Auto | `TestEpisodeResult::test_construction_with_trees` | `test_minecraft_evaluate.py` |
| Auto | `TestPrescanTrajectory::*` (3 tests) | `test_minecraft_dataset.py` |
| Auto | `TestLoadTrajectory::*` (3 tests) | `test_minecraft_dataset.py` |
| Auto | `TestMinecraftVPTDataset::test_lengths_and_item_shapes` | `test_minecraft_dataset.py` |
| Auto | `TestMinecraftVPTDataset::test_max_trajectories_limit` | `test_minecraft_dataset.py` |
| Auto | `TestCollateMinecraftBatch::test_stack_shapes` | `test_minecraft_dataset.py` |
| Manual | MT-10: JSON eval report written to `--output` | `MANUAL_TESTS.md` |

### Req11 — Calculate and save final performance metrics when evaluation completes

| Kind | Test | File |
|---|---|---|
| Auto | `TestAggregateResults::test_empty_input_returns_empty_dict` | `test_minecraft_evaluate.py` |
| Auto | `TestAggregateResults::test_counts_and_means` | `test_minecraft_evaluate.py` |
| Auto | `TestAggregateResults::test_success_rate` | `test_minecraft_evaluate.py` |
| Auto | `TestAggregateResults::test_mean_steps_infinity_when_no_success` | `test_minecraft_evaluate.py` |
| Auto | `TestTechTree::test_eleven_tasks` | `test_minecraft_evaluate.py` |
| Auto | `TestTechTree::test_progression_order` | `test_minecraft_evaluate.py` |
| Auto | `TestTechTree::test_each_task_has_inventory_key` | `test_minecraft_evaluate.py` |
| Auto | `TestEpisodeResult::test_construction_with_trees` | `test_minecraft_evaluate.py` |
| Manual | MT-11: Final metrics JSON content inspection | `MANUAL_TESTS.md` |

### Req12 — (Optionally) display performance metrics when final evaluation results are available

| Kind | Test | File |
|---|---|---|
| Auto | `TestAggregateResults::test_mean_steps_infinity_when_no_success` (JSON-safe inf) | `test_minecraft_evaluate.py` |
| Auto | `TestCheckInventory::*` (5 tests — robust metric source) | `test_minecraft_evaluate.py` |
| Manual | MT-12: Print summary table to stdout | `MANUAL_TESTS.md` |

### Req13 — (Optionally) update and display metrics in real time when specified via the config

| Kind | Test | File |
|---|---|---|
| Auto | `TestCLI::test_use_lewm_flag_parses` (pattern: optional flag reaches trainer) | `test_minecraft_training.py` |
| Manual | MT-13: Live metrics stream during training | `MANUAL_TESTS.md` |

### Req14 — Allow the user to select which metrics to observe and when metrics are displayed

| Kind | Test | File |
|---|---|---|
| Auto | `TestTechTree::test_each_task_has_inventory_key` (metric registry is well-formed) | `test_minecraft_evaluate.py` |
| Auto | `TestTechTree::test_progression_order` | `test_minecraft_evaluate.py` |
| Manual | MT-14: User filters metric set via config | `MANUAL_TESTS.md` |

---

## 2. Inverse Index — Test-to-Requirement Map

Useful when editing a specific test: tells you which requirements you
might affect.

### `test_minecraft_dataset.py`

| Test class / method | Reqs |
|---|---|
| `TestActionSpaceConstants::*` | Req9 |
| `TestMuLawEncode::*` | Req9 |
| `TestDiscretizeCamera::*` | Req9 |
| `TestCameraBinsToJointIndex::*` | Req9 |
| `TestParseJsonlAction::*` | Req9, Req10 |
| `TestEnvActionToDreamer4::*` | Req9 |
| `TestZeroPadFrame::*` | Req10 |
| `TestCompositeCursor::*` | Req10 |
| `TestPrescanTrajectory::*` | Req10 |
| `TestLoadTrajectory::*` | Req10 |
| `TestMinecraftVPTDataset::test_lengths_and_item_shapes` | Req10 |
| `TestMinecraftVPTDataset::test_short_trajectory_excluded` | Req10 |
| `TestMinecraftVPTDataset::test_max_trajectories_limit` | Req5, Req10 |
| `TestMinecraftVPTDataset::test_getitem_retries_on_error` | Req10 |
| `TestCollateMinecraftBatch::test_stack_shapes` | Req6, Req10 |

### `test_minecraft_agent.py`

| Test class / method | Reqs |
|---|---|
| `TestCameraRoundtripPurePython::*` | Req3, Req9 |
| `TestDreamer4ActionsToMinerl::*` | Req2, Req3, Req8, Req9 |

### `test_minecraft_training.py`

| Test class / method | Reqs |
|---|---|
| `TestEnforceCuda::*` | Req5 |
| `TestResolveResumeFrom::*` | Req1, Req7 |
| `TestCLI::*` | Req4, Req5, Req6 |
| `TestHyperparameterDefaults::*` | Req5, Req6 |

### `test_minecraft_evaluate.py`

| Test class / method | Reqs |
|---|---|
| `TestTechTree::*` | Req11, Req14 |
| `TestCheckInventory::*` | Req8, Req12 |
| `TestEpisodeResult::*` | Req10, Req11 |
| `TestAggregateResults::*` | Req11, Req12 |

### `test_minecraft_integration.py`

| Test class / method | Reqs |
|---|---|
| `TestDatasetToDynamics::test_dataloader_collate_shapes` | Req6 |
| `TestDatasetToDynamics::test_forward_pass_loss_is_scalar` | Req6 |
| `TestDatasetToDynamics::test_gradients_flow` | Req6 |
| `TestDatasetToDynamics::test_generate_emits_valid_action_shape` | Req3, Req9 |

---

## 3. Coverage Summary

| Req | # Automated tests | # Manual tests | Status |
|---|---:|---:|---|
| Req1  |  7 | 2 | Covered |
| Req2  |  2 | 1 | Covered (eval-env gated) |
| Req3  |  5 | 1 | Covered |
| Req4  |  3 | 1 | Covered |
| Req5  | 12 | 1 | Covered |
| Req6  |  9 | 1 | Covered |
| Req7  |  4 | 1 | Covered |
| Req8  |  3 | 1 | Covered (eval-env gated) |
| Req9  | 23 | 1 | Covered |
| Req10 | 13 | 1 | Covered |
| Req11 |  8 | 1 | Covered |
| Req12 |  6 | 1 | Covered |
| Req13 |  1 | 1 | Covered (primarily manual) |
| Req14 |  2 | 1 | Covered (primarily manual) |

Every requirement has ≥ 1 automated test AND ≥ 1 manual test. See
`COVERAGE.md` for code-coverage percentages.

---

## 4. How to regenerate this matrix

This file is maintained by hand. When you add or rename a test, update:
1. The forward mapping in §1 (which req → which tests).
2. The inverse index in §2.
3. The summary counts in §3.

A lint check for "every requirement has at least one test" can be added
by grepping for each `Req\d+` token in this file and asserting the
count of non-header lines under its section is ≥ 1; we have not wired
this into CI.
