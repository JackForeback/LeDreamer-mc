"""Tests for Video-Pre-Training/evaluate_dreamer4_minecraft.py.

The module imports MineRL only inside run_worker() (the subprocess body),
so its data-layer helpers are safe to test without the eval-env set up.
"""
import math

import pytest

import evaluate_dreamer4_minecraft as ev


# ─── TECH_TREE_TASKS ────────────────────────────────────────────────

class TestTechTree:
    def test_eleven_tasks(self):
        # Matches the diamond tech tree reported in the Dreamer4 paper.
        assert len(ev.TECH_TREE_TASKS) == 11

    def test_progression_order(self):
        names = [t[0] for t in ev.TECH_TREE_TASKS]
        assert names.index("log") < names.index("planks")
        assert names.index("planks") < names.index("crafting_table")
        assert names.index("wooden_pickaxe") < names.index("stone_pickaxe")
        assert names.index("iron_ingot") < names.index("iron_pickaxe")
        assert names[-1] == "diamond"

    def test_each_task_has_inventory_key(self):
        for name, key in ev.TECH_TREE_TASKS:
            assert isinstance(name, str) and isinstance(key, str) and key


# ─── check_inventory ────────────────────────────────────────────────

class TestCheckInventory:
    def test_dict_with_item(self):
        info = {"inventory": {"log": 3}}
        assert ev.check_inventory(info, "log") is True

    def test_dict_zero_count(self):
        info = {"inventory": {"log": 0}}
        assert ev.check_inventory(info, "log") is False

    def test_dict_missing_item(self):
        info = {"inventory": {"log": 1}}
        assert ev.check_inventory(info, "diamond") is False

    def test_missing_inventory(self):
        assert ev.check_inventory({}, "log") is False

    def test_non_dict_inventory(self):
        # Some MineRL versions give a list — defensive branch should no-op
        assert ev.check_inventory({"inventory": [1, 2, 3]}, "log") is False


# ─── EpisodeResult dataclass ────────────────────────────────────────

class TestEpisodeResult:
    def test_construction_defaults(self):
        r = ev.EpisodeResult(
            episode_id=1, worker_id=0, total_steps=10,
            total_reward=1.5, wall_time_seconds=0.5,
        )
        assert r.tech_tree_steps == {}
        assert r.tech_tree_achieved == {}

    def test_construction_with_trees(self):
        r = ev.EpisodeResult(
            episode_id=1, worker_id=0, total_steps=10,
            total_reward=1.5, wall_time_seconds=0.5,
            tech_tree_steps={"log": 5},
            tech_tree_achieved={"log": True},
        )
        assert r.tech_tree_achieved["log"] is True


# ─── aggregate_results ──────────────────────────────────────────────

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


class TestAggregateResults:
    def test_empty_input_returns_empty_dict(self):
        assert ev.aggregate_results([]) == {}

    def test_counts_and_means(self):
        results = [
            _make_result(0, 10.0, 100, {"log": True, "planks": True}),
            _make_result(1, 20.0, 200, {"log": True}),
            _make_result(2,  0.0,  50, {}),
        ]
        stats = ev.aggregate_results(results)
        assert stats["n_episodes"] == 3
        assert stats["mean_reward"] == pytest.approx(10.0)
        assert stats["mean_steps"] == pytest.approx(116.66666, abs=1e-2)

    def test_success_rate(self):
        results = [
            _make_result(0, 0, 0, {"log": True}),
            _make_result(1, 0, 0, {"log": True}),
            _make_result(2, 0, 0, {}),
            _make_result(3, 0, 0, {}),
        ]
        stats = ev.aggregate_results(results)
        assert stats["tasks"]["log"]["success_rate"] == 0.5
        assert stats["tasks"]["log"]["n_successes"] == 2

    def test_mean_steps_infinity_when_no_success(self):
        results = [_make_result(0, 0, 0, {}) for _ in range(3)]
        stats = ev.aggregate_results(results)
        assert math.isinf(stats["tasks"]["diamond"]["mean_steps_to_success"])
