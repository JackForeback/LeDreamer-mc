"""
Evaluation script for Dreamer4 Minecraft agent using VPT infrastructure.

Runs the trained Dreamer4 agent in MineRL HumanSurvival environment and
tracks diamond tech tree progression. Supports parallel evaluation using
multiple processes with Xvfb for headless rendering.

This is a corrected version of examples/evaluate_dreamer4_parallel.py that:
  - Uses the corrected Dreamer4MinecraftAgent (not the broken DreamerV4Agent)
  - Accepts a single checkpoint path (not three separate checkpoints)
  - Adds proper error handling and timeouts
  - Validates checkpoint existence before spawning workers

Usage:
    # Single process (for debugging):
    python evaluate_dreamer4_minecraft.py \
        --checkpoint ./checkpoints/dreamer4_minecraft.pt \
        --n_episodes 10 \
        --n_workers 1

    # Parallel headless evaluation:
    python evaluate_dreamer4_minecraft.py \
        --checkpoint ./checkpoints/dreamer4_minecraft.pt \
        --n_episodes 100 \
        --n_workers 8 \
        --max_steps 36000 \
        --output_json eval_results.json
"""

import os
import sys
import json
import time
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from multiprocessing import Queue, get_context

import numpy as np

# Add paths for imports — this script lives inside Video-Pre-Training/,
# so go up one level to find the dreamer4 package and minecraft_vpt_dataset.
_this_dir = os.path.dirname(os.path.abspath(__file__))       # .../Video-Pre-Training
_project_root = os.path.dirname(_this_dir)                    # .../dreamer4 (repo root)
sys.path.insert(0, _this_dir)                                 # VPT local imports (agent.py, lib/)
sys.path.insert(0, _project_root)                             # dreamer4 package + minecraft_vpt_dataset


# Diamond tech tree tasks in progression order.
# Matches the tasks reported in the Dreamer4 paper.
TECH_TREE_TASKS = [
    ("log", "log"),
    ("planks", "planks"),
    ("crafting_table", "crafting_table"),
    ("wooden_pickaxe", "wooden_pickaxe"),
    ("cobblestone", "cobblestone"),
    ("stone_pickaxe", "stone_pickaxe"),
    ("iron_ore", "iron_ore"),
    ("furnace", "furnace"),
    ("iron_ingot", "iron_ingot"),
    ("iron_pickaxe", "iron_pickaxe"),
    ("diamond", "diamond"),
]


@dataclass
class EpisodeResult:
    """Results from a single evaluation episode."""
    episode_id: int
    worker_id: int
    total_steps: int
    total_reward: float
    wall_time_seconds: float
    tech_tree_steps: Dict[str, int] = field(default_factory=dict)
    tech_tree_achieved: Dict[str, bool] = field(default_factory=dict)


def check_inventory(info: dict, item_name: str) -> bool:
    """Check if an item is present in the player's inventory."""
    inventory = info.get("inventory", {})
    if isinstance(inventory, dict):
        return inventory.get(item_name, 0) > 0
    return False


def run_worker(
    worker_id: int,
    episode_queue: Queue,
    result_queue: Queue,
    args_dict: dict,
):
    """Worker process for parallel evaluation.

    Each worker:
    1. Starts its own Xvfb display for headless rendering
    2. Creates a MineRL HumanSurvival environment
    3. Loads the Dreamer4 agent
    4. Runs episodes from the queue until poison pill
    5. Reports results
    """
    display_num = 99 + worker_id
    os.environ["DISPLAY"] = f":{display_num}"

    xvfb_proc = None
    if args_dict.get("headless", True):
        try:
            xvfb_proc = subprocess.Popen(
                ["Xvfb", f":{display_num}", "-screen", "0", "1024x768x24", "-ac"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1)
        except FileNotFoundError:
            print(f"[Worker {worker_id}] WARNING: Xvfb not found, using existing display")
            xvfb_proc = None

    try:
        from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
        from agent import ENV_KWARGS
        from dreamer4_minecraft_agent import Dreamer4MinecraftAgent

        env = HumanSurvival(**ENV_KWARGS).make()

        agent = Dreamer4MinecraftAgent(
            env,
            checkpoint_path=args_dict["checkpoint"],
            stochastic=not args_dict.get("deterministic", False),
        )

        while True:
            episode_id = episode_queue.get()
            if episode_id is None:
                break

            print(f"[Worker {worker_id}] Starting episode {episode_id}")
            t0 = time.time()

            obs = env.reset()
            agent.reset()

            total_reward = 0.0
            tech_tree_steps = {name: -1 for name, _ in TECH_TREE_TASKS}
            tech_tree_achieved = {name: False for name, _ in TECH_TREE_TASKS}

            for step in range(args_dict["max_steps"]):
                try:
                    action = agent.get_action(obs)
                    obs, reward, done, info = env.step(action)
                except Exception as e:
                    print(f"[Worker {worker_id}] Error at step {step}: {e}")
                    break

                total_reward += reward

                # Check tech tree progression
                for task_name, item_key in TECH_TREE_TASKS:
                    if not tech_tree_achieved[task_name]:
                        if check_inventory(info, item_key):
                            tech_tree_achieved[task_name] = True
                            tech_tree_steps[task_name] = step
                            print(
                                f"[Worker {worker_id}] Episode {episode_id}: "
                                f"achieved '{task_name}' at step {step}"
                            )

                if done:
                    break

            wall_time = time.time() - t0
            result = EpisodeResult(
                episode_id=episode_id,
                worker_id=worker_id,
                total_steps=step + 1,
                total_reward=total_reward,
                wall_time_seconds=wall_time,
                tech_tree_steps=tech_tree_steps,
                tech_tree_achieved=tech_tree_achieved,
            )
            result_queue.put(result)
            print(
                f"[Worker {worker_id}] Episode {episode_id} done: "
                f"{step+1} steps, reward={total_reward:.2f}, "
                f"time={wall_time:.1f}s"
            )

        env.close()

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if xvfb_proc is not None:
            xvfb_proc.terminate()
            xvfb_proc.wait()


def aggregate_results(results: List[EpisodeResult]) -> Dict:
    """Compute aggregate statistics across all episodes."""
    n = len(results)
    if n == 0:
        return {}

    stats = {
        "n_episodes": n,
        "mean_reward": float(np.mean([r.total_reward for r in results])),
        "std_reward": float(np.std([r.total_reward for r in results])),
        "mean_steps": float(np.mean([r.total_steps for r in results])),
        "tasks": {},
    }

    for task_name, _ in TECH_TREE_TASKS:
        achieved = [r.tech_tree_achieved[task_name] for r in results]
        success_rate = sum(achieved) / n
        success_steps = [
            r.tech_tree_steps[task_name]
            for r in results
            if r.tech_tree_achieved[task_name]
        ]
        mean_steps = float(np.mean(success_steps)) if success_steps else float("inf")

        stats["tasks"][task_name] = {
            "success_rate": success_rate,
            "n_successes": sum(achieved),
            "mean_steps_to_success": mean_steps,
        }

    return stats


def main():
    parser = argparse.ArgumentParser("Dreamer4 Minecraft Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to dreamer4_minecraft.pt")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=36000,
                        help="Max steps per episode (36000 = 30min at 20fps)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_headless", action="store_true",
                        help="Don't start Xvfb (use existing display)")
    parser.add_argument("--output_json", type=str, default="eval_results.json")
    args = parser.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ctx = get_context("spawn")
    episode_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for i in range(args.n_episodes):
        episode_queue.put(i)
    for _ in range(args.n_workers):
        episode_queue.put(None)

    args_dict = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "max_steps": args.max_steps,
        "deterministic": args.deterministic,
        "headless": not args.no_headless,
    }

    workers = []
    for w in range(args.n_workers):
        p = ctx.Process(target=run_worker, args=(w, episode_queue, result_queue, args_dict))
        p.start()
        workers.append(p)

    print(f"Started {args.n_workers} workers for {args.n_episodes} episodes")

    results = []
    for _ in range(args.n_episodes):
        try:
            result = result_queue.get(timeout=3600)  # 1 hour timeout per episode
            results.append(result)
            if len(results) % 10 == 0:
                print(f"Progress: {len(results)}/{args.n_episodes} episodes")
        except Exception:
            print(f"Timeout waiting for results, got {len(results)}/{args.n_episodes}")
            break

    for p in workers:
        p.join(timeout=30)

    if results:
        stats = aggregate_results(results)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Episodes: {stats['n_episodes']}")
        print(f"Mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
        print(f"Mean length: {stats['mean_steps']:.0f} steps")
        print()
        print(f"{'Task':<20} {'Success Rate':>12} {'N Success':>10} {'Mean Steps':>12}")
        print("-" * 56)
        for task_name, _ in TECH_TREE_TASKS:
            t = stats["tasks"][task_name]
            sr = f"{t['success_rate']*100:.1f}%"
            ns = str(t["n_successes"])
            ms = f"{t['mean_steps_to_success']:.0f}" if t["mean_steps_to_success"] != float("inf") else "N/A"
            print(f"{task_name:<20} {sr:>12} {ns:>10} {ms:>12}")

        output = {
            "aggregate": stats,
            "episodes": [asdict(r) for r in results],
            "config": vars(args),
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")
    else:
        print("No results collected!")


if __name__ == "__main__":
    main()
