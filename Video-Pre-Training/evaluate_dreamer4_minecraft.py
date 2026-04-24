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
import atexit
import signal
import argparse
import subprocess
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List
from multiprocessing import Queue, get_context

import numpy as np


# Sentinel result payload signalling a worker exited before producing results.
# Carries the worker id and the raw traceback so the parent can surface it
# instead of silently hanging on result_queue.get() forever.
@dataclass
class WorkerFailure:
    worker_id: int
    stage: str           # "startup", "reset", "step", etc.
    error: str           # repr(exception)
    tb: str              # traceback.format_exc()


# Parent-side registry of live child processes so SIGTERM/SIGINT in the
# driver reaps them. Populated in main(), consumed in _terminate_children.
_CHILDREN = []


def _terminate_children(*_args):
    """Kill any still-running worker processes. Safe to call multiple times."""
    for p in _CHILDREN:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
    # Give them a moment to die, then hard-kill anything left.
    deadline = time.time() + 5
    for p in _CHILDREN:
        try:
            remaining = max(0.0, deadline - time.time())
            p.join(timeout=remaining)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)
        except Exception:
            pass

# Add paths for imports — this script lives inside Video-Pre-Training/,
# so go up one level to find the dreamer4 package and minecraft_vpt_dataset.
_this_dir = os.path.dirname(os.path.abspath(__file__))  # .../Video-Pre-Training
_project_root = os.path.dirname(_this_dir)              # .../dreamer4 (repo root)
sys.path.insert(0, _this_dir)      # VPT local imports (agent.py, lib/)
sys.path.insert(0, _project_root)  # dreamer4 package + minecraft_vpt_dataset


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
    """Check if ``item_name`` is present in the player's MineRL inventory.

    Args:
        info: MineRL ``info`` dict returned by ``env.step``.
        item_name: Minecraft item id (e.g. ``"log"``, ``"iron_pickaxe"``).

    Returns:
        True if the inventory entry for ``item_name`` exists with a
        count greater than zero, False otherwise (including when the
        inventory field is missing or not a dict).
    """
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
    env = None

    def _emit_failure(stage, exc):
        """Send traceback back to parent so it lands in sbatch stdout."""
        try:
            result_queue.put(WorkerFailure(
                worker_id=worker_id,
                stage=stage,
                error=repr(exc),
                tb=traceback.format_exc(),
            ))
        except Exception:
            pass

    # Ensure SIGTERM from the parent kills the Minecraft JVMs before we exit.
    # Without this, an aborted run leaves a tree of Java processes on HPC.
    def _worker_signal_handler(signum, _frame):
        raise SystemExit(f"worker {worker_id} received signal {signum}")
    signal.signal(signal.SIGTERM, _worker_signal_handler)

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
        try:
            from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
            from agent import ENV_KWARGS
            from dreamer4_minecraft_agent import Dreamer4MinecraftAgent
            from minerl.env.malmo import InstanceManager, MinecraftInstance
        except Exception as e:
            # Import-time failure (e.g. the Python 3.9 / PEP 604 issue this file
            # has tripped on before). Surface it through the result queue so the
            # parent can abort instead of waiting an hour per missing result.
            print(f"[Worker {worker_id}] Import error: {e}")
            traceback.print_exc()
            _emit_failure("import", e)
            return

        InstanceManager.configure_malmo_base_port(9000 + worker_id * 100)

        # Reduce JVM heap from 4G to 2G per instance to prevent OOM kills
        _orig_mc_init = MinecraftInstance.__init__
        _mc_max_mem = args_dict["minecraft_mem"]

        def _mc_init_with_reduced_mem(self, port=None, existing=False,
                                      status_dir=None, seed=None,
                                      instance_id=None, max_mem=_mc_max_mem):
            _orig_mc_init(self, port, existing, status_dir, seed,
                          instance_id, max_mem)
        MinecraftInstance.__init__ = _mc_init_with_reduced_mem

        try:
            env = HumanSurvival(**ENV_KWARGS).make()
        except Exception as e:
            print(f"[Worker {worker_id}] env construction failed: {e}")
            traceback.print_exc()
            _emit_failure("env_make", e)
            return

        try:
            agent = Dreamer4MinecraftAgent(
                env,
                checkpoint_path=args_dict["checkpoint"],
                stochastic=not args_dict.get("deterministic", False),
            )
        except Exception as e:
            print(f"[Worker {worker_id}] agent construction failed: {e}")
            traceback.print_exc()
            _emit_failure("agent_init", e)
            return

        while True:
            episode_id = episode_queue.get()
            if episode_id is None:
                break

            print(f"[Worker {worker_id}] Starting episode {episode_id}")
            t0 = time.time()

            # MineRL can transiently fail, so give it a few tries with backoff
            obs = None
            for attempt in range(5):
                try:
                    obs = env.reset()
                    break
                except Exception as e:
                    print(f"[Worker {worker_id}] env.reset() attempt {attempt+1}/5 failed: {e}")
                    if attempt < 4:
                        try:
                            env.close()  # have to make new or will just keep getting error
                        except Exception:
                            pass
                        time.sleep(10 * (attempt + 1))
                        env = HumanSurvival(**ENV_KWARGS).make()
                    else:
                        raise

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

                # Detect MineRL returning a random obs due to a dead Minecraft process
                if 'error' in info:
                    print(f"[Worker {worker_id}] Minecraft connection lost at "
                          f"step {step}, ending episode")
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

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        traceback.print_exc()
        _emit_failure("fatal", e)
    finally:
        # Close the MineRL env first — this shuts down the Minecraft JVM.
        # Without it, the Java child of this worker survives and lingers
        # as a zombie until the Slurm job hits its wall time.
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if xvfb_proc is not None:
            try:
                xvfb_proc.terminate()
                xvfb_proc.wait(timeout=5)
            except Exception:
                try:
                    xvfb_proc.kill()
                except Exception:
                    pass


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
    """CLI entry point: spawn worker processes and collect evaluation results.

    Builds a shared ``episode_queue`` of ``n_episodes`` work items (plus
    one ``None`` poison pill per worker), forks ``n_workers`` worker
    processes running :func:`run_worker`, drains the ``result_queue``
    with a one-hour-per-result timeout, aggregates stats via
    :func:`aggregate_results`, prints a tech-tree progression table,
    and writes the full JSON report to ``--output_json``.
    """
    parser = argparse.ArgumentParser("Dreamer4 Minecraft Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to dreamer4_minecraft.pt")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--minecraft_mem", type=str, default="4G",
                        help="Max JVM heap per Minecraft instance "
                             "(e.g., '2G', '1G'). Enables parallelism "
                             "with low VRAM.")
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
        "minecraft_mem": args.minecraft_mem,
        "deterministic": args.deterministic,
        "headless": not args.no_headless,
    }

    # Install the global cleanup hooks BEFORE any child starts. atexit covers
    # normal exits, the signal handlers cover sbatch SIGTERM / user Ctrl-C.
    # Whatever happens below, worker processes (and therefore their Xvfb and
    # Minecraft JVM children) get reaped rather than becoming zombies.
    atexit.register(_terminate_children)
    signal.signal(signal.SIGTERM, _terminate_children)
    signal.signal(signal.SIGINT, _terminate_children)

    workers = []
    for w in range(args.n_workers):
        p = ctx.Process(target=run_worker, args=(w, episode_queue, result_queue, args_dict))
        p.start()
        workers.append(p)
        _CHILDREN.append(p)
        if w < args.n_workers - 1:
            time.sleep(10)  # Stagger Minecraft JVM launches to avoid resource contention

    print(f"Started {args.n_workers} workers for {args.n_episodes} episodes")

    results = []
    failures = []
    for _ in range(args.n_episodes):
        # Poll result_queue with a short timeout so we can notice that every
        # worker has died (e.g. import error) instead of hanging for an hour.
        while True:
            try:
                item = result_queue.get(timeout=30)
            except Exception:
                # Nothing came through in 30s — are any workers still running?
                if not any(p.is_alive() for p in workers):
                    print(
                        f"[main] All {len(workers)} workers have exited; "
                        f"aborting with {len(results)}/{args.n_episodes} results "
                        f"({len(failures)} worker failure(s) reported)."
                    )
                    break
                continue  # workers still alive, keep waiting
            break

        if isinstance(item, WorkerFailure):
            failures.append(item)
            print(
                f"[main] Worker {item.worker_id} failed during "
                f"'{item.stage}': {item.error}"
            )
            print(item.tb)
            # If every worker has died, no point draining further — abort now.
            if not any(p.is_alive() for p in workers):
                print(
                    f"[main] No workers remain; aborting after "
                    f"{len(failures)} failure(s)."
                )
                break
            continue
        if item is None:
            # treat as timeout-with-dead-workers: already handled above
            break

        results.append(item)
        if len(results) % 10 == 0:
            print(f"Progress: {len(results)}/{args.n_episodes} episodes")

    # Best-effort graceful shutdown; _terminate_children (via atexit) will
    # force-kill anything still alive.
    for p in workers:
        p.join(timeout=30)
    _terminate_children()

    if failures and not results:
        print("\n" + "=" * 60)
        print("EVALUATION ABORTED — all workers failed before producing results")
        print("=" * 60)
        for f in failures[:3]:  # show first few tracebacks
            print(f"\n[Worker {f.worker_id}] stage={f.stage} error={f.error}")
            print(f.tb)
        sys.exit(2)

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
            mean_steps = t["mean_steps_to_success"]
            ms = f"{mean_steps:.0f}" if mean_steps != float("inf") else "N/A"
            print(f"{task_name:<20} {sr:>12} {ns:>10} {ms:>12}")

        output = {
            "aggregate": stats,
            "episodes": [asdict(r) for r in results],
            "config": vars(args),
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")
    else:
        print("No results collected!")


if __name__ == "__main__":
    main()
