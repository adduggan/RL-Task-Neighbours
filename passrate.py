# passrate.py
# Measure pass rate over multiple trials while injecting TASK_SEED per run.
# Usage examples:
#   python passrate.py --trials 10 \
#     --attempt-cmd "python task.py" \
#     --grade-cmd   "python grader.py" \
#     --seed-mode incremental --base-seed 1000 --refresh-data
#
#   python passrate.py --trials 10 \
#     --attempt-cmd "uv run task.py" \
#     --grade-cmd   "uv run grader.py" \
#     --seed-mode random --refresh-data

from __future__ import annotations
import argparse
import os
import random
import shlex
import subprocess
from pathlib import Path

DATA_PATH = Path("data/embeddings.npy")


def run_cmd(cmd: str, extra_env: dict | None = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # Use shell-safe splitting; pass a list to subprocess
    return subprocess.call(shlex.split(cmd), env=env)


def main():
    ap = argparse.ArgumentParser(description="Pass-rate harness for RL task")
    ap.add_argument("--trials", type=int, default=10, help="Number of attempts to run")
    ap.add_argument(
        "--attempt-cmd",
        required=True,
        help="Command that performs ONE attempt (generates output, e.g., `python task.py`)",
    )
    ap.add_argument(
        "--grade-cmd",
        required=True,
        help="Command that runs the grader and returns 0 on PASS (e.g., `python grader.py`)",
    )
    ap.add_argument(
        "--seed-mode",
        choices=["incremental", "random"],
        default="incremental",
        help="How to pick TASK_SEED per trial",
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Starting seed for incremental mode",
    )
    ap.add_argument(
        "--refresh-data",
        action="store_true",
        help="Delete data/embeddings.npy before each trial so task.py regenerates per seed",
    )
    args = ap.parse_args()

    passes = 0

    for i in range(1, args.trials + 1):
        # Choose seed
        if args.seed_mode == "incremental":
            seed = args.base_seed + (i - 1)
        else:
            seed = random.SystemRandom().randint(1, 2**31 - 1)

        print(f"\n=== Trial {i}/{args.trials} | TASK_SEED={seed} ===")

        # Ensure new data if requested
        if args.refresh_data and DATA_PATH.exists():
            try:
                DATA_PATH.unlink()
                print("[prep] Removed data/embeddings.npy to force regeneration")
            except Exception as e:
                print(f"[prep] Warning: could not remove {DATA_PATH}: {e}")

        # 1) One attempt with injected seed
        rc_attempt = run_cmd(args.attempt_cmd, extra_env={"TASK_SEED": str(seed)})
        if rc_attempt != 0:
            print(f"[attempt] non-zero exit ({rc_attempt}) — counting as FAIL")
            continue

        # 2) Grade
        rc_grade = run_cmd(args.grade_cmd)
        if rc_grade == 0:
            print("[grade] PASS")
            passes += 1
        else:
            print(f"[grade] FAIL (exit {rc_grade})")

    rate = 100.0 * passes / max(1, args.trials)
    print(f"\n=== Summary ===\nPasses: {passes}/{args.trials}  ({rate:.1f}%)")


if __name__ == "__main__":
    main()