# # passrate.py
# Measure pass rate over multiple trials. Optionally call Anthropic (Claude)
# each trial to generate solution.py before attempting & grading.
#
# Examples:
#   # Non-API mode (use your existing solution.py)
#   python passrate.py --trials 10 --attempt-cmd "python task.py" --grade-cmd "python grader.py" \
#     --seed-mode incremental --base-seed 1000 --refresh-data
#
#   # Anthropic mode (Claude writes solution.py every trial)
#   python passrate.py --trials 10 --attempt-cmd "python task.py" --grade-cmd "python grader.py" \
#     --seed-mode random --refresh-data \
#     --use-anthropic --model "claude-3-5-sonnet-20241022" --temperature 0.6 --max-tokens 1800
#
# Requires:
#   pip install anthropic
#   export ANTHROPIC_API_KEY="sk-ant-..."
#
from __future__ import annotations
import argparse
import os
import random
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional

DATA_PATH = Path("data/embeddings.npy")
SOLUTION_PATH = Path("solution.py")
README_PATH = Path("README.md")


def run_cmd(cmd: str, extra_env: dict | None = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.call(shlex.split(cmd), env=env)


def _read_text(path: Path, fallback: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return fallback


def _default_task_prompt(seed: int) -> str:
    # Keep the task spec crisp and aligned with grader. Include the seed so trials vary a bit.
    return f"""You are to generate the full contents of a Python file named `solution.py`.

Task summary:
- Input: numpy array at 'data/embeddings.npy' with shape [N, D] (float32).
- Output: Parquet at 'outputs/topk.parquet' with columns:
  - row_id:int64 = 0..N-1 in order
  - nbr_idx:list<int64>[5] (top-5 neighbor indices by cosine similarity, excluding self, sorted desc)
  - nbr_sim:list<float32>[5] (matching similarities, non-increasing, finite)
- Requirements:
  1) Exclude self (no row may list itself as a neighbor).
  2) Handle zero vectors: for rows with L2 norm == 0, all neighbor similarities must be exactly 0.0.
  3) Deterministic output (re-running on same inputs yields identical results).
  4) Use only NumPy (or PyTorch CPU is acceptable), no external ANN libraries.
  5) Avoid O(N^2) Python loops; use vectorized or blocked computation.
- Top-K: K=5.

Implementation constraints:
- Provide a single function: main(input_path: str, output_path: str) -> None.
- Save parquet with pandas/pyarrow.
- Assume the directories already exist.

Important:
- Reply with ONLY a single Markdown code block containing valid Python for solution.py.
- Do not include explanations outside the code block.

Randomness hint for diversity across trials (do NOT use randomness in outputs; this is metadata only): SEED={seed}
"""


def _extract_code_from_markdown(md: str) -> Optional[str]:
    """
    Extract Python code from a Markdown response. Prefer ```python blocks, then any ``` block.
    """
    # ```python ... ```
    m = re.search(r"```python\s+(.*?)```", md, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # ``` ... ```
    m = re.search(r"```\s+(.*?)```", md, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: if there's no fences, return whole text (last resort)
    return md.strip() if md.strip() else None


def _gen_solution_with_anthropic(model: str, temperature: float, max_tokens: int, seed: int,
                                 system_prompt: Optional[str], user_prompt: Optional[str]) -> str:
    # Lazy import so non-API mode doesn't require the package
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")

    client = Anthropic(api_key=api_key)

    sys_msg = system_prompt or (
        "You are an expert Python ML engineer. Output only valid Python inside one code fence. "
        "Your code must run deterministically and match the task spec exactly."
    )

    usr_msg = user_prompt or _default_task_prompt(seed)

    resp = client.messages.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system=sys_msg,
        messages=[{"role": "user", "content": usr_msg}],
    )

    # Concatenate text blocks
    parts = []
    for block in resp.content:
        if block.type == "text":
            parts.append(block.text)
    text = "\n".join(parts).strip()

    code = _extract_code_from_markdown(text)
    if not code:
        raise RuntimeError("Could not extract Python code from Claude's reply.")
    return code


def _write_solution_py(code: str) -> None:
    SOLUTION_PATH.write_text(code, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Pass-rate harness; optional Anthropic code-gen per trial.")
    ap.add_argument("--trials", type=int, default=10, help="Number of attempts to run.")
    ap.add_argument("--attempt-cmd", required=True, help='Runs one attempt (e.g., "python task.py").')
    ap.add_argument("--grade-cmd", required=True, help='Runs grader (e.g., "python grader.py"), returns 0 on PASS.')
    ap.add_argument("--seed-mode", choices=["incremental", "random"], default="incremental",
                    help="How to pick TASK_SEED per trial.")
    ap.add_argument("--base-seed", type=int, default=1000, help="Starting seed for incremental mode.")
    ap.add_argument("--refresh-data", action="store_true",
                    help="Delete data/embeddings.npy before each trial so task.py regenerates per seed.")

    # Anthropic options
    ap.add_argument("--use-anthropic", action="store_true",
                    help="If set, call Anthropic each trial to generate solution.py.")
    ap.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Anthropic model name.")
    ap.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    ap.add_argument("--max-tokens", type=int, default=1800, help="Max tokens for code generation.")
    ap.add_argument("--system-prompt-file", type=str, default=None, help="Optional system prompt file.")
    ap.add_argument("--user-prompt-file", type=str, default=None, help="Optional user prompt file (overrides default).")
    args = ap.parse_args()

    system_prompt = _read_text(Path(args.system_prompt_file)) if args.system_prompt_file else None
    user_prompt = _read_text(Path(args.user_prompt_file)) if args.user_prompt_file else None

    passes = 0

    for i in range(1, args.trials + 1):
        # Pick seed for this trial
        seed = (args.base_seat if False else None)  # placeholder to avoid linter; real logic below
        if args.seed_mode == "incremental":
            seed = args.base_seed + (i - 1)
        else:
            seed = random.SystemRandom().randint(1, 2**31 - 1)

        print(f"\n=== Trial {i}/{args.trials} | TASK_SEED={seed} {'| Anthropic' if args.use_anthropic else ''} ===")

        # Optional: regenerate embeddings each trial
        if args.refresh_data and DATA_PATH.exists():
            try:
                DATA_PATH.unlink()
                print("[prep] Removed data/embeddings.npy to force regeneration")
            except Exception as e:
                print(f"[prep] Warning: could not remove {DATA_PATH}: {e}")

        # If using Anthropic, ask Claude to produce a fresh solution.py for this trial.
        if args.use_anthropic:
            try:
                code = _gen_solution_with_anthropic(
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=seed,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                _write_solution_py(code)
                print("[anthropic] Wrote solution.py from Claude response")
            except Exception as e:
                print(f"[anthropic] Generation failed: {e} — counting as FAIL")
                # Still try to grade? No; without solution it's guaranteed to fail attempt.
                continue

        # 1) Run one attempt (generate outputs/topk.parquet)
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