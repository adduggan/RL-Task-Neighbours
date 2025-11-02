# RL Task: Top-K Cosine Neighbors (Fast, Correct, Deterministic)

## Overview
This reinforcement-learning (RL) task trains a model to compute the **top-5 cosine neighbors** for each embedding vector in `data/embeddings.npy`, excluding self-matches, handling zero vectors correctly, and producing deterministic, efficient outputs under runtime and memory constraints.  
The task teaches **practical ML-engineering skills** such as vectorization, numerical stability, reproducibility, and behavioral evaluation.

---

## Learning Objectives
- Implement efficient cosine-similarity search without O(N²) Python loops  
- Ensure deterministic and reproducible results  
- Handle zero-norm vectors safely  
- Balance correctness, performance, and resource limits under realistic ML conditions  

---

## Repository Structure
rl_task_neighbors/
├─ task.py          # Generates/loads embeddings, calls solution.main()
├─ solution.py      # Model implementation (computes top-K neighbors)
├─ grader.py        # Behavioral grading: correctness, determinism, performance
├─ passrate.py      # Repeated trials with automatic TASK_SEED injection
├─ tools.py         # Helpers for timing and memory conversions
├─ data/            # Contains generated embeddings.npy
└─ outputs/         # Contains generated topk.parquet

---

## Setup
Requires **Python 3.10+** and the following packages:

bash
pip install numpy pandas pyarrow
# optional (for PyTorch-based approaches)
pip install torch --index-url https://download.pytorch.org/whl/cpu

## Running the Task

## Step 1: Generate Data & Run the Model
python task.py

This command:
	1.	Generates a dataset (data/embeddings.npy) using a deterministic random seed.
	2.	Calls solution.main() to compute cosine neighbors.
	3.	Writes outputs/topk.parquet.

Each dataset is reproducible per seed (see TASK_SEED below).


## Step 2: Grade the Result
python grader.py

## Checks performed:
	1.	Schema and column consistency
	2.	Self-exclusion (no row includes itself as a neighbor)
	3.	Zero-vector rule (rows with norm 0 must have all similarities = 0.0)
	4.	Correctness on a random subset (tie-aware)
	5.	Determinism across repeated runs
	6.	Performance and peak memory limits

## Success output:
PASS: All checks succeeded.

## Step 3: Measure Pass-Rate
python passrate.py --trials 10 \
  --attempt-cmd "python task.py" \
  --grade-cmd "python grader.py" \
  --seed-mode incremental --base-seed 1000 --refresh-data

  ## Options:
	•	--seed-mode incremental → uses seeds 1000, 1001, 1002, …
	•	--seed-mode random → draws cryptographically random seeds
	•	--refresh-data → deletes data/embeddings.npy each trial to force regeneration

Target pass-rate: between 10% and 40%
This ensures the task is neither trivial nor impossible, providing a meaningful RL signal.

## Environment Variable: TASK_SEED

The dataset is deterministic per seed.
TASK_SEED=42 python task.py     # default
TASK_SEED=101 python task.py    # generate a new deterministic dataset

## Output Format

outputs/topk.parquet must contain:
Column  Type      Description
row_id   int64    Index of each embedding (0…N-1)
nbr_idx  list[5]  Indices of top-K neighbors (descending similarity)
nbr_sim  list[5]  Corresponding cosine similarities (descending)

Tuning Difficulty / Pass-Rate

Adjust these to control difficulty:
Parameter                 Easier      Harder
N (in task.py)             smaller     larger
X_SECONDS (in grader.py)   higher      lower
Y_MB (in grader.py)        higher      lower


Target: 10–40% pass-rate for a competent model such as Claude-3.5

## Tools

tools.py provides:
	•	bytes_to_mb() → converts byte counts to MB
	•	time_block() → simple timing context manager

These are imported by grader.py and optional for development.







