# RL Task: Top-K Cosine Neighbors (Fast, Correct, Deterministic)


## Overview
This reinforcement-learning task trains models to compute the **top-5 cosine neighbors** for each embedding vector while excluding self-matches, handling zero vectors, and maintaining deterministic behavior under runtime and memory constraints. It demonstrates efficient vectorization, reproducibility, and realistic ML-engineering rigor.

## Setup
```bash
pip install numpy pandas pyarrow anthropic
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxx" 
```

## Run and Grade
```
python task.py
python grader.py
```
task.py generates data/embeddings.npy and writes outputs/topk.parquet.
grader.py verifies correctness, determinism, performance, and resource use.

## Measure Pass-Rate
```
python passrate.py --trials 10 \
  --attempt-cmd "python task.py" \
  --grade-cmd "python grader.py" \
  --seed-mode incremental --refresh-data
```
Target pass-rate: 10–40% for balanced difficulty.

## Anthropic Integration
passrate.py can automatically generate and evaluate new solutions using Claude via the Anthropic API.

```
python passrate.py --trials 10 \
  --attempt-cmd "python task.py" \
  --grade-cmd "python grader.py" \
  --use-anthropic \
  --model "claude-3-5-sonnet-20241022" \
  --temperature 0.6 --max-tokens 1800 \
  --seed-mode random --refresh-data
```
Claude generates a new solution.py per trial, which is then executed and graded automatically.

## Output Format
outputs/topk.parquet
Column		Type		Description
row_id		int64		Index of each embedding (0…N-1)
nbr_idx		list[5]		Indices of top-K neighbors (descending similarity)
nbr_sim		list[5]		Corresponding cosine similarities (descending)

## Tuning Difficulty 
Parameter				Easier			Harder
N (task.py)				smaller			larger
X_SECONDS (grader.py)	higher			lower
Y_MB (grader.py)		higher		lower

The default configuration produces a pass rate of approximately 25–30% for Claude-3.5 Sonnet.

## Code Metrics
File				Functional LOC			Total LOC (incl. docs/comments)
task.py				45						55
solution.py			50						60
grader.py			160						180
tools.py			10						15
passrate.py			130						170
Total				≈395 functional			≈480 total

Core RL task (task, solution, grader, tools) remains under 300 LOC. Anthropic integration adds ~130 LOC as an optional enhancement.

## Submission Summary

This task meets all RL requirements:
	•	Realistic ML-engineering objective
	•	Behavioral grading (no string matching)
	•	Controlled pass-rate (10–40%)
	•	Compact codebase (<300 LOC core)
	•	Optional model-in-the-loop evaluation via Anthropic API

Measured pass-rate: ~25–30% across 10 randomized trials using Claude-3.5 Sonnet.


## Author
Andrew Duggan
AI Architect & ML Systems Engineer
San Diego, CA — 2025



