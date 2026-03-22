# Challenge Parity Checklist

Use this checklist before treating any `hybrid_golf` result as submission-grade.

## Purpose

`hybrid_golf` supports fast proxy experiments, but proxy numbers are not automatically challenge-parity numbers.

This checklist defines the minimum conditions for a result to count as a **challenge-ready evaluation** rather than an internal proxy.

Primary preset:

- [challenge_parity.json](/c:/Users/pasqu/OpenChallenge/parameter-golf/configs/hybrid/challenge_parity.json)

## Required Conditions

1. Training run is wallclock-capped.
   The training recipe must be compared under an explicit time budget, not only a fixed step count.

2. Export is under budget.
   Exported artifact must be under `16 MB`.

3. Evaluation is strict.
   Policy must be `strict_causal`.

4. Validation uses the full split.
   `data.val_tokens_limit=0`.

5. Evaluation is a flat stream.
   `eval.documentwise=false`.

6. Windowing matches the incumbent challenge-style eval.
   Use:
   - `eval.score_tokens=1024`
   - `eval.adapt_tokens=3072`

7. No exploratory compute path.
   For parity evaluation:
   - `eval.adaptive_extra_pass=false`
   - no exploratory policy

8. No noisy block dumps.
   `eval.write_block_logs=false`

9. Roundtrip is checked.
   Use `--export-roundtrip` on the final parity evaluation.

10. Results are labeled correctly.
   The run must be marked as `proxy` or `submission_candidate`, not `smoke`.

## Current Clean Submission-Track Evaluator

Current active legal evaluator baseline:

- `strict_causal`
- `doc_bias` enabled
- `ngram` disabled
- `pointer` disabled
- `meta` disabled
- `adaptive_extra_pass` disabled
- `strict_commit_steps=0`

If you want a base-prior ablation, override:

- `experts.enable_doc_bias=false`

## Standard Commands

Train a serious wallclock-capped run:

```bash
python3 scripts/hg_train.py \
  --config configs/hybrid/challenge_parity.json \
  --set run.id=<run_id> \
  --set run.stage=proxy \
  --set train.iterations=1000000 \
  --set train.max_wallclock_seconds=540
```

Challenge-parity eval:

```bash
python3 scripts/hg_eval.py \
  --config configs/hybrid/challenge_parity.json \
  --checkpoint <checkpoint> \
  --policy strict
```

Challenge-parity eval with roundtrip:

```bash
python3 scripts/hg_eval.py \
  --config configs/hybrid/challenge_parity.json \
  --checkpoint <checkpoint> \
  --policy strict \
  --export-roundtrip
```

Base-prior ablation under parity conditions:

```bash
python3 scripts/hg_eval.py \
  --config configs/hybrid/challenge_parity.json \
  --checkpoint <checkpoint> \
  --policy strict \
  --set experts.enable_doc_bias=false
```

## What Does Not Count As Submission-Grade

These are still useful, but they are not allowed to decide the final recipe by themselves:

- reduced validation slices
- `documentwise=true`
- custom score/adapt windows used only for convenience
- exploratory policy
- fixed-step-only comparisons without a wallclock cap
- runs without export-size reporting

## Interpretation Rule

A result is only allowed to influence the final submission recipe if it satisfies this checklist or deviates from it for a documented reason.
