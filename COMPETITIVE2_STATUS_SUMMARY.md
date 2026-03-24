# Competitive2 Status Summary

Date: 2026-03-24

## Context

The strongest scoring scripts currently available in the repo include several borrowed leaderboard branches:

- `train_gpt_obliterate.py`
- `train_gpt_sub.py`
- `train_gpt_signalrush.py`
- `train_gpt_abaybektursun.py`

Those are useful references, but they are not the main target if the goal is to advance an original branch.

Under that criterion, the main self-authored branch to investigate is:

- `train_gpt_competitive2.py`

## High-Level Conclusion

`train_gpt_competitive2.py` is the strongest current original branch in terms of being the main self-authored candidate worth studying, but the cleaned runs show that the current recipe is not competitive yet.

The branch is now structurally cleaner than before, but the metric is still poor after fixing the evaluation and wallclock issues.

## Patches Made To `train_gpt_competitive2.py`

Recent work on `train_gpt_competitive2.py` included:

- compile-safe `BitLinear` quant path
- compile-safe rotary cache path
- warmup excluded from measured wallclock
- pre-step wallclock cap handling
- optional separation of training compile path from evaluation path
- optional disabling of forced last-step validation
- short-run projection-QAT guard so QAT is not silently disabled in short wallclock runs

Relevant recent commits:

- `60f85c6` Tighten competitive2 eval path
- `48177d5` Enable short-run QAT for competitive2
- `d89e164` Fix competitive2 QAT log ordering

## Important Experimental Results

### Borrowed Reference Scripts

These are not the target branch, but they are the quality references:

| Script | Notes | Result |
| --- | --- | --- |
| `train_gpt_obliterate.py` | best owned-measured borrowed-style line so far | about `1.3857` at `300s` |
| `train_gpt_abaybektursun.py` | leaderboard reference branch | `1.46353839` at `180s` |
| `train_gpt_signalrush.py` | leaderboard reference branch | `1.55533763` at `180s` |

### `competitive2` Runs

| Run ID | Main Settings | Result | Conclusion |
| --- | --- | --- | --- |
| `competitive2_meta_fast_1xh100_180s` | fast adapters + meta + QAT, older evaluation path | about `1.5637` | not trustworthy because the old protocol had heavy eval overhead and did not cleanly exercise the late phase |
| `competitive2_meta_fast_1xh100_180s_v3` | fast adapters + meta + short-run QAT | `3.08601350` | catastrophic once short-run QAT activated |
| `competitive2_meta_fast_1xh100_180s_v4_noqat` | fast adapters + meta, QAT off | `2.10989724` | still bad; meta/fast regime does not survive export well |
| `competitive2_core_1xh100_180s` | fast off, meta off, QAT off | `2.14485563` | core branch is still weak |
| `competitive2_core_1xh100_180s_bestckpt` | core branch + periodic val + restore best checkpoint | `2.12227630` | best-checkpoint restore does not rescue the branch |

## What The Clean Runs Proved

The recent clean `competitive2` runs support the following conclusions:

1. The bad score is not caused by export-only corruption.
   - In the best-checkpoint run, the restored checkpoint and final roundtrip result were almost identical.

2. The bad score is not caused only by short-run projection QAT.
   - QAT made things worse, but turning it off did not make the branch good.

3. The bad score is not caused only by fast-adapter meta training.
   - Fast/meta made things worse, but the stripped core branch was still poor.

4. The branch is now infra-clean enough to judge.
   - Wallclock behavior is fixed.
   - Evaluation overhead is under control.
   - Memory usage is reasonable.
   - Throughput is reasonable.

## Current Assessment

The correct current reading is:

- `competitive2` is the main original branch worth learning from
- `competitive2` in its current form is not a viable competitive recipe
- borrowed branches still outperform it by a large margin

This means the next step should not be more small tuning on the same `competitive2` recipe.

## Recommended Direction

Do not continue treating the current `competitive2` recipe as the mainline candidate.

Instead:

1. Keep `competitive2` as an idea source.
2. Build a new original branch from it rather than trying to rescue the current exact recipe.
3. Only carry forward ideas that have a defensible reason to help.

## Open Design Question

The main unresolved design question is what the next original branch should keep from `competitive2`.

Two broad options:

1. Keep the overall family but replace the weak core training recipe.
2. Start a new original branch that borrows only a few structural ideas from `competitive2`.

What looks unsupported by the data so far:

- fast-adapter meta as the main driver
- short-run projection QAT
- assuming the old `~1.56` result represented the true capability of the branch

## Practical Bottom Line

If the goal is originality, `competitive2` is still the most important self-authored branch examined so far.

If the goal is current performance, `competitive2` in its present form is not good enough and should be treated as a failed recipe that needs a new successor rather than more minor tuning.
