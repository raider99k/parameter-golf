# Moonshot RunPod A5000 Experiment Summary

This file summarizes the post-T4 experiments run on RunPod RTX A5000 hardware against `train_gpt_moonshot.py`, `train_gpt_competitive.py`, and `train_gpt_extreme.py`.

All results below come from the interactive commands and logs captured in the working chat thread. They are not reconstructed from local log files.

## Environment

- Hardware: RunPod RTX A5000
- VRAM: `24 GB`
- Dataset: `./data/datasets/fineweb10B_sp1024`
- Tokenizer: `./data/tokenizers/fineweb_1024_bpe.model`
- Common model family:
  - `VOCAB_SIZE=1024`
  - `NUM_LAYERS=2`
  - `NUM_UNIQUE_ATTN=2`
  - `NUM_UNIQUE_MLP=1`
  - `MODEL_DIM=128`
  - `NUM_HEADS=4`
  - `NUM_KV_HEADS=2`
  - `MLP_MULT=2`
  - `TRAIN_SEQ_LEN=64`
  - `TRAIN_BATCH_TOKENS=1024`
- Common training recipe:
  - `USE_FAST_ADAPTERS=1`
  - `FAST_RANK=2`
  - `FAST_GRAD_CLIP=0.1`
  - `FAST_GATE_INIT=0.05`
  - `TRAIN_META_MODE=replace`
  - `TRAIN_META_EVERY=4`
  - `TRAIN_META_START_FRAC=0.35`
  - `TRAIN_META_END_FRAC=1.0`
  - `TRAIN_META_STEPS=1`
  - `TRAIN_META_FIRST_ORDER=1`
  - `TRAIN_META_MIN_SEQS_PER_SIDE=8`
  - `TRAIN_META_LOSS_A_WEIGHT=0.2`
  - `TRAIN_META_LOSS_B_WEIGHT=0.8`
  - `USE_FACTOR_EMBED=1`
  - `EMBED_DIM=96`
  - `TIED_EMBED_LR=0.001`
  - `MATRIX_LR=0.03`
  - `SCALAR_LR=0.04`
  - `GRAD_CLIP_NORM=1.0`
  - `ITERATIONS=1500`
  - `WARMUP_STEPS=5`
  - `VAL_TOKENS_LIMIT=2048`
  - `EVAL_ADAPT=256`
  - `EVAL_SCORE=64`
  - `USE_COMPILE=0`

## Repo State After the Cleanup

The codebase was simplified before moving onto RunPod:

- `train_gpt.py`
  - baseline/reference script
- `train_gpt_moonshot.py`
  - historical research branch
- `train_gpt_competitive.py`
  - cleaned incumbent derived from Moonshot
  - intended staging script for the real leaderboard path
- `train_gpt_extreme.py`
  - aggressive architecture branch for new ideas
- `train_gpt_mod.py`
  - removed as obsolete

The intent of the split is:

- keep the validated competition path stable in `train_gpt_competitive.py`
- keep aggressive architecture work isolated in `train_gpt_extreme.py`

## Hardware Switch: What Changed

The switch from Colab T4 to RunPod A5000 changed the optimization path materially.

Observed A5000 log differences versus T4:

- `low_precision_dtype:bfloat16`
- `sdp_backends ... flash=True`

These are not cosmetic differences. On A5000, the same incumbent recipe trained to a much stronger result than on T4.

Practical implication:

- T4 results remain useful for the historical path and the architecture search that led to the incumbent
- but A5000 became the correct search platform for further work

## Validated A5000 Incumbent

The first important question on A5000 was whether `train_gpt_competitive.py` was only a cleanup of Moonshot or whether it had changed behavior.

That question is now resolved.

### Competitive vs Moonshot on A5000

Using the same incumbent recipe and `SEED=3407`:

| Script | Params | Float `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| `train_gpt_competitive.py` | `278,292` | `2.5026` | `2.5046` | `245,003` bytes |
| `train_gpt_moonshot.py` | `278,292` | `2.5026` | `2.5046` | `262,965` bytes |

Conclusion:

- the two scripts are behaviorally identical for the incumbent recipe on A5000
- the large gain over T4 comes from the hardware/numerics path, not from the competitive cleanup
- `train_gpt_competitive.py` is still the better submission-facing script because it carries less code overhead for the same model result

### Current Best A5000 Incumbent Result

On `train_gpt_competitive.py`, `SEED=3407`:

- float:
  - `val_loss=4.6201`
  - `val_bpb=2.5026`
- final int8 roundtrip:
  - `val_loss=4.6239`
  - `val_bpb=2.5046`

This is the current best fully validated A5000 result in the working thread so far.

## Extreme Branch

`train_gpt_extreme.py` was introduced as a separate branch for pushing the current design harder without contaminating the incumbent.

Implemented ideas:

- NormFormer-lite branch-local output norms
- recurrent reapplication of the shared stack
- per-layer output modulation
- depth-aware residual initialization
- parity-safe dormant hooks for future larger-model work:
  - low-rank shared-block deltas
  - pass-specific modulation
  - pass-specific QK scaling
  - low-rank logit-head delta

### Parity status

The extreme branch originally failed as a clean ablation harness because the all-off path did not exactly match the incumbent.

A later parity fix corrected that:

- extreme all-off, after the fix:
  - float `2.5026`
  - final int8 `2.5046`

This now exactly matches the incumbent A5000 result for `SEED=3407`.

### A5000 extreme results

All runs below used the same incumbent recipe with `SEED=3407`, changing only the `EXTREME_*` controls.

| Variant | Params | Float `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| incumbent `train_gpt_competitive.py` | `278,292` | `2.5026` | `2.5046` | `245,003` bytes |
| extreme sanity, all features off, **before parity fix** | `278,292` | `2.5180` | `2.5188` | `244,894` bytes |
| extreme sanity, all features off, **after parity fix** | `278,292` | `2.5026` | `2.5046` | `245,003` bytes |
| NormFormer-lite only | `278,804` | `2.5336` | `2.5328` | `245,810` bytes |
| recurrence only | `278,548` | `2.5081` | `2.5088` | `245,826` bytes |
| layer modulation + depth-aware residuals | `279,316` | `2.5493` | `2.5509` | `246,885` bytes |
| full extreme stack | `280,084` | `2.5610` | `2.5627` | `248,686` bytes |

Interpretation:

- every currently implemented extreme variation is worse than the incumbent
- the least harmful idea so far is recurrence-only
- NormFormer-lite, modulation, and the full combined stack are clearly negative in this regime
- after the parity fix, the extreme branch is a valid future ablation harness
- however, none of the currently tested extreme features are worth promoting on the present small-model regime

## Scaling the Incumbent

After the extreme branch failed to improve the current tiny incumbent, the search moved to static-backbone scaling on `train_gpt_competitive.py`.

All runs below kept the same winning A5000 recipe and only changed backbone size:

- `TRAIN_META_MODE=replace`
- `TRAIN_META_EVERY=4`
- `TRAIN_META_START_FRAC=0.35`
- `FAST_RANK=2`
- `USE_FACTOR_EMBED=1`
- `USE_COMPILE=0`

### First scaling sweep on `SEED=3407`

| Variant | Params | Float `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| incumbent `2L 128d e96` | `278,292` | `2.5026` | `2.5046` | `245,003` bytes |
| `4L 128d e96` | `279,956` | `2.5281` | `2.5286` | `283,737` bytes |
| `4L 160d e128` | `414,452` | `2.4752` | `2.4786` | `451,456` bytes |
| `6L 160d e128` | `416,532` | `2.4862` | `2.4875` | `540,492` bytes |

Interpretation:

- scaling works
- depth alone did not help: `4L 128d e96` is worse than the incumbent
- depth plus modest width is the winning direction
- more depth is not automatically better: `6L 160d e128` improves over the incumbent, but loses to `4L 160d e128`

### Multi-seed validation for the best scaled line

The strongest scaled line from the first sweep was:

- `NUM_LAYERS=4`
- `NUM_UNIQUE_ATTN=2`
- `NUM_UNIQUE_MLP=1`
- `MODEL_DIM=160`
- `EMBED_DIM=128`

This was then checked across the three main seeds:

| Seed | Float `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---:|---:|---:|---:|
| `1337` | `2.4752` | `2.4769` | `451,889` bytes |
| `2024` | `2.4886` | `2.4921` | `451,492` bytes |
| `3407` | `2.4752` | `2.4786` | `451,456` bytes |

Three-seed mean, final int8 roundtrip:

- `2.4826`

This is a clear improvement over the old A5000 incumbent:

- old incumbent mean:
  - not fully measured across all three seeds on A5000 in the thread
- but even the single-seed incumbent baseline at `3407` was `2.5046`
- the scaled `4L 160d e128` line beats that on all three tested seeds

Best single observed scaled run so far:

- `SEED=1337`
- float `2.4752`
- final int8 `2.4769`

## Current A5000 Main Line

The A5000 main line is now the scaled competitive recipe:

```bash
RUN_ID=competitive_static_B_s3407_4L_160d_e128 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=3407 \
VOCAB_SIZE=1024 \
NUM_LAYERS=4 \
NUM_UNIQUE_ATTN=2 \
NUM_UNIQUE_MLP=1 \
MODEL_DIM=160 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=2 \
USE_FAST_ADAPTERS=1 \
FAST_RANK=2 \
FAST_GRAD_CLIP=0.1 \
FAST_GATE_INIT=0.05 \
TRAIN_META_MODE=replace \
TRAIN_META_EVERY=4 \
TRAIN_META_START_FRAC=0.35 \
TRAIN_META_END_FRAC=1.0 \
TRAIN_META_STEPS=1 \
TRAIN_META_FIRST_ORDER=1 \
TRAIN_META_MIN_SEQS_PER_SIDE=8 \
TRAIN_META_LOSS_A_WEIGHT=0.2 \
TRAIN_META_LOSS_B_WEIGHT=0.8 \
TRAIN_META_SAME_SHARD=0 \
TRAIN_SEQ_LEN=64 \
TRAIN_BATCH_TOKENS=1024 \
ITERATIONS=1500 \
WARMUP_STEPS=5 \
WARMDOWN_ITERS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_TOKENS_LIMIT=2048 \
EVAL_ADAPT=256 \
EVAL_SCORE=64 \
USE_FACTOR_EMBED=1 \
EMBED_DIM=128 \
TIED_EMBED_LR=0.001 \
MATRIX_LR=0.03 \
SCALAR_LR=0.04 \
GRAD_CLIP_NORM=1.0 \
USE_COMPILE=0 \
python train_gpt_competitive.py
```

## Current Conclusions

1. The T4 document should remain separate from the A5000 work.
2. The A5000 changed the optimization path enough that direct T4-to-A5000 metric comparisons should be treated carefully.
3. The original incumbent recipe itself was strong:
   - `train_gpt_competitive.py` and `train_gpt_moonshot.py` match on quality
   - the current validated A5000 incumbent is `2.5046` final int8 roundtrip on `SEED=3407`
4. The competitive cleanup was worthwhile:
   - same model quality
   - lower code-size overhead in the submission artifact
5. The current extreme implementations are not ready for promotion on the current regime:
   - parity is fixed
   - all tested variants are still worse than the incumbent
6. Scaling is now the dominant source of improvement:
   - `4L 160d e128` clearly beats the old `2L 128d e96` incumbent
   - the gain is repeatable across `1337`, `2024`, and `3407`
7. Compression is still not the primary bottleneck:
   - the new scaled main line is still only about `451 KB` int8+zlib
   - this is far below the `16 MB` competition cap
8. The next serious path is therefore:
   - continue scaling from `train_gpt_competitive.py`
   - keep the same `2x1` sharing philosophy unless scaling disproves it
   - use `train_gpt_extreme.py` only as a future ablation branch for larger-model regimes

## Recommended Next Step

The most defensible next move after these A5000 runs is:

1. treat `4L 160d e128` as the new A5000 main line
2. continue probing nearby scaled variants from `train_gpt_competitive.py`
3. stop spending GPU time on the current extreme features in the tiny-model regime
4. only revisit the dormant extreme hooks after scaling creates a new optimization bottleneck

If a local testing helper is kept, it should point first to the incumbent path and only secondarily to the extreme branch.
